"""Fetch GDELT narrative intensity signals for sectors and institutions.

Usage:
    uv run python scripts/smim_fetch_gdelt.py

No API key required — GDELT is public.

Fetches theme-level and actor-level article counts and tone from the GDELT
DOC API v2 Timeline endpoint, aggregated to weekly frequency.

GDELT data is append-only — no revisions — so pub_date = week_start + 7 days
(data is complete and public by end of the week).

Outputs:
    data/smim/raw/gdelt/<query_id>.parquet        — raw per-query daily data
    data/smim/processed/gdelt_narrative.parquet   — wide weekly table
    data/smim/pit_store/gdelt.parquet             — PIT store (long format)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from quantdsl_backtest.smim.data.pit_store import PointInTimeStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Query definitions ─────────────────────────────────────────────────────────

# Sector themes: each value is a list of GDELT theme codes.
# Each code is queried as `theme:<CODE>` and results are OR-merged per sector.
SECTOR_THEMES: dict[str, list[str]] = {
    "energy": ["ECON_ENERGY", "ENV_ENERGY", "FUEL", "OIL", "GAS"],
    "technology": ["TECH", "CYBER", "AI", "DIGITAL"],
    "financials": ["ECON_BANKING", "ECON_INTEREST_RATE", "FINANCIAL"],
    "healthcare": ["HEALTH", "PHARMA", "MEDICAL"],
    "macro": ["ECON_INFLATION", "ECON_UNEMPLOYMENT", "ECON_GDP"],
}

# Institutional actor queries: free-text keyword searches.
ACTOR_QUERIES: dict[str, str] = {
    "FED": '"federal reserve" OR "central bank"',
    "SEC": '"securities and exchange"',
    "IMF": '"international monetary fund"',
    "BOE": '"bank of england"',
}

# Date range — GDELT v2 DOC API has reliable coverage from 2015
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# GDELT API
_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DT_FMT = "%Y%m%d%H%M%S"
_REQUEST_DELAY_S = 2.0

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_DIR = _ROOT / "data" / "smim" / "raw" / "gdelt"
PROCESSED_PATH = _ROOT / "data" / "smim" / "processed" / "gdelt_narrative.parquet"
PIT_DIR = _ROOT / "data" / "smim" / "pit_store"


# ── GDELT API helpers ─────────────────────────────────────────────────────────

def _gdelt_dt(ts: pd.Timestamp) -> str:
    return ts.strftime(_DT_FMT)


def _fetch_timeline(
    client: object,
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mode: str,
) -> pd.Series | None:
    """Fetch one GDELT timeline series (volume or tone).

    Returns a daily pd.Series indexed by date, or None on failure.
    """
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": _gdelt_dt(start),
        "enddatetime": _gdelt_dt(end),
        "timelinesmooth": "0",
    }
    url = f"{_BASE_URL}?{urlencode(params)}"
    try:
        resp = client.get(url, timeout=60.0)  # type: ignore[attr-defined]
    except Exception as exc:
        log.warning("GDELT request failed (%s mode=%s): %s", query[:50], mode, exc)
        return None

    if resp.status_code != 200:
        log.warning(
            "GDELT HTTP %d for query=%r mode=%s", resp.status_code, query[:50], mode
        )
        return None

    try:
        data = resp.json()
    except Exception as exc:
        log.warning("GDELT JSON parse error (mode=%s): %s", mode, exc)
        return None

    # Response shape: {"timeline": [{"series": [{"date": "...", "value": ...}, ...]}]}
    # or {"timeline": [{"date": "...", "value": ...}, ...]}
    timeline = data.get("timeline") or []
    if not timeline:
        return None

    first = timeline[0]
    if isinstance(first, dict) and "series" in first:
        rows = first["series"]
    else:
        rows = timeline

    records: list[dict] = []
    for row in rows:
        dt_str = row.get("date", "")
        val = row.get("value")
        if dt_str and val is not None:
            try:
                dt = pd.to_datetime(dt_str, format=_DT_FMT)
            except ValueError:
                try:
                    dt = pd.to_datetime(dt_str)
                except ValueError:
                    continue
            records.append({"date": dt.normalize(), "value": float(val)})

    if not records:
        return None

    s = (
        pd.DataFrame(records)
        .set_index("date")["value"]
        .resample("D")
        .mean()
    )
    return s


def _fetch_query(
    client: object,
    query_id: str,
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch volume + tone for one query; aggregate to weekly.

    Returns a DataFrame with columns: article_count, avg_tone, intensity
    indexed by week_start (Monday).  Returns None if volume fetch fails.
    """
    log.info("  vol   %s", query_id)
    vol = _fetch_timeline(client, query, start, end, mode="timelinevol")
    time.sleep(_REQUEST_DELAY_S)

    log.info("  tone  %s", query_id)
    tone = _fetch_timeline(client, query, start, end, mode="timelinecovtone")
    time.sleep(_REQUEST_DELAY_S)

    if vol is None:
        log.warning("No volume data for %s — skipping", query_id)
        return None

    # Resample daily → weekly (week starting Monday)
    # volume: sum over the week; tone: mean over the week
    weekly_vol = vol.resample("W-MON", label="left", closed="left").sum()
    weekly_tone = (
        tone.resample("W-MON", label="left", closed="left").mean()
        if tone is not None
        else pd.Series(dtype="float64", name="avg_tone")
    )

    df = pd.DataFrame(
        {
            "article_count": weekly_vol,
            "avg_tone": weekly_tone,
        }
    )
    # intensity = article_count normalised to [0,1] range across the series
    # (timelinevol already returns % of all GDELT articles — this IS the intensity)
    df["intensity"] = df["article_count"]
    df.index.name = "week_start"
    return df.dropna(subset=["article_count"])


# ── Step 1: Fetch all queries ──────────────────────────────────────────────────

def fetch_all(client: object) -> pd.DataFrame:
    """Fetch all sector themes and actor queries; return combined daily-to-weekly frame.

    Returns a long-form DataFrame with columns:
        theme_or_actor, week_start, article_count, avg_tone, intensity
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE).replace(hour=23, minute=59, second=59)

    frames: list[pd.DataFrame] = []
    zero_coverage: list[str] = []

    # ── Sector themes ──────────────────────────────────────────────────────────
    for sector, theme_codes in SECTOR_THEMES.items():
        query_id = f"sector_{sector}"
        log.info("[theme] %s — codes: %s", sector, theme_codes)
        # Build a compound OR query for all theme codes in the sector
        query = " OR ".join(f"theme:{code}" for code in theme_codes)
        df = _fetch_query(client, query_id, query, start, end)

        if df is None or df.empty:
            zero_coverage.append(query_id)
            log.warning("ZERO COVERAGE for sector theme: %s", sector)
            continue

        raw_path = RAW_DIR / f"{query_id}.parquet"
        df_raw = df.copy()
        df_raw["theme_or_actor"] = query_id
        df_raw.reset_index().to_parquet(raw_path, index=False)

        df["theme_or_actor"] = query_id
        frames.append(df.reset_index())

    # ── Institutional actor queries ───────────────────────────────────────────
    for actor_id, kw_query in ACTOR_QUERIES.items():
        query_id = f"actor_{actor_id}"
        log.info("[actor] %s — query: %s", actor_id, kw_query)
        df = _fetch_query(client, query_id, kw_query, start, end)

        if df is None or df.empty:
            zero_coverage.append(query_id)
            log.warning("ZERO COVERAGE for actor: %s", actor_id)
            continue

        raw_path = RAW_DIR / f"{query_id}.parquet"
        df_raw = df.copy()
        df_raw["theme_or_actor"] = query_id
        df_raw.reset_index().to_parquet(raw_path, index=False)

        df["theme_or_actor"] = query_id
        frames.append(df.reset_index())

    if zero_coverage:
        log.warning("Queries with zero coverage: %s", zero_coverage)

    if not frames:
        log.error("No data retrieved from GDELT.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"])
    combined["article_count"] = combined["article_count"].astype("float64")
    combined["avg_tone"] = combined["avg_tone"].astype("float64")
    combined["intensity"] = combined["intensity"].astype("float64")
    return combined[["theme_or_actor", "week_start", "article_count", "avg_tone", "intensity"]]


# ── Step 3: PIT ingest ────────────────────────────────────────────────────────

def ingest_to_pit(processed: pd.DataFrame) -> None:
    """Reshape wide processed frame to long PIT format and ingest.

    pub_date = week_start + 7 days (data complete and public at end of week).
    Stores three signals per (actor, week): article_count, avg_tone, intensity.
    """
    pit_rows: list[pd.DataFrame] = []

    for metric in ("article_count", "avg_tone", "intensity"):
        chunk = processed[["theme_or_actor", "week_start", metric]].copy()
        chunk = chunk.rename(
            columns={"theme_or_actor": "actor_id", "week_start": "event_date", metric: "value"}
        )
        chunk["signal_id"] = f"gdelt_{metric}"
        chunk["pub_date"] = chunk["event_date"] + pd.Timedelta(days=7)
        chunk["source"] = "gdelt"
        chunk["vintage_id"] = None
        pit_rows.append(chunk)

    pit_df = pd.concat(pit_rows, ignore_index=True)

    store = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store updated at %s", PIT_DIR)


# ── Step 4: Summary ───────────────────────────────────────────────────────────

def print_summary(processed: pd.DataFrame) -> None:
    if processed.empty:
        print("\nNo GDELT data retrieved.\n")
        return

    n_queries = processed["theme_or_actor"].nunique()
    total_obs = len(processed)
    date_min = processed["week_start"].min().date()
    date_max = processed["week_start"].max().date()

    zero = [
        qid
        for qid in (
            [f"sector_{s}" for s in SECTOR_THEMES]
            + [f"actor_{a}" for a in ACTOR_QUERIES]
        )
        if qid not in processed["theme_or_actor"].values
    ]

    print("\n" + "=" * 60)
    print("GDELT narrative signals — fetch summary")
    print("=" * 60)
    print(f"  Themes/actors fetched       : {n_queries}")
    print(f"  Total weekly observations   : {total_obs:,}")
    print(f"  Date range                  : {date_min} to {date_max}")
    print(f"  Processed parquet           : {PROCESSED_PATH}")
    print(f"  PIT store                   : {PIT_DIR}")
    if zero:
        print(f"  WARN — zero coverage        : {zero}")
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required. Install with: pip install httpx"
        ) from exc

    client = httpx.Client(timeout=60.0)

    total_queries = len(SECTOR_THEMES) + len(ACTOR_QUERIES)
    log.info(
        "Step 1: Fetching %d sector themes + %d actor queries from GDELT …",
        len(SECTOR_THEMES),
        len(ACTOR_QUERIES),
    )
    log.info(
        "  %d queries × 2 API calls × %.0fs delay ≈ %.0f s estimated",
        total_queries,
        _REQUEST_DELAY_S,
        total_queries * 2 * _REQUEST_DELAY_S,
    )
    processed = fetch_all(client)

    if processed.empty:
        log.error("No GDELT data retrieved — aborting.")
        sys.exit(1)

    # Step 2: Save processed parquet
    log.info("Step 2: Saving processed parquet …")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(PROCESSED_PATH, index=False)
    log.info("Saved %d weekly rows to %s", len(processed), PROCESSED_PATH)

    # Step 3: PIT ingest
    log.info("Step 3: Ingesting into PIT store …")
    ingest_to_pit(processed)

    # Step 4: Summary
    print_summary(processed)


if __name__ == "__main__":
    main()
