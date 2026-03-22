"""Fetch GDELT narrative intensity signals for SMIM from raw GKG 2.0 CSV files.

PRIMARY DATA SOURCE
    GDELT Global Knowledge Graph (GKG) 2.0 raw CSV files.
    Base URL: http://data.gdeltproject.org/gdeltv2/
    Files updated every 15 minutes, publicly accessible, no authentication.

WHY RAW GKG CSV INSTEAD OF DOC API
    The GDELT DOC API has two fatal problems for this backfill:
      1. Rate-limited to ~1 req/5s per IP; consecutive year-by-year absolute-date
         queries for common theme codes trigger 429s lasting hours.
      2. Several theme codes (ECON_ENERGY, ENV_ENERGY, etc.) return empty timelines
         via the DOC API even when present in the raw GKG data. This is a DOC API
         limitation, not a data gap.
    Direct GKG CSV downloads have neither limitation: they are static files served
    by a standard HTTP server with no rate limits and complete raw theme data.

SAMPLING STRATEGY
    For weekly SMIM signals we need one representative GKG snapshot per week.
    We download the GKG file closest to Monday 12:00 UTC for each ISO week.
    GKG files are produced every 15 minutes; we try slots at 120000, 121500,
    123000, 090000, 150000 and take the first that exists (HTTP 200).

    Each GKG file contains ~200-1000 articles processed in that 15-min window.
    A single file is a statistical sample (~0.15%) of the full week, but it is
    sufficient for a stable normalised intensity signal at weekly frequency.
    All signal values are divided by the total document count in that file,
    so they represent fractional shares — not absolute counts.

COVERAGE
    GDELT GKG 2.0 began 2015-02-19.  Files exist from that date forward.
    This script fetches 2015-02-23 (first Monday) through 2025-12-29.

METHODOLOGY
    For each weekly GKG snapshot file:
      1. Parse V2Themes column: split by ";", take part before "," (= code),
         deduplicate per document. Match against basket sets.
      2. Parse V2Organizations column: split by ";", take part before ",",
         normalise (lowercase, collapse whitespace), match alias lists.
      3. Parse V2Tone column: first comma-delimited field = primary tone.
      4. Count distinct DocumentIdentifiers per signal.
      5. article_count = matched docs in file.
      6. total_docs   = total distinct docs in file.
      7. intensity    = article_count / total_docs  (0.0–1.0 fractional share).
      8. avg_tone     = mean tone across matched docs.

    Results cached per week as data/smim/raw/gdelt/gkg_weekly/{YYYY-Www}.parquet.

GKG FIELD REFERENCE (column indices, 0-based, TSV file):
    4  DocumentIdentifier  — article URL
    8  V2Themes            — "CODE,offset;CODE2,offset;..."
    14 V2Organizations     — "OrgName,offset;OrgName2,offset;..."
    15 V2Tone              — "tone,pos,neg,polarity,actref,sgref,wordcount"

Usage:
    uv run python scripts/smim_fetch_gdelt.py
    uv run python scripts/smim_fetch_gdelt.py --start-date 2023-01-01 --end-date 2023-12-31
    uv run python scripts/smim_fetch_gdelt.py --force-refetch
    uv run python scripts/smim_fetch_gdelt.py --workers 8
    uv run python scripts/smim_fetch_gdelt.py --validate-only

Requirements:
    pip install httpx pyarrow pandas
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

# ── Date range ─────────────────────────────────────────────────────────────────
# GDELT GKG 2.0 began 2015-02-19; first Monday is 2015-02-23.
GKG_V2_START  = pd.Timestamp("2015-02-23")   # first Monday with GKG 2.0
DEFAULT_START = pd.Timestamp("2015-02-23")
DEFAULT_END   = pd.Timestamp("2025-12-29")   # last Monday of 2025

# ── GKG file settings ──────────────────────────────────────────────────────────
GKG_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
# 15-minute slots to try for each Monday, in preference order
MONDAY_SLOTS = ["120000", "121500", "123000", "090000", "150000", "180000"]

# GKG TSV column indices (0-based)
COL_DOCID  = 4
COL_THEMES = 8
COL_ORGS   = 14
COL_TONE   = 15

# ── Sector theme baskets ───────────────────────────────────────────────────────
# GKG 2.0 V2EnhancedThemes codes (column 8 in the TSV).
#
# IMPORTANT: The old simple codes (OIL, GAS, TECH, HEALTH, ECON_BANKING, etc.)
# do NOT appear in GKG 2.0 files — the GDELT taxonomy was replaced by a
# hierarchical WB_/ENV_/EPU_/TAX_ scheme starting in Feb 2015.
# The correct codes are those verified from actual GKG 2.0 file inspection.
#
# Codes verified present in GKG 2.0 files (count per ~1300-doc snapshot):
#   ENV_OIL (540), ECON_OILPRICE (120), FUELPRICES (79), ENV_NATURALGAS (54)
#   WB_507_ENERGY_AND_EXTRACTIVES (387)
#   WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES (453)
#   SOC_INNOVATION (149), TECH_AUTOMATION (46), CYBER_ATTACK (20)
#   ECON_STOCKMARKET (425), WB_1920_FINANCIAL_SECTOR_DEVELOPMENT (164)
#   EPU_CATS_FINANCIAL_REGULATION (122), ECON_DEBT (31)
#   GENERAL_HEALTH (572), MEDICAL (552), WB_1350_PHARMACEUTICALS (405)
#   UNGP_HEALTHCARE (331)
#   ECON_INFLATION (55), WB_442_INFLATION (54)
#   WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT (94)

SECTOR_CODES: dict[str, set[str]] = {
    "sector_energy": {
        "ENV_OIL",
        "ECON_OILPRICE",
        "FUELPRICES",
        "ENV_NATURALGAS",
        "WB_507_ENERGY_AND_EXTRACTIVES",
    },
    "sector_technology": {
        "WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES",
        "SOC_INNOVATION",
        "TECH_AUTOMATION",
        "CYBER_ATTACK",
    },
    "sector_financials": {
        "ECON_STOCKMARKET",
        "WB_1920_FINANCIAL_SECTOR_DEVELOPMENT",
        "EPU_CATS_FINANCIAL_REGULATION",
        "ECON_DEBT",
    },
    "sector_healthcare": {
        "GENERAL_HEALTH",
        "MEDICAL",
        "WB_1350_PHARMACEUTICALS",
        "UNGP_HEALTHCARE",
    },
    "sector_macro": {
        "ECON_INFLATION",
        "WB_442_INFLATION",
        "WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT",
    },
}

# ── Actor alias lists ──────────────────────────────────────────────────────────
# Matched case-insensitively as substrings in normalised V2Organizations entries.
# "central bank" excluded from actor_FED — too ambiguous.
ACTOR_ALIASES: dict[str, list[str]] = {
    "actor_FED": [
        "federal reserve",
        "board of governors of the federal reserve",
        "fomc",
    ],
    "actor_SEC": [
        "securities exchange commission",
        "securities and exchange commission",
        "u.s. securities",
        "us securities",
    ],
    "actor_IMF": [
        "international monetary fund",
    ],
    "actor_BOE": [
        "bank of england",
    ],
}

ALL_SIGNALS = list(SECTOR_CODES.keys()) + list(ACTOR_ALIASES.keys())

# ── Output paths ───────────────────────────────────────────────────────────────
RAW_WEEKLY_DIR  = _ROOT / "data" / "smim" / "raw" / "gdelt" / "gkg_weekly"
PROCESSED_PATH  = _ROOT / "data" / "smim" / "processed" / "gdelt_narrative.parquet"
PIT_DIR         = _ROOT / "data" / "smim" / "pit_store"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_client():
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("httpx required: pip install httpx") from exc
    return httpx.Client(timeout=60.0, follow_redirects=True)


def _download_gkg_file(client, monday: pd.Timestamp) -> bytes | None:
    """Download the GKG zip file for the given Monday, trying multiple time slots.

    Returns the raw zip bytes, or None if no slot has a file.
    """
    date_str = monday.strftime("%Y%m%d")
    for slot in MONDAY_SLOTS:
        url = f"{GKG_BASE_URL}/{date_str}{slot}.gkg.csv.zip"
        try:
            resp = client.get(url, timeout=60.0)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception:
            continue
    return None


# ── Parse one GKG file ─────────────────────────────────────────────────────────

def _extract_theme_codes(v2themes: str) -> set[str]:
    """Extract distinct theme codes from V2Themes field.

    V2Themes format: "CODE1,offset1;CODE2,offset2;..."
    Returns set of CODE strings (before the comma).
    """
    if not v2themes or pd.isna(v2themes):
        return set()
    codes: set[str] = set()
    for entry in v2themes.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        code = entry.split(",")[0].strip()
        if code:
            codes.add(code)
    return codes


def _extract_org_names(v2orgs: str) -> list[str]:
    """Extract normalised organisation names from V2Organizations field.

    V2Organizations format: "OrgName,offset;OrgName2,offset;..."
    Returns list of lowercased, whitespace-collapsed names.
    """
    if not v2orgs or pd.isna(v2orgs):
        return []
    names: list[str] = []
    for entry in v2orgs.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name = entry.split(",")[0].strip().lower()
        # Collapse internal whitespace
        name = " ".join(name.split())
        if name:
            names.append(name)
    return names


def _extract_tone(v2tone: str) -> float | None:
    """Extract primary tone scalar from V2Tone field (first CSV value)."""
    if not v2tone or pd.isna(v2tone):
        return None
    try:
        return float(v2tone.split(",")[0])
    except (ValueError, IndexError):
        return None


def _parse_gkg_bytes(data: bytes) -> pd.DataFrame | None:
    """Unzip and parse a GKG CSV zip file.

    Returns a DataFrame with columns: doc_id, themes_set, org_names, tone.
    Returns None if the file cannot be parsed.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as fh:
                raw = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    dtype=str,
                    on_bad_lines="skip",
                    engine="python",
                )
    except Exception as exc:
        log.warning("parse error: %s", exc)
        return None

    if raw.shape[1] <= COL_TONE:
        log.warning("unexpected column count: %d", raw.shape[1])
        return None

    records = []
    for _, row in raw.iterrows():
        doc_id = str(row.iloc[COL_DOCID]).strip() if pd.notna(row.iloc[COL_DOCID]) else None
        if not doc_id or doc_id == "nan":
            continue
        themes = _extract_theme_codes(str(row.iloc[COL_THEMES]))
        orgs   = _extract_org_names(str(row.iloc[COL_ORGS]))
        tone   = _extract_tone(str(row.iloc[COL_TONE]))
        records.append({"doc_id": doc_id, "themes": themes, "orgs": orgs, "tone": tone})

    if not records:
        return None
    return pd.DataFrame(records)


# ── Compute weekly signal statistics from parsed GKG ──────────────────────────

def _compute_week_stats(
    docs: pd.DataFrame,
    sector_codes: dict[str, set[str]],
    actor_aliases: dict[str, list[str]],
) -> dict[str, dict]:
    """Compute per-signal article_count, avg_tone, total_docs for one week's snapshot.

    Returns: {signal_id: {"article_count": int, "avg_tone": float|nan, "total_docs": int}}
    """
    # Deduplicate on doc_id (same URL may appear in the file twice)
    docs = docs.drop_duplicates(subset=["doc_id"])
    total_docs = len(docs)
    if total_docs == 0:
        return {}

    result: dict[str, dict] = {}

    # Sector signals
    for sig, basket in sector_codes.items():
        mask  = docs["themes"].apply(lambda t: bool(t & basket))
        sub   = docs[mask]
        count = len(sub)
        tones = sub["tone"].dropna()
        result[sig] = {
            "article_count": count,
            "avg_tone":      float(tones.mean()) if len(tones) else float("nan"),
            "total_docs":    total_docs,
        }

    # Actor signals
    for sig, aliases in actor_aliases.items():
        def _matches(org_list: list[str]) -> bool:
            for org in org_list:
                for alias in aliases:
                    if alias in org:
                        return True
            return False

        mask  = docs["orgs"].apply(_matches)
        sub   = docs[mask]
        count = len(sub)
        tones = sub["tone"].dropna()
        result[sig] = {
            "article_count": count,
            "avg_tone":      float(tones.mean()) if len(tones) else float("nan"),
            "total_docs":    total_docs,
        }

    return result


# ── Per-week fetch + cache ────────────────────────────────────────────────────

def _week_cache_path(monday: pd.Timestamp) -> Path:
    return RAW_WEEKLY_DIR / f"{monday.strftime('%Y-W%V')}.parquet"


def _load_week_cache(monday: pd.Timestamp) -> pd.DataFrame | None:
    p = _week_cache_path(monday)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_week_cache(monday: pd.Timestamp, stats: dict[str, dict]) -> None:
    if not stats:
        return
    rows = [
        {"signal": sig, **vals}
        for sig, vals in stats.items()
    ]
    RAW_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_week_cache_path(monday), index=False)


def fetch_week(
    monday: pd.Timestamp,
    force: bool = False,
) -> dict[str, dict] | None:
    """Fetch and process one Monday's GKG file. Returns per-signal stats dict.

    Uses on-disk cache to skip already-processed weeks.
    Returns None if the GKG file cannot be downloaded or parsed.
    """
    if not force:
        cached = _load_week_cache(monday)
        if cached is not None and not cached.empty:
            return {
                row["signal"]: {
                    "article_count": row["article_count"],
                    "avg_tone":      row["avg_tone"],
                    "total_docs":    row["total_docs"],
                }
                for _, row in cached.iterrows()
            }

    client = _get_client()
    data   = _download_gkg_file(client, monday)
    if data is None:
        log.warning("  %s — no GKG file found for any Monday slot", monday.date())
        return None

    docs = _parse_gkg_bytes(data)
    if docs is None or docs.empty:
        log.warning("  %s — GKG file parsed to 0 rows", monday.date())
        return None

    stats = _compute_week_stats(docs, SECTOR_CODES, ACTOR_ALIASES)
    _save_week_cache(monday, stats)
    return stats


# ── Build processed DataFrame ─────────────────────────────────────────────────

def build_processed(
    weekly_stats: dict[pd.Timestamp, dict[str, dict]],
    signals: list[str],
) -> pd.DataFrame:
    """Convert {monday: {signal: stats}} → processed long DataFrame.

    Schema:
        theme_or_actor : str
        week_start     : datetime64[ns]
        article_count  : float64  — docs matching signal in that week's GKG snapshot
        avg_tone       : float64  — mean V2Tone[0] of matching docs
        intensity      : float64  — article_count / total_docs (fractional share)
    """
    rows: list[dict] = []
    for monday, sig_stats in weekly_stats.items():
        for sig in signals:
            s = sig_stats.get(sig)
            if s is None:
                continue
            total = s["total_docs"]
            ac    = s["article_count"]
            rows.append({
                "theme_or_actor": sig,
                "week_start":     monday,
                "article_count":  float(ac),
                "avg_tone":       s["avg_tone"],
                "intensity":      float(ac) / total if total > 0 else float("nan"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    for col in ("article_count", "avg_tone", "intensity"):
        df[col] = df[col].astype("float64")
    return (
        df[["theme_or_actor", "week_start", "article_count", "avg_tone", "intensity"]]
        .sort_values(["theme_or_actor", "week_start"])
        .reset_index(drop=True)
    )


# ── PIT ingest ─────────────────────────────────────────────────────────────────

def ingest_to_pit(processed: pd.DataFrame) -> None:
    pit_rows: list[pd.DataFrame] = []
    for metric in ("article_count", "avg_tone", "intensity"):
        chunk = processed[["theme_or_actor", "week_start", metric]].dropna(
            subset=[metric]
        ).copy()
        chunk = chunk.rename(
            columns={"theme_or_actor": "actor_id", "week_start": "event_date",
                     metric: "value"}
        )
        chunk["signal_id"]  = f"gdelt_{metric}"
        chunk["pub_date"]   = chunk["event_date"] + pd.Timedelta(days=7)
        chunk["source"]     = "gdelt"
        chunk["vintage_id"] = None
        pit_rows.append(chunk)
    pit_df = pd.concat(pit_rows, ignore_index=True)
    store  = PointInTimeStore(root_dir=PIT_DIR)
    store.bulk_ingest([pit_df])
    log.info("PIT store: %d rows → %s", len(pit_df), PIT_DIR)


# ── Validate with a quick single-file test ────────────────────────────────────

def run_validation() -> None:
    """Download one recent GKG file and show sample signal values."""
    log.info("Validation: downloading last Monday's GKG file …")
    # Find last Monday
    today  = pd.Timestamp.now().normalize()
    monday = today - pd.Timedelta(days=today.dayofweek)
    if monday == today:
        monday -= pd.Timedelta(weeks=1)

    client = _get_client()
    data   = _download_gkg_file(client, monday)
    if data is None:
        log.warning("Could not download validation file for %s", monday.date())
        return

    docs = _parse_gkg_bytes(data)
    if docs is None:
        log.warning("Could not parse validation file")
        return

    stats = _compute_week_stats(docs, SECTOR_CODES, ACTOR_ALIASES)
    total = next(iter(stats.values()), {}).get("total_docs", 0) if stats else 0
    log.info("Validation file: %d docs (Monday %s)", total, monday.date())
    for sig, s in stats.items():
        log.info(
            "  %-28s  %3d docs  intensity=%.4f  tone=%.2f",
            sig, s["article_count"], s["article_count"] / max(total, 1),
            s["avg_tone"] if not pd.isna(s["avg_tone"]) else 0,
        )


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(processed: pd.DataFrame, n_files: int, n_skipped: int) -> None:
    if processed.empty:
        print("\nNo data.\n")
        return
    fetched = set(processed["theme_or_actor"].unique())
    missing = [s for s in ALL_SIGNALS if s not in fetched]
    print("\n" + "=" * 68)
    print("GDELT GKG 2.0 raw-file narrative signals — summary")
    print("=" * 68)
    print(f"  Source        : {GKG_BASE_URL}")
    print(f"  Method        : 1 GKG file per week (Monday ~noon UTC sample)")
    print(f"  Files fetched : {n_files}  ({n_skipped} weeks skipped / no file)")
    print(f"  Signals       : {len(fetched)} / {len(ALL_SIGNALS)}")
    print(f"  Weekly rows   : {len(processed):,}")
    print(f"  Date range    : {processed['week_start'].min().date()} "
          f"-> {processed['week_start'].max().date()}")
    print(f"  article_count : docs matching signal in that week's snapshot")
    print(f"  intensity     : article_count / total_docs_in_snapshot")
    print(f"  avg_tone      : mean V2Tone[0] of matching docs")
    print(f"  Cache         : {RAW_WEEKLY_DIR}")
    print(f"  Processed     : {PROCESSED_PATH}")
    print(f"  PIT store     : {PIT_DIR}")
    if missing:
        print(f"  WARN missing  : {missing}")
    print("=" * 68 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GDELT GKG 2.0 narrative signals for SMIM "
            "from raw CSV files (free, no rate limits)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START.strftime("%Y-%m-%d"),
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START.date()}.",
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END.strftime("%Y-%m-%d"),
        help=f"End date. Default: {DEFAULT_END.date()}.",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Concurrent download workers. Default: 5.",
    )
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="Ignore per-week cache and re-download everything.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Download last Monday's GKG file and show signal stats. No full fetch.",
    )
    args = parser.parse_args()

    if args.validate_only:
        run_validation()
        return

    start = max(pd.Timestamp(args.start_date), GKG_V2_START)
    end   = pd.Timestamp(args.end_date)

    # Generate all Mondays in range
    mondays = pd.date_range(start=start, end=end, freq="W-MON")
    log.info(
        "Fetching %d weekly GKG snapshots (%s → %s) with %d workers …",
        len(mondays), mondays[0].date(), mondays[-1].date(), args.workers,
    )

    # Count cache hits
    n_cached = sum(
        1 for m in mondays
        if not args.force_refetch and _week_cache_path(m).exists()
    )
    log.info("Cache hits: %d / %d  (downloading %d files)", n_cached, len(mondays),
             len(mondays) - n_cached)

    # Fetch all weeks (parallelised)
    weekly_stats: dict[pd.Timestamp, dict[str, dict]] = {}
    n_skipped = 0
    completed = 0

    def _do_week(m: pd.Timestamp) -> tuple[pd.Timestamp, dict | None]:
        return m, fetch_week(m, force=args.force_refetch)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_week, m): m for m in mondays}
        for fut in as_completed(futures):
            monday, stats = fut.result()
            completed += 1
            if stats is None:
                n_skipped += 1
            else:
                weekly_stats[monday] = stats
            if completed % 50 == 0 or completed == len(mondays):
                log.info(
                    "  Progress: %d / %d weeks  (%d skipped)",
                    completed, len(mondays), n_skipped,
                )

    log.info(
        "Downloaded %d weeks (%d skipped). Building processed signals …",
        len(weekly_stats), n_skipped,
    )

    processed = build_processed(weekly_stats, ALL_SIGNALS)

    if processed.empty:
        log.error("No data — aborting.")
        sys.exit(1)

    # Log coverage
    for sig in sorted(processed["theme_or_actor"].unique()):
        sub = processed[processed["theme_or_actor"] == sig]
        n   = sub["article_count"].notna().sum()
        log.info(
            "  %-28s  %3d weeks  intensity [%.4f–%.4f]  tone [%.1f–%.1f]",
            sig, n,
            sub["intensity"].min(), sub["intensity"].max(),
            sub["avg_tone"].min(),  sub["avg_tone"].max(),
        )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(PROCESSED_PATH, index=False)
    log.info("Saved %d rows → %s", len(processed), PROCESSED_PATH)

    log.info("Ingesting into PIT store …")
    ingest_to_pit(processed)

    print_summary(processed, len(weekly_stats), n_skipped)


if __name__ == "__main__":
    main()
