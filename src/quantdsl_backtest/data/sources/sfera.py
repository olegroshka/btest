"""
Sfera PostgreSQL data-source adapter for btest.

One URI scheme for everything:

    sfera://schema/table[?date_col=x&ticker_col=y&deprecated_filter=0]

The adapter auto-detects what to return based on DataConfig.kind:
  - kind="market_bars"  → MarketBarsBundle  (OHLCV price data)
  - kind="timeseries"   → TimeSeriesBundle  (any other data: macro, yields, events…)

Sfera is one database — no need for a separate URI prefix.

Examples:
    DataConfig(source="sfera://bbgidx/index_prices",       frequency="1d")
    DataConfig(source="sfera://mxbdprc/bond_market_data",  frequency="1d")
    DataConfig(source="sfera://ecocal/events?deprecated_filter=0")

URI options:
    date_col=<col>       Date/datetime column name  (default: auto-detect)
    ticker_col=<col>     Instrument/ticker column    (default: auto-detect)
    deprecated_filter=1  Add WHERE deprecated_at IS NULL  (default: 1)

Backward-compat:  sfera-bars://schema/table  still accepted (treated as
                  sfera:// with kind=market_bars).

Configuration
─────────────
All connection credentials come from sfera-db (env vars or .env file):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

No credentials are hardcoded here — safe to push to public repos.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd

from ...dsl.universe import Universe
from ..bundles import DataBundle, MarketBarsBundle, TimeSeriesBundle
from ..requests import DataRequest, KIND_MARKET_BARS, KIND_TIME_SERIES
from .base import CacheStore

# ── sfera-db resolution ──────────────────────────────────────────────────────
# sfera-db lives at workspace root/sfera-db — add to path if not already there.

def _ensure_sfera_db_on_path() -> None:
    here = Path(__file__).resolve()
    # Walk up to find workspace root (contains sfera-db/)
    for parent in here.parents:
        candidate = parent / "sfera-db"
        if candidate.is_dir():
            p = str(candidate)
            if p not in sys.path:
                sys.path.insert(0, p)
            return

_ensure_sfera_db_on_path()

# sfera DB client lives in the sfera repo as `connector` (pip-installed; exposes columns()/query()/index_prices()).
import connector as _sfera_db


def _require_sfera_db() -> Any:
    """Return the optional sfera-db module or raise a clear runtime error.

    This keeps non-Sfera providers importable in environments that do not have
    the local `sfera-db` workspace available, while preserving the existing
    error when a Sfera-backed request is actually used.
    """

    if _sfera_db is None:
        raise ImportError(
            "sfera-db package not found. Expected at <workspace>/sfera-db/. "
            "Check your workspace layout."
        )
    return _sfera_db


# ── URI parsing ──────────────────────────────────────────────────────────────

SFERA_PREFIX      = "sfera://"
SFERA_BARS_PREFIX = "sfera-bars://"

_OHLCV_CANDIDATES = ["open", "high", "low", "close", "volume"]
_DATE_CANDIDATES  = ["tradedate", "trade_date", "date", "datetime", "timestamp",
                     "effectivedate", "settlementdate"]
_TICKER_CANDIDATES = ["ticker", "secid", "isin", "instrument", "symbol",
                      "boardid", "entity_id"]


@dataclass(frozen=True)
class _ParsedUri:
    schema: str       # postgres schema name
    table: str        # table name
    date_col: Optional[str]    # explicit override
    ticker_col: Optional[str]  # explicit override
    deprecated_filter: bool


def _parse_uri(source: str) -> _ParsedUri:
    """Parse  sfera[‑bars]://schema/table[?date_col=x&ticker_col=y]"""
    raw = source
    for prefix in (SFERA_BARS_PREFIX, SFERA_PREFIX):
        if source.lower().startswith(prefix):
            source = source[len(prefix):]
            break

    # parse optional query string
    if "?" in source:
        path_part, qs = source.split("?", 1)
    else:
        path_part, qs = source, ""

    parts = path_part.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"sfera URI must be sfera://schema/table, got {raw!r}"
        )
    schema, table = parts[0], parts[1]

    params = parse_qs(qs)
    date_col   = (params.get("date_col")   or [None])[0]
    ticker_col = (params.get("ticker_col") or [None])[0]
    dep_filter = (params.get("deprecated_filter") or ["1"])[0] != "0"

    return _ParsedUri(schema=schema, table=table,
                      date_col=date_col, ticker_col=ticker_col,
                      deprecated_filter=dep_filter)


# ── column auto-detection ────────────────────────────────────────────────────

def _detect_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all non-datetime object columns to numeric where possible."""
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # leave non-numeric columns as-is
    return df


# ── shared fetch ─────────────────────────────────────────────────────────────

def _fetch_raw(uri: _ParsedUri, start: str, end: str) -> pd.DataFrame:
    """Pull rows from sfera within [start, end] on the detected/declared date col."""
    sfera_db = _require_sfera_db()

    # First get column info so we can build the WHERE clause correctly
    cols_df = sfera_db.columns(table=uri.table, schema=uri.schema)
    if cols_df.empty:
        raise ValueError(
            f"Table {uri.schema}.{uri.table} not found in sfera. "
            f"Run sfera_db.tables(schema='{uri.schema}') to list available tables."
        )

    all_cols = cols_df["column_name"].tolist()

    # determine date column
    if uri.date_col:
        date_col = uri.date_col
    else:
        # auto-detect from known candidates
        lower_map = {c.lower(): c for c in all_cols}
        date_col = next(
            (lower_map[c.lower()] for c in _DATE_CANDIDATES if c.lower() in lower_map),
            None,
        )
    if date_col is None:
        raise ValueError(
            f"Cannot detect a date column in {uri.schema}.{uri.table}. "
            f"Columns found: {all_cols}. "
            f"Pass date_col=<name> in the URI, e.g. sfera://{uri.schema}/{uri.table}?date_col=my_date"
        )

    clauses: list[str] = []
    params:  list     = []
    if start:
        clauses.append(f"{date_col} >= %s")
        params.append(start)
    if end:
        clauses.append(f"{date_col} <= %s")
        params.append(end)
    if uri.deprecated_filter and "deprecated_at" in all_cols:
        clauses.append("deprecated_at IS NULL")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql   = f'SELECT * FROM {uri.schema}."{uri.table}" {where} ORDER BY {date_col}'

    df = sfera_db.query(sql, params or None)
    if df.empty:
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)
    df.index.name = "date"
    df = _coerce_numeric(df)
    return df


# ── Unified Sfera adapter ────────────────────────────────────────────────────

@dataclass(slots=True)
class SferaSource:
    """
    Single adapter for all Sfera PostgreSQL tables.

    URI:  sfera://schema/table[?date_col=x&ticker_col=y&deprecated_filter=0]

    The return type is determined by DataConfig.kind (from the strategy):
      - kind="market_bars"  → MarketBarsBundle  (OHLCV price data)
      - kind="timeseries"   → TimeSeriesBundle  (macro, yields, events, …)

    Sfera is one database — one URI scheme is enough.  The old sfera-bars://
    prefix is still accepted for backward compatibility.

    OHLCV column aliases are normalised automatically:
        open_price → open,  close_price → close, etc.

    Examples:
        DataConfig(source="sfera://bbgidx/index_prices",      kind="market_bars")
        DataConfig(source="sfera://mxbdprc/bond_market_data", kind="timeseries")
    """

    name: str = "sfera"

    def can_load(self, request: DataRequest) -> bool:
        src = request.source.lower()
        return src.startswith(SFERA_PREFIX) or src.startswith(SFERA_BARS_PREFIX)

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> DataBundle:
        uri = _parse_uri(request.source)
        df  = _fetch_raw(uri, request.start, request.end)

        # Determine intent from request.kind; sfera-bars:// always means market_bars
        want_bars = (
            request.kind == KIND_MARKET_BARS
            or request.source.lower().startswith(SFERA_BARS_PREFIX)
        )

        # ── common: filter by universe instruments ───────────────────────────
        instr_col = uri.ticker_col or _detect_col(df, _TICKER_CANDIDATES)
        entities_filter = (
            universe.static_instruments
            if universe is not None and universe.static_instruments
            else None
        )
        if instr_col and df is not None and not df.empty and entities_filter:
            df = df[df[instr_col].isin(entities_filter)]

        if df.empty:
            if want_bars:
                return MarketBarsBundle(
                    kind=KIND_MARKET_BARS, source=request.source,
                    start=request.start, end=request.end,
                    frequency=request.frequency,
                    bars={}, instruments=[], fields=[],
                )
            return TimeSeriesBundle(
                kind=KIND_TIME_SERIES, source=request.source,
                start=request.start, end=request.end,
                frequency=request.frequency,
                frames={}, entities=[], fields=[],
            )

        # ── market bars path ─────────────────────────────────────────────────
        if want_bars:
            # normalise OHLCV column names (also handles Bloomberg-style aliases)
            col_map: Dict[str, str] = {}
            lower = {c.lower(): c for c in df.columns}
            for std in _OHLCV_CANDIDATES:
                for alias in [std, f"{std}_price", f"{std[0]}_{std[1:]}"]:
                    if alias in lower:
                        col_map[lower[alias]] = std
                        break
            df = df.rename(columns=col_map)

            if "close" not in df.columns:
                raise ValueError(
                    f"sfera table {uri.schema}.{uri.table} has no close-price column — "
                    f"cannot load as market_bars. Columns found: {list(df.columns)}. "
                    f"Use kind='timeseries' if this is not OHLCV data."
                )

            bars: Dict[str, pd.DataFrame] = {}
            ohlcv_cols = [c for c in _OHLCV_CANDIDATES if c in df.columns]

            if instr_col and instr_col in df.columns:
                for instr, sub in df.groupby(instr_col):
                    sub = sub.drop(columns=[instr_col])
                    keep  = [c for c in ohlcv_cols if c in sub.columns]
                    extra = [c for c in sub.columns if c not in keep]
                    bars[str(instr)] = sub[keep + extra]
            else:
                keep  = [c for c in ohlcv_cols if c in df.columns]
                extra = [c for c in df.columns if c not in keep]
                bars[uri.table] = df[keep + extra]

            instruments = list(bars.keys())
            fields = list(next(iter(bars.values())).columns) if bars else []

            return MarketBarsBundle(
                kind=KIND_MARKET_BARS, source=request.source,
                start=request.start, end=request.end,
                frequency=request.frequency,
                calendar=getattr(request, "calendar", None),
                tz=getattr(request, "tz", None),
                bars=bars, instruments=instruments, fields=fields,
            )

        # ── time-series path ─────────────────────────────────────────────────
        frames: Dict[str, pd.DataFrame] = {}
        if instr_col and instr_col in df.columns:
            for entity, sub in df.groupby(instr_col):
                sub = sub.drop(columns=[instr_col])
                if request.fields:
                    keep = [c for c in request.fields if c in sub.columns]
                    sub = sub[keep] if keep else sub
                frames[str(entity)] = sub
        else:
            frames[f"{uri.schema}/{uri.table}"] = df

        entities = list(frames.keys())
        fields   = list(next(iter(frames.values())).columns) if frames else []

        return TimeSeriesBundle(
            kind=KIND_TIME_SERIES, source=request.source,
            start=request.start, end=request.end,
            frequency=request.frequency,
            frames=frames, entities=entities, fields=fields,
        )


# ── Backward-compat aliases (import name unchanged) ──────────────────────────
SferaMarketBarsSource  = SferaSource   # was: sfera-bars:// only
SferaTimeSeriesSource  = SferaSource   # was: sfera:// only (timeseries)
