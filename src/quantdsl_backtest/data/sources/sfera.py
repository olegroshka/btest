"""
Sfera PostgreSQL data-source adapters for btest.

Two URI schemes:

    sfera://schema/table
        Generic adapter — returns TimeSeriesBundle.
        Works with ANY table in sfera: bond yields, macro, spreads,
        event data, intelligence findings — anything with a date column.

        Example:
            DataConfig(source="sfera://mxbdprc/bond_market_data",
                       frequency="1d")

    sfera-bars://schema/table
        Price adapter — returns MarketBarsBundle (OHLCV shape).
        Use only when the table has open/high/low/close/volume columns
        (or close at minimum), and a ticker/instrument column.

        Example:
            DataConfig(source="sfera-bars://bbgidx/index_prices",
                       frequency="1d")

Configuration
─────────────
All connection credentials come from sfera-db (env vars or .env file):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

No credentials are hardcoded here — safe to push to public repos.

URI options (appended as query-string style params after '?'):
    date_col=<col>      Date/datetime column name  (default: auto-detect)
    ticker_col=<col>    Instrument/ticker column    (default: auto-detect)
    deprecated_filter=1 Add WHERE deprecated_at IS NULL  (default: 1)

Examples with options:
    sfera://mxbdprc/bond_yields?date_col=tradedate&deprecated_filter=1
    sfera-bars://bbgidx/index_prices?ticker_col=ticker&date_col=trade_date
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
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

try:
    import sfera_db
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sfera-db package not found. Expected at <workspace>/sfera-db/. "
        "Check your workspace layout."
    ) from exc


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
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# ── shared fetch ─────────────────────────────────────────────────────────────

def _fetch_raw(uri: _ParsedUri, start: str, end: str) -> pd.DataFrame:
    """Pull rows from sfera within [start, end] on the detected/declared date col."""
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


# ── TimeSeriesBundle adapter (generic — any sfera table) ─────────────────────

@dataclass(slots=True)
class SferaTimeSeriesSource:
    """
    Generic sfera adapter — returns a TimeSeriesBundle.

    URI:  sfera://schema/table[?date_col=x&ticker_col=y&deprecated_filter=0]

    If a ticker/instrument column is detected the data is split per instrument.
    If not (e.g. pure macro series), the whole table is returned as one entity
    keyed by "schema/table".

    Works for:
        • Bond yields / spreads   (mxbdprc.bond_yields, mxbdprc.bond_market_data)
        • Macro / econ calendar   (tvcal.*, bbgidx.*)
        • Intelligence findings   (intelligence.findings)
        • Any future sfera table
    """

    name: str = "sfera_timeseries"

    def can_load(self, request: DataRequest) -> bool:
        src = request.source.lower()
        return src.startswith(SFERA_PREFIX) and not src.startswith(SFERA_BARS_PREFIX)

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> TimeSeriesBundle:
        uri = _parse_uri(request.source)
        df  = _fetch_raw(uri, request.start, request.end)

        if df.empty:
            return TimeSeriesBundle(
                kind=KIND_TIME_SERIES, source=request.source,
                start=request.start, end=request.end,
                frequency=request.frequency,
                frames={}, entities=[], fields=[],
            )

        # apply universe instrument filter if provided
        instr_col = uri.ticker_col or _detect_col(df, _TICKER_CANDIDATES)
        entities_filter = (
            universe.static_instruments
            if universe is not None and universe.static_instruments
            else None
        )
        if instr_col and entities_filter:
            df = df[df[instr_col].isin(entities_filter)]

        # split into per-entity frames (or return whole table as one entity)
        frames: Dict[str, pd.DataFrame] = {}
        if instr_col and instr_col in df.columns:
            for entity, sub in df.groupby(instr_col):
                sub = sub.drop(columns=[instr_col])
                if request.fields:
                    keep = [c for c in request.fields if c in sub.columns]
                    sub = sub[keep] if keep else sub
                frames[str(entity)] = sub
        else:
            entity_key = f"{uri.schema}/{uri.table}"
            frames[entity_key] = df

        entities = list(frames.keys())
        fields   = list(next(iter(frames.values())).columns) if frames else []

        return TimeSeriesBundle(
            kind=KIND_TIME_SERIES, source=request.source,
            start=request.start, end=request.end,
            frequency=request.frequency,
            frames=frames, entities=entities, fields=fields,
        )


# ── MarketBarsBundle adapter (OHLCV tables only) ─────────────────────────────

@dataclass(slots=True)
class SferaMarketBarsSource:
    """
    OHLCV-specific sfera adapter — returns a MarketBarsBundle.

    URI:  sfera-bars://schema/table[?ticker_col=x&date_col=y]

    The table must have at minimum a 'close' column.
    Columns are mapped:  open/high/low/close/volume  (case-insensitive).

    Use for:
        • bbgidx.index_prices    (Bloomberg index OHLCV)
        • Any future equity price table added to sfera
    """

    name: str = "sfera_market_bars"

    def can_load(self, request: DataRequest) -> bool:
        return (request.kind == KIND_MARKET_BARS
                and request.source.lower().startswith(SFERA_BARS_PREFIX))

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> MarketBarsBundle:
        uri = _parse_uri(request.source)
        df  = _fetch_raw(uri, request.start, request.end)

        if df.empty:
            return MarketBarsBundle(
                kind=KIND_MARKET_BARS, source=request.source,
                start=request.start, end=request.end,
                frequency=request.frequency,
                bars={}, instruments=[], fields=[],
            )

        # detect ticker column
        instr_col = uri.ticker_col or _detect_col(df, _TICKER_CANDIDATES)
        entities_filter = (
            universe.static_instruments
            if universe is not None and universe.static_instruments
            else None
        )
        if instr_col and entities_filter:
            df = df[df[instr_col].isin(entities_filter)]

        # normalise OHLCV column names (close is mandatory, rest optional)
        col_map: Dict[str, str] = {}
        lower = {c.lower(): c for c in df.columns}
        for std in _OHLCV_CANDIDATES:
            # also handle Bloomberg-style aliases: close_price → close, etc.
            for alias in [std, f"{std}_price", f"{std[0]}_{std[1:]}"]:
                if alias in lower:
                    col_map[lower[alias]] = std
                    break
        df = df.rename(columns=col_map)

        if "close" not in df.columns:
            raise ValueError(
                f"sfera-bars table {uri.schema}.{uri.table} has no close-price column. "
                f"Columns found: {list(df.columns)}. "
                f"Use sfera:// (TimeSeriesBundle) instead if this is not price data."
            )

        bars: Dict[str, pd.DataFrame] = {}
        ohlcv_cols = [c for c in _OHLCV_CANDIDATES if c in df.columns]

        if instr_col and instr_col in df.columns:
            for instr, sub in df.groupby(instr_col):
                sub = sub.drop(columns=[instr_col])
                keep = [c for c in ohlcv_cols if c in sub.columns]
                extra = [c for c in sub.columns if c not in keep]
                bars[str(instr)] = sub[keep + extra]
        else:
            # single-instrument table — use table name as key
            keep = [c for c in ohlcv_cols if c in df.columns]
            extra = [c for c in df.columns if c not in keep]
            bars[uri.table] = df[keep + extra]

        instruments = list(bars.keys())
        fields = list(next(iter(bars.values())).columns) if bars else []

        return MarketBarsBundle(
            kind=KIND_MARKET_BARS, source=request.source,
            start=request.start, end=request.end,
            frequency=request.frequency, calendar=request.calendar, tz=request.tz,
            bars=bars, instruments=instruments, fields=fields,
        )
