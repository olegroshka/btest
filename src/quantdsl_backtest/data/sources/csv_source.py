"""
CSV market-bars data source for btest.

URI scheme:
    csv://path/to/file.csv
    csv://path/to/file.csv?ticker_col=symbol&date_col=trade_date

The file must have at least a date column and a close column.
If a ticker/instrument column is present, the file is treated as a
multi-instrument file (long format), identical to the parquet source.
If no ticker column is found, the whole file is treated as a
single-instrument series keyed by the filename stem.

Examples:
    DataConfig(source="csv://data/c40_ohlcv.csv", ...)
    DataConfig(source="csv://data/my_prices.csv?ticker_col=symbol", ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd

from ...dsl.universe import Universe
from ..bundles import MarketBarsBundle
from ..requests import DataRequest, KIND_MARKET_BARS
from .base import CacheStore
from .universe_filtering import UniverseFilterEngine


_DATE_CANDIDATES   = ["datetime", "date", "timestamp", "trade_date", "effectivedate"]
_TICKER_CANDIDATES = ["ticker", "symbol", "ric", "instrument", "secid"]
_OHLCV_CANDIDATES  = ["open", "high", "low", "close", "volume"]


def _parse_csv_uri(source: str):
    """Return (path, ticker_col_override, date_col_override)."""
    raw = source[len("csv://"):]
    if "?" in raw:
        path_part, qs = raw.split("?", 1)
    else:
        path_part, qs = raw, ""

    params = parse_qs(qs)
    ticker_col = (params.get("ticker_col") or [None])[0]
    date_col   = (params.get("date_col")   or [None])[0]
    return path_part, ticker_col, date_col


def _detect_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _normalize_index(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Set the date column as a DatetimeIndex."""
    col = date_col or _detect_col(df, _DATE_CANDIDATES)
    if col and col in df.columns:
        df = df.copy()
        df.index = pd.to_datetime(df[col], errors="coerce")
        df = df.drop(columns=[col])
        df.index.name = "datetime"
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"CSV source: cannot find a date column. "
            f"Columns found: {df.columns.tolist()}. "
            f"Pass date_col=<name> in the URI, e.g. csv://file.csv?date_col=my_date"
        )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


@dataclass(slots=True)
class CsvMarketBarsSource:
    """Load OHLCV market bars from a CSV file.

    URI:  csv://relative/or/absolute/path.csv[?ticker_col=x&date_col=y]

    - Long-format CSV (multiple tickers in one file): needs a ticker column.
    - Single-instrument CSV: no ticker column needed; instrument name = filename stem.
    """

    name: str = "csv_market_bars"
    universe_filters: UniverseFilterEngine = field(default_factory=UniverseFilterEngine)

    def can_load(self, request: DataRequest) -> bool:
        return request.kind == KIND_MARKET_BARS and request.source.lower().startswith("csv://")

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> MarketBarsBundle:
        path, ticker_col_override, date_col_override = _parse_csv_uri(request.source)

        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV data file not found: {path!r}")

        table = pd.read_csv(path)
        table = _normalize_index(table, date_col_override)

        # Filter date range
        if request.start:
            table = table.loc[request.start:]
        if request.end:
            table = table.loc[:request.end]

        # Detect ticker column
        id_col = ticker_col_override or (
            universe.id_field if universe is not None and hasattr(universe, "id_field") else None
        ) or _detect_col(table, _TICKER_CANDIDATES)

        # Filter by static_instruments if universe specifies them
        if id_col and id_col in table.columns:
            if universe is not None and universe.static_instruments:
                table = table[table[id_col].isin(universe.static_instruments)]

        bars_by_instr: Dict[str, pd.DataFrame] = {}
        instruments: List[str] = []
        filters = universe.filters if universe is not None else []

        if id_col and id_col in table.columns:
            # multi-instrument long-format file
            for instr, sub in table.groupby(id_col):
                instr = str(instr)
                sub = sub.drop(columns=[id_col]).sort_index()
                sub.index = pd.to_datetime(sub.index).tz_localize(None)

                if filters and not self.universe_filters.passes(sub, filters):
                    continue

                bars_by_instr[instr] = sub
                instruments.append(instr)
        else:
            # single-instrument file — use filename stem as instrument name
            instr = os.path.splitext(os.path.basename(path))[0]
            df = table.sort_index()
            df.index = pd.to_datetime(df.index).tz_localize(None)

            if not filters or self.universe_filters.passes(df, filters):
                bars_by_instr[instr] = df
                instruments.append(instr)

        return MarketBarsBundle(
            kind=KIND_MARKET_BARS,
            source=request.source,
            start=request.start,
            end=request.end,
            frequency=request.frequency,
            calendar=getattr(request, "calendar", None),
            tz=getattr(request, "tz", None),
            bars=bars_by_instr,
            instruments=instruments,
            fields=list(request.fields or []),
        )
