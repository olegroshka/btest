# src/quantdsl_backtest/data/adapters.py

from __future__ import annotations

import os
from typing import Optional, Dict, List, Tuple

import pandas as pd
import vectorbt as vbt  # NEW: for Yahoo / YFData support

from ..dsl.data_config import DataConfig
from ..dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV, UniverseFilter
from .schema import MarketData, InstrumentId
from .market import fetch_fred_series  # FRED helper (uses fredapi)
from .cache_arctic import (
    get_cache_lib,
    cache_has_symbol,
    cache_read_symbol,
    cache_write_symbol,
)


YF_PREFIX = "yf://"
FRED_PREFIX = "fred://"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_market_data(
    data_cfg: DataConfig,
    universe: Optional[Universe] = None,
) -> MarketData:
    """
    Load market data according to the DataConfig and optional Universe.

    Supported:
      - Parquet-backed table:  source="parquet://path/to/file.parquet"
      - Yahoo Finance via vectorbt: source="yf://AAPL,MSFT,SPY" or
        use `Universe.static_instruments` (tickers) with source="yf://".

    Parquet expected schema (can be adapted as you like):
      - date or datetime column (will become the index)
      - instrument ID column (universe.id_field, e.g. "ticker")
      - OHLCV columns: open, high, low, close, volume, etc.
    """

    src = data_cfg.source

    if src.startswith("parquet://"):
        # Existing parquet logic
        path = src[len("parquet://") :]
        table = _read_parquet_table(path)
        return _market_data_from_parquet_table(
            table=table,
            data_cfg=data_cfg,
            universe=universe,
        )

    if src.startswith(YF_PREFIX):
        # NEW: Yahoo / vectorbt loader
        return _load_yahoo_vbt(
            data_cfg=data_cfg,
            universe=universe,
        )

    # FRED provider (single series time series)
    if src.lower().startswith(FRED_PREFIX):
        return _load_fred_series(
            data_cfg=data_cfg,
            universe=universe,
        )

    # FINRA convenience mapping for common series routed via FRED IDs
    if src.upper().startswith("FINRA:"):
        mapped = _map_finra_to_fred(src)
        if mapped is None:
            raise ValueError(
                f"Unknown FINRA source {src!r}. Supported examples: 'FINRA:HY_OAS', 'FINRA:IG_OAS'."
            )
        cfg2 = DataConfig(
            source=mapped,
            calendar=data_cfg.calendar,
            frequency=data_cfg.frequency,
            start=data_cfg.start,
            end=data_cfg.end,
            price_adjustment=data_cfg.price_adjustment,
            fields=data_cfg.fields,
            tz=data_cfg.tz,
            transforms=getattr(data_cfg, "transforms", None),
        )
        return _load_fred_series(cfg2, universe)

    raise ValueError(
        f"Unsupported data source scheme in {src!r}. "
        "Currently supported: 'parquet://...', 'yf://...', 'fred://SERIES_ID', 'FINRA:ALIAS'."
    )


# ---------------------------------------------------------------------------
# Parquet-backed loader (existing logic, wrapped into helper)
# ---------------------------------------------------------------------------


def _market_data_from_parquet_table(
    table: pd.DataFrame,
    data_cfg: DataConfig,
    universe: Optional[Universe],
) -> MarketData:
    """Existing parquet processing logic factored into a helper."""

    # Detect date/datetime column and set as index
    table = _normalize_datetime_index(table)
    # Ensure the datetime index is globally monotonic before any slicing
    # Some datasets may be sorted by other columns (e.g., ticker first),
    # which breaks label-based slicing on a non-monotonic DatetimeIndex.
    table = table.sort_index()

    # Filter by date range
    table = table.loc[data_cfg.start : data_cfg.end]

    if universe is not None:
        id_col = universe.id_field
    else:
        # Fallback: try to guess an ID field
        id_col = _infer_instrument_id_column(table)

    if universe is not None:
        # Apply static instrument restriction (if provided)
        if universe.static_instruments is not None:
            table = table[table[id_col].isin(universe.static_instruments)]

        # Apply dynamic filters
        table = _apply_universe_filters(table, id_col, universe.filters, data_cfg.fields)

    # Now split into per-instrument DataFrames
    bars_by_instr: Dict[InstrumentId, pd.DataFrame] = {}
    instruments: List[InstrumentId] = []

    for instr, df_instr in table.groupby(id_col):
        # Drop the ID column; it's now implicit in the dict key
        df_instr = df_instr.drop(columns=[id_col])

        # Ensure columns are in the configured order (if present)
        cols = [c for c in data_cfg.fields if c in df_instr.columns]
        # plus any extra columns at the end (e.g. sector info)
        extra_cols = [c for c in df_instr.columns if c not in cols]
        df_instr = df_instr[cols + extra_cols]

        bars_by_instr[str(instr)] = df_instr.sort_index()
        instruments.append(str(instr))

    market_data = MarketData(
        bars=bars_by_instr,
        instruments=instruments,
        fields=data_cfg.fields,
        frequency=data_cfg.frequency,
        calendar=data_cfg.calendar,
        tz=data_cfg.tz,
    )

    return market_data


def _read_parquet_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet data file not found: {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Yahoo Finance via vectorbt.YFData (NEW)
# ---------------------------------------------------------------------------


def _load_yahoo_vbt(
    data_cfg: DataConfig,
    universe: Optional[Universe],
) -> MarketData:
    """
    Load OHLCV from Yahoo Finance using vectorbt.YFData.

    Conventions:

      - If universe.static_instruments is not None:
            use that as the list of tickers.

      - Else:
            parse tickers from DataConfig.source after 'yf://', e.g.
            source="yf://AAPL,MSFT,SPY"

      - DataConfig.start / end / frequency are used as date range and interval.
    """
    # 1) Decide tickers
    symbols: List[str] = []
    if universe is not None and universe.static_instruments:
        symbols = list(universe.static_instruments)
    else:
        raw = data_cfg.source[len(YF_PREFIX) :]  # text after "yf://"
        # allow "yf://" with no symbols if universe supplies them; else need raw
        if raw.strip():
            symbols = [s.strip() for s in raw.split(",") if s.strip()]

    if not symbols:
        raise ValueError(
            "No tickers specified for Yahoo Finance source. "
            "Provide Universe.static_instruments or use source='yf://AAPL,MSFT,...'."
        )

    # 2) Interval / frequency mapping
    freq = data_cfg.frequency or "1d"
    freq = freq.lower()
    if freq in ("1d", "d", "daily"):
        interval = "1d"
    elif freq in ("1h", "h", "60min", "60m"):
        interval = "1h"
    elif freq in ("1m", "m", "1min"):
        interval = "1m"
    else:
        # Default to daily
        interval = "1d"

    # 3) Download via vectorbt.YFData
    yf_data = vbt.YFData.download(
        symbols,
        start=data_cfg.start,
        end=data_cfg.end,
        interval=interval,
        missing_index="drop",
    )

    # Get OHLCV panels; they may be Series (single symbol) or DataFrames
    open_df = yf_data.get("Open")
    high_df = yf_data.get("High")
    low_df = yf_data.get("Low")
    close_df = yf_data.get("Close")
    vol_df = yf_data.get("Volume")

    # Normalize to DataFrames
    def _ensure_df(obj, col_name: str) -> pd.DataFrame:
        if isinstance(obj, pd.Series):
            return obj.to_frame(col_name)
        return obj

    open_df = _ensure_df(open_df, symbols[0])
    high_df = _ensure_df(high_df, symbols[0])
    low_df = _ensure_df(low_df, symbols[0])
    close_df = _ensure_df(close_df, symbols[0])
    vol_df = _ensure_df(vol_df, symbols[0])

    # Ensure consistent symbol order and index
    open_df = open_df[symbols]
    high_df = high_df[symbols]
    low_df = low_df[symbols]
    close_df = close_df[symbols]
    vol_df = vol_df[symbols]

    # 4) Build bars dict, apply universe-like filters per instrument
    bars_by_instr: Dict[InstrumentId, pd.DataFrame] = {}
    instruments: List[InstrumentId] = []

    # Universe filters (same semantics as parquet branch)
    filters: List[UniverseFilter] = universe.filters if universe is not None else []

    for sym in symbols:
        df = pd.DataFrame(
            {
                "open": open_df[sym],
                "high": high_df[sym],
                "low": low_df[sym],
                "close": close_df[sym],
                "volume": vol_df[sym],
            }
        ).dropna(how="all")

        # Align to configured date range (just in case)
        df = df.loc[data_cfg.start : data_cfg.end]

        # Apply filters at instrument level
        if filters and not _passes_filters(df, data_cfg.fields, filters):
            continue

        # Respect requested fields ordering (same as parquet loader)
        cols = [c for c in data_cfg.fields if c in df.columns]
        extra_cols = [c for c in df.columns if c not in cols]
        df = df[cols + extra_cols]

        bars_by_instr[sym] = df.sort_index()
        instruments.append(sym)

    market_data = MarketData(
        bars=bars_by_instr,
        instruments=instruments,
        fields=data_cfg.fields,
        frequency=data_cfg.frequency,
        calendar=data_cfg.calendar,
        tz=data_cfg.tz,
    )

    return market_data


# ---------------------------------------------------------------------------
# FRED adapter with ArcticDB caching
# ---------------------------------------------------------------------------


def _parse_fred_source(source: str) -> str:
    # Accept only form like "fred://CPIAUCSL"
    if not source.lower().startswith(FRED_PREFIX):
        raise ValueError(
            f"FRED source must use 'fred://<SERIES_ID>' scheme, got {source!r}"
        )
    series_id = source[len(FRED_PREFIX):]
    return series_id.strip()


def _map_finra_to_fred(source: str) -> Optional[str]:
    """Map a FINRA alias to an underlying FRED series ID."""
    alias = source.split(":", 1)[1].strip().upper()
    mapping = {
        # High Yield OAS
        "HY_OAS": f"{FRED_PREFIX}BAMLH0A0HYM2",
        # Investment Grade OAS
        "IG_OAS": f"{FRED_PREFIX}BAMLC0A0CM",
    }
    fred = mapping.get(alias)
    return fred


def _load_fred_series(
    data_cfg: DataConfig,
    universe: Optional[Universe],
) -> MarketData:
    """
    Load a single FRED series and present it as one instrument with 'close' field.
    Uses ArcticDB local cache for persistence.
    """
    series_id = _parse_fred_source(data_cfg.source)

    # We treat each FRED series as a single instrument
    instrument = series_id

    # ArcticDB cache library per provider/frequency
    lib = get_cache_lib(provider="FRED", frequency=data_cfg.frequency)
    key = instrument  # key per instrument

    start_ts = pd.to_datetime(data_cfg.start)
    end_ts = pd.to_datetime(data_cfg.end)

    df: pd.DataFrame
    if cache_has_symbol(lib, key):
        cached = cache_read_symbol(lib, key)
        # Expect columns: ['value'] and index datetime or a 'date' column
        if "date" in cached.columns:
            cached = cached.copy()
            cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None)
            cached = cached.set_index("date").sort_index()
        else:
            if not isinstance(cached.index, pd.DatetimeIndex):
                # try to coerce
                cached = cached.copy()
                cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()

        last_dt = cached.index.max() if not cached.empty else None
        # If cache covers the end date, use slice directly
        if last_dt is not None and last_dt >= end_ts:
            df_full = cached
        else:
            # Fetch missing tail from last_dt + 1 day (if any)
            fetch_start = start_ts
            if last_dt is not None:
                fetch_start = (last_dt + pd.Timedelta(days=1)).normalize()
            fetched = fetch_fred_series(series_id, fetch_start, end_ts)
            fetched = fetched.rename(columns={"value": "close"})
            fetched["date"] = pd.to_datetime(fetched["date"]).dt.tz_localize(None)
            fetched = fetched.set_index("date").sort_index()
            # Normalize cached columns to 'close' if needed
            if "value" in cached.columns and "close" not in cached.columns:
                cached = cached.rename(columns={"value": "close"})
            df_full = pd.concat([cached, fetched], axis=0)
            df_full = df_full[~df_full.index.duplicated(keep="last")].sort_index()
            # Persist merged
            cache_write_symbol(lib, key, df_full)
        df = df_full
    else:
        fetched = fetch_fred_series(series_id, start_ts, end_ts)
        fetched = fetched.rename(columns={"value": "close"})
        fetched["date"] = pd.to_datetime(fetched["date"]).dt.tz_localize(None)
        df = fetched.set_index("date").sort_index()
        cache_write_symbol(lib, key, df)

    # Map to bars dict with requested fields
    # FRED has only a single value; map to 'close'. Volume not available.
    cols: Dict[str, pd.Series] = {}
    if "close" in data_cfg.fields:
        cols["close"] = df["close"].astype("float64")
    if "open" in data_cfg.fields:
        cols["open"] = df["close"].astype("float64")
    if "high" in data_cfg.fields:
        cols["high"] = df["close"].astype("float64")
    if "low" in data_cfg.fields:
        cols["low"] = df["close"].astype("float64")
    if "volume" in data_cfg.fields:
        cols["volume"] = pd.Series(index=df.index, data=float("nan"))

    df_bars = pd.DataFrame(cols).loc[data_cfg.start : data_cfg.end]

    market_data = MarketData(
        bars={instrument: df_bars},
        instruments=[instrument],
        fields=list(df_bars.columns),
        frequency=data_cfg.frequency,
        calendar=data_cfg.calendar,
        tz=data_cfg.tz,
    )

    return market_data


# ---------------------------------------------------------------------------
# Shared helper functions (unchanged)
# ---------------------------------------------------------------------------


def _normalize_datetime_index(table: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed by a datetime-like column named 'date' or 'datetime'.
    The original column is dropped after being moved to the index.
    """
    # Try a few common timestamp column names
    for col in ("datetime", "date", "timestamp"):
        if col in table.columns:
            dt = pd.to_datetime(table[col])
            table = table.drop(columns=[col])
            table.index = dt
            table.index.name = "datetime"
            return table

    # If already has a DatetimeIndex, just return
    if isinstance(table.index, pd.DatetimeIndex):
        return table

    raise ValueError(
        "Could not find a datetime column ('datetime', 'date', or 'timestamp') "
        "and index is not a DatetimeIndex."
    )


def _infer_instrument_id_column(table: pd.DataFrame) -> str:
    """
    Heuristic to guess the instrument identifier column if Universe.id_field
    was not provided. Prefer common names like 'ticker', 'symbol', 'ric'.
    """
    candidates = ["ticker", "symbol", "ric", "instrument", "secid"]
    for c in candidates:
        if c in table.columns:
            return c
    # As a fallback, just pick the first non-numeric column
    for c in table.columns:
        if not pd.api.types.is_numeric_dtype(table[c]):
            return c
    raise ValueError("Could not infer an instrument ID column from data table.")


def _apply_universe_filters(
    table: pd.DataFrame,
    id_col: str,
    filters: List[UniverseFilter],
    fields: List[str],
) -> pd.DataFrame:
    """
    Apply UniverseFilter objects to the base data table and return a filtered table.

    This is intentionally simple: it's not meant to be maximally efficient,
    just clear and correct enough for the skeleton.
    """

    # Work per-instrument for clarity
    grouped = table.groupby(id_col)
    keep_instruments: List[str] = []

    for instr, df in grouped:
        if _passes_filters(df, fields, filters):
            keep_instruments.append(instr)

    return table[table[id_col].isin(keep_instruments)]


def _passes_filters(
    df_instr: pd.DataFrame,
    fields: List[str],
    filters: List[UniverseFilter],
) -> bool:
    """
    Check if a single instrument's timeseries passes all filters.
    """
    # Extract close & volume where available
    close = df_instr["close"] if "close" in df_instr.columns else None
    volume = df_instr["volume"] if "volume" in df_instr.columns else None

    for f in filters:
        if isinstance(f, HasHistory):
            if len(df_instr) < f.min_days:
                return False

        elif isinstance(f, MinPrice):
            if close is None:
                return False
            # Min over the available history
            if close.min() < f.min_price:
                return False

        elif isinstance(f, MinDollarADV):
            if close is None or volume is None:
                return False
            # Simple ADV proxy: mean(close * volume)
            dollar_adv = (close * volume).mean()
            if dollar_adv < f.min_dollar_adv:
                return False

        # Add more UniverseFilter types here as needed.

    return True
