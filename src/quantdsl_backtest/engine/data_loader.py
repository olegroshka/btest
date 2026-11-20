# src/quantdsl_backtest/engine/data_loader.py

from __future__ import annotations

from typing import Tuple, Iterable

import pandas as pd

from ..dsl.strategy import Strategy
from ..dsl.transforms import DataTransform
from ..data.adapters import load_market_data
from ..data.schema import MarketData
from ..utils.logging import get_logger


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level data cache
# Keyed by (source, start, end) so sweeps sharing the same universe + date
# range only hit the parquet once per process.  Call clear_data_cache() to
# free memory or to force a reload.
# ---------------------------------------------------------------------------
_DATA_CACHE: dict[tuple, tuple] = {}


def clear_data_cache() -> None:
    """Evict all cached market data (e.g. between unrelated backtests)."""
    _DATA_CACHE.clear()


def load_data_for_strategy(strategy: Strategy) -> Tuple[MarketData, pd.DataFrame, pd.DataFrame]:
    """
    Load market data for the given strategy and return:

        market_data: MarketData
        prices: DataFrame [datetime x instrument]  (close prices)
        volumes: DataFrame [datetime x instrument] (volume)

    Assumes that 'close' and 'volume' are present in DataConfig.fields.

    Important: Preserve the original timestamp index from the data adapter
    to maintain exact alignment with vectorbt baselines used in tests.
    """
    _cache_key = (
        str(strategy.data.source),
        str(getattr(strategy.data, "start", None)),
        str(getattr(strategy.data, "end", None)),
        repr(strategy.universe),  # two universes over one file are two different panels
    )
    if _cache_key in _DATA_CACHE:
        log.debug("Data cache hit for %s", strategy.data.source)
        return _DATA_CACHE[_cache_key]

    log.info("Loading market data from %s", strategy.data.source)
    md = load_market_data(strategy.data, strategy.universe)

    instruments = md.instruments

    # Build wide DataFrames directly from adapter output without altering timestamps
    prices = pd.DataFrame({
        instr: md.bars[instr].get("close", pd.Series(dtype="float64"))
        for instr in instruments
    })
    volumes = pd.DataFrame({
        instr: md.bars[instr].get("volume", pd.Series(dtype="float64"))
        for instr in instruments
    })

    # Ensure time is sorted and volumes align to prices' index
    prices = prices.sort_index()
    volumes = volumes.sort_index().reindex(prices.index)

    # Apply any configured data transforms declared in the DSL. This replaces
    # previous source-specific cleaning logic with a generic, composable pipeline.
    transforms: Iterable[DataTransform] = getattr(strategy.data, "transforms", []) or []
    for t in transforms:
        try:
            # Allow users to pass either a transform instance or the class itself.
            # If a class is provided, instantiate with defaults.
            obj = t  # type: ignore[assignment]
            if isinstance(t, type):  # a class, not an instance
                obj = t()  # type: ignore[call-arg]

            prices, volumes = obj.apply(prices, volumes)  # type: ignore[attr-defined]
        except Exception as e:
            name = None
            try:
                name = getattr(t, "name", None)
            except Exception:
                name = None
            if not isinstance(name, str) or not name:
                # Fall back to readable type/instance name
                name = getattr(getattr(t, "__class__", type(t)), "__name__", str(t))
            log.error("Data transform %s failed: %s", name, e)
            raise

    if transforms:
        log.info("Applied %d data transform(s); resulting shape: prices=%s, volumes=%s",
                 len(list(transforms)), prices.shape, volumes.shape)

    # Enforce float64 dtype for numerical stability
    prices = prices.astype("float64")
    volumes = volumes.astype("float64")

    log.info(
        "Loaded data: %d instruments, %d bars",
        len(instruments),
        len(prices.index),
    )
    result = md, prices, volumes
    _DATA_CACHE[_cache_key] = result
    return result


def load_income_for_strategy(strategy: "Strategy", instruments, dates):
    """Optional FIXED-INCOME coupon panel [dates x instruments] = coupon cash paid PER UNIT held per date.

    Reads strategy.data.income_source (long-format parquet: columns date, ticker, income). Returns None when
    not configured (ordinary equity backtests unaffected); 0 where no coupon on that date/instrument.
    """
    import re
    src = getattr(getattr(strategy, "data", None), "income_source", None)
    if not src:
        return None
    path = re.sub(r"^parquet://", "", str(src))
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    panel = df.pivot_table(index="date", columns="ticker", values="income", aggfunc="sum")
    return panel.reindex(index=pd.DatetimeIndex(dates), columns=instruments).fillna(0.0)


def load_funding_for_strategy(strategy: "Strategy", instruments, dates):
    """Perp funding panel [dates x instruments] = SUM of funding rates settling inside each bar's
    holding window, or None when `strategy.costs.funding` is not set.

    Timing: engine dates are bar OPEN times (naive UTC) and a bar's close is the next bar's open, so the
    position carried into bar t (held from close[t-1] = open[t] to close[t] = open[t] + step) pays the
    events with funding_ts in (open[t], open[t] + step]. An event exactly at a bar boundary belongs to the
    bar that ENDS there — the position held up to that instant is the one that pays.
    """
    import re
    fc = getattr(getattr(strategy, "costs", None), "funding", None)
    if fc is None:
        return None
    src = str(fc.source)
    if src.startswith("sfera://"):
        from ..data.sources.sfera import _require_sfera_db
        schema, table = src[len("sfera://"):].split("?")[0].strip("/").split("/")[:2]
        df = _require_sfera_db().query(
            f"SELECT symbol AS ticker, funding_ts AS ts, funding_rate FROM {schema}.{table} "
            f"WHERE venue = %s AND symbol = ANY(%s) AND deprecated_at IS NULL",
            [fc.venue, [str(i) for i in instruments]],
        )
    else:
        df = pd.read_parquet(re.sub(r"^parquet://", "", src))
    idx = pd.DatetimeIndex(dates)
    if df is None or df.empty or len(idx) < 2:
        return pd.DataFrame(0.0, index=idx, columns=instruments)
    ts = pd.to_datetime(df["ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    step = pd.Series(idx).diff().median()
    # bar t owns (open[t], open[t] + step]  →  locate open[t] via the event time minus 1ns
    pos = idx.searchsorted(ts - pd.Timedelta(1, "ns"), side="right") - 1
    ok = (pos >= 0) & (ts.values <= (idx[pos.clip(0)] + step).values)
    out = pd.DataFrame({"date": idx[pos[ok]], "ticker": df["ticker"].values[ok],
                        "rate": pd.to_numeric(df["funding_rate"]).values[ok]})
    panel = out.pivot_table(index="date", columns="ticker", values="rate", aggfunc="sum")
    return panel.reindex(index=idx, columns=instruments).fillna(0.0).astype("float64")


def load_field_panel(md: "MarketData", instruments: "pd.Index", field: str) -> "pd.DataFrame":
    """A wide panel [datetime x instrument] of one data field. Instruments without the column get NaN."""
    panel = pd.DataFrame({
        instr: md.bars[instr].get(field, pd.Series(dtype="float64"))
        for instr in instruments
    })
    return panel.sort_index().astype("float64")


def load_open_prices(md: "MarketData", instruments: "pd.Index") -> "pd.DataFrame":
    """Extract open prices from already-loaded MarketData.

    Returns a DataFrame [datetime x instrument] aligned to the same index as
    the close prices.  Instruments missing an "open" column get NaN.
    """
    open_df = pd.DataFrame({
        instr: md.bars[instr].get("open", pd.Series(dtype="float64"))
        for instr in instruments
    })
    return open_df.sort_index().astype("float64")
