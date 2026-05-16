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
