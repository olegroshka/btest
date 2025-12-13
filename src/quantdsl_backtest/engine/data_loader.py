# src/quantdsl_backtest/engine/data_loader.py

from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..dsl.strategy import Strategy
from ..data.adapters import load_market_data
from ..data.schema import MarketData
from ..utils.logging import get_logger


log = get_logger(__name__)


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

    # Enforce float64 dtype for numerical stability
    prices = prices.astype("float64")
    volumes = volumes.astype("float64")

    log.info(
        "Loaded data: %d instruments, %d bars",
        len(instruments),
        len(prices.index),
    )
    return md, prices, volumes
