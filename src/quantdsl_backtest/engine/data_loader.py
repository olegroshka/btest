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
    """
    log.info("Loading market data from %s", strategy.data.source)
    md = load_market_data(strategy.data, strategy.universe)

    # Build wide panels for 'close' and 'volume'
    instruments = md.instruments
    # Construct index as union of all bar indices
    all_indices = sorted({ts for df in md.bars.values() for ts in df.index})
    idx = pd.DatetimeIndex(all_indices, name="datetime")

    prices = pd.DataFrame(index=idx, columns=instruments, dtype="float64")
    volumes = pd.DataFrame(index=idx, columns=instruments, dtype="float64")

    for instr, df in md.bars.items():
        prices[instr] = df.get("close", pd.Series(index=df.index, dtype="float64")).reindex(idx)
        volumes[instr] = df.get("volume", pd.Series(index=df.index, dtype="float64")).reindex(idx)

    log.info(
        "Loaded data: %d instruments, %d bars",
        len(instruments),
        len(idx),
    )
    return md, prices, volumes
