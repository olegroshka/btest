# src/quantdsl_backtest/data/adapters.py

from __future__ import annotations

from typing import Optional

from ..dsl.data_config import DataConfig
from ..dsl.universe import Universe
from .schema import MarketData
from .loaders import load_data_config_as_market_data


def load_market_data(
    data_cfg: DataConfig,
    universe: Optional[Universe] = None,
) -> MarketData:
    """Backwards-compatible market data loader.

    This is a stable facade used by the engine.
    Internally it delegates to the new provider-based data source layer.
    """

    return load_data_config_as_market_data(data_cfg, universe)
