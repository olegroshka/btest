# src/quantdsl_backtest/dsl/data_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


PriceAdjustment = Literal["none", "split_only", "split_dividend"]


@dataclass(slots=True)
class DataConfig:
    """
    Configuration for market data used in a backtest.

    This is purely declarative: the engine will interpret this to
    load the right data from the right place.
    """

    source: str                          # e.g. "parquet://equities/sp500_daily"
    calendar: str                        # e.g. "XNYS"
    frequency: Literal["1d", "1m", "5m"] # etc.
    start: str                           # ISO date, e.g. "2015-01-01"
    end: str                             # ISO date, e.g. "2025-01-01"
    price_adjustment: PriceAdjustment = "split_dividend"
    fields: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # Optional: additional metadata
    tz: Optional[str] = None             # e.g. "America/New_York"
