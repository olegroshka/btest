# src/quantdsl_backtest/dsl/data_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from .frequency import Frequency

if TYPE_CHECKING:
    from .transforms import DataTransform


PriceAdjustment = str  # keep as string-literal-like values; validated elsewhere if needed.


@dataclass(slots=True)
class DataConfig:
    """
    Configuration for market data used in a backtest.

    This is purely declarative: the engine will interpret this to
    load the right data from the right place.
    """

    source: str
    calendar: str

    # Bar frequency, canonical format like "1d", "5m", "15m", "1h".
    # We validate in __post_init__ via Frequency.parse.
    frequency: str

    start: str
    end: str

    price_adjustment: str = "split_dividend"

    fields: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # Optional: additional metadata
    tz: Optional[str] = None

    # Optional: data transforms to apply after loading
    transforms: Optional[List["DataTransform"]] = None

    # Optional FIXED-INCOME coupon stream. Path to a long-format parquet with columns (date, ticker, income)
    # giving the coupon cash paid PER UNIT held on each date (0 / absent elsewhere). When set, the engine
    # credits sum(positions * income) to cash each bar. Pair with a DIRTY price source (close = clean+accrued)
    # for correct, smooth bond total return: the dirty mark books daily accrual, the coupon date's dirty drop
    # is offset by this cash credit. None = ordinary equity behaviour.
    income_source: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate/normalize frequency early so invalid configs fail fast.
        # Preserve original string, but ensure it parses and is in canonical form.
        self.frequency = str(Frequency.parse(self.frequency))

        # Normalize common price_adjustment values
        self.price_adjustment = (self.price_adjustment or "").strip().lower()
