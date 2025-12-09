# src/quantdsl_backtest/dsl/portfolio.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# --- Selectors & weighting --------------------------------------------------


@dataclass(slots=True)
class Selector:
    """Base class for how we pick instruments into a book."""
    pass


@dataclass(slots=True)
class TopN(Selector):
    factor_name: str
    n: int
    mask_name: Optional[str] = None


@dataclass(slots=True)
class BottomN(Selector):
    factor_name: str
    n: int
    mask_name: Optional[str] = None


@dataclass(slots=True)
class Weighting:
    """Base class for within-book weighting schemes."""
    pass


@dataclass(slots=True)
class EqualWeight(Weighting):
    """Equal notional weights across selected instruments."""
    pass


# --- Constraints / risk-style knobs ----------------------------------------


@dataclass(slots=True)
class SectorNeutral:
    sector_field: str = "sector"
    tolerance: float = 0.02            # +/- tolerance in absolute weights


@dataclass(slots=True)
class TurnoverLimit:
    window_bars: int = 1
    max_fraction: float = 0.30         # fraction of *gross* per window


# --- Books and overall portfolio -------------------------------------------


@dataclass(slots=True)
class Book:
    """
    One side of a long/short portfolio: long book or short book.
    """

    name: str
    selector: Selector
    weighting: Weighting


@dataclass(slots=True, kw_only=True)
class LongShortPortfolio:
    """
    Declarative spec for a daily cross-sectional long/short strategy.
    """

    # Books (required, must appear before any fields with defaults)
    long_book: Book
    short_book: Book

    # Rebalance policy (defaults)
    rebalance_frequency: Literal["1d", "1w", "1m"]
    rebalance_at: Literal["market_open", "market_close"] = "market_close"
    signal_delay_bars: int = 0

    # Exposure / constraints
    target_gross_leverage: float = 2.0
    target_net_exposure: float = 0.0
    max_abs_weight_per_name: float = 0.03

    # Optional constraints
    sector_neutral: Optional[SectorNeutral] = None
    turnover_limit: Optional[TurnoverLimit] = None
