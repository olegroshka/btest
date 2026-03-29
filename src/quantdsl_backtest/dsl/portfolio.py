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
    # If True (default), when the mask yields fewer than n names,
    # fill the remainder from the unmasked ranked universe (current default behavior).
    # If False, return only the masked selection (may be < n), making masks bite harder.
    fill_from_unmasked: bool = True


@dataclass(slots=True)
class MaskSelector(Selector):
    """
    Select instruments where a named boolean signal (mask) is True.

    Used in single-instrument timing and filtered long-only books where the
    universe itself acts as the positionable set.

    Example::

        long_book = Book(
            name="long",
            selector=MaskSelector(signal_name="entry_signal"),
            weighting=EqualWeight(),
        )
    """

    signal_name: str


@dataclass(slots=True)
class BottomN(Selector):
    factor_name: str
    n: int
    mask_name: Optional[str] = None
    # See TopN.fill_from_unmasked
    fill_from_unmasked: bool = True


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
    debugging: bool = False


@dataclass(slots=True, kw_only=True)
class TimingPortfolio:
    """
    Single-instrument (or small-set) market-timing portfolio.

    Instead of ranking a universe and picking top/bottom N, the portfolio holds
    a 100% long position in ``instrument`` whenever ``signal_name`` evaluates to
    True, and is fully flat otherwise.  Only long exposure is taken; no shorts.

    Typical use-case: timed exposure to a single index / ETF where the entry/exit
    decision comes from an ML model or regime filter (e.g. TKAN + IVol gate).

    Parameters
    ----------
    signal_name:
        Name of the boolean signal defined in ``Strategy.signals`` that drives
        the timing decision.  True → long, False → flat.
    instrument:
        Ticker of the instrument to trade (must appear in the loaded universe /
        data).
    rebalance_frequency:
        How often to re-evaluate positions.  Supports ``"1d"``, ``"1w"``, ``"1m"``.
    signal_delay_bars:
        Number of bars to lag the signal before acting (default 1 = next-day
        fill after signal is generated at the close).
    target_leverage:
        Gross leverage when the signal is on (default 1.0 = fully invested).

    Example::

        portfolio = TimingPortfolio(
            signal_name="entry_signal",
            instrument="CACT",
            rebalance_frequency="1d",
            signal_delay_bars=1,
        )
    """

    signal_name: str
    instrument: str

    rebalance_frequency: Literal["1d", "1w", "1m"] = "1d"
    rebalance_at: Literal["market_open", "market_close"] = "market_close"
    signal_delay_bars: int = 1
    target_leverage: float = 1.0
