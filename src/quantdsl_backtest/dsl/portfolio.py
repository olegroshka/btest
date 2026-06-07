# src/quantdsl_backtest/dsl/portfolio.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import pandas as pd


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


@dataclass(slots=True)
class SignalWeight(Weighting):
    """Weight instruments proportionally to their signal values."""

    signal_name: str  # Name of the signal (typically the rank or composite)


# --- Constraints / risk-style knobs ----------------------------------------


@dataclass(slots=True)
class SectorNeutral:
    sector_field: str = "sector"
    tolerance: float = 0.02  # +/- tolerance in absolute weights


@dataclass(slots=True)
class TurnoverLimit:
    window_bars: int = 1
    max_fraction: float = 0.30  # fraction of *gross* per window


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


@dataclass(slots=True, kw_only=True)
class TargetWeights:
    """
    Generic target-weights portfolio: the single primitive that subsumes
    timing, long/short, rotation, leverage and hedging.

    Everything a portfolio does reduces to a ``[date x instrument]`` matrix of
    target weights:

    - **sign** = long (+) / short (-)
    - **magnitude** = position size (fraction of equity)
    - **row sum** = net exposure;  **sum of abs** = gross leverage

    With this one class:

    - ``TimingPortfolio``        = one-hot ``{0, 1}`` on a single column
    - ``LongShortPortfolio``     = ranked ``+/-`` weights across a universe
    - a rotation (e.g. r_lev_001) = one-hot across an instrument menu
    - 2x leverage               = weights whose abs-sum is 2.0
    - a hedge / pair            = mixed signs

    The execution engine (event-driven and vectorized) is fully generic: it
    applies ``signal_delay_bars`` + execution timing, computes
    ``weights.shift(delay) . asset_returns``, charges turnover costs on
    ``|delta weights|`` and financing on the borrowed/short notional. None of
    that needs to know how the weights were produced.

    **Two ways to supply weights** (exactly one must be set):

    1. ``weights`` — a precomputed ``DataFrame`` indexed by date, columns =
       instrument tickers. This is the maximally-general escape hatch: compute
       the matrix any way you like (even in plain pandas — a rotation state
       machine, a vol-target overlay, a leverage multiplier) and run it through
       the engine with correct costs / financing / execution.

    2. ``weights_signal`` — the name of an entry in ``Strategy.signals`` whose
       panel *is* the ``[date x instrument]`` weights matrix. This is the
       DSL-native path: the signal graph produces the weights.

    Optional post-processing (applied per row, after the delay shift):

    - ``target_gross_leverage`` — if set, each row is rescaled so its abs-sum
      equals this value (``None`` = use the matrix as-is, the default).
    - ``max_abs_weight_per_name`` — clip any single weight to ``+/-`` this.
    - ``turnover_limit`` — cap per-rebalance turnover (scales the move toward
      target), reusing the same machinery as ``LongShortPortfolio``.

    Example (precomputed rotation matrix)::

        import pandas as pd

        w = pd.DataFrame(0.0, index=dates, columns=["TQQQ", "VIXY", "IEF", "DBC"])
        # ... fill one-hot per the rotation state machine ...

        portfolio = TargetWeights(
            weights=w,
            rebalance_frequency="1d",
            signal_delay_bars=1,          # decide at T, fill at T+1 (MOO with fill_on="open")
            target_gross_leverage=2.0,    # optional: lever the whole book 2x
        )

    Example (DSL-native, weights produced by the signal graph)::

        portfolio = TargetWeights(
            weights_signal="rotation_weights",   # a [date x instrument] signal panel
            rebalance_frequency="1d",
            signal_delay_bars=1,
        )
    """

    # Exactly one of these supplies the weights.
    weights: Optional["pd.DataFrame"] = None
    weights_signal: Optional[str] = None

    # Rebalance / timing policy (mirrors the other portfolio types)
    rebalance_frequency: Literal["1d", "1w", "1m"] = "1d"
    rebalance_at: Literal["market_open", "market_close"] = "market_close"
    signal_delay_bars: int = 1

    # Event-driven rebalancing. When True, only trade when the target allocation actually CHANGES
    # (e.g. a regime flip). Between changes the exact share count is held — no daily drift/cost-funding
    # micro-trades. Use for rotation/timing strategies that should "hold IEF until the signal flips".
    # When False (default), the portfolio rebalances on every `rebalance_frequency` bar back to the
    # target (correct for fixed-weight books like 60/40 that must rebalance back as prices drift).
    rebalance_on_change: bool = False

    # Optional per-row post-processing
    target_gross_leverage: Optional[float] = None
    max_abs_weight_per_name: Optional[float] = None
    turnover_limit: Optional[TurnoverLimit] = None

    debugging: bool = False

    def __post_init__(self) -> None:
        has_matrix = self.weights is not None
        has_signal = self.weights_signal is not None
        if has_matrix == has_signal:
            raise ValueError(
                "TargetWeights requires exactly one of `weights` (a precomputed "
                "DataFrame) or `weights_signal` (a signal name); "
                f"got weights={'set' if has_matrix else 'None'}, "
                f"weights_signal={self.weights_signal!r}."
            )
