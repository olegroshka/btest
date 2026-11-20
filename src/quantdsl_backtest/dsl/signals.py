# src/quantdsl_backtest/dsl/signals.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any, Protocol, Sequence, runtime_checkable


# For now, we keep the expression types loose (Any / object) so the engine
# can accept:
#   - string references to factors/signals ("mom_6m", "vol_20d", "rank")
#   - other SignalNode instances
#   - numeric constants (floats, ints)
#
# Later, if you want, we can tighten this up with proper union types.


Expr = Any  # placeholder alias for clarity


@runtime_checkable
class Signal(Protocol):
    """
    Lightweight interface for all DSL signal nodes.

    Engines can call `evaluate(engine)` for dynamic dispatch without
    long isinstance chains. Implementations are free to delegate back
    to the engine to keep business logic centralized (double-dispatch).
    """

    def evaluate(self, engine: Any) -> Any:  # return is a pandas.DataFrame
        ...


@dataclass(slots=True)
class SignalNode:
    """
    Base marker for all signal / expression DSL nodes.

    Note: we do NOT put `name` here, so that subclasses can choose their
    own constructor argument order. That way, usages like NotNull("mom_6m")
    still work nicely.
    """
    # just a marker; no fields for now
    pass


# ---------------------------------------------------------------------------
# Simple boolean / validity checks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NotNull(SignalNode):
    """
    Signal that is True where the referenced factor/signal is non-null.

    Example:
        NotNull("mom_6m")
    """

    factor_name: str
    name: Optional[str] = None

    # Double-dispatch entry
    def evaluate(self, engine: Any) -> Any:
        return engine._eval_notnull(self)


# ---------------------------------------------------------------------------
# Logical composition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class And(SignalNode):
    """
    Logical AND of two boolean expressions.

    `left` and `right` can be:
      - other SignalNode instances,
      - string names referencing signals/factors,
      - nested comparison nodes.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_and(self)


@dataclass(slots=True)
class Or(SignalNode):
    """
    Logical OR of two boolean expressions.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_or(self)


@dataclass(slots=True)
class Not(SignalNode):
    """
    Logical NOT of a boolean expression.
    """

    expr: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_not(self)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LessEqual(SignalNode):
    """
    Comparison: left <= right.

    `left` / `right` can be:
      - factor names (e.g. "vol_20d"),
      - other expressions (e.g. Quantile(...)),
      - numeric constants.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_less_equal(self)


@dataclass(slots=True)
class GreaterEqual(SignalNode):
    """
    Comparison: left >= right.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_greater_equal(self)


# Additional strict comparisons


@dataclass(slots=True)
class Less(SignalNode):
    """
    Comparison: left < right.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_less(self)


@dataclass(slots=True)
class Greater(SignalNode):
    """
    Comparison: left > right.
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_greater(self)


# ---------------------------------------------------------------------------
# Cross-sectional operations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CrossSectionAggregate(SignalNode):
    """
    Cross-sectional aggregate over instruments at each timestamp.

    Example:
        CrossSectionAggregate(source="mom_126", op="mean", name="avg_mom_126")

    Semantics:
      for each date t:
        value(t) = agg(op, source(t, universe) where mask is True and not NaN)

    The engine will broadcast the scalar time series across columns so that
    it can be used in comparisons and boolean masks consistently.
    """

    source: str                      # factor or signal name to aggregate
    op: Literal["mean", "median", "sum", "min", "max"] = "mean"
    mask_name: Optional[str] = None  # optional mask to restrict universe
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_cross_section_aggregate(self)


@dataclass(slots=True)
class Quantile(SignalNode):
    """
    Cross-sectional quantile of a factor at each time point.

    Example:
        Quantile(factor_name="vol_20d", q=0.9)

    The engine will interpret this as:
      for each date t:
        q_value(t) = quantile_q of vol_20d(t, universe)
    """

    factor_name: str
    q: float                           # 0.0 - 1.0
    within_mask: Optional[str] = None  # restrict to subset, by mask name
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_quantile(self)


@dataclass(slots=True)
class CrossSectionRank(SignalNode):
    """
    Cross-sectional rank of a factor.

    method:
      - "percentile": 0..1 rank
      - "demeaned": the percentile rank minus one half, -0.5..0.5 (a signal
        whose sign is the side of the book and whose size is the distance
        from the middle; its negation is the rank of the other side)
      - "midpoint": (rank - 0.5) / n - 0.5, strictly inside -0.5..0.5 (each name
        sits at the middle of its rank cell, so the mean over the day is zero
        and the spread does not depend on how many names are ranked)
      - "zscore": z-scored version

    ``factor_name`` may name a factor or another signal (a lagged or rolled
    panel, a composite). ``ascending=False`` ranks the negated values, so the
    lowest value gets the highest rank.
    """

    factor_name: str
    mask_name: Optional[str] = None
    method: Literal["percentile", "demeaned", "midpoint", "zscore"] = "percentile"
    name: Optional[str] = None
    ascending: bool = True

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rank(self)


@dataclass(slots=True)
class Composite(SignalNode):
    """
    Equal-weighted mean of several panels, cell by cell, over the members
    that have a value there.

    A cell is NaN unless at least ``min_count`` members are present, so a
    name scored by too few signals is not ranked on the strength of one.

    Example (three rank scores, at least two present):
        Composite(members=["r_mom", "r_vol", "r_gap"], min_count=2, name="score")
    """

    members: Sequence[Expr]
    min_count: int = 1
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.members) == 0:
            raise ValueError("Composite needs at least one member")
        if not 1 <= int(self.min_count) <= len(self.members):
            raise ValueError("min_count must be between 1 and the number of members")

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_composite(self)


@dataclass(slots=True)
class MaskFromBoolean(SignalNode):
    """
    Wrapper indicating that the given boolean expression should be stored
    as a named mask (True/False per instrument/time).

    Example:
        MaskFromBoolean(
            name="long_candidates",
            expr=LessEqual(...),
        )
    """

    expr: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_mask_from_boolean(self)


# ---------------------------------------------------------------------------
# Time-series primitives (for macro series and transforms)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TimeSeries(SignalNode):
    """
    Load a single time series (e.g., FRED) and broadcast across instruments.

    `source` can be:
      - a DataConfig-like dict or object understood by the data loader
      - a string URI such as "fred://BAMLH0A0HYM2"
    """

    source: Any
    field: str = "close"
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_time_series(self)


@dataclass(slots=True)
class EWMMean(SignalNode):
    base: Expr
    span: int
    min_periods: int = 1
    adjust: bool = False
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_ewm_mean(self)


@dataclass(slots=True)
class RollingMean(SignalNode):
    base: Expr
    window: int
    min_periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rolling_mean(self)


@dataclass(slots=True)
class Identity(SignalNode):
    """
    A factor (or another signal) used as a signal, unchanged.

    The panel itself, copied: for an ``ExternalFactor`` that already is the
    score a book weights by, or to give an expression a name. Replaces the
    ``RollingMean(window=1)`` idiom.

    Example:
        Identity(base="s_bear", name="bear")
    """

    base: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_identity(self)


@dataclass(slots=True)
class RollingSum(SignalNode):
    """Sum over the trailing ``window`` bars, per instrument; NaN with fewer than ``min_periods`` values."""

    base: Expr
    window: int
    min_periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rolling_sum(self)


@dataclass(slots=True)
class Lag(SignalNode):
    """
    The panel as it stood ``periods`` bars earlier, per instrument.

    The point-in-time operator: a value that is only complete at a bar's
    close (a close-to-close return, a volatility over closes) must be lagged
    one bar before it may drive a decision taken before that close.
    ``periods`` cannot be negative: that would read the future.

    Example (volatility over the 20 closes ending yesterday):
        Lag(base=RollingStd(base="ret_1d", window=20, min_periods=15), periods=1, name="vol20")
    """

    base: Expr
    periods: int = 1
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.periods) < 0:
            raise ValueError("Lag.periods cannot be negative: that would read the future")

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_lag(self)


@dataclass(slots=True)
class RollingStd(SignalNode):
    base: Expr
    window: int
    min_periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rolling_std(self)


@dataclass(slots=True)
class Diff(SignalNode):
    base: Expr
    periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_diff(self)


@dataclass(slots=True)
class PctChange(SignalNode):
    base: Expr
    periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_pct_change(self)


@dataclass(slots=True)
class ZScoreRolling(SignalNode):
    base: Expr
    window: int
    min_periods: int = 1
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_zscore_rolling(self)


@dataclass(slots=True)
class RiskMultiplierFromZ(SignalNode):
    """
    Map a z-score to [0,1] as: 1 - clip(z, 0, max_z)/max_z, then clip to [0,1].
    """

    z: Expr
    max_z: float = 2.5
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_risk_multiplier_from_z(self)
