# src/quantdsl_backtest/dsl/signals.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any, Protocol, runtime_checkable


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
      - "zscore": z-scored version
    """

    factor_name: str
    mask_name: Optional[str] = None
    method: Literal["percentile", "zscore"] = "percentile"
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rank(self)


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


# ---------------------------------------------------------------------------
# Arithmetic operators (element-wise on aligned panels)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Add(SignalNode):
    """
    Element-wise addition: left + right.

    Both operands are resolved with _resolve_expr, so either can be a factor
    name, another SignalNode, or a numeric constant.

    Example — spread minus index basis:
        Add(left="issuer_spread", right=EWMMean(base="cdx_spread", span=60))
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_add(self)


@dataclass(slots=True)
class Sub(SignalNode):
    """
    Element-wise subtraction: left - right.

    Example — cash-index basis:
        Sub(left="issuer_spread", right="cdx_spread")
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_sub(self)


@dataclass(slots=True)
class Mul(SignalNode):
    """
    Element-wise multiplication: left * right.

    Useful for sign-flipping a signal by a constant or another series:
        Mul(left="my_signal", right=-1.0)
    """

    left: Expr
    right: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_mul(self)


# ---------------------------------------------------------------------------
# Univariate transforms
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Sign(SignalNode):
    """
    Element-wise sign: +1 / 0 / -1 (NaN-preserving).

    Example — credit-quality signal from spread direction:
        Sign(base="spread_diff")
    """

    base: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_sign(self)


@dataclass(slots=True)
class Clip(SignalNode):
    """
    Element-wise clip to [lower, upper].

    Example:
        Clip(base="zscore", lower=-3.0, upper=3.0)
    """

    base: Expr
    lower: float = -3.0
    upper: float = 3.0
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_clip(self)


@dataclass(slots=True)
class Abs(SignalNode):
    """
    Element-wise absolute value.
    """

    base: Expr
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_abs(self)


# ---------------------------------------------------------------------------
# Rolling quantile (time-series, per-instrument)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RollingQuantile(SignalNode):
    """
    Per-instrument rolling quantile over a look-back window.

    This is the **time-series** (per-row) analog of the cross-sectional
    ``Quantile`` node.  It answers: "what is the q-th percentile of this
    instrument's own history over the last `window` bars?"

    Typical uses
    ------------
    * VIX term-structure threshold  (S11):
        RollingQuantile(base="vix_slope", window=252, q=0.5)
    * Cash-dispersion quantile filter (S3):
        RollingQuantile(base="cdr_slow", window=504, q=0.05)
    * Cash-synthetic basis suppression (S9):
        RollingQuantile(base="basis", window=252, q=0.95)

    Parameters
    ----------
    base : Expr
        Input series / factor name.
    window : int
        Rolling window length in bars.
    q : float
        Quantile level in [0, 1].  0.5 = median.
    min_periods : int
        Minimum non-NaN observations required; defaults to ``window``.
    name : Optional[str]
        Optional cache key.
    """

    base: Expr
    window: int
    q: float
    min_periods: Optional[int] = None
    name: Optional[str] = None

    def evaluate(self, engine: Any) -> Any:
        return engine._eval_rolling_quantile(self)
