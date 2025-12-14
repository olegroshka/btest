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


# ---------------------------------------------------------------------------
# Cross-sectional operations
# ---------------------------------------------------------------------------


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
