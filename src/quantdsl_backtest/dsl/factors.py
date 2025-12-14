# src/quantdsl_backtest/dsl/factors.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# Factor method literals
ReturnMethod = Literal["log", "simple"]
VolMethod = Literal["realized", "stdev"]


@dataclass(slots=True)
class FactorNode:
    """
    Base class / marker for factor DSL nodes.
    Each concrete factor (e.g. ReturnFactor) inherits from this.
    """
    name: str


@dataclass(slots=True)
class ReturnFactor(FactorNode):
    """
    Simple return factor over a lookback window.
    Engine will compute returns from a given price field.
    """

    field: str = "close"
    lookback: int = 1
    method: ReturnMethod = "simple"


@dataclass(slots=True)
class VolatilityFactor(FactorNode):
    """
    Realized volatility / standard deviation of returns over a window.
    """

    field: str = "close"
    lookback: int = 20
    method: VolMethod = "realized"
    # Optional: you might later add annualization, etc.
    annualize: bool = True
    # Allow loosening of rolling window requirement in sparse calendars.
    # If None, defaults to `lookback`.
    min_periods: int | None = None


@dataclass(slots=True)
class FiboRetraceFactor(FactorNode):
    """
    Example extension factor: Fibonacci retracement level over a window.
    Included for completeness with the example script, though not required
    for the minimal backtest.
    """

    field_high: str = "high"
    field_low: str = "low"
    lookback: int = 50
    level: float = 0.618  # e.g. 61.8%
    # Optionally allow custom output name separate from `name`
    output_name: Optional[str] = None


@dataclass(slots=True)
class OvernightReturnFactor(FactorNode):
    """
    Rolling average of overnight returns: log(open_t / close_{t-1}).
    """

    field_open: str = "open"
    field_close: str = "close"
    lookback: int = 20
    method: ReturnMethod = "log"


@dataclass(slots=True)
class IntradayReturnFactor(FactorNode):
    """
    Rolling average of intraday returns: log(close_t / open_t).
    """

    field_open: str = "open"
    field_close: str = "close"
    lookback: int = 20
    method: ReturnMethod = "log"


@dataclass(slots=True)
class WinsorizedFactor(FactorNode):
    """
    Wraps another factor node and applies cross-sectional per-date symmetric
    winsorization to its output before downstream usage (e.g., ranking).

    Parameters
    - base: FactorNode to evaluate first
    - z: float, symmetric clipping threshold (mean ± z * std) per date
    """

    base: FactorNode
    z: float = 3.0


@dataclass(slots=True)
class RatioFactor(FactorNode):
    """
    Element-wise ratio of two factor nodes: numerator / denominator.
    Both inputs are evaluated first and aligned on [datetime x instrument].

    Notes:
    - No implicit epsilon/floor is applied; NaNs and infs propagate naturally.
      Upstream/downstream winsorization or masking should be used if desired.
    """

    numerator: FactorNode
    denominator: FactorNode
