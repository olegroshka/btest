# src/quantdsl_backtest/dsl/factors.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


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


@dataclass(slots=True)
class ExternalFactor(FactorNode):
    """
    Load pre-computed factor values from a file (pickle, parquet, or CSV).

    **Broadcast mode** (default, ``per_instrument=False``):
      The file contains a ``pd.Series`` with a ``DatetimeIndex``; the value is
      broadcast to all instruments in the universe.  Use this for universe-wide
      scalars (macro signals, index-level ML predictions, etc.).

      - ``pd.Series``           → used directly
      - ``pd.DataFrame``        → column ``column`` or first column → Series

    **Per-instrument mode** (``per_instrument=True``):
      The file contains a ``pd.DataFrame`` with a ``DatetimeIndex`` and **columns
      named after the instrument tickers** in the universe.  Each column is
      assigned to its matching instrument, so different instruments can carry
      different pre-computed signals.

      Use this when signals are fundamentally per-asset (e.g. QQQ SMA filter for
      TQQQ, TLT SMA filter for TMF).  Any instrument not present in the file
      columns receives ``NaN`` for the full period.

      Example — SMA trend-filter for a 3-asset universe::

          sma_filter = ExternalFactor(
              name="sma_eligible",
              path="data/signals/sma_eligible.parquet",
              # parquet columns: TQQQ, TMF, IEF (one boolean per day per ticker)
              per_instrument=True,
          )

    Primary use-case (broadcast): ML model outputs (e.g. TKAN predictions).
    Primary use-case (per_instrument): per-asset SMA / trend filters loaded from
    pre-computed parquet.

    Example (broadcast)::

        tkan_pred = ExternalFactor(
            name="tkan_pred",
            path="research/Index Directional/tkan/v3/weights/pred_cache.pkl",
            column=None,       # file is a plain Series
        )
    """

    path: str
    column: Optional[str] = None
    per_instrument: bool = False
    # Optional callable: loader(obj: Any) -> pd.Series  (broadcast mode)
    #                 or loader(obj: Any) -> pd.DataFrame  (per_instrument mode)
    # Where obj is the raw deserialized object from disk.
    # Use this for non-standard formats (TKAN prediction tuples, HDF5
    # compound objects, custom NN output structures, etc.).
    # If None, the engine applies standard rules.
    loader: Optional[Callable[[Any], "pd.Series"]] = field(default=None, compare=False, repr=False)


@dataclass(slots=True)
class FieldFactor(FactorNode):
    """
    Extract a non-standard field from the loaded data as a factor.

    Used to expose auxiliary columns that are loaded alongside OHLCV but are
    not covered by any of the purpose-built factor types (e.g. implied volatility,
    fundamental ratios, macro time-series already joined into the price DataFrame).

    Example::

        ivol = FieldFactor(name="ivol_3m", field="3m_50d_ivol")
    """

    field: str
