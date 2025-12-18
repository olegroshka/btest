# src/quantdsl_backtest/dsl/backtest_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TYPE_CHECKING

# Avoid runtime dependency on engine.analytics in the DSL package.
# Use TYPE_CHECKING to only import for typing purposes.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..engine.analytics.types import SignalAnalyticsConfig  # noqa: F401
    from ..engine.analytics.types import StrategyAnalyticsConfig  # noqa: F401


@dataclass(slots=True)
class MarginConfig:
    """
    Margin requirements for long and short positions.
    """

    long_initial: float = 0.5           # fraction of notional
    short_initial: float = 0.5
    maintenance: float = 0.3


@dataclass(slots=True)
class RiskChecks:
    """
    High-level risk/tail-stop controls for the backtest.
    """

    # DEPRECATED: prefer `drawdown` policy below. If set, treated as a
    # hard kill-switch threshold (liquidate and halt trading at breach).
    max_drawdown: Optional[float] = None       # e.g. 0.25 = 25% DD
    max_gross_leverage: Optional[float] = None # e.g. <= 3x gross
    max_daily_loss: Optional[float] = None     # e.g. 0.05 = -5% daily loss triggers cooldown
    # New drawdown policy (preferred over max_drawdown)
    drawdown: Optional["DrawdownPolicy"] = None


@dataclass(slots=True)
class DrawdownPolicy:
    """
    Configurable drawdown risk behaviour.

    Modes:
      - "none": no global DD limit; rely on other controls (e.g., max_daily_loss).
      - "hard_kill": if DD magnitude >= `threshold`, liquidate and halt trading.
      - "soft_scale": as DD increases beyond `start`, scale gross exposure from 1→0,
                       reaching 0 at `full`. Scaling curve can be linear/quadratic/sqrt.
    """

    mode: Literal["none", "hard_kill", "soft_scale"] = "none"

    # hard_kill parameter
    threshold: Optional[float] = None  # 0.35 → 35%

    # soft_scale parameters
    start: Optional[float] = None      # begin de-risking at e.g. 0.10 (10%)
    full: Optional[float] = None       # fully flat at e.g. 0.35 (35%)
    curve: Literal["linear", "quadratic", "sqrt"] = "linear"


@dataclass(slots=True)
class Reporting:
    """
    Output controls: what to store, what metrics to compute.
    """

    store_trades: bool = True
    store_positions: bool = True
    metrics: List[str] = field(default_factory=list)
    # Preferred location for configuring signal analytics / attribution.
    # Annotated as string to avoid runtime import dependency on engine.analytics.
    signal_analytics: Optional["SignalAnalyticsConfig"] = None

    # Strategy-level analytics (QuantStats): metrics summary + HTML tearsheet
    # Optional and only evaluated if present and enabled.
    strategyAnalytics: Optional["StrategyAnalyticsConfig"] = None


@dataclass(slots=True)
class BacktestConfig:
    """
    Global runtime configuration for the backtest engine.
    """

    engine: Literal["event_driven", "vectorized"] = "event_driven"
    cash_initial: float = 1_000_000.0
    margin: MarginConfig = field(default_factory=MarginConfig)
    risk_checks: RiskChecks = field(default_factory=RiskChecks)
    reporting: Reporting = field(default_factory=Reporting)

    # Optional metadata / knobs
    #
    # Use this dictionary for advanced/experimental switches that affect engine behavior
    # but are not yet promoted to first‑class fields. Known keys:
    #
    #   - "hold_when_no_targets": bool (default False)
    #       Controls how the event‑driven engine behaves on a rebalance date
    #       when the selector produces an empty set of targets (i.e., desired
    #       target weights are all zeros) and trading is not halted by risk policy.
    #
    #       False (default, legacy behavior):
    #           Liquidate to cash on that day (engine rebalances into zero weights).
    #           This can reduce exposure on sporadic empty‑selection days but may
    #           introduce “spurious” flat days if emptiness is caused by data gaps
    #           or overly strict filters.
    #
    #       True:
    #           Carry forward the previous positions instead of liquidating when the
    #           selection is empty. This avoids unexpected flat days due to temporary
    #           selection emptiness, but increases the number of invested days and can
    #           change returns/Sharpe if those additional days are adverse.
    #
    #       Note: Currently applied by the event‑driven engine; the vectorized engine
    #       may not honor this flag yet.
    extra: Dict[str, object] = field(default_factory=dict)

    # DEPRECATED: prefer Reporting.signal_analytics. Kept for backward compatibility.
    # Defined in engine.analytics.types.SignalAnalyticsConfig; annotated as string to avoid import cycle.
    signal_analytics: Optional["SignalAnalyticsConfig"] = None
