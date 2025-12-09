# src/quantdsl_backtest/dsl/backtest_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


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

    max_drawdown: Optional[float] = None       # e.g. 0.25 = 25% DD
    max_gross_leverage: Optional[float] = None # e.g. <= 3x gross


@dataclass(slots=True)
class Reporting:
    """
    Output controls: what to store, what metrics to compute.
    """

    store_trades: bool = True
    store_positions: bool = True
    metrics: List[str] = field(default_factory=list)


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
    extra: Dict[str, object] = field(default_factory=dict)
