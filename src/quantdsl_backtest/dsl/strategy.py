# src/quantdsl_backtest/dsl/strategy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .data_config import DataConfig
from .signals import SignalNode
from .universe import Universe
from .factors import FactorNode
from .portfolio import LongShortPortfolio
from .execution import Execution
from .costs import Costs
from .backtest_config import BacktestConfig


@dataclass(slots=True)
class Strategy:
    """
    Top-level DSL object describing a full trading strategy
    and how it should be backtested.
    """

    name: str

    data: DataConfig
    universe: Universe

    # Named factor and signal definitions.
    factors: Dict[str, FactorNode]
    signals: Dict[str, SignalNode]          # you’ll replace `object` with a proper base class later

    portfolio: LongShortPortfolio
    execution: Execution
    costs: Costs
    backtest: BacktestConfig
