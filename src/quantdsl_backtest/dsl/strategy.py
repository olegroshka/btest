# src/quantdsl_backtest/dsl/strategy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from .backtest_config import BacktestConfig
from .costs import Costs
from .data_config import DataConfig
from .execution import Execution
from .factors import FactorNode
from .portfolio import LongShortPortfolio, TargetWeights, TimingPortfolio
from .signals import SignalNode
from .universe import Universe


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
    signals: Dict[
        str, SignalNode
    ]  # you’ll replace `object` with a proper base class later

    portfolio: Union[LongShortPortfolio, TimingPortfolio, TargetWeights]
    execution: Execution
    costs: Costs
    backtest: BacktestConfig
