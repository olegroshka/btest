# src/quantdsl_backtest/engine/__init__.py

from .backtest_runner import run_backtest  # re-export for convenience
from .results import BacktestResult

__all__ = ["run_backtest", "BacktestResult"]
