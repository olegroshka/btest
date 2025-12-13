from __future__ import annotations

import math

import pandas as pd

from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest


def test_indices_example_trades_and_has_variance():
    """
    Smoke test for the indices example strategy:
      - Runs the strategy over a recent 5-year window
      - Asserts that turnover and return variance are non-zero

    This guards against regressions where sparse ranks/masks yield zero target
    weights and a flat equity curve.
    """
    strat = build_strategy()

    # Constrain to a period with good overlap; keep engine event-driven
    strat.data.start = "2018-01-01"
    strat.data.end = "2023-01-01"
    strat.backtest.engine = "event_driven"

    result = run_backtest(strat)

    # Basic sanity
    assert len(result.equity) > 10
    assert len(result.returns) == len(result.equity)

    # Non-zero turnover and some variance in returns
    metrics = result.metrics
    turnover_annual = float(metrics.get("turnover_annual", 0.0))
    assert turnover_annual > 0.0, "Expected non-zero turnover for indices example"

    std_ret = float(pd.Series(result.returns).std())
    assert std_ret > 0.0 and math.isfinite(std_ret), "Returns should have non-zero std"
