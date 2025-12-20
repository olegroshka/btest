import numpy as np
import pandas as pd
import logging

from quantdsl_backtest.engine.accounting import compute_basic_metrics


def test_metrics_clipping_negative_warns_and_caps_max_dd(caplog):
    # Construct returns with an extreme negative outlier (-90%)
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])  # 3 days
    # Start with a flat day (0.0) to mirror engine behavior (first day ret = 0)
    returns = pd.Series([0.0, -0.9, 0.0], index=idx, dtype="float64")

    # Equity path (raw), used only for total_return here
    equity = pd.Series([100.0, 100.0, 10.0], index=idx, dtype="float64")

    with caplog.at_level(logging.WARNING):
        metrics = compute_basic_metrics(returns=returns, equity=equity, weights=None)

    # Expect a clipping warning
    warnings = [rec for rec in caplog.records if "Clipped" in rec.getMessage()]
    assert len(warnings) >= 1

    # Max drawdown should reflect clipped limit (-30%)
    assert abs(metrics["max_drawdown"] - (-0.3)) < 1e-9


def test_metrics_clipping_positive_warns(caplog):
    # Construct returns with an extreme positive outlier (+500%)
    idx = pd.to_datetime(["2020-02-01", "2020-02-02"])  # 2 days
    returns = pd.Series([0.0, 5.0], index=idx, dtype="float64")

    # Equity path (raw), used only for total_return
    equity = pd.Series([100.0, 100.0], index=idx, dtype="float64")

    with caplog.at_level(logging.WARNING):
        metrics = compute_basic_metrics(returns=returns, equity=equity, weights=None)

    # Expect a clipping warning due to +500% return
    warnings = [rec for rec in caplog.records if "Clipped" in rec.getMessage()]
    assert len(warnings) >= 1

    # Metrics should be finite numbers
    for k in ("sharpe", "sortino", "max_drawdown", "total_return"):
        v = metrics[k]
        assert v == v  # not NaN