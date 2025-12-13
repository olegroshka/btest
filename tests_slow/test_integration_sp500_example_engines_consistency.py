import numpy as np
import pytest

from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.accounting import compute_basic_metrics


# Use the example strategy definition directly
from quantdsl_backtest.examples.momentum_long_short_sp500 import build_strategy


@pytest.mark.slow
def test_sp500_example_event_vs_vectorized_engines():
    """
    Build the S&P 500 momentum long/short example strategy twice with different
    engines (event_driven vs vectorized) and assert that equity/returns/metrics
    are nearly identical. This mirrors other engine-consistency tests but uses
    the full example strategy with frictions.

    This test is expected to FAIL currently due to discrepancies reported in
    manual runs (e.g., turnover reported as 0.00 in the vectorized engine).
    """

    # Build two independent strategy instances to avoid shared mutation
    strat_event = build_strategy()
    strat_event.backtest.engine = "event_driven"

    strat_vec = build_strategy()
    strat_vec.backtest.engine = "vectorized"

    # Run both engines
    res_event = run_backtest(strat_event)
    res_vec = run_backtest(strat_vec)

    # Align indices
    equity_event = res_event.equity
    equity_vec = res_vec.equity.reindex(equity_event.index).ffill()

    returns_event = res_event.returns
    returns_vec = res_vec.returns.reindex(returns_event.index).fillna(0.0)

    # Sanity
    assert len(equity_event) > 0 and len(equity_vec) > 0
    assert len(returns_event) > 0 and len(returns_vec) > 0

    # Compute comparable metrics using shared accounting logic
    metrics_event = compute_basic_metrics(returns_event, equity_event, res_event.weights)
    metrics_vec = compute_basic_metrics(returns_vec, equity_vec, res_vec.weights)

    # Equity curves should be almost identical
    np.testing.assert_allclose(
        equity_event.values,
        equity_vec.values,
        rtol=1e-6,
        atol=1e-2,  # a few cents on 100mm
        err_msg="Equity curves differ between event_driven and vectorized engines (SP500 example)",
    )

    # Daily returns should be almost identical
    np.testing.assert_allclose(
        returns_event.values,
        returns_vec.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Daily returns differ between event_driven and vectorized engines (SP500 example)",
    )

    # Total return, max drawdown, and turnover should be close
    assert abs(metrics_event["total_return"] - metrics_vec["total_return"]) < 1e-6
    assert abs(metrics_event["max_drawdown"] - metrics_vec["max_drawdown"]) < 1e-6
    assert abs(metrics_event.get("turnover_annual", 0.0) - metrics_vec.get("turnover_annual", 0.0)) < 1e-6
