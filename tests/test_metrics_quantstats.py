# tests/test_metrics_quantstats.py

import math

import numpy as np
import pandas as pd
import pytest

qs = pytest.importorskip("quantstats")  # skip all tests here if quantstats missing

from quantdsl_backtest.engine.metrics_quantstats import compute_quantstats_metrics
from quantdsl_backtest.engine.results import BacktestResult


def _make_dummy_backtest_result(n_days: int = 252) -> BacktestResult:
    """Construct a minimal BacktestResult for testing QuantStats integration."""
    index = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Simple random-walk equity with positive drift
    rng = np.random.default_rng(42)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    equity = 100.0 * (1.0 + pd.Series(daily_rets, index=index)).cumprod()

    # Dummy exposures / positions / weights
    zero_series = pd.Series(0.0, index=index)
    zero_df = pd.DataFrame(0.0, index=index, columns=["DUMMY"])

    return BacktestResult(
        equity=equity,
        returns=pd.Series(daily_rets, index=index),
        cash=zero_series.copy(),
        gross_exposure=zero_series.copy(),
        net_exposure=zero_series.copy(),
        long_exposure=zero_series.copy(),
        short_exposure=zero_series.copy(),
        leverage=zero_series.copy(),
        positions=zero_df.copy(),
        weights=zero_df.copy(),
        trades=pd.DataFrame(
            columns=[
                "datetime",
                "instrument",
                "side",
                "quantity",
                "price",
                "notional",
                "slippage_bps",
                "commission",
                "fees",
                "realized_pnl",
            ]
        ),
        metrics={},
        start_date=index[0],
        end_date=index[-1],
        benchmark=None,
        metadata={"strategy_name": "dummy_strategy"},
    )


def test_compute_quantstats_metrics_basic():
    """compute_quantstats_metrics should return scalar floats for known metrics."""
    index = pd.date_range("2020-01-01", periods=252, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, size=len(index)), index=index)

    metric_names = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility"]

    metrics = compute_quantstats_metrics(
        returns=rets,
        metric_names=metric_names,
        benchmark=None,
        risk_free=0.0,
    )

    # We must get exactly the requested keys
    assert set(metrics.keys()) == set(metric_names)

    # And all values must be finite floats
    for name, value in metrics.items():
        assert isinstance(value, float)
        assert math.isfinite(value), f"Metric {name} is not finite: {value!r}"


def test_backtestresult_quantstats_metrics_wrapper():
    """BacktestResult.quantstats_metrics should delegate to compute_quantstats_metrics."""
    result = _make_dummy_backtest_result()
    metric_names = ["cagr", "sharpe", "max_drawdown"]

    # Call the wrapper
    qs_series = result.quantstats_metrics(metric_names, risk_free=0.0)

    # Keys should be prefixed with 'qs_' by default
    expected_index = {f"qs_{m}" for m in metric_names}
    assert set(qs_series.index) == expected_index

    # Compare against direct call to compute_quantstats_metrics
    direct = compute_quantstats_metrics(result.returns, metric_names, benchmark=None, risk_free=0.0)

    for m in metric_names:
        wrapped_val = qs_series[f"qs_{m}"]
        direct_val = direct[m]
        # They should be numerically very close
        assert math.isfinite(wrapped_val)
        assert math.isfinite(direct_val)
        assert abs(wrapped_val - direct_val) < 1e-9


def test_backtestresult_quantstats_tearsheet(tmp_path):
    """quantstats_tearsheet should generate an HTML file without errors."""
    result = _make_dummy_backtest_result()

    output_path = tmp_path / "tearsheet.html"
    # Just ensure this doesn't raise and file is created
    result.quantstats_tearsheet(
        output=str(output_path),
        title="Dummy Strategy Tearsheet",
    )

    assert output_path.exists()
    # Basic sanity: file non-empty
    assert output_path.stat().st_size > 0
