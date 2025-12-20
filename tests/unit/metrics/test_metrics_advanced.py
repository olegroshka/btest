import math

import numpy as np
import pandas as pd

from quantdsl_backtest.engine.metrics_advanced import compute_advanced_metrics_from_result
from quantdsl_backtest.engine.results import BacktestResult


def _mk_result(*, equity: pd.Series, returns: pd.Series, leverage: pd.Series | None = None, gross_exposure: pd.Series | None = None, metrics: dict | None = None) -> BacktestResult:
    idx = equity.index
    instruments = ["A"]
    zeros = pd.Series(np.zeros(len(idx)), index=idx)

    if leverage is None:
        leverage = zeros
    if gross_exposure is None:
        gross_exposure = zeros

    return BacktestResult(
        equity=equity,
        returns=returns,
        cash=equity * 0.0,
        gross_exposure=gross_exposure,
        net_exposure=zeros,
        long_exposure=zeros,
        short_exposure=zeros,
        leverage=leverage,
        positions=pd.DataFrame(0.0, index=idx, columns=instruments),
        weights=pd.DataFrame(0.0, index=idx, columns=instruments),
        trades=pd.DataFrame(),
        metrics=metrics or {},
        start_date=idx[0],
        end_date=idx[-1],
        benchmark=None,
        metadata={},
    )


def test_advanced_metrics_win_rate_profit_factor_tail_ratio():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    # returns: +, +, -, 0, -
    rets = pd.Series([0.01, 0.02, -0.01, 0.0, -0.02], index=idx)
    eq = (1.0 + rets).cumprod()

    res = _mk_result(equity=eq, returns=rets)
    m = compute_advanced_metrics_from_result(res)

    assert math.isfinite(m["win_rate"])
    assert abs(m["win_rate"] - (2 / 5)) < 1e-12

    # profit factor: sum(pos)=0.03, sum(neg)=-0.03 => 1.0
    assert math.isfinite(m["profit_factor"])
    assert abs(m["profit_factor"] - 1.0) < 1e-12

    # tail ratio should be finite for this simple series
    assert "tail_ratio" in m


def test_advanced_metrics_ulcer_index_zero_for_monotonic_equity():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([1.0, 1.1, 1.2, 1.25, 1.3], index=idx)
    rets = eq.pct_change().fillna(0.0)

    res = _mk_result(equity=eq, returns=rets)
    m = compute_advanced_metrics_from_result(res)

    assert math.isfinite(m["ulcer_index"])
    assert abs(m["ulcer_index"]) < 1e-12


def test_advanced_metrics_calmar_uses_existing_max_drawdown_and_cagr():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    # Force a drawdown: 1.0 -> 1.2 -> 0.9 -> 1.0 -> 1.1
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 1.1], index=idx)
    rets = eq.pct_change().fillna(0.0)

    # max_drawdown here is -0.25 (from 1.2 to 0.9)
    res = _mk_result(equity=eq, returns=rets, metrics={"max_drawdown": -0.25, "cagr": 0.10})
    m = compute_advanced_metrics_from_result(res)

    assert math.isfinite(m["calmar"])
    assert abs(m["calmar"] - (0.10 / 0.25)) < 1e-12


def test_advanced_metrics_leverage_and_days_in_market():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    eq = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    rets = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    lev = pd.Series([0.0, 1.0, 2.0, 1.0], index=idx)
    gross = pd.Series([0.0, 10.0, 10.0, 0.0], index=idx)

    res = _mk_result(equity=eq, returns=rets, leverage=lev, gross_exposure=gross)
    m = compute_advanced_metrics_from_result(res)

    assert abs(m["avg_leverage"] - 1.0) < 1e-12
    assert abs(m["max_leverage"] - 2.0) < 1e-12
    assert abs(m["pct_days_in_market"] - 0.5) < 1e-12

