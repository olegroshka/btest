import numpy as np
import pandas as pd

from quantdsl_backtest.engine.accounting import (
    mark_to_market,
    apply_carry_costs,
    compute_exposures,
    compute_basic_metrics,
)
from quantdsl_backtest.dsl.costs import BorrowCost, FinancingCost


def test_mark_to_market_long_short_and_equity_before():
    names = ["A", "B"]
    prev_pos = pd.Series({"A": 10.0, "B": -5.0})
    prev_px = pd.Series({"A": 100.0, "B": 50.0})
    curr_px = pd.Series({"A": 110.0, "B": 45.0})
    prev_cash = 1000.0

    equity_before, price_pnl = mark_to_market(prev_pos, prev_px, curr_px, prev_cash)

    # Price PnL = 10*(110-100) + (-5)*(45-50) = 100 + 25 = 125
    assert np.isclose(price_pnl, 125.0)

    # Equity before trades = cash + sum(curr_px * positions)
    # = 1000 + (10*110 + (-5)*45) = 1000 + (1100 - 225) = 1875
    assert np.isclose(equity_before, 1875.0)


def test_apply_carry_costs_borrow_on_shorts_and_financing_on_cash():
    # One short, one long; ensure borrow costs apply to shorts and financing on cash
    positions = pd.Series({"A": -5.0, "B": 10.0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    cash = 2000.0

    borrow = BorrowCost(default_annual_rate=0.25)  # 25% p.a. on shorts
    financing = FinancingCost(base_rate_curve="SOFR", spread_bps=100.0)  # 1% p.a. on cash

    new_cash, borrow_cost, fin_pnl = apply_carry_costs(positions, prices, cash, borrow, financing)

    # Short notional = 5 * 100 = 500 -> borrow cost ~ 500 * 0.25 / 252
    expected_borrow = 500.0 * 0.25 / 252.0
    # Financing on cash = 2000 * 0.01 / 252
    expected_fin = 2000.0 * 0.01 / 252.0

    assert np.isclose(borrow_cost, expected_borrow)
    assert np.isclose(fin_pnl, expected_fin)
    assert np.isclose(new_cash, cash - expected_borrow + expected_fin)


def test_compute_exposures_values():
    positions = pd.Series({"A": 2.0, "B": -3.0, "C": 0.0})
    prices = pd.Series({"A": 10.0, "B": 20.0, "C": 5.0})

    exps = compute_exposures(positions, prices)

    # Notionals: A=20, B=-60, C=0 -> long=20, short=-60
    assert np.isclose(exps["long_exposure"], 20.0)
    assert np.isclose(exps["short_exposure"], -60.0)
    assert np.isclose(exps["gross_exposure"], 80.0)
    assert np.isclose(exps["net_exposure"], -40.0)


def test_compute_basic_metrics_total_return_drawdown_turnover():
    # Construct small returns and equity series
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    rets = pd.Series([0.0, 0.01, -0.005, 0.02], index=idx)

    equity = pd.Series(index=idx, dtype=float)
    equity.iloc[0] = 100.0
    for i in range(1, len(idx)):
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + rets.iloc[i])

    # Weights to generate non-zero turnover
    weights = pd.DataFrame(
        [
            {"A": 0.5, "B": -0.5},
            {"A": 0.0, "B": 0.0},
            {"A": -0.5, "B": 0.5},
            {"A": -0.5, "B": 0.5},
        ],
        index=idx,
        dtype=float,
    )

    metrics = compute_basic_metrics(rets, equity, weights)

    # Total return from equity series
    expected_tr = equity.iloc[-1] / equity.iloc[0] - 1.0
    assert np.isclose(metrics["total_return"], expected_tr)

    # Max drawdown from cumulative returns
    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    expected_dd = float(dd.min())
    assert np.isclose(metrics["max_drawdown"], expected_dd)

    # Sharpe should be finite and non-negative for these returns
    assert np.isfinite(metrics["sharpe"]) and metrics["sharpe"] >= 0.0

    # Turnover: 0.5 * mean(sum(abs(diff))) daily, annualized.
    # Align with implementation which effectively treats the first NaN as 0 when averaging.
    diffs = weights.diff().abs().sum(axis=1) * 0.5
    expected_turnover_annual = diffs.fillna(0.0).mean() * 252.0
    assert np.isclose(metrics["turnover_annual"], expected_turnover_annual)
