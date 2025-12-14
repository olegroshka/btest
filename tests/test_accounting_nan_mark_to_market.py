import numpy as np
import pandas as pd

from quantdsl_backtest.engine.accounting import mark_to_market


def test_mark_to_market_nan_curr_prices_keeps_prev_mark():
    instruments = ["A", "B", "C"]
    prev_positions = pd.Series([10.0, -5.0, 0.0], index=instruments, dtype="float64")
    prev_prices = pd.Series([100.0, 200.0, 300.0], index=instruments, dtype="float64")
    curr_prices = pd.Series([110.0, np.nan, 300.0], index=instruments, dtype="float64")
    prev_cash = 1_000.0

    equity_before, price_pnl = mark_to_market(
        prev_positions=prev_positions,
        prev_prices=prev_prices,
        curr_prices=curr_prices,
        prev_cash=prev_cash,
    )

    # PnL: only A contributes (10 * (110-100) = 100). B is NaN today -> PnL 0.
    assert abs(price_pnl - 100.0) < 1e-9

    # Equity_before: cash + previous notional (10*100 + -5*200 + 0*300 = 0) + pnl(100) = 1100
    assert abs(equity_before - 1100.0) < 1e-9


def test_mark_to_market_prev_nan_excludes_from_pnl_and_prev_notional():
    instruments = ["A", "B", "C"]
    prev_positions = pd.Series([0.0, 10.0, 0.0], index=instruments, dtype="float64")
    prev_prices = pd.Series([100.0, np.nan, 300.0], index=instruments, dtype="float64")
    curr_prices = pd.Series([105.0, 210.0, 310.0], index=instruments, dtype="float64")
    prev_cash = 0.0

    equity_before, price_pnl = mark_to_market(
        prev_positions=prev_positions,
        prev_prices=prev_prices,
        curr_prices=curr_prices,
        prev_cash=prev_cash,
    )

    # B is invalid due to prev NaN -> no PnL from B; A and C have zero position -> PnL 0.
    assert abs(price_pnl - 0.0) < 1e-12

    # Equity_before: only previous marks with known prev prices count, but positions there are 0 -> 0
    assert abs(equity_before - 0.0) < 1e-12
