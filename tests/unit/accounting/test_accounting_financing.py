from __future__ import annotations

import pandas as pd

from quantdsl_backtest.engine.accounting import apply_carry_costs
from quantdsl_backtest.dsl.costs import BorrowCost, FinancingCost


def test_financing_applies_on_positive_cash():
    # With positive cash and non-zero spread_bps, financing should accrue (earnings)
    positions = pd.Series({"A": 0.0})
    prices = pd.Series({"A": 100.0})
    cash0 = 1_000_000.0
    borrow = BorrowCost(default_annual_rate=0.0)
    financing = FinancingCost(base_rate_curve="SOFR", spread_bps=300.0)

    new_cash, borrow_cost, financing_pnl = apply_carry_costs(
        positions=positions,
        prices=prices,
        cash=cash0,
        borrow=borrow,
        financing=financing,
        dt_years=1.0 / 252.0,
    )

    assert borrow_cost == 0.0
    assert financing_pnl > 0.0
    assert new_cash > cash0


def test_financing_cost_when_cash_negative():
    # With negative cash and non-zero spread_bps, financing should apply as a cost (negative PnL)
    positions = pd.Series({"A": 0.0})
    prices = pd.Series({"A": 100.0})
    cash0 = -1_000_000.0
    borrow = BorrowCost(default_annual_rate=0.0)
    financing = FinancingCost(base_rate_curve="SOFR", spread_bps=300.0)

    new_cash, borrow_cost, financing_pnl = apply_carry_costs(
        positions=positions,
        prices=prices,
        cash=cash0,
        borrow=borrow,
        financing=financing,
        dt_years=1.0 / 252.0,
    )

    assert borrow_cost == 0.0
    assert financing_pnl < 0.0  # cost on negative cash
    assert new_cash < cash0
