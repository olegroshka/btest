"""Per-instrument borrow rates in BorrowCost, by mapping or by file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.costs import BorrowCost, FinancingCost
from quantdsl_backtest.engine.accounting import apply_carry_costs, resolve_borrow_rates


def test_rates_mapping_charges_each_short_at_its_own_rate_and_the_rest_at_the_default() -> (
    None
):
    positions = pd.Series({"A": -5.0, "B": 10.0, "C": -2.0})
    prices = pd.Series({"A": 100.0, "B": 50.0, "C": 25.0})
    borrow = BorrowCost(default_annual_rate=0.01, rates={"A": 0.10})
    financing = FinancingCost(base_rate_curve="SOFR", spread_bps=0.0)
    new_cash, cost, fin = apply_carry_costs(
        positions, prices, 1000.0, borrow, financing
    )
    expected = (
        500.0 * 0.10 + 50.0 * 0.01
    ) / 252.0  # A at its own rate, C at the default, B long
    assert (
        np.isclose(cost, expected)
        and fin == 0.0
        and np.isclose(new_cash, 1000.0 - expected)
    )
    rates = resolve_borrow_rates(borrow, positions.index)
    assert rates.tolist() == pytest.approx([0.10, 0.01, 0.01])


def test_rates_file_is_read_once_and_a_bad_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rates.parquet"
    pd.DataFrame({"ticker": ["A", "C"], "fee_rate": [0.05, 0.02]}).to_parquet(
        path, index=False
    )
    borrow = BorrowCost(default_annual_rate=0.01, rates_path=str(path))
    positions = pd.Series({"A": -1.0, "B": -1.0, "C": -1.0})
    prices = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    _, cost, _ = apply_carry_costs(
        positions, prices, 0.0, borrow, FinancingCost(base_rate_curve="SOFR")
    )
    assert np.isclose(cost, (100.0 * 0.05 + 100.0 * 0.01 + 100.0 * 0.02) / 252.0)
    path.unlink()  # the second call must not read the file again
    _, again, _ = apply_carry_costs(
        positions, prices, 0.0, borrow, FinancingCost(base_rate_curve="SOFR")
    )
    assert np.isclose(again, cost)
    csv = tmp_path / "rates.csv"
    csv.write_text("symbol,rate\nA,0.5\n", encoding="utf-8")
    custom = BorrowCost(
        rates_path=str(csv), instrument_column="symbol", rate_column="rate"
    )
    assert resolve_borrow_rates(custom, pd.Index(["A", "B"])).tolist() == pytest.approx(
        [0.5, 0.0]
    )
    with pytest.raises(ValueError, match="column"):
        resolve_borrow_rates(BorrowCost(rates_path=str(csv)), pd.Index(["A"]))


def test_flat_rate_is_unchanged_without_per_name_rates() -> None:
    positions = pd.Series({"A": -5.0, "B": 10.0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    borrow = BorrowCost(default_annual_rate=0.25)
    _, cost, _ = apply_carry_costs(
        positions, prices, 0.0, borrow, FinancingCost(base_rate_curve="SOFR")
    )
    assert np.isclose(cost, 500.0 * 0.25 / 252.0)
    assert resolve_borrow_rates(borrow, positions.index).tolist() == [0.25, 0.25]
