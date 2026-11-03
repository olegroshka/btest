"""Costs and limits that only bite a small retail account: per-order minimums, a monthly activity fee, no borrowing."""
import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.costs import BorrowCost, Commission, Costs, FinancingCost, StaticFees, TransactionTax
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import TargetWeights
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe
import quantdsl_backtest.engine.backtest_runner as br
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.execution_engine import rebalance_to_target_weights
from quantdsl_backtest.engine.vectorized_engine import run_backtest_vectorized
from quantdsl_backtest.models.costs import build_cost_model


def _execution(**policy) -> Execution:
    return Execution(
        order_policy=OrderPolicy(**policy),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),
    )


def _rebalance(*, cash, prev, target, commission=None, tax=None, prices=None, **policy):
    names = list(target)
    px = pd.Series(prices or {n: 100.0 for n in names})
    return rebalance_to_target_weights(
        pd.Timestamp("2024-01-02"),
        _execution(**policy),
        commission or Commission(type="bps_notional", amount=0.0),
        StaticFees(),
        cash,
        px,
        pd.Series({n: 1e9 for n in names}),
        pd.Series(prev, dtype=float).reindex(names).fillna(0.0),
        pd.Series(target, dtype=float),
        tax=tax,
    )


@pytest.mark.parametrize(
    "notional, expected",
    [
        (100.0, 3.0),  # the floor
        (6000.0, 3.0),  # exactly where 5 bps meets the floor
        (20000.0, 10.0),  # 5 bps above it
    ],
)
def test_commission_is_the_larger_of_rate_and_minimum(notional, expected):
    model = build_cost_model(Commission(type="bps_notional", amount=5.0, min_per_order=3.0))
    assert model.commission_from_trade(qty=notional / 10.0, exec_price=10.0) == pytest.approx(expected)


def test_commission_minimum_without_a_rate_and_no_charge_without_a_trade():
    model = build_cost_model(Commission(type="bps_notional", amount=0.0, min_per_order=3.0))
    assert model.commission_from_trade(qty=1.0, exec_price=10.0) == pytest.approx(3.0)
    assert model.commission_from_trade(qty=0.0, exec_price=10.0) == 0.0


def test_commission_without_a_minimum_is_unchanged():
    model = build_cost_model(Commission(type="bps_notional", amount=10.0))
    assert model.commission_from_trade(qty=1.0, exec_price=100.0) == pytest.approx(0.1)


def test_fx_charge_has_a_minimum_and_touches_only_the_named_instruments():
    fx = TransactionTax(rate=0.0003, on="both", min_per_trade=1.5, instruments=frozenset({"F"}))
    _, _, trades = _rebalance(cash=20000.0, prev={}, target={"F": 0.05, "H": 0.05}, tax=[fx])
    by_name = trades.set_index("instrument")["tax"]
    assert by_name["F"] == pytest.approx(1.5)  # 3 bps of 1,000 is 0.30, below the floor
    assert by_name["H"] == 0.0


def test_fx_charge_above_the_minimum_and_on_a_sale():
    fx = TransactionTax(rate=0.0003, on="both", min_per_trade=1.5, instruments=frozenset({"F"}))
    _, _, buy = _rebalance(cash=20000.0, prev={}, target={"F": 1.0}, tax=[fx])
    assert buy["tax"].iloc[0] == pytest.approx(6.0)
    _, _, sell = _rebalance(cash=0.0, prev={"F": 100.0}, target={"F": 0.0}, tax=[fx])
    assert sell["tax"].iloc[0] == pytest.approx(3.0)


def test_stamp_duty_style_tax_is_unchanged():
    stamp = TransactionTax(rate=0.005, on="buy")
    _, _, trades = _rebalance(cash=10000.0, prev={}, target={"A": 1.0}, tax=stamp)
    assert trades["tax"].iloc[0] == pytest.approx(50.0)


# Fully invested at 10,000. A, B and C are each 2 points over target, inside a 5 point band, so they are not
# sold; D is 6 points under, so it buys with no cash to spend. This is what the gate exists to stop.
DRIFTED = dict(cash=0.0, prev={"A": 27.0, "B": 27.0, "C": 27.0, "D": 19.0}, target={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}, no_trade_band=0.05)


def test_without_cash_only_a_blocked_sale_lets_the_buy_borrow():
    _, delta, trades = _rebalance(**DRIFTED)
    assert list(trades["instrument"]) == ["D"]
    assert delta == pytest.approx(-600.0)


def test_cash_only_stops_that_borrowing():
    _, delta, trades = _rebalance(**DRIFTED, cash_only=True)
    assert delta == 0.0
    assert trades.empty


def test_cash_only_spends_sale_proceeds_after_all_costs():
    commission = Commission(type="bps_notional", amount=5.0, min_per_order=3.0)
    fx = TransactionTax(rate=0.0003, on="both", min_per_trade=1.5, instruments=frozenset({"B"}))
    # Fully invested, moving 40 points from A to B: costs must come out of what is bought.
    _, delta, trades = _rebalance(
        cash=0.0,
        prev={"A": 80.0, "B": 20.0},
        target={"A": 0.5, "B": 0.5},
        commission=commission,
        tax=[fx],
        cash_only=True,
    )
    assert trades.iloc[0]["side"] == "SELL"  # sells run first
    assert delta >= -1e-9  # cash cannot go negative
    assert delta < 15.0  # and the proceeds were spent, bar what the conservative cost estimate holds back


def test_cash_only_shares_scarce_cash_pro_rata():
    new, delta, trades = _rebalance(cash=1000.0, prev={"A": 90.0}, target={"A": 0.9, "B": 0.5, "C": 0.5}, cash_only=True)
    bought = trades.set_index("instrument")["notional"]
    assert bought["B"] == pytest.approx(bought["C"])
    assert delta == pytest.approx(-1000.0)


def test_cash_only_never_overdraws_when_the_floor_exceeds_the_budget():
    commission = Commission(type="bps_notional", amount=5.0, min_per_order=3.0)
    new, delta, trades = _rebalance(cash=2.0, prev={"A": 100.0}, target={"A": 10000.0 / 10002.0, "B": 0.2}, commission=commission, cash_only=True)
    assert delta >= -2.0 - 1e-9
    assert new["B"] == 0.0


IDX = pd.bdate_range("2024-01-02", "2024-04-30")


def _strategy(amount=0.0, floor=0.0, monthly=0.0, frequency="1m"):
    weights = pd.DataFrame(1.0, index=IDX, columns=["A"])
    return Strategy(
        name="small_account",
        data=DataConfig(source="dummy://", calendar="XLON", frequency="1d", start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="u", static_instruments=["A"]),
        factors={"ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple")},
        signals={},
        portfolio=TargetWeights(weights=weights, rebalance_frequency=frequency, signal_delay_bars=0, rebalance_on_change=True),
        execution=_execution(fill_on="close", cash_only=True),
        costs=Costs(
            commission=Commission(type="bps_notional", amount=amount, min_per_order=floor),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(min_monthly_fee=monthly),
        ),
        backtest=BacktestConfig(cash_initial=15000.0),
    )


def _run(monkeypatch, strategy):
    strategy.backtest.reporting.output_dir = None
    frame = pd.DataFrame({"close": 100.0, "volume": 1e9}, index=IDX)
    market = MarketData(bars={"A": frame}, instruments=["A"], fields=["close", "volume"], frequency="1d", calendar="XLON")
    monkeypatch.setattr(
        br,
        "load_data_for_strategy",
        lambda _s: (market, pd.DataFrame({"A": 100.0}, index=IDX), pd.DataFrame({"A": 1e9}, index=IDX)),
    )
    monkeypatch.setattr(br, "load_income_for_strategy", lambda _s, instruments, dates: None)
    return run_backtest(strategy)


def test_monthly_fee_is_charged_at_each_month_end(monkeypatch):
    # Flat price, one buy on day one that costs nothing, then Jan, Feb and Mar close (April is not finished).
    result = _run(monkeypatch, _strategy(monthly=3.0))
    assert result.equity.iloc[0] - result.equity.iloc[-1] == pytest.approx(9.0, abs=1e-6)


def test_commission_paid_offsets_the_monthly_fee(monkeypatch):
    # The day-one order pays a 2.00 floor: January owes 1.00, February and March 3.00 each.
    result = _run(monkeypatch, _strategy(floor=2.0, monthly=3.0))
    assert 15000.0 - result.equity.iloc[-1] == pytest.approx(2.0 + 1.0 + 3.0 + 3.0, abs=1e-6)


def test_no_monthly_fee_no_regression(monkeypatch):
    result = _run(monkeypatch, _strategy())
    assert np.isclose(result.equity.iloc[-1], 15000.0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s.costs.commission, "min_per_order", 3.0),
        lambda s: setattr(s.costs.fees, "min_monthly_fee", 3.0),
        lambda s: setattr(s.costs, "tax", TransactionTax(rate=0.0, min_per_trade=1.0)),
        lambda s: setattr(s.execution.order_policy, "cash_only", True),
    ],
)
def test_vectorized_engine_refuses_what_it_cannot_model(mutate):
    strategy = _strategy()
    strategy.execution.order_policy.cash_only = False
    mutate(strategy)
    with pytest.raises(NotImplementedError):
        run_backtest_vectorized(strategy)
