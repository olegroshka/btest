"""Per-share commission as a broker charges it: a minimum, a cap, and shares counted at the printed price."""
import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.costs import BorrowCost, Commission, Costs, FinancingCost, StaticFees
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.portfolio import TargetWeights
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe
import quantdsl_backtest.engine.backtest_runner as br
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.execution_engine import rebalance_to_target_weights
from quantdsl_backtest.engine.vectorized_engine import run_backtest_vectorized
from quantdsl_backtest.models.costs import PerShareCommission, build_cost_model

FIXED = dict(type="per_share", amount=0.005, min_per_order=1.0, max_fraction_of_notional=0.01)


def _execution(**policy) -> Execution:
    return Execution(
        order_policy=OrderPolicy(**policy),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),
    )


def _rebalance(*, cash, prev, target, commission, prices, share_prices=None, **policy):
    names = list(target)
    return rebalance_to_target_weights(
        pd.Timestamp("2024-01-02"),
        _execution(**policy),
        commission,
        StaticFees(),
        cash,
        pd.Series(prices, dtype=float),
        pd.Series({n: 1e12 for n in names}),
        pd.Series(prev, dtype=float).reindex(names).fillna(0.0),
        pd.Series(target, dtype=float),
        share_prices=None if share_prices is None else pd.Series(share_prices, dtype=float),
    )


# --------------------------------------------------------------------------- the model


@pytest.mark.parametrize(
    "shares, price, expected",
    [
        (100.0, 50.0, 1.0),  # 0.50 by the rate: the minimum
        (200.0, 50.0, 1.0),  # exactly where the rate meets the minimum
        (1000.0, 50.0, 5.0),  # the rate
        (100.0, 0.5, 0.5),  # 1% of 50 dollars: the cap wins over the minimum
        (10000.0, 0.25, 25.0),  # 50 by the rate, capped at 1% of 2,500
    ],
)
def test_minimum_then_cap(shares, price, expected):
    model = build_cost_model(Commission(**FIXED))
    assert model.commission_from_trade(qty=shares, exec_price=price) == pytest.approx(expected)
    assert model.commission_from_trade(qty=-shares, exec_price=price) == pytest.approx(expected)


def test_plain_per_share_is_unchanged_and_nothing_is_charged_without_a_trade():
    plain = build_cost_model(Commission(type="per_share", amount=0.02))
    assert isinstance(plain, PerShareCommission)
    assert plain.commission_from_trade(qty=-50.0, exec_price=8.0) == pytest.approx(1.0)
    assert build_cost_model(Commission(**FIXED)).commission_from_trade(qty=0.0, exec_price=10.0) == 0.0


def test_minimum_alone_builds_a_per_share_model():
    model = build_cost_model(Commission(type="per_share", amount=0.0, min_per_order=0.35))
    assert model.commission_from_trade(qty=3.0, exec_price=100.0) == pytest.approx(0.35)


def test_share_count_overrides_quantity():
    model = build_cost_model(Commission(type="per_share", amount=0.005))
    # 1,000 units of an adjusted price are really 250 shares
    assert model.commission_from_trade(qty=1000.0, exec_price=25.0, shares=250.0) == pytest.approx(1.25)


def test_cap_applies_to_a_bps_commission_too():
    model = build_cost_model(Commission(type="bps_notional", amount=5.0, min_per_order=3.0, max_fraction_of_notional=0.01))
    assert model.commission_from_trade(qty=1.0, exec_price=100.0) == pytest.approx(1.0)  # 1% of 100 beats the 3 floor
    assert model.commission_from_trade(qty=100.0, exec_price=100.0) == pytest.approx(5.0)


# --------------------------------------------------------------------------- inside a rebalance


def test_shares_come_from_the_printed_price_not_the_adjusted_one():
    # The adjusted price is 25 where the share really traded at 100 (a later 4-for-1 split).
    # 100,000 dollars buy 4,000 adjusted units, which were 1,000 real shares: 5.00, not 20.00.
    commission = Commission(type="per_share", amount=0.005)
    _, cash_delta, trades = _rebalance(
        cash=100_000.0, prev={}, target={"A": 1.0}, commission=commission, prices={"A": 25.0}, share_prices={"A": 100.0}
    )
    row = trades.iloc[0]
    assert row["quantity"] == pytest.approx(4000.0)
    assert row["shares"] == pytest.approx(1000.0)
    assert row["commission"] == pytest.approx(5.0)
    assert cash_delta == pytest.approx(-100_000.0 - 5.0)

    _, _, plain = _rebalance(cash=100_000.0, prev={}, target={"A": 1.0}, commission=commission, prices={"A": 25.0})
    assert plain.iloc[0]["commission"] == pytest.approx(20.0)
    assert plain.iloc[0]["shares"] == pytest.approx(4000.0)


def test_a_missing_printed_price_falls_back_to_the_fill_price():
    commission = Commission(type="per_share", amount=0.005)
    _, _, trades = _rebalance(
        cash=10_000.0,
        prev={},
        target={"A": 0.5, "B": 0.5},
        commission=commission,
        prices={"A": 25.0, "B": 50.0},
        share_prices={"A": np.nan, "B": 100.0},
    )
    by_name = trades.set_index("instrument")
    assert by_name.loc["A", "shares"] == pytest.approx(200.0)  # 5,000 / 25
    assert by_name.loc["B", "shares"] == pytest.approx(50.0)  # 5,000 / 100
    assert by_name.loc["B", "commission"] == pytest.approx(0.25)


def test_printed_prices_never_size_or_fill_an_order():
    commission = Commission(type="per_share", amount=0.0)
    a, _, ta = _rebalance(cash=10_000.0, prev={}, target={"A": 1.0}, commission=commission, prices={"A": 25.0})
    b, _, tb = _rebalance(
        cash=10_000.0, prev={}, target={"A": 1.0}, commission=commission, prices={"A": 25.0}, share_prices={"A": 777.0}
    )
    assert a["A"] == b["A"] and ta.iloc[0]["price"] == tb.iloc[0]["price"]


def test_cash_only_buy_stays_inside_the_cash_with_a_minimum_per_share_commission():
    new, cash_delta, trades = _rebalance(
        cash=1_000.0,
        prev={},
        target={"A": 0.5, "B": 0.5},
        commission=Commission(**FIXED),
        prices={"A": 20.0, "B": 10.0},
        share_prices={"A": 80.0, "B": 10.0},
        cash_only=True,
    )
    assert trades["commission"].tolist() == pytest.approx([1.0, 1.0])
    assert 1_000.0 + cash_delta >= -1e-9
    assert (new * pd.Series({"A": 20.0, "B": 10.0})).sum() == pytest.approx(998.0)


# --------------------------------------------------------------------------- through the runner

IDX = pd.bdate_range("2024-01-02", periods=4)
OPEN = pd.DataFrame({"A": [10.0, 11.0, 10.0, 10.0]}, index=IDX)
CLOSE = pd.DataFrame({"A": [10.0, 10.0, 10.0, 10.0]}, index=IDX)
RAW_OPEN = OPEN * 4.0  # the share printed at four times the adjusted price
RAW_CLOSE = CLOSE * 5.0  # a different ratio on purpose: the two fields must not be mixed up


def _strategy(commission: Commission, engine: str = "event_driven") -> Strategy:
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, index=IDX)
    return Strategy(
        name="per_share",
        data=DataConfig(source="dummy://", calendar="XNYS", frequency="1d", start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="u", static_instruments=["A"]),
        factors={},
        signals={},
        portfolio=TargetWeights(weights=w, rebalance_frequency="1d", signal_delay_bars=0),
        execution=_execution(default_order_type="MOC", fill_on="close", exit_on="next_open"),
        costs=Costs(
            commission=commission,
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(),
        ),
        backtest=BacktestConfig(cash_initial=10_000.0, engine=engine),
    )


def _run(monkeypatch, strategy: Strategy, with_raw: bool = True):
    strategy.backtest.reporting.output_dir = None
    cols = {"open": OPEN["A"], "close": CLOSE["A"], "volume": 1e12}
    if with_raw:
        cols |= {"raw_open": RAW_OPEN["A"], "raw_close": RAW_CLOSE["A"]}
    market = MarketData(bars={"A": pd.DataFrame(cols)}, instruments=["A"], fields=list(cols), frequency="1d", calendar="XNYS")
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: (market, CLOSE.copy(), pd.DataFrame(1e12, index=IDX, columns=["A"])))
    monkeypatch.setattr(br, "load_income_for_strategy", lambda _s, instruments, dates: None)
    return run_backtest(strategy)


def test_runner_uses_the_close_field_for_the_buy_and_the_open_field_for_the_sale(monkeypatch):
    commission = Commission(type="per_share", amount=0.01, share_price_field_close="raw_close", share_price_field_open="raw_open")
    res = _run(monkeypatch, _strategy(commission))
    buy, sell = res.trades.iloc[0], res.trades.iloc[1]
    # Buy: 10,000 dollars at an adjusted 10 = 1,000 units = 200 shares at the printed 50.
    assert buy["side"] == "BUY" and buy["shares"] == pytest.approx(200.0) and buy["commission"] == pytest.approx(2.0)
    # Sale at the open: 1,000 units at an adjusted 11, printed 44 = 250 shares.
    assert sell["side"] == "SELL" and sell["fill_at"] == "open"
    assert sell["shares"] == pytest.approx(250.0) and sell["commission"] == pytest.approx(2.5)
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 + 1_000.0 - 2.0 - 2.5)


def test_runner_without_the_fields_counts_units_as_shares(monkeypatch):
    res = _run(monkeypatch, _strategy(Commission(type="per_share", amount=0.01)))
    assert res.trades["commission"].tolist() == pytest.approx([10.0, 10.0])


def test_runner_refuses_a_field_that_is_not_in_the_data(monkeypatch):
    commission = Commission(type="per_share", amount=0.01, share_price_field_close="raw_close")
    with pytest.raises(ValueError, match="raw_close"):
        _run(monkeypatch, _strategy(commission), with_raw=False)


@pytest.mark.parametrize(
    "extra",
    [dict(max_fraction_of_notional=0.01), dict(share_price_field_close="raw_close"), dict(share_price_field_open="raw_open")],
)
def test_vectorized_engine_refuses_what_it_cannot_charge(extra):
    strategy = _strategy(Commission(type="per_share", amount=0.01, **extra), engine="vectorized")
    strategy.execution = _execution()
    with pytest.raises(NotImplementedError):
        run_backtest_vectorized(strategy)
