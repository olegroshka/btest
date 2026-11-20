"""OrderPolicy.exit_on="next_open": buy at a bar's close, sell at the next bar's open, flat during the day."""
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
from quantdsl_backtest.engine.vectorized_engine import run_backtest_vectorized

IDX = pd.bdate_range("2024-01-02", periods=5)
#        day:      0      1      2      3      4
OPEN = pd.DataFrame({"A": [100.0, 102.0, 99.0, 101.0, 100.0], "B": [50.0, 49.0, 50.0, 52.0, 51.0]}, index=IDX)
CLOSE = pd.DataFrame({"A": [100.0, 110.0, 100.0, 90.0, 100.0], "B": [50.0, 50.0, 40.0, 50.0, 50.0]}, index=IDX)


def _strategy(weights, *, exit_on="next_open", commission_bps=0.0, floor=0.0, cash=10_000.0, frequency="1d", engine="event_driven", **policy):
    return Strategy(
        name="overnight",
        data=DataConfig(source="dummy://", calendar="XNYS", frequency="1d", start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="u", static_instruments=list(weights.columns)),
        factors={},
        signals={},
        portfolio=TargetWeights(weights=weights, rebalance_frequency=frequency, signal_delay_bars=0),
        execution=Execution(
            order_policy=OrderPolicy(default_order_type="MOC", fill_on="close", exit_on=exit_on, **policy),
            latency=LatencyModel(),
            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
            volume_limits=VolumeParticipation(max_participation=1.0),
        ),
        costs=Costs(
            commission=Commission(type="bps_notional", amount=commission_bps, min_per_order=floor),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(),
        ),
        backtest=BacktestConfig(cash_initial=cash, engine=engine),
    )


def _run(monkeypatch, strategy, open_=OPEN, close=CLOSE):
    strategy.backtest.reporting.output_dir = None
    names = list(close.columns)
    bars = {n: pd.DataFrame({"open": open_[n], "close": close[n], "volume": 1e12}) for n in names}
    market = MarketData(bars=bars, instruments=names, fields=["open", "close", "volume"], frequency="1d", calendar="XNYS")
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: (market, close.copy(), pd.DataFrame(1e12, index=close.index, columns=names)))
    monkeypatch.setattr(br, "load_income_for_strategy", lambda _s, instruments, dates: None)
    return run_backtest(strategy)


def test_exit_needs_a_close_fill_and_a_known_value():
    with pytest.raises(ValueError):
        OrderPolicy(fill_on="open", exit_on="next_open")
    with pytest.raises(ValueError):
        OrderPolicy(exit_on="noon")
    assert OrderPolicy().exit_on == "hold"


def test_only_the_overnight_move_is_earned(monkeypatch):
    # Hold A every night: close 100 -> open 102, 110 -> 99, 100 -> 101, 90 -> 100. The daytime moves are not held.
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    res = _run(monkeypatch, _strategy(w))
    expected = 10_000.0 * np.cumprod([1.0, 102 / 100, 99 / 110, 101 / 100, 100 / 90])
    np.testing.assert_allclose(res.equity.to_numpy(), expected, rtol=1e-12)
    # fully invested at every close, flat in between
    np.testing.assert_allclose(res.weights["A"].to_numpy(), 1.0, atol=1e-12)


def test_without_the_exit_the_daytime_move_is_held_too(monkeypatch):
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    res = _run(monkeypatch, _strategy(w, exit_on="hold"))
    np.testing.assert_allclose(res.equity.to_numpy(), 10_000.0 * CLOSE["A"].to_numpy() / 100.0, rtol=1e-12)


def test_trades_are_tagged_and_priced_at_the_right_auction(monkeypatch):
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    tr = _run(monkeypatch, _strategy(w)).trades
    sells, buys = tr[tr["fill_at"] == "open"], tr[tr["fill_at"] == "close"]
    assert (sells["side"] == "SELL").all() and (buys["side"] == "BUY").all()
    assert len(buys) == 5 and len(sells) == 4                       # nothing to sell on the first bar
    np.testing.assert_allclose(sells["raw_price"].to_numpy(), OPEN["A"].to_numpy()[1:])
    np.testing.assert_allclose(buys["raw_price"].to_numpy(), CLOSE["A"].to_numpy())
    # each sale closes exactly what the previous close bought
    np.testing.assert_allclose(-sells["quantity"].to_numpy(), buys["quantity"].to_numpy()[:-1], rtol=1e-12)


def test_commission_is_paid_on_both_legs(monkeypatch):
    c = 1e-3  # 10 bp of notional per order
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    res = _run(monkeypatch, _strategy(w, commission_bps=10.0, cash_only=True))
    # cash_only sizes the buy with the commission of the full-size order held back: of `cash`, cash * (1 - c)
    # is invested and c * cash * (1 - c) is paid, so equity after a buy is cash * (1 - c + c * c).
    after_buy = lambda cash: (cash * (1 - c), cash * (1 - c + c * c))  # (position value, equity)
    position, equity0 = after_buy(10_000.0)
    assert res.equity.iloc[0] == pytest.approx(equity0, rel=1e-12)
    cash1 = (equity0 - position) + position * 1.02 * (1 - c)  # sold at the open of day 1, 10 bp on the proceeds
    assert res.equity.iloc[1] == pytest.approx(after_buy(cash1)[1], rel=1e-12)
    assert res.trades["commission"].iloc[:2].sum() == pytest.approx(c * position * (1 + 1.02), rel=1e-12)
    assert res.cash.min() >= -1e-9


def test_a_minimum_commission_per_order_is_charged_twice_a_night(monkeypatch):
    flat = pd.DataFrame(100.0, index=IDX, columns=["A", "B"])
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=IDX)
    res = _run(monkeypatch, _strategy(w, floor=1.0, cash_only=True), open_=flat, close=flat)
    tr = res.trades
    assert len(tr) == 2 * 5 + 2 * 4                                  # two buys a close, two sells an open
    assert tr["commission"].sum() == pytest.approx(18.0)
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 - 18.0, abs=1e-6)


def test_a_weekly_book_is_held_for_one_night_only(monkeypatch):
    idx = pd.bdate_range("2024-01-02", periods=8)                    # Tue .. Thu of the following week
    open_ = pd.DataFrame({"A": np.linspace(101.0, 108.0, 8)}, index=idx)
    close = pd.DataFrame({"A": 100.0}, index=idx)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    res = _run(monkeypatch, _strategy(w, frequency="1w"), open_=open_, close=close)
    buys = res.trades[res.trades["fill_at"] == "close"]
    assert list(buys["datetime"]) == [idx[0], idx[4]]                # first bar, then the first bar of the next week
    held = res.weights["A"].to_numpy() > 0.5
    assert held.tolist() == [True, False, False, False, True, False, False, False]
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 * (102 / 100) * (106 / 100), rel=1e-12)


def test_a_no_trade_band_does_not_block_the_exit(monkeypatch):
    w = pd.DataFrame({"A": 0.04, "B": 0.0}, index=IDX)               # a 4-point position, inside a 5-point band
    w.iloc[0] = [1.0, 0.0]
    res = _run(monkeypatch, _strategy(w, no_trade_band=0.05))
    assert res.positions["A"].iloc[0] > 0 and (res.positions.iloc[1:].abs().sum(axis=1) == 0).all()
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 * 1.02, rel=1e-12)


def test_matches_a_two_bars_per_session_run(monkeypatch):
    """The native mode equals the long-hand version: an 'am' bar that closes at the open with zero target,
    and a 'pm' bar that closes at the close with the target, both filled at their own close."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-01-02", periods=60)
    names = ["A", "B", "C", "D"]
    steps = rng.normal(0.0, 0.01, (120, 4)).cumsum(axis=0)
    path = 50.0 * np.exp(steps)                                      # am0, pm0, am1, pm1, ...
    open_ = pd.DataFrame(path[0::2], index=idx, columns=names)
    close = pd.DataFrame(path[1::2], index=idx, columns=names)
    raw = pd.DataFrame(rng.random((60, 4)), index=idx, columns=names)
    w = raw.where(raw.rank(axis=1) > 2, 0.0)
    w = w.div(w.sum(axis=1), axis=0)

    kw = dict(commission_bps=2.0, floor=0.5, cash=25_000.0, cash_only=True)
    native = _run(monkeypatch, _strategy(w, **kw), open_=open_, close=close)

    am, pm = idx + pd.Timedelta("09:30:00"), idx + pd.Timedelta("16:00:00")
    two = pd.concat([open_.set_axis(am), close.set_axis(pm)]).sort_index()
    w2 = pd.concat([pd.DataFrame(0.0, index=am, columns=names), w.set_axis(pm)]).sort_index()
    longhand = _run(monkeypatch, _strategy(w2, exit_on="hold", **kw), open_=two, close=two)

    np.testing.assert_allclose(native.equity.to_numpy(), longhand.equity.loc[pm].to_numpy(), rtol=1e-12)
    assert native.trades["commission"].sum() == pytest.approx(longhand.trades["commission"].sum(), rel=1e-12)
    assert len(native.trades) == len(longhand.trades)


def test_the_vectorized_engine_refuses(monkeypatch):
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=IDX)
    with pytest.raises(NotImplementedError):
        run_backtest_vectorized(_strategy(w, engine="vectorized"))
