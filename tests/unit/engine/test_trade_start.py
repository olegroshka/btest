"""BacktestConfig.trade_start: earlier bars warm the signals up and are not simulated."""
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
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import Book, EqualWeight, LongShortPortfolio, TopN
from quantdsl_backtest.dsl.signals import CrossSectionRank
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe
import quantdsl_backtest.engine.backtest_runner as br
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.vectorized_engine import run_backtest_vectorized

IDX = pd.bdate_range("2024-01-02", periods=10)
RNG = np.random.default_rng(7)
CLOSE = pd.DataFrame(
    100.0 * np.exp(np.cumsum(RNG.normal(0.0, 0.02, (10, 3)), axis=0)), index=IDX, columns=["A", "B", "C"]
)
START = IDX[5]


def _strategy(trade_start=None, engine="event_driven") -> Strategy:
    return Strategy(
        name="warmup",
        data=DataConfig(source="dummy://", calendar="XNYS", frequency="1d", start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="u", static_instruments=list(CLOSE.columns)),
        factors={"mom": ReturnFactor(name="mom", field="close", lookback=4, method="simple")},
        signals={"rank": CrossSectionRank(factor_name="mom", name="rank")},
        portfolio=LongShortPortfolio(
            long_book=Book(name="long", selector=TopN(factor_name="rank", n=1), weighting=EqualWeight()),
            short_book=None,
            rebalance_frequency="1d",
            signal_delay_bars=0,
            target_gross_leverage=1.0,
            target_net_exposure=1.0,
            max_abs_weight_per_name=1.0,
            min_names_per_book=1,
        ),
        execution=Execution(
            order_policy=OrderPolicy(),
            latency=LatencyModel(),
            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
            volume_limits=VolumeParticipation(max_participation=1.0),
        ),
        costs=Costs(
            commission=Commission(type="bps_notional", amount=0.0),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(),
        ),
        backtest=BacktestConfig(cash_initial=10_000.0, engine=engine, trade_start=trade_start),
    )


def _run(monkeypatch, strategy: Strategy, close: pd.DataFrame = CLOSE):
    strategy.backtest.reporting.output_dir = None
    names = list(close.columns)
    bars = {n: pd.DataFrame({"open": close[n], "close": close[n], "volume": 1e12}) for n in names}
    market = MarketData(bars=bars, instruments=names, fields=["open", "close", "volume"], frequency="1d", calendar="XNYS")
    volumes = pd.DataFrame(1e12, index=close.index, columns=names)
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: (market, close.copy(), volumes))
    monkeypatch.setattr(br, "load_income_for_strategy", lambda _s, instruments, dates: None)
    return run_backtest(strategy)


def test_series_start_at_the_trading_start(monkeypatch):
    res = _run(monkeypatch, _strategy(trade_start=str(START.date())))
    assert res.equity.index[0] == START and len(res.equity) == 5
    assert res.start_date == START
    assert res.equity.iloc[0] == pytest.approx(10_000.0)
    assert res.trades["datetime"].min() == START


def test_the_first_simulated_bar_already_has_a_signal(monkeypatch):
    # A 4-bar momentum needs 4 earlier closes. With the history as warm-up the book is invested on the first
    # simulated bar; given only the bars from the start date it would wait four more.
    warm = _run(monkeypatch, _strategy(trade_start=str(START.date())))
    cold = _run(monkeypatch, _strategy(), close=CLOSE.loc[START:])
    assert warm.weights.iloc[0].abs().sum() == pytest.approx(1.0)
    assert cold.weights.iloc[0].abs().sum() == 0.0
    best = CLOSE.pct_change(4).loc[START].idxmax()
    assert warm.weights.iloc[0][best] == pytest.approx(1.0)


def test_returns_match_the_run_that_simulated_the_history_too(monkeypatch):
    warm = _run(monkeypatch, _strategy(trade_start=str(START.date())))
    full = _run(monkeypatch, _strategy())
    # Same holdings from the start date on, no costs: the same returns after the first simulated bar.
    np.testing.assert_allclose(warm.returns.iloc[1:], full.returns.loc[warm.returns.index[1:]], atol=1e-12)
    pd.testing.assert_frame_equal(warm.weights, full.weights.loc[START:], atol=1e-12)


def test_without_a_trading_start_nothing_changes(monkeypatch):
    res = _run(monkeypatch, _strategy())
    assert res.equity.index[0] == IDX[0] and len(res.equity) == len(IDX)


def test_a_start_after_the_data_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="trade_start"):
        _run(monkeypatch, _strategy(trade_start="2030-01-01"))


def test_vectorized_engine_refuses_a_trading_start():
    with pytest.raises(NotImplementedError, match="trade_start"):
        run_backtest_vectorized(_strategy(trade_start=str(START.date()), engine="vectorized"))
