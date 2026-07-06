"""Fixed-income / dividend coupon-income accounting in the engine.

A bond marked at flat DIRTY price has zero capital P&L, so its total return must come entirely from the
coupon income stream (credited per unit held on the coupon date). Equities (no income stream) are unaffected.
"""
import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import TargetWeights
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.backtest_runner import run_backtest
import quantdsl_backtest.engine.backtest_runner as br

IDX = pd.date_range("2024-01-01", periods=12, freq="D")
COUPON_DAY, COUPON_PER_UNIT = IDX[6], 5.0   # 5.0 per unit on a 100-priced bond -> +5% of a fully-invested book


def _flat_dirty_md():
    """1 bond, DIRTY price flat at 100 (no capital move) so all TR must be the coupon."""
    df = pd.DataFrame({"close": 100.0, "volume": 1_000_000.0}, index=IDX)
    md = MarketData(bars={"BND": df}, instruments=["BND"], fields=["close", "volume"],
                    frequency="1d", calendar="XNYS")
    return md, pd.DataFrame({"BND": 100.0}, index=IDX), pd.DataFrame({"BND": 1_000_000.0}, index=IDX)


def _strategy():
    W = pd.DataFrame(1.0, index=IDX, columns=["BND"])
    return Strategy(
        name="FItest",
        data=DataConfig(source="dummy://", calendar="XNYS", frequency="1d",
                        start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="FI", static_instruments=["BND"]),
        factors={"ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple")},
        signals={},
        portfolio=TargetWeights(weights=W, rebalance_frequency="1d", signal_delay_bars=0),
        execution=Execution(order_policy=OrderPolicy(fill_on="close"), latency=LatencyModel(),
                            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
                            volume_limits=VolumeParticipation(max_participation=1.0)),
        costs=Costs(commission=Commission(type="bps_notional", amount=0.0),
                    borrow=BorrowCost(default_annual_rate=0.0),
                    financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0), fees=StaticFees()),
        backtest=BacktestConfig(cash_initial=100_000.0),
    )


def _run(monkeypatch, income_panel):
    strat = _strategy(); strat.backtest.reporting.output_dir = None
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: _flat_dirty_md())
    monkeypatch.setattr(br, "load_income_for_strategy", lambda _s, instruments, dates: income_panel)
    return run_backtest(strat)


def test_coupon_income_credited_exactly(monkeypatch):
    income = pd.DataFrame(0.0, index=IDX, columns=["BND"]); income.loc[COUPON_DAY, "BND"] = COUPON_PER_UNIT
    res = _run(monkeypatch, income)
    # flat dirty price + one coupon of 5/unit on a 100-priced fully-invested book => exactly +5%
    assert np.isclose(res.equity.iloc[-1] / res.equity.iloc[0] - 1.0, 0.05, atol=1e-4)


def test_no_income_no_regression(monkeypatch):
    res = _run(monkeypatch, None)
    # flat dirty price, no income stream => flat equity (income path is a no-op for equities)
    assert np.isclose(res.equity.iloc[-1] / res.equity.iloc[0] - 1.0, 0.0, atol=1e-6)
