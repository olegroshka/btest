"""OrderPolicy.limit_buy_pct — limit-on-open buys (fill_on="open").

A buy is a limit at prev_close × (1 + pct): it fills at the open only if open ≤ limit, otherwise it is cancelled
for the day and retried the next morning (the target is re-queued every bar). Sells are never limited.
"""
import pandas as pd

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import TargetWeights
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.backtest_runner import run_backtest, _apply_limit_buys
import quantdsl_backtest.engine.backtest_runner as br

IDX = pd.date_range("2024-01-01", periods=6, freq="B")
CLOSE = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
OPEN = [100.0, 110.0, 100.0, 100.0, 100.0, 100.0]          # +10% gap into the first fill day


def _strategy(limit):
    W = pd.DataFrame({"X": [1.0] * len(IDX)}, index=IDX)
    return Strategy(
        name="limit_test",
        data=DataConfig(source="dummy://", calendar="XNYS", frequency="1d", start=str(IDX[0].date()), end=str(IDX[-1].date())),
        universe=Universe(name="U", static_instruments=["X"]),
        factors={"ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple")},
        signals={},
        portfolio=TargetWeights(weights=W, rebalance_frequency="1d", signal_delay_bars=0),
        execution=Execution(order_policy=OrderPolicy(fill_on="open", limit_buy_pct=limit), latency=LatencyModel(),
                            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0), volume_limits=VolumeParticipation(max_participation=1.0)),
        costs=Costs(commission=Commission(type="bps_notional", amount=0.0), borrow=BorrowCost(default_annual_rate=0.0),
                    financing=FinancingCost(spread_bps=0.0), fees=StaticFees()),
        backtest=BacktestConfig(cash_initial=100_000.0),
    )


def _md():
    df = pd.DataFrame({"open": OPEN, "close": CLOSE, "volume": 1e9}, index=IDX)
    md = MarketData(bars={"X": df}, instruments=["X"], fields=["open", "close", "volume"], frequency="1d", calendar="XNYS")
    return md, pd.DataFrame({"X": CLOSE}, index=IDX), pd.DataFrame({"X": 1e9}, index=IDX)


def _run(monkeypatch, limit):
    s = _strategy(limit); s.backtest.reporting.output_dir = None
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: _md())
    return run_backtest(s)


def test_no_limit_buys_into_the_gap(monkeypatch):
    r = _run(monkeypatch, None)
    assert r.positions["X"].iloc[1] > 0                      # filled at the 110 open
    assert "limit_buy_cancels" not in r.metrics


def test_gap_above_limit_cancels_then_retries(monkeypatch):
    r = _run(monkeypatch, 0.03)
    assert r.positions["X"].iloc[1] == 0                     # 110 > 100 × 1.03 → cancelled, stays cash
    assert r.positions["X"].iloc[2] > 0                      # next open 100 ≤ 103 → filled
    assert r.metrics["limit_buy_cancels"] == 1


def test_gap_within_limit_fills(monkeypatch):
    r = _run(monkeypatch, 0.15)                              # 110 ≤ 115 → fills on the gap day
    assert r.positions["X"].iloc[1] > 0
    assert r.metrics["limit_buy_cancels"] == 0


def test_helper_only_touches_buys_past_the_limit():
    idx = pd.Index(["A", "B", "C"])
    w = pd.Series([0.5, 0.0, 0.5], index=idx)                # buy A, sell B (held), buy C
    q = pd.Series([0.0, 500.0, 0.0], index=idx)              # 500 B held at 100 = 50k, cash 50k
    out = _apply_limit_buys(w, {"A": 0.03, "C": 0.03}, pd.Series([110.0, 100.0, 101.0], index=idx),
                            pd.Series([100.0, 100.0, 100.0], index=idx), q, 50_000.0)
    assert out["A"] == 0.0                                   # gapped past the limit → held weight (0)
    assert out["B"] == 0.0                                   # the sell is untouched
    assert out["C"] == 0.5                                   # within the limit → kept
    assert out.attrs["cancelled"] == 1
