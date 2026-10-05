"""Crypto support: perp FUNDING cost + configurable periods_per_year.

Funding: a position carried over a bar pays q * price * Σrate for the funding events that settle inside
that bar's holding window (open, open + step]. Longs pay positive rates, shorts receive them.
periods_per_year: every annualisation (Sharpe, turnover, vol factor, carry dt) follows the config; the
default stays 252 so existing strategies are unchanged.
"""
import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import TargetWeights
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation)
from quantdsl_backtest.dsl.costs import (
    Costs, Commission, BorrowCost, FinancingCost, StaticFees, FundingCost)
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.accounting import apply_funding, compute_basic_metrics
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.data_loader import load_funding_for_strategy
import quantdsl_backtest.engine.backtest_runner as br

IDX = pd.date_range("2024-01-01", periods=10, freq="D")     # daily bar OPEN times, naive UTC


def _events(times, rate=0.0001):
    return pd.DataFrame({"ts": pd.to_datetime(times, utc=True), "ticker": "BTCUSDT", "funding_rate": rate})


def _strategy(weight, funding_path=None, ppy=252.0):
    W = pd.DataFrame(weight, index=IDX, columns=["BTCUSDT"])
    return Strategy(
        name="perp_test",
        data=DataConfig(source="dummy://", calendar="24/7", frequency="1d",
                        start="2024-01-01", end="2024-12-31"),
        universe=Universe(name="C", static_instruments=["BTCUSDT"]),
        factors={"ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple")},
        signals={},
        portfolio=TargetWeights(weights=W, rebalance_frequency="1d", signal_delay_bars=0),
        execution=Execution(order_policy=OrderPolicy(fill_on="close"), latency=LatencyModel(),
                            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
                            volume_limits=VolumeParticipation(max_participation=1.0)),
        costs=Costs(commission=Commission(type="bps_notional", amount=0.0),
                    borrow=BorrowCost(default_annual_rate=0.0),
                    financing=FinancingCost(spread_bps=0.0), fees=StaticFees(),
                    funding=FundingCost(source=f"parquet://{funding_path}") if funding_path else None),
        backtest=BacktestConfig(cash_initial=100_000.0, periods_per_year=ppy),
    )


def _flat_md():
    df = pd.DataFrame({"close": 100.0, "volume": 1e9}, index=IDX)
    md = MarketData(bars={"BTCUSDT": df}, instruments=["BTCUSDT"], fields=["close", "volume"],
                    frequency="1d", calendar="24/7")
    return md, pd.DataFrame({"BTCUSDT": 100.0}, index=IDX), pd.DataFrame({"BTCUSDT": 1e9}, index=IDX)


# ── bucketing ────────────────────────────────────────────────────────────────

def test_events_bucket_into_holding_window(tmp_path):
    """Bar D owns (D 00:00, D+1 00:00]: D 08:00, D 16:00 and D+1 00:00 → bar D; D 00:00 → bar D-1."""
    p = tmp_path / "f.parquet"
    _events(["2024-01-03 00:00", "2024-01-03 08:00", "2024-01-03 16:00", "2024-01-04 00:00"]).to_parquet(p)
    panel = load_funding_for_strategy(_strategy(1.0, p), ["BTCUSDT"], IDX)
    assert np.isclose(panel.loc["2024-01-02", "BTCUSDT"], 0.0001)       # 01-03 00:00 closes bar 01-02
    assert np.isclose(panel.loc["2024-01-03", "BTCUSDT"], 0.0003)       # 08:00 + 16:00 + 01-04 00:00
    assert np.isclose(panel["BTCUSDT"].sum(), 0.0004)


def test_tz_aware_events_are_converted_to_utc(tmp_path):
    """sfera returns timestamptz in Asia/Dubai; 2024-01-03 12:00+04:00 = 08:00 UTC → bar 01-03."""
    p = tmp_path / "f.parquet"
    df = _events(["2024-01-03 08:00"]); df["ts"] = df["ts"].dt.tz_convert("Asia/Dubai"); df.to_parquet(p)
    panel = load_funding_for_strategy(_strategy(1.0, p), ["BTCUSDT"], IDX)
    assert np.isclose(panel.loc["2024-01-03", "BTCUSDT"], 0.0001)


# ── sign convention ──────────────────────────────────────────────────────────

def test_apply_funding_sign():
    px, rate = pd.Series({"A": 100.0}), pd.Series({"A": 0.001})
    assert np.isclose(apply_funding(pd.Series({"A": 10.0}), px, rate), 1.0)     # long pays
    assert np.isclose(apply_funding(pd.Series({"A": -10.0}), px, rate), -1.0)   # short receives
    assert apply_funding(pd.Series({"A": 10.0}), px, None) == 0.0


# ── end-to-end through the engine ────────────────────────────────────────────

def _run(monkeypatch, strat):
    strat.backtest.reporting.output_dir = None
    monkeypatch.setattr(br, "load_data_for_strategy", lambda _s: _flat_md())
    return run_backtest(strat)


def test_long_pays_positive_funding_end_to_end(monkeypatch, tmp_path):
    """Flat price, fully long, 3 events/day at +1bp over 9 held bars → equity ≈ −27bp (compounded)."""
    p = tmp_path / "f.parquet"
    times = [d + pd.Timedelta(hours=h) for d in IDX[1:] for h in (8, 16, 24)]
    _events(times).to_parquet(p)
    res = _run(monkeypatch, _strategy(1.0, p))
    held_bars = len(IDX) - 1
    expected = (1 - 0.0003) ** held_bars - 1
    assert np.isclose(res.equity.iloc[-1] / res.equity.iloc[0] - 1.0, expected, atol=2e-5)
    assert res.metrics["funding_paid"] > 0


def test_short_receives_positive_funding(monkeypatch, tmp_path):
    p = tmp_path / "f.parquet"
    _events([d + pd.Timedelta(hours=8) for d in IDX[1:]]).to_parquet(p)
    res = _run(monkeypatch, _strategy(-1.0, p))
    assert res.equity.iloc[-1] > res.equity.iloc[0]
    assert res.metrics["funding_paid"] < 0


def test_no_funding_no_regression(monkeypatch):
    res = _run(monkeypatch, _strategy(1.0, None))
    assert np.isclose(res.equity.iloc[-1] / res.equity.iloc[0] - 1.0, 0.0, atol=1e-9)
    assert "funding_paid" not in res.metrics


# ── periods_per_year ─────────────────────────────────────────────────────────

def test_sharpe_scales_with_periods_per_year():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.02, 500), index=pd.date_range("2020", periods=500))
    eq = (1 + r).cumprod()
    s252 = compute_basic_metrics(r, eq, None)["sharpe"]
    s365 = compute_basic_metrics(r, eq, None, periods_per_year=365.0)["sharpe"]
    assert np.isclose(s365 / s252, np.sqrt(365 / 252))


def test_default_periods_per_year_is_252():
    assert BacktestConfig().periods_per_year == 252.0
