import numpy as np
import pandas as pd
import types

import pytest

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, NotNull
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.signal_engine import SignalEngine
from quantdsl_backtest.engine.factor_engine import FactorEngine
from quantdsl_backtest.engine.analytics.signal_analytics import assign_quantiles
from quantdsl_backtest.engine.analytics.attribution import (
    contrib_return_panel,
    contrib_by_quantile,
)


def _make_synth_prices(n_days=30, symbols=("A", "B", "C", "D")):
    idx = pd.date_range("2021-01-01", periods=n_days, freq="B")
    cols = list(symbols)
    # Controlled trends:
    # A strong up, B flat, C down, D mild up
    base = np.array([100.0, 100.0, 100.0, 100.0])
    rets = np.vstack([
        np.full(len(idx), 0.005),   # A +0.5%
        np.full(len(idx), 0.000),   # B 0%
        np.full(len(idx), -0.005),  # C -0.5%
        np.full(len(idx), 0.002),   # D +0.2%
    ]).T  # [t x 4]
    prices = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    prices.iloc[0] = base
    for t in range(1, len(idx)):
        prices.iloc[t] = prices.iloc[t - 1] * (1.0 + rets[t])
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=cols, dtype="float64")
    # Minimal MarketData stub with .bars for non-'close' fields if requested (not used here)
    md = types.SimpleNamespace(bars={})
    return md, prices, volumes


def _build_strategy(signal_delay_bars=0) -> Strategy:
    data = DataConfig(
        source="mock://synthetic",
        calendar="XNYS",
        frequency="1d",
        start="2021-01-01",
        end="2021-03-31",
        fields=["close"],
    )
    universe = Universe(name="U", id_field="ticker", filters=[])

    # Factor: 1-bar simple return on close
    ret1 = ReturnFactor(name="ret1", field="close", lookback=1, method="simple")

    # Signals: rank of ret1 + validity mask
    rank_s1 = CrossSectionRank(factor_name="ret1", mask_name=None, method="percentile", name="rank_s1")
    valid = MaskFromBoolean(name="valid", expr=NotNull(factor_name="ret1"))

    long_book = Book(name="long", selector=TopN(factor_name="rank_s1", n=1), weighting=EqualWeight())
    short_book = Book(name="short", selector=BottomN(factor_name="rank_s1", n=1), weighting=EqualWeight())
    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=signal_delay_bars,
        target_gross_leverage=1.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=1.0,
        turnover_limit=None,
    )

    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000.0,
        reporting=Reporting(store_trades=True, store_positions=True, metrics=["sharpe", "max_drawdown"]),
    )

    # Enable analytics for the tested signal
    bt.extra["signal_analytics"] = {
        "signals": ["rank_s1"],
        "horizons": [1],
        "quantiles": 3,
        "within_mask": "valid",
        "signal_delay_bars": signal_delay_bars,
        "store_values": False,
        "store_rank": False,
        "store_quantile": True,
    }

    # Minimal execution with no latency, no slippage, and full volume participation
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0, use_intraday_vol=False),
        volume_limits=VolumeParticipation(max_participation=1.0, mode="proportional", min_fill_notional=0.0),
    )

    strategy = Strategy(
        name="test_attrib_synth",
        data=data,
        universe=universe,
        factors={"ret1": ret1},
        signals={"rank_s1": rank_s1, "valid": valid},
        portfolio=portfolio,
        execution=execution,
        costs=Costs(
            commission=Commission(type="bps_notional", amount=0.0),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
        ),
        backtest=bt,
    )
    return strategy


def test_end_to_end_attribution_matches_manual(monkeypatch):
    # Patch data loader to return our synthetic panels
    from quantdsl_backtest.engine import backtest_runner as br
    md, prices, volumes = _make_synth_prices(n_days=25)

    def _stub_loader(_strategy):
        return md, prices, volumes

    monkeypatch.setattr(br, "load_data_for_strategy", _stub_loader)

    # Avoid file writes from HTML renderers during test
    import quantdsl_backtest.engine.analytics.render_tearsheets as rt
    monkeypatch.setattr(rt, "render_signal_tearsheet_html", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rt, "render_portfolio_signal_tearsheet_html", lambda *a, **k: None, raising=False)

    strat = _build_strategy(signal_delay_bars=0)
    result = run_backtest(strat)

    # Recompute signals exactly as engine does to build manual quantiles
    fe = FactorEngine(md, prices)
    factor_panels = fe.compute_all(strat.factors)
    se = SignalEngine(factor_panels, strat.signals)
    signal_panels = se.compute_all()

    used_panel = signal_panels["rank_s1"].shift(strat.portfolio.signal_delay_bars)
    mask_df = signal_panels["valid"].astype(bool)
    q = 3
    qdf = assign_quantiles(used_panel, q=q, mask=mask_df).astype("float32")

    contrib = contrib_return_panel(result.weights, prices)
    manual_by_q, manual_ls = contrib_by_quantile(contrib, qdf, q=q)

    attr = result.signal_attribution["rank_s1"]
    # Align indices just in case and compare closely
    common_index = manual_by_q.index.intersection(attr.contrib_ret_by_q.index)
    mbq = manual_by_q.loc[common_index]
    abq = attr.contrib_ret_by_q.loc[common_index]
    # Fill NaNs (e.g., first row) with 0.0 for comparison purposes
    mbq_f = mbq.fillna(0.0)
    abq_f = abq.fillna(0.0)
    # Compare with tolerance for floating differences
    assert np.isclose(mbq_f.values, abq_f.values, atol=1e-12, rtol=1e-9).all()

    mls = manual_ls.loc[common_index].fillna(0.0)
    als = attr.contrib_ret_ls.loc[common_index].fillna(0.0)
    assert np.isclose(mls.values, als.values, atol=1e-12, rtol=1e-9).all()
