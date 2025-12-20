from pathlib import Path

import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.portfolio import Book, BottomN, EqualWeight, LongShortPortfolio, TopN
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import BorrowCost, Commission, Costs, FinancingCost, StaticFees
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.engine.analytics.types import (
    PortfolioSignalAttribution,
    SignalAnalyticsConfig,
    SignalTearsheetData,
)
from quantdsl_backtest.engine.backtest_runner import ReportingContext, ReportSiteIndexRenderer
from quantdsl_backtest.engine.results import BacktestResult


def _dummy_result() -> BacktestResult:
    idx = pd.date_range("2020-01-01", periods=3)
    instruments = ["A", "B"]
    zeros = pd.Series(np.zeros(len(idx)), index=idx)
    weights = pd.DataFrame(0.0, index=idx, columns=instruments)
    positions = pd.DataFrame(0.0, index=idx, columns=instruments)

    res = BacktestResult(
        equity=pd.Series([1.0, 1.0, 1.0], index=idx),
        returns=zeros,
        cash=pd.Series([1.0, 1.0, 1.0], index=idx),
        gross_exposure=zeros,
        net_exposure=zeros,
        long_exposure=zeros,
        short_exposure=zeros,
        leverage=zeros,
        positions=positions,
        weights=weights,
        trades=pd.DataFrame(),
        metrics={},
        start_date=idx[0],
        end_date=idx[-1],
        metadata={"strategy_name": "S"},
    )

    cfg = SignalAnalyticsConfig(signals=["sig1"], horizons=[1], quantiles=5)
    rep = SignalTearsheetData(name="sig1", config=cfg)
    rep.coverage = pd.Series([0.5, 0.6, 0.7], index=idx)
    rep.quantile_turnover = pd.Series([0.1, 0.2, 0.15], index=idx)
    rep.rank_ic = {1: pd.Series([0.01, 0.02, 0.00], index=idx)}
    res.signal_reports = {"sig1": rep}

    contrib_by_q = pd.DataFrame({1: [0.01, 0.02, 0.0], 2: [-0.01, 0.0, 0.0]}, index=idx)
    ls = pd.Series([0.02, 0.01, 0.0], index=idx)
    res.signal_attribution = {"sig1": PortfolioSignalAttribution(contrib_ret_by_q=contrib_by_q, contrib_ret_ls=ls)}

    return res


def test_report_site_index_writes_index_and_links(tmp_path):
    res = _dummy_result()

    portfolio = LongShortPortfolio(
        long_book=Book(name="L", selector=TopN(factor_name="f", n=1), weighting=EqualWeight()),
        short_book=Book(name="S", selector=BottomN(factor_name="f", n=1), weighting=EqualWeight()),
        rebalance_frequency="1d",
    )

    strat = Strategy(
        name="S",
        data=DataConfig(source="test", calendar="XNYS", frequency="1d", start="2020-01-01", end="2020-01-03"),
        universe=Universe(name="TEST", static_instruments=["A", "B"]),
        factors={},
        signals={},
        portfolio=portfolio,
        execution=Execution(
            order_policy=OrderPolicy(),
            latency=LatencyModel(),
            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0),
            volume_limits=VolumeParticipation(max_participation=1.0),
        ),
        costs=Costs(
            commission=Commission(type="bps_notional", amount=0.0),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(spread_bps=0.0),
            fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
        ),
        backtest=BacktestConfig(reporting=Reporting()),
    )

    ctx = ReportingContext(strategy=strat, output_dir=tmp_path)
    ReportSiteIndexRenderer().render(res, ctx)

    p = tmp_path / "index.html"
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "tearsheet.html" in txt
    assert "signals/sig1/signal_tearsheet.html" in txt
    assert "attribution/sig1/portfolio_signal_tearsheet.html" in txt
