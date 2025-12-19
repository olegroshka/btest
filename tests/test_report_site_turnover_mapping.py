import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.dsl.costs import BorrowCost, Commission, Costs, FinancingCost, StaticFees
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import Execution, LatencyModel, OrderPolicy, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.portfolio import Book, BottomN, EqualWeight, LongShortPortfolio, TopN
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import ReportSiteIndexRenderer, ReportingContext
from quantdsl_backtest.engine.results import BacktestResult


def test_turnover_metric_prefers_turnover_annual(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3)
    instruments = ["A", "B"]
    zeros = pd.Series(np.zeros(len(idx)), index=idx)

    res = BacktestResult(
        equity=pd.Series([1.0, 1.0, 1.0], index=idx),
        returns=pd.Series([0.0, 0.0, 0.0], index=idx),
        cash=pd.Series([1.0, 1.0, 1.0], index=idx),
        gross_exposure=zeros,
        net_exposure=zeros,
        long_exposure=zeros,
        short_exposure=zeros,
        leverage=zeros,
        positions=pd.DataFrame(0.0, index=idx, columns=instruments),
        weights=pd.DataFrame(0.0, index=idx, columns=instruments),
        trades=pd.DataFrame(),
        metrics={"turnover_annual": 0.1234},
        start_date=idx[0],
        end_date=idx[-1],
        metadata={"strategy_name": "S"},
    )

    portfolio = LongShortPortfolio(
        long_book=Book(name="L", selector=TopN(factor_name="f", n=1), weighting=EqualWeight()),
        short_book=Book(name="S", selector=BottomN(factor_name="f", n=1), weighting=EqualWeight()),
        rebalance_frequency="1d",
    )

    strat = Strategy(
        name="S",
        data=DataConfig(source="test", calendar="XNYS", frequency="1d", start="2020-01-01", end="2020-01-03"),
        universe=Universe(name="TEST", static_instruments=instruments),
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
        backtest=BacktestConfig(
            reporting=Reporting(
                strategyAnalytics=StrategyAnalyticsConfig(
                    enabled=True,
                    metrics=["turnover"],
                    print_metrics=False,
                    write_tearsheet=False,
                )
            )
        ),
    )

    # ensure we don't accidentally hit quantstats
    def _no_qs(self, *args, **kwargs):
        raise AssertionError("quantstats_metrics should not be called for turnover")

    monkeypatch.setattr(BacktestResult, "quantstats_metrics", _no_qs, raising=True)

    ReportSiteIndexRenderer().render(res, ReportingContext(strategy=strat, output_dir=tmp_path))

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Turnover" in html
    # turnover is formatted as scalar (annualized fraction), not percent
    assert "0.123" in html

