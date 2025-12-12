# tests_slow/test_integration_engines_costs_consistency.py

import numpy as np
import pytest

from quantdsl_backtest.dsl.backtest_config import BacktestConfig, MarginConfig, RiskChecks, Reporting
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, LessEqual
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.engine.accounting import compute_basic_metrics
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.utils.logging import get_logger


DATA_SOURCE = "parquet://equities/sp500_daily"
log = get_logger(__name__)


def _build_strategy_with_costs(*, engine: str, per_share: float, base_bps: float, k: float, exponent: float) -> Strategy:
    data_cfg = DataConfig(
        source=DATA_SOURCE,
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-01-01",
        price_adjustment="split_dividend",
        fields=["open", "high", "low", "close", "volume"],
    )

    universe = Universe(
        name="SP500",
        filters=[HasHistory(min_days=252), MinPrice(min_price=5.0), MinDollarADV(min_dollar_adv=1_000_000.0)],
        id_field="ticker",
    )

    mom_6m = ReturnFactor(name="mom_6m", field="close", lookback=126, method="log")
    factors = {"mom_6m": mom_6m}

    rank = CrossSectionRank(factor_name="mom_6m", mask_name=None, method="percentile", name="rank")
    long_candidates = MaskFromBoolean(name="long_candidates", expr=LessEqual(left=0.9, right="rank"))
    short_candidates = MaskFromBoolean(name="short_candidates", expr=LessEqual(left="rank", right=0.1))

    signals = {"rank": rank, "long_candidates": long_candidates, "short_candidates": short_candidates}

    long_book = Book(name="long_book", selector=TopN(factor_name="rank", n=50, mask_name="long_candidates"), weighting=EqualWeight())
    short_book = Book(name="short_book", selector=BottomN(factor_name="rank", n=50, mask_name="short_candidates"), weighting=EqualWeight())

    portfolio = LongShortPortfolio(
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,
        long_book=long_book,
        short_book=short_book,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.05,
        sector_neutral=None,
        turnover_limit=None,
    )

    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=float(base_bps), k=float(k), exponent=float(exponent), use_intraday_vol=False),
        volume_limits=VolumeParticipation(max_participation=1.0, mode="proportional", min_fill_notional=0.0),
        book_model=None,
    )

    costs = Costs(
        commission=Commission(type="per_share", amount=float(per_share)),
        borrow=BorrowCost(default_annual_rate=0.0, curve_name=None),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    backtest_cfg = BacktestConfig(
        engine=engine,
        cash_initial=100_000_000.0,
        margin=MarginConfig(long_initial=0.5, short_initial=0.5, maintenance=0.3),
        risk_checks=RiskChecks(max_drawdown=None, max_gross_leverage=None),
        reporting=Reporting(store_trades=True, store_positions=True, metrics=[]),
    )

    return Strategy(
        name=f"test_mom_ls_costs_{engine}",
        data=data_cfg,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest_cfg,
    )


@pytest.mark.slow
def test_event_vs_vectorized_with_per_share_and_powerlaw():
    # Non-zero frictions
    per_share = 0.003  # $0.003/share
    base_bps = 1.0
    k = 30.0
    exponent = 1.0

    strat_event = _build_strategy_with_costs(engine="event_driven", per_share=per_share, base_bps=base_bps, k=k, exponent=exponent)
    strat_vec = _build_strategy_with_costs(engine="vectorized", per_share=per_share, base_bps=base_bps, k=k, exponent=exponent)

    res_event = run_backtest(strat_event)
    res_vec = run_backtest(strat_vec)

    equity_event = res_event.equity
    equity_vec = res_vec.equity.reindex(equity_event.index).ffill()

    returns_event = res_event.returns
    returns_vec = res_vec.returns.reindex(returns_event.index).fillna(0.0)

    # Sanity
    assert len(equity_event) > 0 and len(equity_vec) > 0
    assert len(returns_event) > 0 and len(returns_vec) > 0

    # Log metrics for CI inspection
    metrics_event = compute_basic_metrics(returns_event, equity_event, res_event.weights)
    metrics_vec = compute_basic_metrics(returns_vec, equity_vec, res_vec.weights)
    log.info("[Engine=event_driven] metrics (with costs): %s", metrics_event)
    log.info("[Engine=vectorized]   metrics (with costs): %s", metrics_vec)

    # Equity curves should be close; allow slightly looser tolerance due to rounding/order aggregation
    np.testing.assert_allclose(
        equity_event.values,
        equity_vec.values,
        rtol=1e-5,
        atol=30_000.0,  # allow up to $30k drift on a $100mm book due to execution model diffs
        err_msg="Equity curves differ between event_driven and vectorized engines under costs",
    )

    # Daily returns should be close
    np.testing.assert_allclose(
        returns_event.values,
        returns_vec.values,
        rtol=1e-6,
        atol=1e-5,
        err_msg="Daily returns differ between event_driven and vectorized engines under costs",
    )

    # Total return within small bps
    total_event = res_event.total_return
    total_vec = res_vec.total_return
    assert abs(total_event - total_vec) < 5e-4  # within 5 bps
