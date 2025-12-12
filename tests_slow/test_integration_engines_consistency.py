# tests_slow/test_integration_engines_consistency.py

import numpy as np
import pytest

from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    MarginConfig,
    RiskChecks,
    Reporting,
)
from quantdsl_backtest.dsl.costs import (
    Costs,
    Commission,
    BorrowCost,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    OrderPolicy,
    LatencyModel,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio,
    Book,
    TopN,
    BottomN,
    EqualWeight,
)
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, LessEqual
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.engine.accounting import compute_basic_metrics
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.utils.logging import get_logger

# Reuse the same data source as other integration tests
DATA_SOURCE = "parquet://equities/sp500_daily"

log = get_logger(__name__)


def _build_simple_momentum_ls_strategy(engine: str) -> Strategy:
    """
    Same simple cross-sectional momentum L/S strategy used in the
    vectorbt integration test, but parameterised by engine
    ("event_driven" or "vectorized").
    """
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
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=1_000_000.0),
        ],
        id_field="ticker",
    )

    # Factor: 6m log momentum
    mom_6m = ReturnFactor(
        name="mom_6m",
        field="close",
        lookback=126,
        method="log",
    )
    factors = {"mom_6m": mom_6m}

    # Signals:
    # - "rank": percentile rank of momentum
    # - "long_candidates": top 10% by rank
    # - "short_candidates": bottom 10% by rank
    rank = CrossSectionRank(
        factor_name="mom_6m",
        mask_name=None,
        method="percentile",
        name="rank",
    )

    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=LessEqual(
            left=0.9,   # rank >= 0.9 (top 10%)
            right="rank",
        ),
    )
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(
            left="rank",
            right=0.1,  # rank <= 0.1 (bottom 10%)
        ),
    )

    signals = {
        "rank": rank,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    long_book = Book(
        name="long_book",
        selector=TopN(
            factor_name="rank",
            n=50,
            mask_name="long_candidates",
        ),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name="short_book",
        selector=BottomN(
            factor_name="rank",
            n=50,
            mask_name="short_candidates",
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,           # no signal delay for this test
        long_book=long_book,
        short_book=short_book,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.05,  # not binding in most cases
        sector_neutral=None,
        turnover_limit=None,
    )

    # No execution frictions
    execution = Execution(
        order_policy=OrderPolicy(
            default_order_type="MOC",
            time_in_force="DAY",
        ),
        latency=LatencyModel(
            signal_to_order_delay_bars=0,
            market_latency_ms=0,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=0.0,
            k=0.0,
            exponent=1.0,
            use_intraday_vol=False,
        ),
        volume_limits=VolumeParticipation(
            max_participation=1.0,
            mode="proportional",
            min_fill_notional=0.0,
        ),
        book_model=None,
    )

    # No costs
    costs = Costs(
        commission=Commission(
            type="per_share",
            amount=0.0,
        ),
        borrow=BorrowCost(
            default_annual_rate=0.0,
            curve_name=None,
        ),
        financing=FinancingCost(
            base_rate_curve="SOFR",
            spread_bps=0.0,
        ),
        fees=StaticFees(
            nav_fee_annual=0.0,
            perf_fee_fraction=0.0,
        ),
    )

    backtest_cfg = BacktestConfig(
        engine=engine,
        cash_initial=100_000_000.0,
        margin=MarginConfig(
            long_initial=0.5,
            short_initial=0.5,
            maintenance=0.3,
        ),
        risk_checks=RiskChecks(
            max_drawdown=None,
            max_gross_leverage=None,
        ),
        reporting=Reporting(
            store_trades=True,
            store_positions=True,
            metrics=[],
        ),
    )

    return Strategy(
        name=f"test_simple_mom_ls_{engine}",
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
def test_simple_momentum_long_short_event_vs_vectorized():
    """
    Integration test:

      - Build *identical* DSL strategy twice, with different engines:
          * engine='event_driven'
          * engine='vectorized'
      - Run both through run_backtest(...)
      - Assert that equity curves, daily returns, and key metrics match
        within tight tolerances.
    """

    # Build identical strategies with different engines
    strat_event = _build_simple_momentum_ls_strategy(engine="event_driven")
    strat_vec = _build_simple_momentum_ls_strategy(engine="vectorized")

    # Run both engines
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
    log.info("[Engine=event_driven] metrics: %s", metrics_event)
    log.info("[Engine=vectorized]   metrics: %s", metrics_vec)

    # Equity curves should be almost identical
    np.testing.assert_allclose(
        equity_event.values,
        equity_vec.values,
        rtol=1e-6,
        atol=1e-2,  # a few cents on 100mm
        err_msg="Equity curves differ between event_driven and vectorized engines",
    )

    # Daily returns should be almost identical
    np.testing.assert_allclose(
        returns_event.values,
        returns_vec.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Daily returns differ between event_driven and vectorized engines",
    )

    # Total return sanity
    total_event = res_event.total_return
    total_vec = res_vec.total_return
    assert abs(total_event - total_vec) < 1e-6
