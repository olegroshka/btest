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

from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.accounting import compute_basic_metrics


DATA_SOURCE = "parquet://equities/sp500_daily"


def _build_simple_strategy_vectorized() -> Strategy:
    """
    Minimal daily-rebalanced L/S momentum strategy with no costs and
    volume limits set to 100% to ensure the vectorized engine path is used.
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

    from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV

    universe = Universe(
        name="SP500",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=1_000_000.0),
        ],
        id_field="ticker",
    )

    mom_6m = ReturnFactor(
        name="mom_6m",
        field="close",
        lookback=126,
        method="log",
    )
    factors = {"mom_6m": mom_6m}

    rank = CrossSectionRank(
        factor_name="mom_6m",
        mask_name=None,
        method="percentile",
        name="rank",
    )
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=LessEqual(left=0.9, right="rank"),  # rank >= 0.9
    )
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(left="rank", right=0.1),  # rank <= 0.1
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
            n=1,
            mask_name="long_candidates",
        ),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name="short_book",
        selector=BottomN(
            factor_name="rank",
            n=1,
            mask_name="short_candidates",
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,
        long_book=long_book,
        short_book=short_book,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=1.0,
        sector_neutral=None,
        turnover_limit=None,
    )

    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0, use_intraday_vol=False),
        volume_limits=VolumeParticipation(max_participation=1.0, mode="proportional", min_fill_notional=0.0),
        book_model=None,
    )

    costs = Costs(
        commission=Commission(type="per_share", amount=0.0),
        borrow=BorrowCost(default_annual_rate=0.0, curve_name=None),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    backtest_cfg = BacktestConfig(
        engine="vectorized",
        cash_initial=100_000_000.0,
        margin=MarginConfig(long_initial=0.5, short_initial=0.5, maintenance=0.3),
        risk_checks=RiskChecks(max_drawdown=None, max_gross_leverage=None),
        reporting=Reporting(store_trades=True, store_positions=True, metrics=[]),
    )

    return Strategy(
        name="unit_vectorized_turnover_nonzero",
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
def test_vectorized_turnover_is_nonzero():
    """
    Regression test: turnover reported as 0.00 in vectorized engine.

    With daily rebalancing and changing selections, portfolio weights should
    vary over time, leading to strictly positive turnover_annual.
    """
    strat_vec = _build_simple_strategy_vectorized()
    res_vec = run_backtest(strat_vec)

    # Compute metrics via shared accounting logic
    metrics = compute_basic_metrics(res_vec.returns, res_vec.equity, res_vec.weights)

    # Direct sanity check on weights to ensure they change over time
    wt = res_vec.weights
    assert wt.shape[0] > 1
    diff = wt.diff().abs().sum(axis=1)
    assert float((diff > 1e-12).sum()) > 0, "Weights never change; turnover cannot be zero here"

    # Turnover must be positive
    assert metrics.get("turnover_annual", 0.0) > 0.0
