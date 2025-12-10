# tests_slow/test_integration_simple_momentum_vectorbt.py

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, LessEqual
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio,
    Book,
    TopN,
    BottomN,
    EqualWeight,
)
from quantdsl_backtest.dsl.execution import (
    Execution,
    OrderPolicy,
    LatencyModel,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import (
    Costs,
    Commission,
    BorrowCost,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    MarginConfig,
    RiskChecks,
    Reporting,
)
from quantdsl_backtest.dsl.strategy import Strategy

from quantdsl_backtest.data.adapters import load_market_data
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.accounting import compute_basic_metrics
from quantdsl_backtest.utils.logging import get_logger


# Path to your S&P500 parquet (same as in momentum_long_short_sp500.py)
DATA_SOURCE = "parquet://equities/sp500_daily"


log = get_logger(__name__)


def build_simple_momentum_ls_strategy() -> Strategy:
    """
    Very simple long/short cross-sectional momentum strategy:

        - Factor: 6m log momentum
        - Long top 50 names, short bottom 50 names
        - Equal weights, target gross 2.0, net 0.0
        - No slippage, no commission, no borrow, no financing
        - Rebalance daily, no signal delay

    This is designed specifically for integration testing vs vectorbt.
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
            left=0.9,      # rank >= 0.9 (top 10%)
            right="rank",
        ),
    )
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(
            left="rank",
            right=0.1,      # rank <= 0.1 (bottom 10%)
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
            max_participation=1.0,  # no real limit
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
        engine="event_driven",
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
        name="test_simple_mom_ls",
        data=data_cfg,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest_cfg,
    )


def _build_vectorbt_baseline(prices: pd.DataFrame) -> vbt.Portfolio:
    """
    Build a vectorbt portfolio that implements the exact same
    simple momentum L/S strategy as build_simple_momentum_ls_strategy().
    """

    # 6m log momentum
    mom = np.log(prices / prices.shift(126))

    # For each date, long top 50, short bottom 50, equal-weight,
    # target gross 2.0, net 0.0 (so long book +1, short book -1).
    dates = prices.index
    instruments = prices.columns
    weights = pd.DataFrame(0.0, index=dates, columns=instruments, dtype=float)

    for ts in dates:
        row = mom.loc[ts]
        valid = row.dropna()
        if len(valid) < 100:
            # not enough names, just stay flat
            continue

        top = valid.sort_values(ascending=False).head(50).index
        bottom = valid.sort_values(ascending=True).head(50).index

        long_gross = 1.0
        short_gross = 1.0

        w_long = long_gross / len(top)
        w_short = -short_gross / len(bottom)

        weights.loc[ts, top] = w_long
        weights.loc[ts, bottom] = w_short

    # Build a portfolio using target-percent sizing, no fees/slippage
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        init_cash=100_000_000.0,
        fees=0.0,
        slippage=0.0,
        cash_sharing=True,
        call_seq="auto",
        freq="1D",
    )

    return pf


@pytest.mark.slow
def test_simple_momentum_long_short_matches_vectorbt():
    """
    Integration test:

        - Runs the simple momentum long/short strategy via our DSL + engine
        - Runs an equivalent implementation via vectorbt directly
        - Asserts that equity curves and returns match within reasonable tolerances.
    """
    # Build DSL strategy and run our engine
    strategy = build_simple_momentum_ls_strategy()
    result = run_backtest(strategy)

    # Load same data (using the same adapter the engine uses)
    md = load_market_data(strategy.data, strategy.universe)
    
    # Construct wide price panel for vectorbt baseline
    instruments = md.instruments
    prices = pd.DataFrame(
        {instr: md.bars[instr]["close"] for instr in instruments}
    ).sort_index()

    # Run vectorbt baseline
    pf = _build_vectorbt_baseline(prices)

    # Aggregate baseline equity / returns to a single series
    equity_vbt = pf.value(group_by=True)
    returns_vbt = pf.returns(group_by=True)
    # Per-instrument weights from realized asset values
    weights_vbt = pf.asset_value().div(pf.value(group_by=True), axis=0).fillna(0.0)

    # Align indices
    equity_dsl = result.equity  # from our BacktestResult
    returns_dsl = result.returns

    equity_vbt = equity_vbt.reindex(equity_dsl.index).ffill()
    returns_vbt = returns_vbt.reindex(returns_dsl.index).fillna(0.0)

    # Basic sanity
    assert len(equity_dsl) > 0
    assert len(returns_dsl) > 0

    # Log comparable metrics for both frameworks for visual comparison in CI logs
    metrics_dsl = compute_basic_metrics(returns_dsl, equity_dsl, result.weights)
    metrics_vbt = compute_basic_metrics(returns_vbt, equity_vbt, weights_vbt)
    log.info("[Metrics] DSL: %s", metrics_dsl)
    log.info("[Metrics] vectorbt: %s", metrics_vbt)

    # Compare equity curves and cumulative returns
    # Use reasonably strict tolerances (they should be nearly identical)
    np.testing.assert_allclose(
        equity_dsl.values,
        equity_vbt.values,
        rtol=1e-6,
        atol=1e-2,  # a few cents on 100mm
        err_msg="Equity curves differ between DSL engine and vectorbt baseline",
    )

    np.testing.assert_allclose(
        returns_dsl.values,
        returns_vbt.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Daily returns differ between DSL engine and vectorbt baseline",
    )

    # Compare total returns
    total_ret_dsl = result.total_return
    total_ret_vbt = float((equity_vbt.iloc[-1] / equity_vbt.iloc[0]) - 1.0)

    assert abs(total_ret_dsl - total_ret_vbt) < 1e-6
