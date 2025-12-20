import numpy as np
import pandas as pd

import pytest

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import MaskFromBoolean, NotNull, CrossSectionRank, GreaterEqual
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, EqualWeight, TurnoverLimit
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting, RiskChecks
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest


DATA_SOURCE = "parquet://equities/indicies.parquet"


def _build_equal_weight_strategy(engine: str) -> Strategy:
    data = DataConfig(
        source=DATA_SOURCE,
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-12-12",
        price_adjustment="split_dividend",
        fields=["close"],
    )

    universe = Universe(
        name="Indices",
        id_field="ticker",
        filters=[HasHistory(min_days=252)],
    )

    # Dummy factor just to drive rank
    mom_20 = ReturnFactor(
        name="mom_20",
        field="close",
        lookback=20,
        method="log",
    )

    factors = {"mom_20": mom_20}

    # Valid mask: just non-null factor
    valid = MaskFromBoolean(
        name="valid",
        expr=NotNull(factor_name="mom_20"),
    )

    rank = CrossSectionRank(
        factor_name="mom_20",
        mask_name="valid",
        method="percentile",
        name="rank",
    )

    # Long everything that is valid (TopN with n large)
    long_mask = MaskFromBoolean(
        name="long_candidates",
        expr=GreaterEqual(left="rank", right=0.0),
    )

    signals = {
        "valid": valid,
        "rank": rank,
        "long_candidates": long_mask,
    }

    long_book = Book(
        name="long_book",
        selector=TopN(
            factor_name="rank",
            n=100,            # effectively, all valid names
            mask_name="long_candidates",
        ),
        weighting=EqualWeight(),
    )

    # No shorts
    dummy_short = Book(
        name="short_book",
        selector=TopN(
            factor_name="rank",
            n=0,
            mask_name=None,
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=dummy_short,
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,
        target_gross_leverage=1.0,
        target_net_exposure=1.0,
        max_abs_weight_per_name=1.0,
        sector_neutral=None,
        turnover_limit=TurnoverLimit(
            window_bars=1,
            max_fraction=1.0,
        ),
    )

    execution = Execution(
        order_policy=OrderPolicy(),
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
    )

    costs = Costs(
        commission=Commission(type="bps_notional", amount=0.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    bt = BacktestConfig(
        engine=engine,
        cash_initial=1_000_000,
        risk_checks=RiskChecks(),
        reporting=Reporting(
            store_trades=True,
            store_positions=True,
            metrics=["sharpe", "max_drawdown", "turnover"],
        ),
    )

    return Strategy(
        name=f"indices_equal_weight_{engine}",
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


@pytest.mark.slow
def test_equal_weight_indices_event_vs_vectorized():
    strat_ev = _build_equal_weight_strategy(engine="event_driven")
    strat_vec = _build_equal_weight_strategy(engine="vectorized")

    res_ev = run_backtest(strat_ev)
    res_vec = run_backtest(strat_vec)

    # Align indices
    rets_ev = res_ev.returns
    rets_vec = res_vec.returns.reindex(rets_ev.index).fillna(0.0)

    # Check sanity: no crazy daily returns in either engine
    assert rets_ev.abs().max() < 0.3  # 30% daily is already extreme for indices
    assert rets_vec.abs().max() < 0.3

    # Check engines roughly agree
    import numpy as np
    np.testing.assert_allclose(
        rets_ev.values,
        rets_vec.values,
        rtol=1e-6,
        atol=1e-4,
    )
