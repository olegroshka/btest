"""Disclosed positioning: cost realism for abnormal short accumulation (the ``z_inventory`` score).

Inputs are written by datacli (``scripts/export_btest_inputs.py``) under the
positioning root: a daily price panel of the short-ladder symbols with a median
dollar volume of at least 5 million since 2018, and the per-symbol EWM z-score
of log short inventory residualised on reversal and momentum, carried forward
from the day after each FINRA publication (point in time).

Orientation: heavy or growing short inventory is bearish. The
portfolio shorts the top decile of residual accumulation and buys the bottom
decile, equal weight, dollar neutral, rebalanced weekly, with a commission,
a price-impact slippage model, a 10 percent volume cap and a flat borrow rate (the broker's borrow file
does not exist yet). The point of the run is the size of the claim after
costs, not whether the ordering exists (the evaluation settled that).

Run from the btest root:  uv run python strategies/dpos/short_accumulation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    DrawdownPolicy,
    Reporting,
    RiskChecks,
)
from quantdsl_backtest.dsl.costs import (
    BorrowCost,
    Commission,
    Costs,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.factors import ExternalFactor
from quantdsl_backtest.dsl.portfolio import (
    Book,
    BottomN,
    EqualWeight,
    LongShortPortfolio,
    TopN,
)
from quantdsl_backtest.dsl.signals import (
    And,
    CrossSectionRank,
    GreaterEqual,
    LessEqual,
    MaskFromBoolean,
    NotNull,
    Quantile,
)
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import HasHistory, MinDollarADV, MinPrice, Universe
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

_REPO = (
    Path(__file__).resolve().parents[2]
)  # the btest root; the datacli exports live under data/raw/ (a local copy, not committed)
EXPORTS = Path(
    os.environ.get(
        "DATACLI_BTEST_EXPORTS",
        str(_REPO / "data" / "raw" / "positioning" / "exports" / "btest"),
    )
)
VARIANT = os.environ.get("DPOS_SIGNAL", "short_z_inventory")  # or short_z_inventory_raw
DECILE = 0.10
NAMES_PER_SIDE = int(os.environ.get("DPOS_NAMES", "150"))
START = os.environ.get("DPOS_START", "2019-01-01")
END = os.environ.get("DPOS_END", "2026-10-01")


def build() -> Strategy:
    data = DataConfig(
        source=f"parquet://{(EXPORTS / 'prices_daily.parquet').as_posix()}",
        calendar="XNYS",
        frequency="1d",
        start=START,
        end=END,
    )
    universe = Universe(
        name="short_ladder_liquid",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=5_000_000.0),
        ],
    )
    accumulation = ExternalFactor(
        name="z_inv",
        path=(EXPORTS / f"{VARIANT}.parquet").as_posix(),
        per_instrument=True,
    )
    rank = CrossSectionRank(factor_name="z_inv", name="rank")
    valid = MaskFromBoolean(name="valid", expr=NotNull("z_inv"))
    # accumulation is bearish, so the top decile of z_inv is the short book
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=And(
            left="valid",
            right=LessEqual(
                left=Quantile(factor_name="rank", q=1 - DECILE), right="rank"
            ),
        ),
    )
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=And(
            left="valid",
            right=GreaterEqual(
                left=Quantile(factor_name="rank", q=DECILE), right="rank"
            ),
        ),
    )
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=BottomN(
                factor_name="rank",
                n=NAMES_PER_SIDE,
                mask_name="long_candidates",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=TopN(
                factor_name="rank",
                n=NAMES_PER_SIDE,
                mask_name="short_candidates",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1w",
        rebalance_at="market_close",
        signal_delay_bars=0,  # the external signal is already shifted to the day after publication
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.02,
    )
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=2.0, k=20.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.1, mode="proportional"),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=1.0),
        borrow=BorrowCost(
            default_annual_rate=0.01
        ),  # 1 percent flat until the borrow file exists
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=10_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=2.5, drawdown=DrawdownPolicy(mode="none")
        ),
        reporting=Reporting(
            output_dir=f"outputs/dpos_short_accumulation_{VARIANT}",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=f"Disclosed positioning: short accumulation ({VARIANT})"
            ),
        ),
        extra={"hold_when_no_targets": True},
    )
    return Strategy(
        name=f"dpos_short_accumulation_{VARIANT}",
        data=data,
        universe=universe,
        factors={"z_inv": accumulation},
        signals={
            "rank": rank,
            "valid": valid,
            "short_candidates": short_candidates,
            "long_candidates": long_candidates,
        },
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    result = run_backtest(build())
    print(result.summary())
    sys.exit(0)
