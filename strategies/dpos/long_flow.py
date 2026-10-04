"""Disclosed positioning, second iteration: the long side (the hedge-fund cohort's net buying) sized.

Inputs are written by datacli (``scripts/export_btest_inputs.py --second``):
``prices_daily_long.parquet`` (the long-ladder symbols with a median dollar
volume of at least 5 million since 2018), ``long_flow_cohort.parquet`` (the
per-period demeaned rank in [-0.5, 0.5] of the hedge-fund cohort's quarterly
net buying by continuing managers, residualised on reversal, momentum,
ownership share and size), ``long_level_cohort.parquet`` (the same for the
cohort's ownership share given reversal, momentum and size), their ``_neg``
negations, and ``long_size_tercile_liquid.parquet`` (the size tercile within
this liquid universe at each period, 0 = smallest third; the evaluation
universe's own smallest third holds only a few percent of the liquid names).
The ranks are taken within this universe so that both books are populated.
Every signal is carried forward from the day after the period's publication
(point in time).

Orientation: a positive rank is bullish. The long book is the top
30 percent of the ranks weighted by the rank, the short book the bottom 30
percent; dollar neutral, monthly rebalance with a turnover limit, the first
iteration's cost model. ``DPOS_SCOPE=small`` restricts both books to the
liquid universe's smallest size third, the investable reading of where
the evaluation located the flow information.

Run from the btest root:  uv run python strategies/dpos/long_flow.py
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
    LongShortPortfolio,
    SignalWeight,
    TopN,
    TurnoverLimit,
)
from quantdsl_backtest.dsl.signals import (
    And,
    GreaterEqual,
    LessEqual,
    MaskFromBoolean,
    NotNull,
    RollingMean,
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
SIGNAL = os.environ.get("DPOS_SIGNAL", "long_flow_cohort")  # or long_level_cohort
SCOPE = os.environ.get("DPOS_SCOPE", "all")  # or small
BOOK_EDGE = float(os.environ.get("DPOS_EDGE", "0.2"))
NAMES_PER_SIDE = int(os.environ.get("DPOS_NAMES", "300"))
TURNOVER = float(os.environ.get("DPOS_TURNOVER", "0.3"))
START = os.environ.get("DPOS_START", "2019-01-01")
END = os.environ.get("DPOS_END", "2026-10-01")


def build() -> Strategy:
    data = DataConfig(
        source=f"parquet://{(EXPORTS / 'prices_daily_long.parquet').as_posix()}",
        calendar="XNYS",
        frequency="1d",
        start=START,
        end=END,
    )
    universe = Universe(
        name="long_ladder_liquid",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=5_000_000.0),
        ],
    )
    factors = {
        "s_bull": ExternalFactor(
            name="s_bull",
            path=(EXPORTS / f"{SIGNAL}.parquet").as_posix(),
            per_instrument=True,
        ),
        "s_bear": ExternalFactor(
            name="s_bear",
            path=(EXPORTS / f"{SIGNAL}_neg.parquet").as_posix(),
            per_instrument=True,
        ),
        "tercile": ExternalFactor(
            name="tercile",
            path=(EXPORTS / "long_size_tercile_liquid.parquet").as_posix(),
            per_instrument=True,
        ),
    }
    scope_expr = (
        NotNull("s_bull")
        if SCOPE == "all"
        else And(left=NotNull("s_bull"), right=LessEqual(left="tercile", right=0.5))
    )
    signals = {
        "bull": RollingMean(base="s_bull", window=1, name="bull"),
        "bear": RollingMean(base="s_bear", window=1, name="bear"),
        "in_scope": MaskFromBoolean(name="in_scope", expr=scope_expr),
        "long_candidates": MaskFromBoolean(
            name="long_candidates",
            expr=And(left="in_scope", right=GreaterEqual(left="bull", right=BOOK_EDGE)),
        ),
        "short_candidates": MaskFromBoolean(
            name="short_candidates",
            expr=And(left="in_scope", right=GreaterEqual(left="bear", right=BOOK_EDGE)),
        ),
    }
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="bull",
                n=NAMES_PER_SIDE,
                mask_name="long_candidates",
                fill_from_unmasked=False,
            ),
            weighting=SignalWeight(signal_name="bull"),
        ),
        short_book=Book(
            name="short",
            selector=TopN(
                factor_name="bear",
                n=NAMES_PER_SIDE,
                mask_name="short_candidates",
                fill_from_unmasked=False,
            ),
            weighting=SignalWeight(signal_name="bear"),
        ),
        rebalance_frequency="1m",
        rebalance_at="market_close",
        signal_delay_bars=0,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.02,
        turnover_limit=TurnoverLimit(window_bars=1, max_fraction=TURNOVER),
    )
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=2.0, k=20.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.1, mode="proportional"),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=1.0),
        borrow=BorrowCost(default_annual_rate=0.01),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    name = f"dpos_long_{SIGNAL}_{SCOPE}_n{NAMES_PER_SIDE}"
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=10_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=2.5, drawdown=DrawdownPolicy(mode="none")
        ),
        reporting=Reporting(
            output_dir=f"outputs/{name}",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=f"Disclosed positioning: long side ({SIGNAL}, {SCOPE})"
            ),
        ),
        extra={"hold_when_no_targets": True},
    )
    return Strategy(
        name=name,
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    result = run_backtest(build())
    print(result.summary())
    sys.exit(0)
