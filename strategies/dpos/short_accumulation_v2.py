"""Disclosed positioning, second iteration: rank-weighted books of abnormal short accumulation.

Inputs are written by datacli (``scripts/export_btest_inputs.py --second``):
the first iteration's daily price panel and ``short_z_inventory_size.parquet``,
the per-date demeaned rank in [-0.5, 0.5] of the short-inventory z-score residualised
on reversal, momentum and log market capitalisation, carried forward from the
day after each FINRA publication (point in time); ``_neg`` is its negation.

Against the first iteration (equal-weight deciles, weekly): the books are the
top and bottom 30 percent of the ranks, weighted by the demeaned rank itself
(a 2.5:1 tilt from the edge of the book to the extreme), rebalanced monthly
with a turnover limit, and the signal is residualised on size as well. Costs
as before: commission, square-root impact, a 10 percent volume cap and a flat
borrow rate; the per-name Interactive Brokers rates are overlaid afterwards by
datacli (``scripts/btest_borrow_overlay.py``).

Run from the btest root:  uv run python strategies/dpos/short_accumulation_v2.py
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
SIGNAL = os.environ.get("DPOS_SIGNAL", "short_z_inventory_size")
BOOK_EDGE = float(
    os.environ.get("DPOS_EDGE", "0.2")
)  # demeaned rank at the edge of each book (0.2 = the top 30 percent)
NAMES_PER_SIDE = int(os.environ.get("DPOS_NAMES", "500"))
TURNOVER = float(
    os.environ.get("DPOS_TURNOVER", "0.3")
)  # fraction of gross per rebalance
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
    # Accumulation is bearish. s_bear is the demeaned rank of the residual z-score, s_bull its negation.
    factors = {
        "s_bear": ExternalFactor(
            name="s_bear",
            path=(EXPORTS / f"{SIGNAL}.parquet").as_posix(),
            per_instrument=True,
        ),
        "s_bull": ExternalFactor(
            name="s_bull",
            path=(EXPORTS / f"{SIGNAL}_neg.parquet").as_posix(),
            per_instrument=True,
        ),
    }
    signals = {
        "bear": RollingMean(
            base="s_bear", window=1, name="bear"
        ),  # the factor as a signal (identity)
        "bull": RollingMean(base="s_bull", window=1, name="bull"),
        "valid": MaskFromBoolean(name="valid", expr=NotNull("s_bear")),
        "short_candidates": MaskFromBoolean(
            name="short_candidates",
            expr=And(left="valid", right=GreaterEqual(left="bear", right=BOOK_EDGE)),
        ),
        "long_candidates": MaskFromBoolean(
            name="long_candidates",
            expr=And(left="valid", right=GreaterEqual(left="bull", right=BOOK_EDGE)),
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
        signal_delay_bars=0,  # the external signal is already shifted to the day after publication
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.01,
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
        borrow=BorrowCost(
            default_annual_rate=0.01
        ),  # flat; the per-name rates are overlaid afterwards
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    name = f"dpos_short_v2_{SIGNAL}_n{NAMES_PER_SIDE}"
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
                title=f"Disclosed positioning: short accumulation v2 ({SIGNAL})"
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
