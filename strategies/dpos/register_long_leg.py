"""Disclosed positioning: the long leg of the register profit ordering alone, with costs.

datacli computes the leg's daily weights (``scripts/register_long_leg.py``,
``positioning.register_long_leg``: on the first trading day of each week the
euro-register names whose disclosed shorts are losing the most, the bottom
30 percent of the profit ranks, weighted by the rank, capped at 4 percent,
held to the next rebalance) and writes ``register_long_leg_weights.parquet``
next to the register profit inputs. This strategy hands that matrix to the engine
(``TargetWeights``) so the costs are the engine's: commission, square-root
impact, a volume cap; no borrow, long only, fully invested.

The weights at row ``d`` are set from the signal shown on ``d`` and are
held from the close of ``d``, so ``signal_delay_bars=0``; the engine trades
only when the matrix changes (``rebalance_on_change``), that is weekly.

Run from the btest root:  uv run python strategies/dpos/register_long_leg.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

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
from quantdsl_backtest.dsl.portfolio import TargetWeights
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
WEIGHTS = os.environ.get("DPOS_WEIGHTS", "register_long_leg_weights")
MIN_ADV = float(os.environ.get("DPOS_MIN_ADV", "1000000"))
START = os.environ.get("DPOS_START", "2015-01-01")
END = os.environ.get("DPOS_END", "2026-10-02")
NO_COSTS = os.environ.get("DPOS_NO_COSTS") == "1"


def build() -> Strategy:
    data = DataConfig(
        source=f"parquet://{(EXPORTS / 'register_prices_daily.parquet').as_posix()}",
        calendar="XETR",
        frequency="1d",
        start=START,
        end=END,
    )
    universe = Universe(
        name="euro_register_issuers",
        filters=[
            HasHistory(min_days=126),
            MinPrice(min_price=1.0),
            MinDollarADV(min_dollar_adv=MIN_ADV),
        ],
    )
    weights = pd.read_parquet(EXPORTS / f"{WEIGHTS}.parquet")
    weights.index = pd.DatetimeIndex(weights.index)
    portfolio = TargetWeights(
        weights=weights,
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,
        rebalance_on_change=True,
        target_gross_leverage=1.0,
    )
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(
            base_bps=0.0 if NO_COSTS else 2.0, k=0.0 if NO_COSTS else 20.0, exponent=0.5
        ),
        volume_limits=VolumeParticipation(max_participation=0.1, mode="proportional"),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=0.0 if NO_COSTS else 1.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    name = f"dpos_register_long_leg_{WEIGHTS}" + ("_nocost" if NO_COSTS else "")
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=10_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=1.5, drawdown=DrawdownPolicy(mode="none")
        ),
        reporting=Reporting(
            output_dir=f"outputs/{name}",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title="Disclosed positioning: register long leg alone"
            ),
        ),
        extra={"hold_when_no_targets": True},
    )
    return Strategy(
        name=name,
        data=data,
        universe=universe,
        factors={},
        signals={},
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    result = run_backtest(build())
    print(result.summary())
    sys.exit(0)
