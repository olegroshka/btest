"""
CAC Total Return — 139-day momentum timing strategy.

Signal:  log(close_T / close_{T-139}) >= 0.005  → long
         else flat (no short)

Execution: MOC order at T (signal computed at close T, submitted before auction)
           First return captured: close_T → close_{T+1}  (signal_delay_bars=1)

Backtest:  2000-01-01 → 2026-03-31  |  XPAR calendar  |  1bp commission

Best sweep params:  w=139, theta=0.005  →  Sharpe ~0.47, CAGR ~6%, MaxDD ~22%
"""

from __future__ import annotations

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import (
    MaskFromBoolean,
    GreaterEqual,
)
from quantdsl_backtest.dsl.portfolio import TimingPortfolio
from quantdsl_backtest.dsl.execution import (
    Execution,
    OrderPolicy,
    LatencyModel,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    RiskChecks,
    DrawdownPolicy,
    Reporting,
)
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

# ── Parameters ────────────────────────────────────────────────────────────────
MOM_WINDOW = 139       # trading days  (~6.6 months)
MOM_THRESHOLD = 0.005  # log-return threshold  (~+0.50% simple)
START = "2000-01-01"
END   = "2026-03-31"


def build_strategy() -> Strategy:
    # 1. Data — CACT from sfera
    data = DataConfig(
        source="sfera://bbgidx/index_total_return",
        calendar="XPAR",
        frequency="1d",
        start=START,
        end=END,
    )

    # 2. Universe — single instrument
    universe = Universe(name="CACT_TR", static_instruments=["CACT"])

    # 3. Factor — lookback momentum on close
    mom = ReturnFactor(
        name=f"mom_{MOM_WINDOW}",
        field="close",
        lookback=MOM_WINDOW,
        method="log",
    )

    # 4. Signal — long when momentum >= threshold
    entry_signal = MaskFromBoolean(
        name="entry_signal",
        expr=GreaterEqual(left=f"mom_{MOM_WINDOW}", right=MOM_THRESHOLD),
    )

    # 5. Timing portfolio — single instrument, daily MOC
    portfolio = TimingPortfolio(
        signal_name="entry_signal",
        instrument="CACT",
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=1,   # signal at close T → position from close T+1
        target_leverage=1.0,
    )

    # 6. Execution — MOC, minimal slippage (index, not single stock)
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC"),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),
    )

    # 7. Costs — 1bp round-trip commission
    costs = Costs(
        commission=Commission(type="bps_notional", amount=1.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    # 8. Backtest config
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=1.0,
            drawdown=DrawdownPolicy(mode="none"),
        ),
        reporting=Reporting(
            output_dir="outputs/cact_momentum_timing",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=f"CACT Momentum Timing  w={MOM_WINDOW}  θ={MOM_THRESHOLD}"
            ),
        ),
    )

    return Strategy(
        name="cact_momentum_timing",
        data=data,
        universe=universe,
        factors={f"mom_{MOM_WINDOW}": mom},
        signals={"entry_signal": entry_signal},
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    strategy = build_strategy()
    result = run_backtest(strategy)
    print(result.summary())
