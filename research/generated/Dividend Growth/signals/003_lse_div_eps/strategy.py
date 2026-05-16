"""
LSE Dividend EPS — Attempt 002 (Fixed Universe)
================================================
Changes vs 001_baseline:
  • Prices: bad ticks (|1d return| > 50%) removed at build time
  • Universe: MinPrice ≥ 100 GBX + MinHistory ≥ 252 days (engine + signal level)
  • Portfolio: Top-30 (wider than Top-20 → less concentration in 250-ticker pool)
  • Costs: unchanged (10 bps commission, realistic slippage)

Run from btest/ root:
    uv run python "research/generated/LSE Dividend EPS/002_fixed/strategy.py"
"""
from __future__ import annotations

from pathlib import Path

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
from quantdsl_backtest.dsl.factors import ExternalFactor, WinsorizedFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank, NotNull
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio, Book, TopN, BottomN, EqualWeight,
)
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel,
    PowerLawSlippageModel, VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

ATTEMPT_ROOT = Path(__file__).resolve().parent
ATTEMPT_REL  = "research/generated/Dividend Growth/signals/003_lse_div_eps"
SHARED_DATA  = "research/generated/Dividend Growth/shared_data"
OUTPUT_DIR   = f"{ATTEMPT_REL}/outputs"

START = "2015-01-01"
END   = "2026-01-01"
TOP_N = 30          # wider than 001 — less concentration in the smaller universe


def build_strategy() -> Strategy:
    # ── 1. Data ───────────────────────────────────────────────────────────────
    data = DataConfig(
        source=f"parquet://{SHARED_DATA}/lse_prices.parquet",
        calendar="XLON",
        frequency="1d",
        start=START,
        end=END,
    )

    # ── 2. Universe ───────────────────────────────────────────────────────────
    # MinPrice ≥ 100 GBX (1 GBP) mirrors the signal-build filter so the engine
    # sees only tickers that already have a clean composite score.
    # MinHistory 252 = 1 year → consistent with signal-build MIN_DAYS.
    universe = Universe(
        name="LSE_fixed",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=100.0),   # GBX — matches signal build filter
        ],
    )

    # ── 3. Factors ────────────────────────────────────────────────────────────
    composite_raw = ExternalFactor(
        name="composite_raw",
        path=str(ATTEMPT_ROOT / "data" / "composite.parquet"),
        per_instrument=True,
    )
    composite = WinsorizedFactor(
        name="composite",
        base=composite_raw,
        z=3.0,
    )

    # ── 4. Signals ────────────────────────────────────────────────────────────
    rank  = CrossSectionRank(factor_name="composite", method="percentile", name="rank_cs")
    valid = NotNull("composite", name="valid")

    # ── 5. Portfolio: Long-only Top-30 ────────────────────────────────────────
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank_cs",
                n=TOP_N,
                mask_name="valid",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(factor_name="rank_cs", n=0),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1m",
        signal_delay_bars=1,
        target_gross_leverage=1.0,
        target_net_exposure=1.0,
        max_abs_weight_per_name=0.08,   # 8% cap — tighter than 001 (3.3× equal weight)
    )

    # ── 6. Execution ──────────────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0),
        slippage=PowerLawSlippageModel(base_bps=2.0, k=20.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.05, mode="proportional"),
    )

    # ── 7. Costs ──────────────────────────────────────────────────────────────
    costs = Costs(
        commission=Commission(type="bps_notional", amount=10.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0),
    )

    # ── 8. Backtest config ────────────────────────────────────────────────────
    config = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000.0,
        reporting=Reporting(
            output_dir=OUTPUT_DIR,
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title="LSE Dividend EPS — Top-30 Long-Only (002 Fixed)"
            ),
        ),
    )

    # ── 9. Assemble ───────────────────────────────────────────────────────────
    return Strategy(
        name="lse_dividend_eps_002",
        data=data,
        universe=universe,
        factors={"composite_raw": composite_raw, "composite": composite},
        signals={"rank_cs": rank, "valid": valid},
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=config,
    )


if __name__ == "__main__":
    strategy = build_strategy()
    run_backtest(strategy)
