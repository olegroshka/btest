"""
LSE Dividend EPS — Long-Only
============================
Signal   : 0.4 × trailing-12M dividend yield rank
         + 0.6 × YoY EPS growth rank (point-in-time, using report_date)
Universe : LSE stocks with dividend + EPS history (non-payers get NaN → excluded)
Portfolio: Long-only Top-20, equal weight, monthly rebalance
Costs    : 10 bps commission (LSE is less liquid than NYSE), realistic slippage
Calendar : XLON (London Stock Exchange)

Return attribution
------------------
The notebook decomposes portfolio returns into:
  • Dividend income  = adj_close return − unadjusted close return
  • Capital gain     = unadjusted close return
Both series are tracked using lse_prices.parquet (has close + close_unadj).

Run from btest/ root:
    uv run python "research/LSE Dividend EPS/001_baseline/strategy.py"
"""
from __future__ import annotations

from pathlib import Path

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
from quantdsl_backtest.dsl.factors import ExternalFactor, WinsorizedFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    NotNull,
)
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
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

BTEST_ROOT    = Path(__file__).resolve().parents[2]   # …/btest
ATTEMPT_ROOT  = Path(__file__).resolve().parent        # …/001_baseline
ATTEMPT_REL   = "research/LSE Dividend EPS/001_baseline"

COMPOSITE_PATH = str(ATTEMPT_ROOT / "data" / "composite.parquet")
PRICES_PATH    = str(ATTEMPT_ROOT / "data" / "lse_prices.parquet")
OUTPUT_DIR     = f"{ATTEMPT_REL}/outputs"

START = "2015-01-01"
END   = "2026-01-01"
TOP_N = 20


def build_strategy() -> Strategy:
    # ── 1. Data ───────────────────────────────────────────────────────────────
    data = DataConfig(
        source=f"parquet://{ATTEMPT_REL}/data/lse_prices.parquet",
        calendar="XLON",
        frequency="1d",
        start=START,
        end=END,
    )

    # ── 2. Universe ───────────────────────────────────────────────────────────
    universe = Universe(
        name="LSE",
        filters=[
            HasHistory(min_days=126),   # ~6 months price history
            MinPrice(min_price=0.05),   # 5p minimum (LSE stocks can be low-priced)
        ],
    )

    # ── 3. Factors ────────────────────────────────────────────────────────────
    # composite = 0.4 × yield rank + 0.6 × EPS YoY rank (pre-computed)
    composite_raw = ExternalFactor(
        name="composite_raw",
        path=COMPOSITE_PATH,
        per_instrument=True,
    )
    composite = WinsorizedFactor(
        name="composite",
        base=composite_raw,
        z=3.0,
    )

    # ── 4. Signals ────────────────────────────────────────────────────────────
    # Cross-section rank: 1.0 = highest composite score
    rank = CrossSectionRank(
        factor_name="composite",
        method="percentile",
        name="rank_cs",
    )

    # Validity: composite must be non-null (non-payers and no-EPS tickers excluded)
    valid = NotNull("composite", name="valid")

    # ── 5. Portfolio: Long-only Top-20 ────────────────────────────────────────
    # Long-only: target_gross_leverage=1.0, target_net_exposure=1.0
    # Short book picks n=0 → no short positions taken
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
            selector=BottomN(
                factor_name="rank_cs",
                n=0,
            ),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1m",       # monthly (closest to quarterly in DSL)
        signal_delay_bars=1,            # signal at close[T] → trade at close[T+1]
        target_gross_leverage=1.0,      # 100% invested long-only
        target_net_exposure=1.0,        # no short offset
        max_abs_weight_per_name=0.10,   # max 10% per name (2× equal weight for 20 stocks)
    )

    # ── 6. Execution ──────────────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(
            default_order_type="MOC",
            time_in_force="DAY",
        ),
        latency=LatencyModel(signal_to_order_delay_bars=0),
        slippage=PowerLawSlippageModel(
            base_bps=2.0,    # LSE spreads are wider than NYSE
            k=20.0,
            exponent=0.5,
        ),
        volume_limits=VolumeParticipation(
            max_participation=0.05,  # conservative — LSE smaller volumes
            mode="proportional",
        ),
    )

    # ── 7. Costs ──────────────────────────────────────────────────────────────
    costs = Costs(
        commission=Commission(type="bps_notional", amount=10.0),   # 10bps stamp duty + broker
        borrow=BorrowCost(default_annual_rate=0.0),                 # long-only, no borrow
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
                title="LSE Dividend EPS — Top-20 Long-Only"
            ),
        ),
    )

    # ── 9. Assemble ───────────────────────────────────────────────────────────
    return Strategy(
        name="lse_dividend_eps_001",
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
