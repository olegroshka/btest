"""
LSE PEAD — 001_baseline — strategy.py
======================================
Post-Earnings Announcement Drift on LSE companies.

Signal: SUE (Standardised Unexpected Earnings)
    = eps_difference / rolling_std(eps_actual, 8 quarters), clipped ±5σ
    Forward-filled for 21 trading days after report_date.
    Signal = NaN after 21 days → position exits on next daily rebalance.

Universe : LSE companies with analyst coverage + clean earnings history
Portfolio: Top-15 by SUE rank among active signals. Equal-weight, long-only.
Rebalance : Daily — catches new events and drops expired signals.
Execution : MOC, signal_delay_bars=1 (trade close of day after report_date).

Run from btest/ root:
    uv run python "research/generated/LSE PEAD/001_baseline/strategy.py"
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
ATTEMPT_REL  = "research/generated/Dividend Growth/signals/004_lse_pead"
SHARED_DATA  = "research/generated/Dividend Growth/shared_data"
OUTPUT_DIR   = f"{ATTEMPT_REL}/outputs"

START = "2015-01-01"
END   = "2026-01-01"
TOP_N = 8    # reduced from 15 to avoid gating on sparse PEAD days (median ~13 active tickers)


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
    universe = Universe(
        name="LSE_earnings",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=100.0),  # GBX — matches build_signals filter
        ],
    )

    # ── 3. Factors ────────────────────────────────────────────────────────────
    # sue_raw = NaN when no event is active for this ticker (outside 21-day window)
    sue_raw = ExternalFactor(
        name="sue_raw",
        path=str(ATTEMPT_ROOT / "data" / "sue_signal.parquet"),
        per_instrument=True,
    )
    sue = WinsorizedFactor(name="sue", base=sue_raw, z=3.0)

    # ── 4. Signals ────────────────────────────────────────────────────────────
    # valid = True only when this ticker has a live event (sue_raw is not NaN)
    valid = NotNull("sue_raw", name="valid")
    rank  = CrossSectionRank(factor_name="sue", method="percentile", name="rank_cs")

    # ── 5. Portfolio ──────────────────────────────────────────────────────────
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank_cs",
                n=TOP_N,
                mask_name="valid",
                fill_from_unmasked=False,   # never pick tickers with no active event
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(factor_name="rank_cs", n=0),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=1,       # report_date signal → trade at next close (MOC)
        target_gross_leverage=1.0,
        target_net_exposure=1.0,   # long-only
        max_abs_weight_per_name=None,  # no per-name cap — EqualWeight handles sizing
    )

    # ── 6. Execution ──────────────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC"),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=2.0, k=20.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.1),
    )

    # ── 7. Costs ──────────────────────────────────────────────────────────────
    costs = Costs(
        commission=Commission(type="bps_notional", amount=10.0),  # 10bps/trade
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve=None, spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    # ── 8. Backtest config ────────────────────────────────────────────────────
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000.0,
        reporting=Reporting(
            output_dir=OUTPUT_DIR,
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(title="LSE PEAD 001 Baseline"),
        ),
    )

    return Strategy(
        name="lse_pead_001_baseline",
        data=data,
        universe=universe,
        factors={"sue_raw": sue_raw, "sue": sue},
        signals={"valid": valid, "rank_cs": rank},
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    result = run_backtest(build_strategy())
    print(result.summary())
