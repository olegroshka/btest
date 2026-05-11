"""
Dividend Growth Long/Short — QuantDSL
======================================
Signal   : 0.4 × trailing-12M yield rank  +  0.6 × YoY dividend growth rank
           Pre-computed per-instrument (wide parquet) via scripts/build_div_signals.py
Universe : SP500 — dividend payers only (non-payers have NaN composite → excluded)
Portfolio: Dollar-neutral L/S — Top-50 long / Bottom-20 short, weekly rebalance
Costs    : 5 bps commission + 50 bps short borrow + realistic slippage
"""
from __future__ import annotations

import os

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
from quantdsl_backtest.dsl.factors import ExternalFactor, WinsorizedFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    MaskFromBoolean,
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


def build_strategy() -> Strategy:
    # ── 1. Data & universe ────────────────────────────────────────────────────
    data = DataConfig(
        source="parquet://equities/sp500_daily",
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-01-01",
    )
    universe = Universe(
        name="SP500",
        filters=[HasHistory(min_days=252), MinPrice(min_price=5.0)],
    )

    # ── 2. Factors ────────────────────────────────────────────────────────────
    # Per-instrument wide parquet: index=trading_date, columns=ticker
    # Values = composite (0.4 * yield_pct + 0.6 * growth_pct), NaN = non-payer
    composite_raw = ExternalFactor(
        name="composite_raw",
        path="data/dividends/div_composite.parquet",
        per_instrument=True,
    )
    composite = WinsorizedFactor(
        name="composite",
        base=composite_raw,
        z=3.0,
    )

    # ── 3. Signals ────────────────────────────────────────────────────────────
    # is_payer: True where composite is non-NaN (paid a dividend in last 12M)
    is_payer = MaskFromBoolean(
        name="is_payer",
        expr=NotNull("composite"),
    )
    # Percentile rank within payers universe only
    rank = CrossSectionRank(
        factor_name="composite",
        mask_name="is_payer",
        method="percentile",
        name="rank",
    )

    factors = {"composite_raw": composite_raw, "composite": composite}
    signals = {"is_payer": is_payer, "rank": rank}

    # ── 4. Portfolio ──────────────────────────────────────────────────────────
    # Long: top-50 highest yield + growth payers
    # Short: bottom-20 — weakest growers / highest relative yield stress
    # Both restricted to dividend payers (fill_from_unmasked=False)
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank",
                n=50,
                mask_name="is_payer",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(
                factor_name="rank",
                n=20,
                mask_name="is_payer",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1w",
        signal_delay_bars=1,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.04,
    )

    # ── 5. Execution & costs ──────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC"),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=1.0, k=5.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.1),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=5.0),
        borrow=BorrowCost(default_annual_rate=0.005),
        financing=FinancingCost(),
        fees=StaticFees(),
    )

    # ── 6. Backtest config ────────────────────────────────────────────────────
    bt = BacktestConfig(
        cash_initial=1_000_000.0,
        reporting=Reporting(
            output_dir="outputs/dividend_growth_ls",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title="Dividend Growth L/S  |  Yield 40% + Growth 60%",
            ),
        ),
    )

    return Strategy(
        name="dividend_growth_ls",
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


def main() -> None:
    strategy = build_strategy()
    result = run_backtest(strategy)
    print(result.summary())


if __name__ == "__main__":
    main()
