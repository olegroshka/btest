"""
Lagging indices example: explore timezone lead/lag and short‑lived inefficiencies.

This example builds a small cross‑sectional strategy over major equity indices
using the parquet dataset saved under equities/indicies.parquet. It combines short‑term
momentum windows intended to capture potential timezone lead/lag effects.

Notes/assumptions:
- We use a single calendar (XNYS) and daily frequency to keep the example
  simple and compatible with the current engine. Extending to native calendars
  per index can be a follow‑up.
- Many index series may lack intraday fields (open/high/low). The strategy
  relies primarily on close‑to‑close returns across multiple windows.
- Costs approximate Interactive Brokers style by using bps on notional,
  plus slippage and financing/borrow models.

Outputs (tearsheet, etc.) are written to outputs/lagging_indecies/.
"""

from __future__ import annotations

import os

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.transforms import CleaningTransform
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
from quantdsl_backtest.dsl.factors import ReturnFactor, VolatilityFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    Quantile,
    LessEqual,
    GreaterEqual,
    MaskFromBoolean,
    And,
    NotNull,
)
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio,
    Book,
    TopN,
    BottomN,
    EqualWeight,
    TurnoverLimit,
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
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.data_loader import load_data_for_strategy


def build_strategy() -> Strategy:
    # 1) Data config: use indices parquet created earlier
    data = DataConfig(
        source="parquet://equities/indicies.parquet",
        calendar="XNYS",  # single calendar; timezone effects approximated via returns
        frequency="1d",
        start="2015-01-01",
        end="2025-12-31",
        price_adjustment="split_dividend",
        fields=["open", "high", "low", "close", "volume"],
        transforms=[CleaningTransform]
    )

    # 2) Universe: require some history and loose price floor
    universe = Universe(
        name="Indices",
        id_field="ticker",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=1.0),
        ],
    )

    # 3) Factors
    # Short / medium horizon momentum
    mom_1 = ReturnFactor(name="mom_1", field="close", lookback=1, method="log")
    mom_5 = ReturnFactor(name="mom_5", field="close", lookback=5, method="log")
    mom_20 = ReturnFactor(name="mom_20", field="close", lookback=20, method="log")

    # Realized volatility as simple risk proxy
    vol_20 = VolatilityFactor(
        name="vol_20",
        field="close",
        lookback=20,
        method="realized",
        annualize=True,
    )

    factors = {
        "mom_1": mom_1,
        "mom_5": mom_5,
        "mom_20": mom_20,
        "vol_20": vol_20,
    }

    # 4) Signals
    #
    # Core ranking signal: cross-sectional rank of 5-day momentum.
    # This is what the portfolio uses to pick top/bottom names.
    rank = CrossSectionRank(
        factor_name="mom_5",
        mask_name=None,
        method="percentile",
        name="rank",
    )

    # Quantile thresholds for "strong" / "weak" on each horizon.
    # We keep them moderately tight so we almost always have candidates
    # even with a small universe of ~7 indices.
    mom5_high = Quantile(factor_name="mom_5", q=0.65)  # strong short-term trend
    mom5_low = Quantile(factor_name="mom_5", q=0.35)   # weak short-term trend

    mom1_high = Quantile(factor_name="mom_1", q=0.65)  # strong recent move
    mom1_low = Quantile(factor_name="mom_1", q=0.35)   # weak / negative recent move

    # "Strong trend" / "weak trend" masks based on mom_5:
    strong_trend = MaskFromBoolean(
        name="strong_trend",
        expr=GreaterEqual(left="mom_5", right=mom5_high),
    )
    weak_trend = MaskFromBoolean(
        name="weak_trend",
        expr=LessEqual(left="mom_5", right=mom5_low),
    )

    # "Strong recent move" / "weak recent move" based on mom_1:
    strong_recent = MaskFromBoolean(
        name="strong_recent",
        expr=GreaterEqual(left="mom_1", right=mom1_high),
    )
    weak_recent = MaskFromBoolean(
        name="weak_recent",
        expr=LessEqual(left="mom_1", right=mom1_low),
    )

    # Basic "valid data" mask: non-null momentum and volatility.
    valid = MaskFromBoolean(
        name="valid",
        expr=And(
            left=NotNull(factor_name="mom_5"),
            right=NotNull(factor_name="vol_20"),
        ),
    )

    # LONG CANDIDATES:
    # - strong short-term trend (mom_5 high)
    # - but weak/negative very recent move (mom_1 low)
    # Interpreted as "trend that just lagged or pulled back": potential catch-up.
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=And(
            left=And(left="strong_trend", right="weak_recent"),
            right="valid",
        ),
    )

    # SHORT CANDIDATES:
    # - weak short-term trend (mom_5 low)
    # - but strong positive very recent move (mom_1 high)
    # Interpreted as "overreaction against a weak trend": potential give-back.
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=And(
            left=And(left="weak_trend", right="strong_recent"),
            right="valid",
        ),
    )

    signals = {
        "rank": rank,
        "strong_trend": strong_trend,
        "weak_trend": weak_trend,
        "strong_recent": strong_recent,
        "weak_recent": weak_recent,
        "valid": valid,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    # 5) Portfolio: small L/S, daily rebalance, no signal delay
    #
    # The Book selectors use 'rank' (mom_5 percentile) to pick names
    # *within* the long/short candidate masks defined above.
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank",   # name of the signal we rank on
                n=2,
                mask_name="long_candidates",
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(
                factor_name="rank",
                n=2,
                mask_name="short_candidates",
            ),
            weighting=EqualWeight(),
        ),
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,         # use today’s signals for today’s close
        target_gross_leverage=1.5,   # a bit more active than 1.0
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.75,
        sector_neutral=None,
        turnover_limit=TurnoverLimit(window_bars=1, max_fraction=1.0),
    )

    # 6) Execution & costs — keep roughly as you had, but keep frictions modest
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(
            signal_to_order_delay_bars=0,
            market_latency_ms=0,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=1.0,   # modest slippage on indices
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
        commission=Commission(type="bps_notional", amount=0.5),
        borrow=BorrowCost(default_annual_rate=0.01),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    # 7) Backtest runtime and reporting
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000,
        reporting=Reporting(
            store_trades=True,
            store_positions=True,
            metrics=[
                "daily_returns",
                "sharpe",
                "sortino",
                "max_drawdown",
                "turnover",
            ],
        ),
    )

    # 8) Compose strategy
    strategy = Strategy(
        name="lagging_indecies_v2",
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )
    return strategy


def main() -> None:
    strategy = build_strategy()
    # Run with the selected engine (vectorized by default for this example)
    result = run_backtest(strategy)
    print(result.summary())

    try:
        qs_metric_names = [
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "skew",
            "kurtosis",
            "var",
            "cvar",
        ]
        qs_metrics = result.quantstats_metrics(qs_metric_names, risk_free=0.0)
        print("\n=== QuantStats metrics ===")
        # Nice aligned printing
        print(qs_metrics.to_string(float_format=lambda x: f"{x:0.4f}"))
    except RuntimeError as e:
        # quantstats not installed
        print(f"\nQuantStats metrics skipped: {e}")

    # Write QuantStats outputs if configured in BacktestConfig.Reporting
    out_dir = os.path.join("outputs", "lagging_indecies")
    os.makedirs(out_dir, exist_ok=True)
    try:
        html_path = os.path.join(out_dir, "tearsheet.html")
        result.quantstats_tearsheet(output=html_path, title="Lagging Indecies L/S (QuantDSL)")
        print(f"QuantStats HTML report written to: {html_path}")
    except RuntimeError as e:
        print(f"QuantStats outputs skipped: {e}")

    # ------------------------------------------------------------------
    # Existing plots (equity / exposures / drawdowns)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        ax_eq = result.plot_equity()
        ax_exp = result.plot_exposures()
        ax_dd = result.plot_drawdowns()

        plot_files = [
            ("equity.png", ax_eq),
            ("exposures.png", ax_exp),
            ("drawdowns.png", ax_dd),
        ]

        for fname, ax in plot_files:
            fig = ax.figure
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)

        print(f"Plots saved under {out_dir}")
    except RuntimeError as e:
        print(f"Plotting skipped: {e}")

    # ------------------------------------------------------------------
    # Export detailed tabular outputs
    # ------------------------------------------------------------------
    result.to_parquet(out_dir)


if __name__ == "__main__":
    main()
