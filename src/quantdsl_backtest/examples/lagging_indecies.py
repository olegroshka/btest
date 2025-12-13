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
    # 1) Data config: indices parquet
    data = DataConfig(
        source="parquet://equities/indicies.parquet",
        calendar="XNYS",  # single calendar; temporal effects via returns
        frequency="1d",
        start="2015-01-01",
        end="2025-12-31",
        price_adjustment="split_dividend",
        fields=["open", "high", "low", "close", "volume"],
    )

    # 2) Universe: minimal sanity filters
    universe = Universe(
        name="Indices",
        id_field="ticker",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
        ],
    )

    # 3) Factors: multi-horizon momentum + vol
    mom_20 = ReturnFactor(
        name="mom_20",
        field="close",
        lookback=20,
        method="log",
    )
    mom_60 = ReturnFactor(
        name="mom_60",
        field="close",
        lookback=60,
        method="log",
    )
    vol_20 = VolatilityFactor(
        name="vol_20",
        field="close",
        lookback=20,
        method="realized",
        annualize=True,
    )

    factors = {
        "mom_20": mom_20,
        "mom_60": mom_60,
        "vol_20": vol_20,
    }

    # 4) Signals
    #
    # Rank 20d and 60d momentum cross-sectionally.
    rank_20 = CrossSectionRank(
        factor_name="mom_20",
        mask_name=None,
        method="percentile",
        name="rank_20",
    )
    rank_60 = CrossSectionRank(
        factor_name="mom_60",
        mask_name=None,
        method="percentile",
        name="rank_60",
    )

    # We'll approximate "combined score" as the average of the two ranks
    # inside the signal engine by composing boolean masks on each:
    combined_strong = MaskFromBoolean(
        name="combined_strong",
        expr=And(
            left=GreaterEqual(left="rank_20", right=0.5),
            right=GreaterEqual(left="rank_60", right=0.5),
        ),
    )
    combined_weak = MaskFromBoolean(
        name="combined_weak",
        expr=And(
            left=LessEqual(left="rank_20", right=0.5),
            right=LessEqual(left="rank_60", right=0.5),
        ),
    )

    # Valid data mask: non-null mom and reasonable volatility
    valid = MaskFromBoolean(
        name="valid",
        expr=And(
            left=NotNull(factor_name="mom_20"),
            right=NotNull(factor_name="mom_60"),
        ),
    )

    # LONG CANDIDATES:
    # indices with above-median 20d AND 60d momentum.
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=And(
            left="combined_strong",
            right="valid",
        ),
    )

    # SHORT CANDIDATES:
    # indices with below-median 20d AND 60d momentum.
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=And(
            left="combined_weak",
            right="valid",
        ),
    )

    signals = {
        "rank_20": rank_20,
        "rank_60": rank_60,
        "combined_strong": combined_strong,
        "combined_weak": combined_weak,
        "valid": valid,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    # 5) Portfolio: small long/short, trend following
    #
    # Books use rank_60 as selector key (slower horizon), but are masked
    # by 'long_candidates' / 'short_candidates' that also require 20d trend.
    long_book = Book(
        name="long_book",
        selector=TopN(
            factor_name="rank_60",
            n=2,
            mask_name="long_candidates",
        ),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name="short_book",
        selector=BottomN(
            factor_name="rank_60",
            n=2,
            mask_name="short_candidates",
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=0,
        target_gross_leverage=1.2,   # a bit conservative vs 1.5–2.0
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.75,
        sector_neutral=None,
        turnover_limit=TurnoverLimit(
            window_bars=5,
            max_fraction=2.0,  # allow moderate churning but not crazy
        ),
    )

    # 6) Execution & costs (leave modest but not crazy)
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(
            signal_to_order_delay_bars=0,
            market_latency_ms=0,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=0.5,  # tighter on highly liquid indices
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
        commission=Commission(
            type="bps_notional",
            amount=0.2,   # keep frictions modest to avoid domination in example
        ),
        borrow=BorrowCost(
            default_annual_rate=0.01,
        ),
        financing=FinancingCost(
            base_rate_curve="SOFR",
            spread_bps=0.0,
        ),
        fees=StaticFees(
            nav_fee_annual=0.0,
            perf_fee_fraction=0.0,
        ),
    )

    # 7) Backtest config & reporting
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=1_000_000,
        reporting=Reporting(
            store_trades=True,
            store_positions=True,
            metrics=[
                "sharpe",
                "sortino",
                "max_drawdown",
                "turnover",
                "daily_returns",
            ],
        ),
    )

    # 8) Compose strategy
    strategy = Strategy(
        name="lagging_indecies_trend_cs",
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
