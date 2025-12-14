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
from quantdsl_backtest.dsl.factors import (
    ReturnFactor,
    VolatilityFactor,
    OvernightReturnFactor,
    IntradayReturnFactor,
)
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
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting, RiskChecks, DrawdownPolicy
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.data_loader import load_data_for_strategy


def build_strategy() -> Strategy:
    # 1) Data config: same indices parquet
    data = DataConfig(
        source="parquet://equities/indicies.parquet",
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-12-12",
        price_adjustment="split_dividend",
        fields=["open", "close", "volume"],
    )

    # 2) Universe: basic sanity
    universe = Universe(
        name="Indices",
        id_field="ticker",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
        ],
    )

    # 3) Factors: medium-term momentum (6m) + slower (12m) for regime + vol + timezone tilt (ON/DAY)
    mom_126 = ReturnFactor(
        name="mom_126",    # ~6 months (252 trading days / 2)
        field="close",
        lookback=126,
        method="log",
    )
    mom_252 = ReturnFactor(
        name="mom_252",    # ~12 months
        field="close",
        lookback=252,
        method="log",
    )
    vol_20 = VolatilityFactor(
        name="vol_20",
        field="close",
        lookback=20,
        method="realized",
        annualize=True,
    )

    # Timezone-flavoured factors
    on_20 = OvernightReturnFactor(
        name="on_20",
        lookback=20,
        method="log",
    )
    day_20 = IntradayReturnFactor(
        name="day_20",
        lookback=20,
        method="log",
    )

    factors = {
        "mom_126": mom_126,
        "mom_252": mom_252,
        "vol_20": vol_20,
        "on_20": on_20,
        "day_20": day_20,
    }

    # 4) Signals

    # Cross-sectional percentile rank of 6m momentum
    rank_126 = CrossSectionRank(
        factor_name="mom_126",
        mask_name=None,
        method="percentile",
        name="rank_126",
    )

    # Per-name validity
    valid = MaskFromBoolean(
        name="valid",
        expr=NotNull(factor_name="mom_126"),
    )

    # Simple per-name regime filter: only trade indices whose own 12m mom >= 0
    risk_on_name = MaskFromBoolean(
        name="risk_on_name",
        expr=GreaterEqual(left="mom_252", right=0.0),
    )

    # Universe-level regime: "is the average 6m momentum across all indices > 0?"
    # We approximate this via a signal:
    #   - avg_mom_126 = mean across instruments of mom_126
    #   - risk_on_global = avg_mom_126 > 0
    #
    # Current DSL you don't yet have an explicit "cross-sectional aggregate"
    # node, so for now we will *only* use the per-name regime and leave the
    # global regime idea as a possible future extension.
    #
    # Ranks for timezone components
    rank_on = CrossSectionRank(
        factor_name="on_20",
        mask_name=None,
        method="percentile",
        name="rank_on_20",
    )
    rank_day = CrossSectionRank(
        factor_name="day_20",
        mask_name=None,
        method="percentile",
        name="rank_day_20",
    )

    # "Timezone leaders": high overnight, low intraday
    tz_long_candidates = MaskFromBoolean(
        name="tz_long_candidates",
        expr=And(
            left=GreaterEqual(left="rank_on_20", right=0.8),
            right=LessEqual(left="rank_day_20", right=0.2),
        ),
    )

    # Require both trend and timezone tilt for long candidates
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=And(
            left="valid",
            right=And(left="risk_on_name", right="tz_long_candidates"),
        ),
    )

    signals = {
        "rank_126": rank_126,
        "valid": valid,
        "risk_on_name": risk_on_name,
        "rank_on_20": rank_on,
        "rank_day_20": rank_day,
        "tz_long_candidates": tz_long_candidates,
        "long_candidates": long_candidates,
    }

    # 5) Portfolio: long-only, top-3 by 6m momentum, weekly rebalance

    long_book = Book(
        name="long_book",
        selector=TopN(
            factor_name="rank_126",
            n=3,                      # long strongest half of the basket
            mask_name="tz_long_candidates",
            fill_from_unmasked=False,
        ),
        weighting=EqualWeight(),
    )

    # Dummy short book (no shorts)
    short_book = Book(
        name="short_book",
        selector=BottomN(
            factor_name="rank_126",
            n=0,
            mask_name=None,
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="5d",      # rebalance roughly weekly
        rebalance_at="market_close",
        signal_delay_bars=0,
        target_gross_leverage=1.0,     # fully invested when risk-on
        target_net_exposure=1.0,       # long-only
        max_abs_weight_per_name=0.6,
        sector_neutral=None,
        turnover_limit=TurnoverLimit(
            window_bars=5,
            max_fraction=1.0,          # allow a full rotation over a week
        ),
    )

    # 6) Execution & costs: keep small but non-zero
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(
            signal_to_order_delay_bars=0,
            market_latency_ms=0,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=0.25,     # 0.25 bps per side
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
            amount=0.1,        # 0.1 bps per trade
        ),
        borrow=BorrowCost(
            default_annual_rate=0.0,   # long-only indices: no borrow
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
        risk_checks=RiskChecks(
            # Prefer soft scale down de-risking for this toy index trend strategy
            drawdown= DrawdownPolicy(
                mode="soft_scale",
                start=0.10,
                full=1.0,
                curve="linear",
            ),
            max_gross_leverage=2.0,
            max_daily_loss=0.10,
        ),
    )

    strategy = Strategy(
        name="indices_mom_6m_long_only_weekly",
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
    # Export detailed tabular outputs
    # ------------------------------------------------------------------
    result.to_parquet(out_dir)

    invested_days = (result.weights.abs().sum(axis=1) > 1e-6).sum()
    print("Invested days:", invested_days, "out of", len(result.returns))

    import numpy as np

    w = result.weights

    print("Max abs weight in sample:", np.nanmax(np.abs(w.values)))
    print("95th percentile of abs weights:", np.nanpercentile(np.abs(w.values), 95))
    print("Number of weights > 10:", (np.abs(w.values) > 10).sum())
    print("Number of weights > 100:", (np.abs(w.values) > 100).sum())
    print("Any NaN rows:", w.isna().all(axis=1).sum())

    debug_worst_days(result)
    print("Invested days:", (result.weights.abs().sum(axis=1) > 1e-6).sum(),
          "out of", len(result.returns))


def debug_worst_days(result, n=10):
    """Print the worst and best daily returns with dates and equity levels."""
    rets = result.returns
    eq = result.equity

    worst = rets.nsmallest(n)
    best = rets.nlargest(n)

    print("\n=== Worst daily returns ===")
    for dt, r in worst.items():
        print(
            f"{dt.date()}  ret={r: .3%}  equity={eq.loc[dt]:,.2f}"
        )

    print("\n=== Best daily returns ===")
    for dt, r in best.items():
        print(
            f"{dt.date()}  ret={r: .3%}  equity={eq.loc[dt]:,.2f}"
        )


if __name__ == "__main__":
    main()
