"""
Example: Cross-sectional momentum long/short strategy on S&P 500

This script builds a Strategy DSL object and runs a backtest via the engine.
It assumes all DSL primitives and the run_backtest() function are implemented.
"""

from dataclasses import asdict
import os

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.factors import (
    ReturnFactor,
    VolatilityFactor,
    FiboRetraceFactor,   # included just to show extensibility
)
from quantdsl_backtest.dsl.signals import (
    And,
    NotNull,
    LessEqual,
    CrossSectionRank,
    Quantile,
    MaskFromBoolean,
)
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio,
    Book,
    TopN,
    BottomN,
    EqualWeight,
    SectorNeutral,
    TurnoverLimit,
)
from quantdsl_backtest.dsl.execution import (
    Execution,
    OrderPolicy,
    LatencyModel,
    PowerLawSlippageModel,
    VolumeParticipation,
    LimitOrderBookModel,
)
from quantdsl_backtest.dsl.costs import (
    Costs,
    Commission,
    BorrowCost,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    MarginConfig,
    RiskChecks,
    Reporting,
)

from quantdsl_backtest.engine.backtest_runner import run_backtest


def build_strategy() -> Strategy:
    """
    Construct the DSL Strategy object for a cross-sectional
    6m momentum long/short strategy on the S&P 500.
    """

    # -----------------------
    # 1. Data configuration
    # -----------------------
    data_cfg = DataConfig(
        source="parquet://equities/sp500_daily",  # implement the adapter later
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-01-01",
        price_adjustment="split_dividend",       # fully adjusted OHLCV
        fields=["open", "high", "low", "close", "volume"],
    )

    # -----------------------
    # 2. Universe definition
    # -----------------------
    universe = Universe(
        name="SP500",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=1_000_000.0),   # >= 1M USD ADV
        ],
        id_field="ticker",
    )

    # -----------------------
    # 3. Factor definitions
    # -----------------------
    # Core factor: 6-month momentum
    mom_6m = ReturnFactor(
        name="mom_6m",
        field="close",
        lookback=126,          # ~6m trading days
        method="log",
    )

    # Volatility filter: 20-day realized volatility
    vol_20d = VolatilityFactor(
        name="vol_20d",
        field="close",
        lookback=20,
        method="realized",
    )

    # Optional: Fibonacci retracement just to demonstrate extensibility
    fibo_50d_618 = FiboRetraceFactor(
        name="fibo_50d_618",
        field_high="high",
        field_low="low",
        lookback=50,
        level=0.618,
    )

    factors = {
        "mom_6m": mom_6m,
        "vol_20d": vol_20d,
        "fibo_50d_618": fibo_50d_618,
    }

    # -----------------------
    # 4. Signal definitions
    # -----------------------
    # Eligible universe: has momentum defined AND vol <= 90th vol quantile
    eligible = And(
        left=NotNull("mom_6m"),
        right=LessEqual(
            left="vol_20d",
            right=Quantile(factor_name="vol_20d", q=0.9),
        ),
        name="eligible",
    )

    # Cross-sectional rank of momentum (0–1 percentile rank)
    rank = CrossSectionRank(
        factor_name="mom_6m",
        mask_name="eligible",
        method="percentile",
        name="rank",
    )

    # Boolean masks for top/bottom momentum names
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=LessEqual(
            # we want rank >= 0.9 (top 10%)
            left=Quantile(factor_name="rank", q=0.9),
            right="rank",
        ),
    )

    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(
            # we want rank <= 0.1 (bottom 10%)
            left="rank",
            right=Quantile(factor_name="rank", q=0.1),
        ),
    )

    signals = {
        "eligible": eligible,
        "rank": rank,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    # -----------------------
    # 5. Portfolio construction
    # -----------------------
    long_book = Book(
        name="long_book",
        selector=TopN(
            factor_name="rank",
            n=50,
            mask_name="long_candidates",
        ),
        weighting=EqualWeight(),
    )

    short_book = Book(
        name="short_book",
        selector=BottomN(
            factor_name="rank",
            n=50,
            mask_name="short_candidates",
        ),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=1,             # use yesterday's signals
        long_book=long_book,
        short_book=short_book,
        target_gross_leverage=2.0,       # 200% gross
        target_net_exposure=0.0,         # approximately market neutral
        max_abs_weight_per_name=0.03,    # 3% NAV per leg
        sector_neutral=SectorNeutral(
            sector_field="gics_sector",
            tolerance=0.02,              # +/- 2% sector dev
        ),
        turnover_limit=TurnoverLimit(
            window_bars=1,
            max_fraction=0.30,           # <= 30% of gross per day
        ),
    )

    # -----------------------
    # 6. Execution model
    # -----------------------
    execution = Execution(
        order_policy=OrderPolicy(
            default_order_type="MOC",     # Market-On-Close
            time_in_force="DAY",
        ),
        latency=LatencyModel(
            signal_to_order_delay_bars=1,   # additional delay if you want
            market_latency_ms=50,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=1.0,
            k=20.0,
            exponent=0.5,
            use_intraday_vol=True,
        ),
        volume_limits=VolumeParticipation(
            max_participation=0.10,        # <= 10% ADV
            mode="proportional",           # partial fills
            min_fill_notional=1_000.0,
        ),
        book_model=LimitOrderBookModel(
            levels=3,
            queue_priority="pro_rata",
            use_spread=True,
        ),
    )

    # -----------------------
    # 7. Costs & financing
    # -----------------------
    costs = Costs(
        commission=Commission(
            type="per_share",
            amount=0.005,                 # $0.005 / share
        ),
        borrow=BorrowCost(
            default_annual_rate=0.02,     # 2% annual borrow
            curve_name="stock_loan_rates",
        ),
        financing=FinancingCost(
            base_rate_curve="SOFR",
            spread_bps=50,
        ),
        fees=StaticFees(
            nav_fee_annual=0.002,         # 20 bps management fee
        ),
    )

    # -----------------------
    # 8. Backtest runtime config
    # -----------------------
    backtest_cfg = BacktestConfig(
        engine="vectorized", #"event_driven",           # or "vectorized"
        cash_initial=100_000_000.0,      # 100mm
        margin=MarginConfig(
            long_initial=0.5,
            short_initial=0.5,
            maintenance=0.3,
        ),
        risk_checks=RiskChecks(
            max_drawdown=0.25,           # hard stop
            max_gross_leverage=3.0,
        ),
        reporting=Reporting(
            store_trades=True,
            store_positions=True,
            metrics=[
                "daily_returns",
                "sharpe",
                "sortino",
                "max_drawdown",
                "turnover",
                "gross_exposure",
                "net_exposure",
            ],
        ),
    )

    # -----------------------
    # 9. Compose the Strategy
    # -----------------------
    strategy = Strategy(
        name="xsec_momentum_long_short_sp500",
        data=data_cfg,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest_cfg,
    )

    return strategy


def main():
    strategy = build_strategy()

    # For debugging you can inspect the DSL tree as plain dict
    print("Strategy spec (truncated):")
    # Beware: asdict() will explode on large trees; this is just conceptual
    print(asdict(strategy)["name"], "configured")

    # Run the backtest
    result = run_backtest(strategy)

    # The BacktestResult object is up to you; here’s a likely interface:
    print("\nBacktest summary:")
    print(f"Start date:  {result.start_date}")
    print(f"End date:    {result.end_date}")
    print(f"Total return: {result.total_return:.2%}")
    print(f"Annualized Sharpe: {result.metrics['sharpe']:.2f}")
    print(f"Max drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Turnover (annualized): {result.metrics['turnover_annual']:.2f}")

    # Outputs directory (used for both images and parquet exports)
    out_dir = "outputs/mom_long_short_sp500/"
    os.makedirs(out_dir, exist_ok=True)

    # Generate plots and save them as PNG files in outputs
    try:
        import matplotlib.pyplot as plt

        ax_eq = result.plot_equity_curve()
        ax_exp = result.plot_exposures()
        ax_dd = result.plot_drawdowns()

        plot_files = [
            ("equity_curve.png", ax_eq),
            ("exposures.png", ax_exp),
            ("drawdowns.png", ax_dd),
        ]
        for fname, ax in plot_files:
            fpath = os.path.join(out_dir, fname)
            ax.figure.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"Saved plot: {fpath}")

        # Optionally show plots if the environment variable is set
        if str(os.environ.get("SHOW_PLOTS", "0")).lower() in ("1", "true", "yes"): 
            plt.show()
        else:
            # Close figures when not showing to free resources
            for _, ax in plot_files:
                plt.close(ax.figure)
    except RuntimeError as e:
        # If matplotlib is not installed, continue without plotting
        print(f"Plotting skipped: {e}")

    # Export detailed tabular outputs
    result.to_parquet(out_dir)


if __name__ == "__main__":
    main()
