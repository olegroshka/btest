"""
Tiny momentum long/short example used in README.

It builds a simple cross‑sectional momentum strategy on the S&P 500 and runs a
daily event‑driven backtest. Optionally produces a QuantStats HTML tear sheet
and a few PNG charts. Outputs are saved under outputs/tiny_momentum_ls/.
"""

from __future__ import annotations

import os

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    Quantile,
    LessEqual,
    MaskFromBoolean,
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
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest


def build_strategy() -> Strategy:
    # 1) Data & universe
    data = DataConfig(
        source="parquet://equities/sp500_daily",
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-01-01",
    )
    universe = Universe(name="SP500")

    # 2) Factor: 6‑month momentum on close prices
    mom_126 = ReturnFactor(name="mom_126", field="close", lookback=126, method="log")

    # 3) Signals: rank momentum; pick top/bottom 50 names
    rank = CrossSectionRank(factor_name="mom_126", name="rank")
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=LessEqual(left=Quantile(factor_name="rank", q=0.9), right="rank"),
    )
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(left="rank", right=Quantile(factor_name="rank", q=0.1)),
    )

    factors = {"mom_126": mom_126}
    signals = {
        "rank": rank,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    # 4) Portfolio: equal‑weight long/short, daily rebalance
    portfolio = LongShortPortfolio(
        long_book=Book(name="long", selector=TopN(factor_name="rank", n=50), weighting=EqualWeight()),
        short_book=Book(name="short", selector=BottomN(factor_name="rank", n=50), weighting=EqualWeight()),
        rebalance_frequency="1d",
    )

    # 5) Execution & costs (kept simple)
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),  # no strict cap
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=1.0),  # 1bp per trade
        borrow=BorrowCost(),
        financing=FinancingCost(),
        fees=StaticFees(),
    )

    # 6) Backtest runtime
    bt = BacktestConfig(engine="event_driven", cash_initial=1_000_000)

    # 7) Compose
    strategy = Strategy(
        name="tiny_momentum_ls",
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

    # Run
    result = run_backtest(strategy)
    print(result.summary())

    # Outputs directory
    out_dir = os.path.join("outputs", "tiny_momentum_ls")
    os.makedirs(out_dir, exist_ok=True)

    # Optional: QuantStats metrics and HTML tear sheet
    try:
        # metrics
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
        print("\n=== QuantStats metrics (subset) ===")
        print(
            result.quantstats_metrics(qs_metric_names).to_string(
                float_format=lambda x: f"{x:0.4f}"
            )
        )

        # Full HTML report
        html_path = os.path.join(out_dir, "tearsheet.html")
        # If BacktestResult already has a benchmark configured, it will be used by default.
        result.quantstats_tearsheet(
            output=html_path,
            title="Tiny Momentum L/S (QuantDSL)",
        )
        print(f"QuantStats HTML report written to: {html_path}")
    except RuntimeError as e:
        # quantstats is optional
        print(f"QuantStats outputs skipped: {e}")

if __name__ == "__main__":
    main()
