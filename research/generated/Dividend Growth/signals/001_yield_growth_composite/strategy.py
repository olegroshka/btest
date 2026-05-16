"""
strategy.py - LSE_FUNDAMENTAL_DIVGROWTH_S001 (LSE rewrite)
=============================================================
Signal   : 40% trailing-12M yield rank  +  60% YoY dividend growth rank
           Both from eodhd.dividends (LSE). Distinct from S003 which uses EPS growth.
Universe : LSE dividend payers (~1,347 canonical, non-payers have NaN -> excluded)
Portfolio: Long-only Top-30, monthly rebalance
Costs    : 10 bps commission, realistic slippage

Plan ID  : LSE_FUNDAMENTAL_DIVGROWTH_S001

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/001_yield_growth_composite/strategy.py"
"""
from __future__ import annotations

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

# ── Parameters ────────────────────────────────────────────────────────────────
LONG_N        = 30
YIELD_WEIGHT  = 0.4   # weight on yield rank; growth_weight = 1 - yield_weight
START         = "2015-01-01"
END           = "2026-01-01"
SIGNAL_DIR    = "research/generated/Dividend Growth/signals/001_yield_growth_composite"
SHARED_DATA   = "research/generated/Dividend Growth/shared_data"


def build_strategy(
    long_n: int = LONG_N,
    yield_weight: float = YIELD_WEIGHT,
    rebalance: str = "1m",
    start: str = START,
    end: str = END,
    suppress_output: bool = False,       # True → no parquet / html writing
) -> Strategy:
    """Build the strategy. Parameters exposed for sweep.py."""

    # ── 1. Data & universe ────────────────────────────────────────────────────
    data = DataConfig(
        source=f"parquet://{SHARED_DATA}/lse_prices.parquet",
        calendar="XLON",
        frequency="1d",
        start=start,
        end=end,
    )
    universe = Universe(
        name="LSE_div_growth",
        filters=[HasHistory(min_days=252), MinPrice(min_price=100.0)],  # 100 GBX = 1 GBP
    )

    # ── 2. Factors ────────────────────────────────────────────────────────────
    # Wide parquet: index=trading_date (tz-naive), columns=ticker
    # Default composite = 0.4 × yield_rank + 0.6 × growth_rank.
    # When yield_weight != default, use a loader to blend raw rank files on-the-fly
    # so no temp files are needed (avoids Windows temp-path issues in sweeps).
    _yw = round(yield_weight, 6)
    _gw = round(1.0 - _yw, 6)
    if abs(_yw - YIELD_WEIGHT) < 1e-9:
        # Use pre-computed composite parquet directly (fastest path)
        composite_raw = ExternalFactor(
            name="composite_raw",
            path=f"{SIGNAL_DIR}/data/div_composite.parquet",
            per_instrument=True,
        )
    else:
        import pandas as _pd
        _yield_rank_path  = f"{SIGNAL_DIR}/data/yield_rank.parquet"
        _growth_rank_path = f"{SIGNAL_DIR}/data/div_growth_rank.parquet"
        def _composite_loader(_yield_df, _yw=_yw, _gw=_gw, _grp=_growth_rank_path):
            _growth = _pd.read_parquet(_grp)
            return _yw * _yield_df + _gw * _growth
        composite_raw = ExternalFactor(
            name="composite_raw",
            path=_yield_rank_path,
            per_instrument=True,
            loader=_composite_loader,
        )
    composite = WinsorizedFactor(
        name="composite",
        base=composite_raw,
        z=3.0,
    )

    # ── 3. Signals ────────────────────────────────────────────────────────────
    is_payer = MaskFromBoolean(
        name="is_payer",
        expr=NotNull("composite"),
    )
    rank = CrossSectionRank(
        factor_name="composite",
        mask_name="is_payer",
        method="percentile",
        name="rank",
    )

    factors = {"composite_raw": composite_raw, "composite": composite}
    signals = {"is_payer": is_payer, "rank": rank}

    # ── 4. Portfolio: Long-only Top-30 ────────────────────────────────────────
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank",
                n=long_n,
                mask_name="is_payer",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(factor_name="rank", n=0),
            weighting=EqualWeight(),
        ),
        rebalance_frequency=rebalance,
        signal_delay_bars=1,
        target_gross_leverage=1.0,
        target_net_exposure=1.0,
        max_abs_weight_per_name=0.10,
    )

    # ── 5. Execution & costs ──────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC"),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=1.0, k=5.0, exponent=0.5),
        volume_limits=VolumeParticipation(max_participation=0.1),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=10.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(),
        fees=StaticFees(),
    )

    # ── 6. Backtest config ────────────────────────────────────────────────────
    _growth_weight = round(1.0 - yield_weight, 4)
    _out_dir = None if suppress_output else f"{SIGNAL_DIR}/outputs"
    bt = BacktestConfig(
        cash_initial=1_000_000.0,
        reporting=Reporting(
            output_dir=_out_dir,
            store_trades=not suppress_output,
            store_positions=not suppress_output,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=(
                    f"LSE Yield+DivGrowth | {yield_weight:.0%} Yield + {_growth_weight:.0%} Growth"
                    f" | Top{long_n} Long-only"
                ),
            ),
        ),
    )

    return Strategy(
        name="lse_div_growth_s001",
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
