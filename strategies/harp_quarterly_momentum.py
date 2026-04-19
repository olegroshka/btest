"""
HARP Quarterly Signal — Risk-Adjusted Momentum Proxy
=====================================================
Source: https://github.com/olegroshka/harp
Paper : "Global Persistence, Local Residual Structure:
         Forecasting Heterogeneous Investment Panels" (Roshka, 2026)

What the HARP paper shows (Table 12)
-------------------------------------
The best portfolio signal is the G1-M2 *disagreement*:
  score = pred_g1 - pred_m2

where:
  G1  = global pooled AR(1) + global PCA+ridge prediction of CapEx/Assets rank
  M2  = same, but the residuals are decomposed BY BLOCK (sector):
        block-specific local PCA+ridge captures sector dynamics the
        global model misses.

Key result: within the tech/health block (25 firms) this disagreement
signal produces Sharpe 1.06 vs equal-weight 0.98, Active IR +0.47.
The full-panel L/S Sharpe depends on signal; the G1-M2 disagreement
consistently beats pure momentum in the paper's backtest.

Translation to btest DSL (no HARP model needed)
------------------------------------------------
The paper's strongest *pure price* signal is 4-quarter momentum (lagged 1Q).
The G1-M2 disagreement is economically equivalent to residual/idiosyncratic
momentum after removing the global factor (sector-neutral momentum).

This strategy uses:
  1. risk_adj_mom = 252d log return / 252d realized vol  (Winsorized ±2σ)
     → captures the conservative-investment premium + momentum + low-vol anomaly
  2. Monthly rebalancing  (original paper: quarterly)
  3. Top-quintile long, bottom-quintile short
  4. Sector-neutral option: set SECTOR_NEUTRAL = True below

If you have run Oleg's pipeline and have results/portfolio/predictions.parquet,
replace the ExternalFactor TODO below to use the actual M2/G1 predictions.

Run
---
cd "c:\\Personal\\Business & Investments\\Python codes\\btest"
uv run python strategies\\harp_quarterly_momentum.py
"""

# ── Tune these ──────────────────────────────────────────────────────────────
MOM_LOOKBACK      = 252    # days ≈ 4 quarters  (HARP uses 4Q cum-return)
VOL_LOOKBACK      = 252    # trailing vol window for risk-adjustment
SKIP_DAYS         = 63    # inner reversal skip ≈ 1 quarter (not subtracted
                           # in DSL but momentum lookback already starts here)
REBAL_FREQ        = "1m"  # monthly — quarterly ("1q") not in DSL yet
N_LONG            = 50    # top-N long (≈ Q5 in HARP's quintile sort)
N_SHORT           = 50    # bottom-N short (≈ Q1)
SECTOR_NEUTRAL    = False  # True → apply SectorNeutral to both books
HAS_HISTORY_DAYS  = 300   # minimum price history
MIN_PRICE         = 5.0
MIN_DOLLAR_ADV    = 5_000_000.0
START             = "2010-01-01"
END               = "2025-01-01"
# ────────────────────────────────────────────────────────────────────────────

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.factors import (
    ReturnFactor,
    VolatilityFactor,
    WinsorizedFactor,
    RatioFactor,
)
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    Quantile,
    MaskFromBoolean,
    And,
    NotNull,
    LessEqual,
    GreaterEqual,
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
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    RiskChecks,
    DrawdownPolicy,
    Reporting,
)
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest


def build_strategy() -> Strategy:

    # ── 1. Data ──────────────────────────────────────────────────────────────
    data = DataConfig(
        source="parquet://equities/sp500_daily",
        calendar="XNYS",
        frequency="1d",
        start=START,
        end=END,
        price_adjustment="split_dividend",
        fields=["open", "high", "low", "close", "volume"],
    )

    # ── 2. Universe ──────────────────────────────────────────────────────────
    universe = Universe(
        name="SP500",
        filters=[
            HasHistory(min_days=HAS_HISTORY_DAYS),
            MinPrice(min_price=MIN_PRICE),
            MinDollarADV(min_dollar_adv=MIN_DOLLAR_ADV),
        ],
    )

    # ── 3. Factors ───────────────────────────────────────────────────────────
    # 3a. 252-day log momentum  (≈ HARP's 4-quarter trailing return)
    #     "skip" the last quarter is approximated by the monthly rebal lag —
    #     for a tighter skip, subtract a 63d ReturnFactor from the rank signal.
    mom = ReturnFactor(
        name="mom_252",
        field="close",
        lookback=MOM_LOOKBACK,
        method="log",
    )

    # 3b. 252-day realised volatility  (annualised)
    vol = VolatilityFactor(
        name="vol_252",
        field="close",
        lookback=VOL_LOOKBACK,
        method="realized",
        annualize=True,
    )

    # 3c. Risk-adjusted momentum = mom / vol   (Sharpe-like per-stock)
    #     Maps both the conservative-investment premium (low vol → high score)
    #     and the momentum anomaly into a single factor.
    #     This is the closest DSL proxy to the HARP G1-M2 disagreement signal.
    mom_adj_raw = RatioFactor(
        name="mom_adj_raw",
        numerator=mom,
        denominator=vol,
    )
    mom_adj = WinsorizedFactor(
        name="mom_adj",          # ← this is the factor key used in signals
        base=mom_adj_raw,
        z=2.0,
    )

    # ── 4. Signals ───────────────────────────────────────────────────────────
    # Cross-sectional rank of risk-adjusted momentum (0 = worst, 1 = best)
    rank = CrossSectionRank(
        factor_name="mom_adj",
        method="percentile",
        name="rank",
    )

    # Validity mask: factor must be non-null for both legs
    valid = MaskFromBoolean(
        name="valid",
        expr=NotNull("mom_adj"),
    )

    # Long candidates: top quintile AND valid
    long_cands = MaskFromBoolean(
        name="long_cands",
        expr=And(
            left="valid",
            right=GreaterEqual(
                left="rank",
                right=Quantile(factor_name="rank", q=0.80),
            ),
        ),
    )

    # Short candidates: bottom quintile AND valid
    short_cands = MaskFromBoolean(
        name="short_cands",
        expr=And(
            left="valid",
            right=LessEqual(
                left="rank",
                right=Quantile(factor_name="rank", q=0.20),
            ),
        ),
    )

    # ── 5. Portfolio ─────────────────────────────────────────────────────────
    sector_neutral = SectorNeutral(sector_field="sector") if SECTOR_NEUTRAL else None

    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank",
                n=N_LONG,
                mask_name="long_cands",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        short_book=Book(
            name="short",
            selector=BottomN(
                factor_name="rank",
                n=N_SHORT,
                mask_name="short_cands",
                fill_from_unmasked=False,
            ),
            weighting=EqualWeight(),
        ),
        rebalance_frequency=REBAL_FREQ,
        rebalance_at="market_close",
        signal_delay_bars=1,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.05,
        sector_neutral=sector_neutral,
        turnover_limit=TurnoverLimit(window_bars=5, max_fraction=0.50),
    )

    # ── 6. Execution ─────────────────────────────────────────────────────────
    execution = Execution(
        order_policy=OrderPolicy(
            default_order_type="MOC",
            time_in_force="DAY",
        ),
        latency=LatencyModel(
            signal_to_order_delay_bars=0,
            market_latency_ms=0,
        ),
        slippage=PowerLawSlippageModel(
            base_bps=2.0,
            k=20.0,
            exponent=0.5,
        ),
        volume_limits=VolumeParticipation(
            max_participation=0.10,
            mode="proportional",
        ),
    )

    # ── 7. Costs ─────────────────────────────────────────────────────────────
    # HARP TC sensitivity: 5bps flattens most signals; use 5bp here.
    costs = Costs(
        commission=Commission(type="bps_notional", amount=5.0),
        borrow=BorrowCost(default_annual_rate=0.0050),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    # ── 8. Backtest config ───────────────────────────────────────────────────
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=10_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=3.0,
            drawdown=DrawdownPolicy(
                mode="soft_scale",
                start=0.12,
                full=0.35,
                curve="linear",
            ),
        ),
        reporting=Reporting(
            output_dir="outputs/harp_quarterly_momentum",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title="HARP Quarterly — Risk-Adj Momentum Proxy"
            ),
        ),
        extra={"hold_when_no_targets": True},
    )

    # ── 9. Assemble ──────────────────────────────────────────────────────────
    return Strategy(
        name="harp_quarterly_momentum",
        data=data,
        universe=universe,
        factors={
            "mom_252":      mom,
            "vol_252":      vol,
            "mom_adj_raw":  mom_adj_raw,
            "mom_adj":      mom_adj,
        },
        signals={
            "rank":        rank,
            "valid":       valid,
            "long_cands":  long_cands,
            "short_cands": short_cands,
        },
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    strategy = build_strategy()
    result = run_backtest(strategy)
    print(result.summary())
