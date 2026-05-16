"""
strategies/s001_vol_control.py  –  S001 + Cross-Sectional Vol Control
======================================================================
Base signal : 50% yield rank + 50% growth rank  (best params from sweep)
Portfolio   : Top-10 long-only, monthly rebalance
Vol control : cross-sectional avg 21d-realized-vol  → rolling 252d z-score
              when vol_z > VOL_Z_THRESHOLD (default 1.5) → regime_ok = False
              → no longs selected → portfolio exits to cash at next rebalance
Safety net  : DrawdownPolicy(soft_scale, start=10%, full=35%) — reactive layer

Baseline without vol control (yw=0.5, n=10): Sharpe=1.201, CAGR=14.7%, MaxDD=-28.7%

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/001_yield_growth_composite/strategy_vol_control.py"
"""
from __future__ import annotations

from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
from quantdsl_backtest.dsl.factors import ExternalFactor, WinsorizedFactor, VolatilityFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    CrossSectionAggregate,
    MaskFromBoolean,
    NotNull,
    And,
    Less,
    ZScoreRolling,
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
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    RiskChecks,
    DrawdownPolicy,
    Reporting,
)
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

# ── Parameters ────────────────────────────────────────────────────────────────
LONG_N           = 10
YIELD_WEIGHT     = 0.5     # best from sweep: equal yield+growth weighting
PRECOMP_YW       = 0.4     # yield weight of the pre-computed div_composite.parquet
VOL_Z_THRESHOLD  = 2.5     # go flat when cross-sect avg vol z-score exceeds this
START            = "2015-01-01"
END              = "2026-01-01"
SIGNAL_DIR       = "research/generated/Dividend Growth/signals/001_yield_growth_composite"
SHARED_DATA      = "research/generated/Dividend Growth/shared_data"


def build_strategy(
    long_n: int = LONG_N,
    yield_weight: float = YIELD_WEIGHT,
    vol_z_threshold: float = VOL_Z_THRESHOLD,
    rebalance: str = "1m",
    start: str = START,
    end: str = END,
    suppress_output: bool = False,
) -> Strategy:

    # ── 1. Data & universe ────────────────────────────────────────────────────
    data = DataConfig(
        source=f"parquet://{SHARED_DATA}/lse_prices.parquet",
        calendar="XLON",
        frequency="1d",
        start=start,
        end=end,
    )

    # Load the pre-built equity allowlist (Common Stock only, no ETFs).
    # Generated once by sfera/_tmp_save_equity_list.py from eodhd.exchange_tickers.
    # Excludes leveraged ETFs (GraniteShares 3x/WisdomTree 5x etc.) whose bad data
    # would pollute the cross-sectional vol regime signal even with median.
    _equity_ticker_path = f"{SHARED_DATA}/lse_equity_tickers.txt"
    with open(_equity_ticker_path) as _f:
        _equity_tickers = [line.strip() for line in _f if line.strip()]

    universe = Universe(
        name="LSE_div_growth",
        filters=[HasHistory(min_days=252), MinPrice(min_price=100.0)],
        static_instruments=_equity_tickers,
    )

    # ── 2. Factors ────────────────────────────────────────────────────────────
    _yw = round(yield_weight, 6)
    _gw = round(1.0 - _yw, 6)
    if abs(_yw - PRECOMP_YW) < 1e-9:
        # Fast path: pre-computed composite parquet (0.4 yield / 0.6 growth)
        composite_raw = ExternalFactor(
            name="composite_raw",
            path=f"{SIGNAL_DIR}/data/div_composite.parquet",
            per_instrument=True,
        )
    else:
        import pandas as _pd
        _yp = f"{SIGNAL_DIR}/data/yield_rank.parquet"
        _gp = f"{SIGNAL_DIR}/data/div_growth_rank.parquet"
        def _loader(_ydf, _yw=_yw, _gw=_gw, _gp=_gp):
            return _yw * _ydf + _gw * _pd.read_parquet(_gp)
        composite_raw = ExternalFactor(
            name="composite_raw",
            path=_yp,
            per_instrument=True,
            loader=_loader,
        )
    composite = WinsorizedFactor(name="composite", base=composite_raw, z=3.0)

    # 21-day realized vol per instrument — cross-sectional MEDIAN = robust market vol proxy
    # (mean is polluted by leveraged ETFs / bad data; median of 1000+ tickers is immune)
    vol_21 = VolatilityFactor(name="vol_21", lookback=21, annualize=True)

    # ── 3. Signals ────────────────────────────────────────────────────────────
    is_payer = MaskFromBoolean(name="is_payer", expr=NotNull("composite"))
    rank = CrossSectionRank(
        factor_name="composite",
        mask_name="is_payer",
        method="percentile",
        name="rank",
    )

    # Cross-sectional MEDIAN vol (scalar per date) → rolling z-score
    # Using median not mean: a single bad-data ticker (e.g. 3SVP: 0.1→2335 GBX in one day)
    # would drag the mean vol to 8456% and corrupt the z-score window for 252 days.
    mkt_vol   = CrossSectionAggregate(source="vol_21", op="median", name="mkt_vol")
    mkt_vol_z = ZScoreRolling(base="mkt_vol", window=252, min_periods=63, name="mkt_vol_z")

    # Regime: risk-on when vol is NOT elevated (z < threshold)
    # When False for all instruments → TopN selects 0 → portfolio holds cash
    regime_ok = MaskFromBoolean(
        name="regime_ok",
        expr=Less(left="mkt_vol_z", right=vol_z_threshold),
    )

    # Entry gate: dividend payer AND low-vol regime (regime_ok broadcasts across instruments)
    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=And(left="is_payer", right="regime_ok"),
    )

    factors = {
        "composite_raw": composite_raw,
        "composite": composite,
        "vol_21": vol_21,
    }
    signals = {
        "is_payer": is_payer,
        "rank": rank,
        "mkt_vol": mkt_vol,
        "mkt_vol_z": mkt_vol_z,
        "regime_ok": regime_ok,
        "long_candidates": long_candidates,
    }

    # ── 4. Portfolio ──────────────────────────────────────────────────────────
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="rank",
                n=long_n,
                mask_name="long_candidates",   # regime-gated (was "is_payer")
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
    _gw_d = round(1.0 - yield_weight, 4)
    _out_dir = None if suppress_output else f"{SIGNAL_DIR}/outputs/vol_control"
    bt = BacktestConfig(
        cash_initial=1_000_000.0,
        risk_checks=RiskChecks(
            drawdown=DrawdownPolicy(
                mode="soft_scale",
                start=0.10,
                full=0.35,
                curve="linear",
            ),
        ),
        reporting=Reporting(
            output_dir=_out_dir,
            store_trades=not suppress_output,
            store_positions=not suppress_output,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=(
                    f"S001+VolCtrl | {yield_weight:.0%}Y+{_gw_d:.0%}G"
                    f" | Top{long_n} | VZ<{vol_z_threshold}"
                ),
            ),
        ),
    )

    return Strategy(
        name="lse_div_growth_s001_vc",
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
