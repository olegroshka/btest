"""
TKAN v4 × CAC 40 Momentum — Market Timing Strategy
====================================================
Entry : TKAN max predicted ratio ≥ 1.5% above anchor
        AND 139-day log momentum ≥ θ  (regime filter)
Exit  : either condition turns off
Exec  : signal at close[T] → position opens close[T+1]
        (next-open in practice — close-to-close approximation)

Variants
--------
  1x   — long 1× plain CACT TR
  2x   — long 2× CACT, ESTER financing deducted on borrowed half

Run
---
    cd btest && uv run python strategies/tkan_v4_momentum_timing.py

ESTER note
----------
The ECB €STR / ESTER rate is approximated as a flat 2.5% p.a. (ESTER_ANNUAL).
This is conservative for 2015–2021 (rate was ~0%) and slightly low for 2023–2024
(rate was ~3.5–4%). Adjust ESTER_ANNUAL to test sensitivity.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────
_HERE     = pathlib.Path(__file__).parent.resolve()
_BTEST    = _HERE.parent                            # .../btest/
_ROOT     = _BTEST.parent                           # .../Python codes/
_SRC      = _BTEST / "src"
_SFERA_DB = _ROOT / "sfera-db"
_SIGNUM   = _ROOT / "signum"

for _p in [str(_SRC), str(_SFERA_DB), str(_SIGNUM)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sfera_db
from quantdsl_backtest.dsl.factors import ExternalFactor
from quantdsl_backtest.dsl.signals import GreaterEqual, And, MaskFromBoolean
from quantdsl_backtest.dsl.portfolio import TimingPortfolio
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel,
    PowerLawSlippageModel, VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.runners.single_asset import SingleAssetRunner

# ── Strategy parameters ────────────────────────────────────────────────────
ENTRY_THRESHOLD = 1.015    # TKAN: max(d1..d10 predicted ratio) ≥ 1.5%
MOM_WINDOW      = 139      # momentum lookback window (locked from sweep)
MOM_THETA       = 0.08     # momentum threshold (log-return) — tune here
ESTER_ANNUAL    = 0.025    # €STR financing cost approximation (2.5% p.a.)
ROLL_WINDOW     = 126      # rolling Sharpe window (~6 months)

# ── File paths ─────────────────────────────────────────────────────────────
WEIGHTS_DIR = (_BTEST / "research" / "Index Directional" / "signals"
               / "tkan" / "versions" / "v4" / "weights")
MOM_FILE    = _ROOT / "sfera" / "data" / "signals" / "cact_momentum.parquet"


# ── DSL factor loaders ─────────────────────────────────────────────────────

def _pred_loader(df: pd.DataFrame) -> pd.Series:
    """Reduce pred_cache (DataFrame d1..d10) to max ratio per day."""
    cols = sorted(
        [c for c in df.columns if c.startswith("d") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    return df[cols].max(axis=1)


def _mom_loader(df: pd.DataFrame) -> pd.Series:
    """Extract pre-computed 139d log momentum from cact_momentum.parquet."""
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"])).set_index("date")
    return df.sort_index()["momentum"]


# ── Build Strategy object ──────────────────────────────────────────────────

def build_strategy(theta: float = MOM_THETA) -> Strategy:
    factors = {
        "tkan_max": ExternalFactor(
            name="tkan_max",
            path=str(WEIGHTS_DIR / "pred_cache.pkl"),
            loader=_pred_loader,
        ),
        "mom_logret": ExternalFactor(
            name="mom_logret",
            path=str(MOM_FILE),
            loader=_mom_loader,
        ),
    }

    # Signals must be in dependency order (earlier ones resolve first)
    signals = {
        "tkan_fire": MaskFromBoolean(
            name="tkan_fire",
            expr=GreaterEqual(left="tkan_max", right=ENTRY_THRESHOLD),
        ),
        "mom_on": MaskFromBoolean(
            name="mom_on",
            expr=GreaterEqual(left="mom_logret", right=theta),
        ),
        "entry_signal": MaskFromBoolean(
            name="entry_signal",
            expr=And(left="tkan_fire", right="mom_on"),
        ),
    }

    portfolio = TimingPortfolio(
        signal_name="entry_signal",
        instrument="CACT",
        rebalance_frequency="1d",
        signal_delay_bars=1,    # signal at close[T] → position opens close[T+1]
        target_leverage=1.0,    # 1× base; 2× computed manually below
    )

    execution = Execution(
        order_policy=OrderPolicy(
            default_order_type="MKT",    # next bar fill
            time_in_force="DAY",
        ),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=0.5, k=0.0),   # 0.5bp for liquid index
        volume_limits=VolumeParticipation(max_participation=1.0),
    )

    costs = Costs(
        commission=Commission(type="bps_notional", amount=0.1),   # 0.1bp, ETF-like
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="ESTER", spread_bps=0.0),
        fees=StaticFees(),
    )

    return Strategy(
        name=f"tkan_v4_mom_timing_theta{theta:.3f}",
        data=DataConfig(
            source="sfera://bbgidx/index_total_return",
            calendar="XPAR",
            frequency="1d",
            start="2015-01-01",
            end="2026-12-31",
        ),
        universe=Universe(name="CACT", static_instruments=["CACT"]),
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=BacktestConfig(
            engine="event_driven",
            cash_initial=1_000_000.0,
            reporting=Reporting(output_dir=str(_BTEST / "outputs" / "tkan_v4_mom_timing")),
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _equity(ret_s: pd.Series, start: float = 100.0) -> pd.Series:
    return (np.exp(ret_s.cumsum()) * start).rename("equity")


def _rolling_sharpe(ret_s: pd.Series, window: int = ROLL_WINDOW) -> pd.Series:
    mu  = ret_s.rolling(window, min_periods=window // 2).mean() * 252
    vol = ret_s.rolling(window, min_periods=window // 2).std()  * np.sqrt(252)
    return mu / vol.where(vol > 1e-9)


def _print_stats(ret_s: pd.Series, label: str, ann: int = 252) -> None:
    cagr = ret_s.mean() * ann
    vol  = ret_s.std()  * np.sqrt(ann)
    sr   = cagr / vol if vol > 0 else 0.0
    eq   = _equity(ret_s)
    mdd  = ((eq - eq.cummax()) / eq.cummax()).min()
    rs   = _rolling_sharpe(ret_s).dropna()
    print(
        f"  {label:40s}  CAGR={cagr*100:+5.1f}%  Vol={vol*100:4.1f}%  "
        f"Sharpe={sr:.2f}  MaxDD={mdd*100:.1f}%  "
        f"RollingSR_med={rs.median():+.2f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("TKAN v4 × CAC 40 Momentum — Market Timing Backtest")
    print("=" * 70)

    # 1. Load CACT price series from sfera_db
    cact_close = (
        sfera_db.query(
            "SELECT trade_date AS date, close_price AS close "
            "FROM bbgidx.index_total_return WHERE ticker='CACT' ORDER BY trade_date"
        )
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")["close"]
    )
    log_ret = np.log(cact_close / cact_close.shift(1)).fillna(0)

    # 2. Build strategy and run via SingleAssetRunner
    strategy = build_strategy(theta=MOM_THETA)
    runner   = SingleAssetRunner(strategy, btest_root=_BTEST)

    result   = runner.run(price_close=cact_close)

    position  = result["position"]     # 0/1 series
    strat_ret = result["strat_ret"]    # 1× net of commission + slippage

    # Align to dates where both price and signals exist
    common = strat_ret.index[strat_ret.notna()]
    strat_ret = strat_ret.loc[common]
    log_ret_c = log_ret.reindex(common).fillna(0)
    position_c = position.reindex(common).fillna(0)

    print(f"\nOOS period : {common[0].date()} → {common[-1].date()}  ({len(common):,}d)")
    print(f"In-position: {position_c.sum():,} / {len(position_c):,}  ({position_c.mean():.1%})\n")

    # 3. Compute 2× variant with ESTER financing drag
    # When in position: gross return = 2× daily log-return
    # Financing drag   = ESTER_ANNUAL / 252 per day on the borrowed fraction (1× NAV)
    ester_daily  = ESTER_ANNUAL / 252.0
    strat_ret_2x = (
        position_c * 2.0 * log_ret_c        # 2× gross
        - position_c * ester_daily           # financing on borrowed half
        - result["cost_ret"].reindex(common).fillna(0)  # same commission / slippage
    )

    bh_ret = log_ret_c   # plain buy & hold

    # 4. Performance comparison
    print("Performance:")
    _print_stats(strat_ret,    f"TKAN+Mom 1×  (θ={MOM_THETA})")
    _print_stats(strat_ret_2x, f"TKAN+Mom 2×  ESTER~{ESTER_ANNUAL:.1%}")
    _print_stats(bh_ret,       "Buy & Hold CACT")

    # 5. Save outputs
    out_dir = _BTEST / "outputs" / "tkan_v4_mom_timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame({
        "tkan_1x": _equity(strat_ret),
        "tkan_2x": _equity(strat_ret_2x),
        "bh":      _equity(bh_ret),
    })
    equity_df.to_parquet(out_dir / "equity.parquet")

    rs_df = pd.DataFrame({
        "tkan_1x": _rolling_sharpe(strat_ret),
        "tkan_2x": _rolling_sharpe(strat_ret_2x),
        "bh":      _rolling_sharpe(bh_ret),
    })
    rs_df.to_parquet(out_dir / "rolling_sharpe.parquet")

    signals_df = pd.DataFrame(result["signals"]).reindex(common)
    signals_df.to_parquet(out_dir / "signals.parquet")

    print(f"\nOutputs saved → {out_dir}")

    # 6. Signum tearsheet (price + equity + rolling Sharpe)
    try:
        from signum import Chart, Dashboard

        def _td(s, name="value"):
            return pd.DataFrame({"time": s.index.strftime("%Y-%m-%d"), name: s.values})

        # Pane 0: price with position shading
        in_pos_df = pd.DataFrame({
            "time": common.strftime("%Y-%m-%d"),
            "position": position_c.astype(int).values,
        })
        p0 = (
            Chart(theme="midnight", height=360, watermark="CACT TR")
            .line(_td(cact_close.reindex(common), "close"), name="CACT TR", color="#5b9cf6", width=1)
            .shade(in_pos_df, position_col="position", color="#00e676", opacity=0.20)
        )

        # Pane 1: equity curves
        p1 = (
            Chart(theme="midnight", height=220)
            .line(_td(_equity(strat_ret)),    name="TKAN+Mom 1×",            color="#00e676", width=2)
            .line(_td(_equity(strat_ret_2x)), name=f"TKAN+Mom 2× (ESTER {ESTER_ANNUAL:.1%})", color="#ff9800", width=2)
            .line(_td(_equity(bh_ret)),       name="B&H CACT",               color="#5b9cf6", width=1)
        )

        # Pane 2: rolling Sharpe
        p2 = (
            Chart(theme="midnight", height=200, watermark=f"{ROLL_WINDOW}d Rolling Sharpe")
            .line(_td(_rolling_sharpe(strat_ret)),    name="TKAN+Mom 1×",            color="#00e676", width=2)
            .line(_td(_rolling_sharpe(strat_ret_2x)), name=f"TKAN+Mom 2×", color="#ff9800", width=1)
            .line(_td(_rolling_sharpe(bh_ret)),       name="B&H CACT",               color="#5b9cf6", width=1)
            .price_line(0.0, title="",      color="rgba(255,255,255,0.15)", line_style=1)
            .price_line(1.0, title="SR=1",  color="rgba(0,230,118,0.25)",   line_style=2)
        )

        dash = Dashboard(
            panes=[p0, p1, p2],
            titles=[
                "CACT TR  |  green = in position (TKAN+Mom combined)",
                "Equity (base=100)  |  1× vs 2× vs B&H",
                f"{ROLL_WINDOW}d Rolling Sharpe  |  horizontal dashes = 0 and SR=1",
            ],
            theme="midnight",
        )

        html_path = out_dir / "tearsheet.html"
        dash.save(str(html_path))
        print(f"Tearsheet saved → {html_path}")

    except Exception as exc:
        print(f"Signum chart skipped: {exc}")


if __name__ == "__main__":
    main()
