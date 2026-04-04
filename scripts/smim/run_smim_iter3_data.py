#!/usr/bin/env python
"""
SMIM Iteration 3 — Phase 0: Data Construction + Validation.

Builds two daily-frequency panels for Paper 2 (emergence testing):
  Panel A: 155 US-LC stocks, 60-day cross-sectional momentum rank
  Panel B: 28 sector/macro actors (11 ETFs + 8 FRED + 9 GDELT)

Also computes step-interpolated CapEx diagnostic and runs validation gates V-1..V-5.

Usage::
    uv run python scripts/smim/run_smim_iter3_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
FRED_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "fred_signals.parquet"
GDELT_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "gdelt_narrative_daily.parquet"
INTENSITY_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "smim" / "intensities"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Daily FRED series IDs (those with >5000 observations = genuinely daily)
FRED_DAILY_IDS = [
    "BAA10Y", "BAMLH0A0HYM2", "DCOILBRENTEU", "DCOILWTICO",
    "DFF", "DTWEXBGS", "T10Y2Y", "VIXCLS",
]

LOOKBACK_DAYS = [20, 60, 120]
DEFAULT_LOOKBACK = 60


# =========================================================================
# Panel A: Stock-level momentum rank
# =========================================================================


def build_panel_a(lookback: int = DEFAULT_LOOKBACK) -> tuple[pd.DataFrame, list[str]]:
    """Build daily momentum rank panel for US-LC stocks with CapEx data.

    Returns:
        panel_wide: DataFrame indexed by date, columns = ticker, values in [0,1]
        tickers: list of tickers in panel
    """
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  PANEL A: Stock-level daily momentum rank (L={lookback})")
    print(f"{'='*80}")

    # Load OHLCV
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    print(f"  OHLCV: {ohlcv['ticker'].nunique()} tickers, "
          f"{ohlcv['date'].min().date()} to {ohlcv['date'].max().date()}")

    # Identify tickers with CapEx data in EDGAR
    edgar = pd.read_parquet(EDGAR_PATH)
    capex_tickers = set(
        edgar[edgar["tag"].str.contains("Payments", na=False)]["ticker"].unique()
    )
    ohlcv_tickers = set(ohlcv["ticker"].unique())
    overlap = sorted(ohlcv_tickers & capex_tickers)
    print(f"  OHLCV tickers: {len(ohlcv_tickers)}, CapEx tickers: {len(capex_tickers)}, "
          f"overlap: {len(overlap)}")

    # Filter OHLCV to overlap tickers and pivot to wide
    ohlcv_filt = ohlcv[ohlcv["ticker"].isin(overlap)].copy()
    close_wide = ohlcv_filt.pivot_table(index="date", columns="ticker", values="close")
    close_wide = close_wide.sort_index()

    # Filter to 2010+ (need lookback buffer before 2010)
    buffer_start = pd.Timestamp("2009-01-01")
    close_wide = close_wide[close_wide.index >= buffer_start]
    print(f"  Close panel: {close_wide.shape[0]} days x {close_wide.shape[1]} tickers "
          f"({close_wide.index.min().date()} to {close_wide.index.max().date()})")

    # Compute daily log returns
    log_ret = np.log(close_wide / close_wide.shift(1))

    # Compute L-day rolling cumulative return (point-in-time: uses t-L to t-1)
    # shift(1) ensures we use returns UP TO yesterday, not including today
    rolling_ret = log_ret.shift(1).rolling(window=lookback, min_periods=lookback).sum()

    # Cross-sectional percentile rank per day (values in [0, 1])
    # rank method='average' then divide by count of non-NaN
    momentum_rank = rolling_ret.rank(axis=1, method="average", pct=True)

    # Filter to 2010+ (drop buffer period)
    panel_start = pd.Timestamp("2010-01-01")
    momentum_rank = momentum_rank[momentum_rank.index >= panel_start]

    # Drop tickers with >20% missing data
    coverage = momentum_rank.notna().mean()
    good_tickers = coverage[coverage > 0.80].index.tolist()
    momentum_rank = momentum_rank[good_tickers]
    print(f"  After coverage filter (>80%): {len(good_tickers)} tickers")
    print(f"  Panel shape: {momentum_rank.shape[0]} days x {momentum_rank.shape[1]} tickers")
    print(f"  Date range: {momentum_rank.index.min().date()} to {momentum_rank.index.max().date()}")
    print(f"  NaN fraction: {momentum_rank.isna().mean().mean():.4f}")
    print(f"  Value range: [{momentum_rank.min().min():.4f}, {momentum_rank.max().max():.4f}]")
    print(f"  Built in {time.time() - t0:.1f}s")

    return momentum_rank, good_tickers


def panel_a_to_long(panel_wide: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Convert wide panel to long format matching existing intensity schema."""
    long = panel_wide.stack().reset_index()
    long.columns = ["period", "actor_id", "intensity_value"]
    long["normalisation_method"] = f"momentum_{lookback}d_xsrank"
    return long


# =========================================================================
# Panel B: Sector/macro daily panel
# =========================================================================


def build_panel_b() -> pd.DataFrame:
    """Build daily panel for sector ETFs + FRED daily + GDELT daily.

    Returns panel_wide: DataFrame indexed by date, columns = actor_id, values in [0,1].
    """
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  PANEL B: Sector/macro daily panel (28 actors)")
    print(f"{'='*80}")

    # --- FRED daily ---
    fred_raw = pd.read_parquet(FRED_PATH)
    fred_daily = fred_raw[fred_raw["signal_id"].isin(FRED_DAILY_IDS)].copy()
    fred_daily["event_date"] = pd.to_datetime(fred_daily["event_date"])

    # Pivot to wide
    fred_wide = fred_daily.pivot_table(
        index="event_date", columns="signal_id", values="value"
    ).sort_index()

    # Rolling 252-day z-score -> CDF for [0,1] normalisation
    fred_mean = fred_wide.rolling(252, min_periods=60).mean()
    fred_std = fred_wide.rolling(252, min_periods=60).std()
    fred_z = (fred_wide - fred_mean) / fred_std.replace(0, np.nan)
    fred_norm = fred_z.apply(lambda col: stats.norm.cdf(col))
    fred_norm.columns = [f"fred_{c}" for c in fred_norm.columns]
    print(f"  FRED daily: {fred_norm.shape[1]} series, "
          f"{fred_norm.index.min().date()} to {fred_norm.index.max().date()}")

    # --- GDELT daily ---
    gdelt = pd.read_parquet(GDELT_PATH)
    gdelt["event_date"] = pd.to_datetime(gdelt["event_date"])

    # Use 'intensity' column, pivot to wide
    gdelt_wide = gdelt.pivot_table(
        index="event_date", columns="theme_or_actor", values="intensity"
    ).sort_index()

    # Rolling 252-day z-score -> CDF
    gd_mean = gdelt_wide.rolling(252, min_periods=60).mean()
    gd_std = gdelt_wide.rolling(252, min_periods=60).std()
    gd_z = (gdelt_wide - gd_mean) / gd_std.replace(0, np.nan)
    gdelt_norm = gd_z.apply(lambda col: stats.norm.cdf(col))
    gdelt_norm.columns = [f"gdelt_{c}" for c in gdelt_norm.columns]
    print(f"  GDELT daily: {gdelt_norm.shape[1]} actors, "
          f"{gdelt_norm.index.min().date()} to {gdelt_norm.index.max().date()}")

    # --- Sector ETFs (from US-LC OHLCV, compute sector averages) ---
    # Instead of downloading ETFs, compute sector-level momentum from stock data
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Daily sector return (equal-weight average of stock returns per sector)
    ohlcv_sorted = ohlcv.sort_values(["ticker", "date"])
    ohlcv_sorted["ret"] = ohlcv_sorted.groupby("ticker")["close"].pct_change()
    sector_ret = ohlcv_sorted.groupby(["date", "sector"])["ret"].mean().unstack("sector")
    sector_ret = sector_ret.sort_index()

    # 60-day rolling sector return, then cross-sectional rank within sectors
    sector_cum = sector_ret.rolling(60, min_periods=30).sum()
    sector_rank = sector_cum.rank(axis=1, method="average", pct=True)
    sector_rank.columns = [f"sector_{c.lower().replace(' ', '_')}" for c in sector_rank.columns]
    print(f"  Sector daily: {sector_rank.shape[1]} sectors, "
          f"{sector_rank.index.min().date()} to {sector_rank.index.max().date()}")

    # --- Merge all on common date range ---
    # GDELT starts 2015, so Panel B is 2015+
    common_start = pd.Timestamp("2015-03-01")  # allow 252-day warmup from GDELT start
    common_end = pd.Timestamp("2025-12-31")

    panel_b = pd.concat([sector_rank, fred_norm, gdelt_norm], axis=1)
    panel_b = panel_b[(panel_b.index >= common_start) & (panel_b.index <= common_end)]

    # Keep only trading days (from sector_rank index)
    trading_days = sector_rank.index
    panel_b = panel_b[panel_b.index.isin(trading_days)]

    # Forward-fill FRED (weekdays only) and GDELT (may have weekend gaps)
    panel_b = panel_b.ffill(limit=5)

    print(f"  Panel B merged: {panel_b.shape[0]} days x {panel_b.shape[1]} actors")
    print(f"  Date range: {panel_b.index.min().date()} to {panel_b.index.max().date()}")
    print(f"  NaN fraction: {panel_b.isna().mean().mean():.4f}")
    print(f"  Built in {time.time() - t0:.1f}s")

    return panel_b


# =========================================================================
# Step-interpolated CapEx diagnostic
# =========================================================================


def build_capex_step_interpolated(tickers: list[str]) -> pd.DataFrame:
    """Build step-interpolated daily CapEx/Assets rank (diagnostic only)."""
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"  DIAGNOSTIC: Step-interpolated daily CapEx rank")
    print(f"{'='*80}")

    # Load quarterly intensities
    quarterly = pd.read_parquet(INTENSITY_PATH)
    quarterly = quarterly[quarterly["normalisation_method"] == "capex_assets_xsrank"]
    quarterly = quarterly[quarterly["actor_id"].isin(tickers)]

    if quarterly.empty:
        print("  WARNING: No CapEx intensity data found for these tickers")
        return pd.DataFrame()

    # Pivot to wide quarterly panel
    q_wide = quarterly.pivot_table(
        index="period", columns="actor_id", values="intensity_value"
    ).sort_index()

    # Create daily date range and forward-fill (step-interpolate)
    daily_idx = pd.date_range(q_wide.index.min(), "2025-12-31", freq="D")
    daily_panel = q_wide.reindex(daily_idx).ffill()

    # Keep only trading days (roughly: exclude weekends)
    daily_panel = daily_panel[daily_panel.index.dayofweek < 5]
    daily_panel = daily_panel[daily_panel.index >= "2010-01-01"]

    print(f"  Quarterly input: {q_wide.shape[0]} quarters x {q_wide.shape[1]} actors")
    print(f"  Daily output: {daily_panel.shape[0]} days x {daily_panel.shape[1]} actors")
    print(f"  1-day autocorrelation (median): {daily_panel.apply(lambda c: c.autocorr(1)).median():.6f}")
    print(f"  Built in {time.time() - t0:.1f}s")

    return daily_panel


# =========================================================================
# Validation Gates
# =========================================================================


def run_validation_gates(
    panel_a: pd.DataFrame,
    capex_panel: pd.DataFrame,
    lookback: int,
) -> dict:
    """Run V-1 through V-5 validation gates.

    Returns dict with gate results and pass/fail status.
    """
    print(f"\n{'='*80}")
    print(f"  VALIDATION GATES (L={lookback})")
    print(f"{'='*80}")

    results = {}

    # V-1: Per-actor 1-day autocorrelation
    rho_per_actor = panel_a.apply(lambda c: c.dropna().autocorr(1))
    rho_median = rho_per_actor.median()
    rho_mean = rho_per_actor.mean()
    v1_pass = rho_median < 0.995
    results["V1_rho_median"] = rho_median
    results["V1_rho_mean"] = rho_mean
    results["V1_pass"] = v1_pass
    print(f"\n  V-1: Per-actor 1-day autocorrelation")
    print(f"    Median rho: {rho_median:.6f}")
    print(f"    Mean rho:   {rho_mean:.6f}")
    print(f"    Range:      [{rho_per_actor.min():.4f}, {rho_per_actor.max():.4f}]")
    print(f"    Gate (median < 0.995): {'PASS' if v1_pass else 'FAIL'}")

    # V-2: Daily cross-sectional rank turnover
    rank_diff = panel_a.diff(1).abs()
    daily_turnover = (rank_diff > 0.01).mean(axis=1)  # fraction of actors changing per day
    turnover_median = daily_turnover.median()
    v2_pass = turnover_median > 0.50
    results["V2_turnover_median"] = turnover_median
    results["V2_pass"] = v2_pass
    print(f"\n  V-2: Daily cross-sectional rank turnover (frac actors changing by >0.01)")
    print(f"    Median daily turnover: {turnover_median:.4f}")
    print(f"    Range: [{daily_turnover.quantile(0.05):.4f}, {daily_turnover.quantile(0.95):.4f}]")
    print(f"    Gate (median > 0.50): {'PASS' if v2_pass else 'FAIL'}")

    # V-3: Effective degrees of freedom
    # T_eff = T * (1 - rho_median) per 5yr window
    T_per_5yr = 252 * 5  # ~1260 trading days
    T_eff = T_per_5yr * (1 - rho_median)
    v3_pass = T_eff > 100
    results["V3_T_eff"] = T_eff
    results["V3_pass"] = v3_pass
    print(f"\n  V-3: Effective degrees of freedom (per 5yr window)")
    print(f"    T_nominal = {T_per_5yr}")
    print(f"    T_eff = T * (1 - rho) = {T_eff:.1f}")
    print(f"    Gate (T_eff > 100): {'PASS' if v3_pass else 'FAIL'}")

    # V-4: Momentum rank vs CapEx rank correlation
    if capex_panel is not None and not capex_panel.empty:
        common_tickers = sorted(set(panel_a.columns) & set(capex_panel.columns))
        if common_tickers:
            # Sample at quarter-end dates
            quarter_dates = pd.date_range("2010-01-01", "2025-12-31", freq="QS")
            spearman_vals = []
            for qd in quarter_dates:
                # Find nearest trading day
                mask_a = panel_a.index[panel_a.index >= qd]
                mask_c = capex_panel.index[capex_panel.index >= qd]
                if len(mask_a) == 0 or len(mask_c) == 0:
                    continue
                day_a = mask_a[0]
                day_c = mask_c[0]
                mom = panel_a.loc[day_a, common_tickers].dropna()
                cap = capex_panel.loc[day_c, common_tickers].dropna()
                both = pd.concat([mom, cap], axis=1, keys=["mom", "cap"]).dropna()
                if len(both) > 10:
                    rho_s, _ = stats.spearmanr(both["mom"], both["cap"])
                    spearman_vals.append(rho_s)

            if spearman_vals:
                spearman_mean = np.mean(spearman_vals)
                v4_pass = spearman_mean > 0.10
                results["V4_spearman_mean"] = spearman_mean
                results["V4_pass"] = v4_pass
                print(f"\n  V-4: Momentum rank vs CapEx rank cross-sectional Spearman")
                print(f"    Mean Spearman: {spearman_mean:.4f}")
                print(f"    Quarters measured: {len(spearman_vals)}")
                print(f"    Range: [{min(spearman_vals):.4f}, {max(spearman_vals):.4f}]")
                print(f"    Gate (mean > 0.10): {'PASS' if v4_pass else 'FAIL'}")
            else:
                results["V4_pass"] = None
                print(f"\n  V-4: SKIPPED (no overlapping quarter dates)")
        else:
            results["V4_pass"] = None
            print(f"\n  V-4: SKIPPED (no overlapping tickers)")
    else:
        results["V4_pass"] = None
        print(f"\n  V-4: SKIPPED (no CapEx panel)")

    # V-5: Momentum rank predicts next-Q CapEx rank change
    if capex_panel is not None and not capex_panel.empty:
        common_tickers = sorted(set(panel_a.columns) & set(capex_panel.columns))
        if common_tickers:
            quarter_dates = pd.date_range("2010-01-01", "2024-12-31", freq="QS")
            mom_vals = []
            capex_changes = []
            for i, qd in enumerate(quarter_dates[:-1]):
                qd_next = quarter_dates[i + 1]
                mask_a = panel_a.index[panel_a.index >= qd]
                mask_c = capex_panel.index[capex_panel.index >= qd]
                mask_c_next = capex_panel.index[capex_panel.index >= qd_next]
                if len(mask_a) == 0 or len(mask_c) == 0 or len(mask_c_next) == 0:
                    continue
                mom = panel_a.loc[mask_a[0], common_tickers]
                cap_now = capex_panel.loc[mask_c[0], common_tickers]
                cap_next = capex_panel.loc[mask_c_next[0], common_tickers]
                delta_cap = cap_next - cap_now
                both = pd.concat([mom, delta_cap], axis=1, keys=["mom", "dcap"]).dropna()
                if len(both) > 10:
                    mom_vals.extend(both["mom"].tolist())
                    capex_changes.extend(both["dcap"].tolist())

            if mom_vals:
                from scipy.stats import pearsonr
                r_val, p_val = pearsonr(mom_vals, capex_changes)
                # Approximate t-stat
                n = len(mom_vals)
                t_stat = r_val * np.sqrt(n - 2) / np.sqrt(1 - r_val**2) if abs(r_val) < 1 else 0
                v5_pass = abs(t_stat) > 2.0
                results["V5_r"] = r_val
                results["V5_t"] = t_stat
                results["V5_n"] = n
                results["V5_pass"] = v5_pass
                print(f"\n  V-5: Momentum rank predicts next-Q CapEx rank change")
                print(f"    Pearson r: {r_val:.4f}")
                print(f"    t-stat:    {t_stat:.2f}")
                print(f"    N obs:     {n}")
                print(f"    Gate (|t| > 2.0): {'PASS' if v5_pass else 'FAIL'}")
            else:
                results["V5_pass"] = None
                print(f"\n  V-5: SKIPPED (insufficient data)")
        else:
            results["V5_pass"] = None
            print(f"\n  V-5: SKIPPED (no overlapping tickers)")
    else:
        results["V5_pass"] = None
        print(f"\n  V-5: SKIPPED (no CapEx panel)")

    # Summary
    signal_gates = [results.get("V1_pass"), results.get("V2_pass"), results.get("V3_pass")]
    signal_ok = all(g for g in signal_gates if g is not None)
    invest_gates = [results.get("V4_pass"), results.get("V5_pass")]
    invest_ok = any(g for g in invest_gates if g is not None and g)

    print(f"\n  {'='*60}")
    print(f"  SUMMARY (L={lookback}):")
    print(f"    Signal quality gates (V1-V3): {'ALL PASS' if signal_ok else 'FAIL'}")
    print(f"    Investment relevance (V4-V5): {'SUPPORTED' if invest_ok else 'NOT SUPPORTED'}")
    if not signal_ok:
        print(f"    RECOMMENDATION: Try shorter lookback or weekly aggregation")
    if not invest_ok:
        print(f"    NOTE: Paper 2 reframed as 'equity dynamics' (still publishable)")
    print(f"  {'='*60}")

    results["signal_ok"] = signal_ok
    results["invest_ok"] = invest_ok
    return results


# =========================================================================
# Main
# =========================================================================


def main():
    t_total = time.time()
    print("=" * 80)
    print("  SMIM Iteration 3 — Phase 0: Data Construction + Validation")
    print("=" * 80)

    # --- Build Panel A for each lookback ---
    panels_a = {}
    for L in LOOKBACK_DAYS:
        panel, tickers = build_panel_a(lookback=L)
        panels_a[L] = panel

        # Save to parquet (long format)
        long_df = panel_a_to_long(panel, L)
        out_path = OUTPUT_DIR / f"iter3_panel_a_L{L}.parquet"
        long_df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Primary panel
    panel_a = panels_a[DEFAULT_LOOKBACK]
    tickers = panel_a.columns.tolist()

    # --- Build Panel B ---
    panel_b = build_panel_b()
    # Save Panel B
    panel_b_long = panel_b.stack().reset_index()
    panel_b_long.columns = ["period", "actor_id", "intensity_value"]
    panel_b_long["normalisation_method"] = "daily_zscore_cdf"
    out_path_b = OUTPUT_DIR / "iter3_panel_b.parquet"
    panel_b_long.to_parquet(out_path_b, index=False)
    print(f"  Saved: {out_path_b.relative_to(PROJECT_ROOT)}")

    # --- Step-interpolated CapEx diagnostic ---
    capex_panel = build_capex_step_interpolated(tickers)
    if not capex_panel.empty:
        capex_long = capex_panel.stack().reset_index()
        capex_long.columns = ["period", "actor_id", "intensity_value"]
        capex_long["normalisation_method"] = "capex_step_interpolated"
        out_path_c = OUTPUT_DIR / "iter3_capex_step.parquet"
        capex_long.to_parquet(out_path_c, index=False)
        print(f"  Saved: {out_path_c.relative_to(PROJECT_ROOT)}")

    # --- Validation Gates ---
    print("\n\n" + "#" * 80)
    print("#  VALIDATION GATES")
    print("#" * 80)

    all_results = {}
    for L in LOOKBACK_DAYS:
        results = run_validation_gates(panels_a[L], capex_panel, lookback=L)
        all_results[L] = results

    # --- Final recommendation ---
    print("\n\n" + "#" * 80)
    print("#  LOOKBACK COMPARISON")
    print("#" * 80)
    print(f"\n  {'L':>5s}  {'rho_med':>8s}  {'turnover':>9s}  {'T_eff':>7s}  {'V1':>5s}  {'V2':>5s}  {'V3':>5s}  {'invest':>7s}")
    print(f"  {'---':>5s}  {'---':>8s}  {'---':>9s}  {'---':>7s}  {'---':>5s}  {'---':>5s}  {'---':>5s}  {'---':>7s}")
    for L in LOOKBACK_DAYS:
        r = all_results[L]
        print(f"  {L:>5d}  {r['V1_rho_median']:>8.5f}  {r['V2_turnover_median']:>9.4f}  "
              f"{r['V3_T_eff']:>7.1f}  "
              f"{'PASS' if r['V1_pass'] else 'FAIL':>5s}  "
              f"{'PASS' if r['V2_pass'] else 'FAIL':>5s}  "
              f"{'PASS' if r['V3_pass'] else 'FAIL':>5s}  "
              f"{'YES' if r.get('invest_ok') else 'NO':>7s}")

    # Select best lookback
    for L in [DEFAULT_LOOKBACK, 20, 120]:
        if all_results[L]["signal_ok"]:
            print(f"\n  SELECTED LOOKBACK: L={L} (signal gates pass)")
            break
    else:
        print(f"\n  WARNING: No lookback passes all signal gates. Consider weekly aggregation.")

    # Save validation summary
    val_df = pd.DataFrame(all_results).T
    val_df.index.name = "lookback"
    val_path = METRICS_DIR / "iter3_validation_gates.csv"
    val_df.to_csv(val_path)
    print(f"  Saved: {val_path.relative_to(PROJECT_ROOT)}")

    print(f"\n  Total time: {time.time() - t_total:.1f}s")
    print("  DONE")


if __name__ == "__main__":
    main()
