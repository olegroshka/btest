#!/usr/bin/env python
"""
SMIM Iteration 5: Dividend Signal Enrichment.

Phase 0: Fetch dividend data, build quarterly dividend yield intensity panel.
Phase 1: Run SMIM (DIAMOND config) on dividend intensity.
Phase 2: Combined CapEx + dividend panel.
Phase 3: Cross-signal economic validation.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
INTENSITY_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "smim" / "intensities"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_TARGET = 8
F_REG = 0.99
Q_INIT = 0.5
LAMBDA_Q = 0.3
EWM_HL = 8
TEST_YEARS = list(range(2015, 2025))

warnings.filterwarnings("ignore", category=FutureWarning)


def ewm_mean(obs, halflife=EWM_HL):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(obs_dm, k):
    N = obs_dm.shape[1]
    k_use = min(k, N - 2)
    if obs_dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(k + 5, N))
        return mf.basis[:, :min(k_use, mf.K)].real
    except Exception:
        return None


def sph_r(obs_dm, U):
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * max(np.mean(resid ** 2), 1e-8)


def kf_step(obs, U, F, R, alpha, P, Q):
    N, K = U.shape
    ap = F @ alpha
    Pp = F @ P @ F.T + Q
    pred = U @ ap
    S = U @ Pp @ U.T + R
    try:
        Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        Kg = np.zeros((K, N))
    af = ap + Kg @ (obs - pred)
    Pf = (np.eye(K) - Kg @ U) @ Pp
    si = af - ap
    Qn = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(si, si)
    Qn = (Qn + Qn.T) / 2 + np.eye(K) * 1e-8
    return pred, af, Pf, Qn


def subspace_angle(U1, U2, k=None):
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.degrees(np.mean(np.arccos(np.clip(svals, -1, 1)))))


def run_smim_rolling(panel_wide, test_year, K_use=K_TARGET, train_years=5):
    """DIAMOND-config rolling SMIM on any quarterly panel. Returns (smim_r2, ar1_r2)."""
    ts = pd.Timestamp(f"{test_year - train_years}-01-01")
    te_tr = pd.Timestamp(f"{test_year - 1}-12-31")
    te_te = pd.Timestamp(f"{test_year}-12-31")

    all_d = panel_wide[(panel_wide.index >= ts) & (panel_wide.index <= te_te)].copy()
    cov = all_d.loc[all_d.index <= te_tr].notna().mean()
    valid = cov[cov > 0.50].index
    all_d = all_d[valid].fillna(all_d[valid].mean())
    N = len(valid)
    if N < 10:
        return None, None

    train = all_d[all_d.index <= te_tr].values.astype(np.float64)
    test_dates = all_d.index[all_d.index > te_tr]
    test = all_d.loc[test_dates].values.astype(np.float64)
    if len(train) < 8 or len(test) == 0:
        return None, None

    # AR(1) baseline
    mu_a = train.mean(0)
    dm_a = train - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])

    # SMIM rolling
    mu = ewm_mean(train)
    dm = train - mu
    U = dmd_basis(dm, K_use)
    if U is None:
        return None, None
    K = U.shape[1]
    R_s = sph_r(dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT
    a = np.zeros(K)
    P = np.eye(K)
    Qr = Q.copy()

    preds_s, preds_a, acts = [], [], []
    prev = train[-1]
    exp = train.copy()

    for i in range(len(test)):
        obs = test[i]
        obs_dm = obs - mu.ravel()
        sp, a, P, Qr = kf_step(obs_dm, U, F, R_s, a, P, Qr)
        preds_s.append(sp + mu.ravel())
        preds_a.append(mu_a + rho * (prev - mu_a))
        acts.append(obs)
        prev = obs

        # Rolling basis (expand each quarter)
        exp = np.vstack([exp, obs.reshape(1, -1)])
        mu = ewm_mean(exp)
        dm_new = exp - mu
        U_new = dmd_basis(dm_new, K_use)
        if U_new is not None:
            Kn = U_new.shape[1]
            a = U_new.T @ obs_dm
            P = np.eye(Kn)
            Qr = np.eye(Kn) * Q_INIT
            F = np.eye(Kn) * F_REG
            R_s = sph_r(dm_new, U_new)
            U, K = U_new, Kn

    ps = np.array(preds_s)
    pa = np.array(preds_a)
    ac = np.array(acts)
    return (
        float(oos_r_squared(ps.ravel(), ac.ravel())),
        float(oos_r_squared(pa.ravel(), ac.ravel())),
    )


# =========================================================================
# Phase 0: Data Construction
# =========================================================================


def fetch_dividends():
    """Fetch dividend history for US-LC tickers from Yahoo Finance."""
    import yfinance as yf

    ohlcv = pd.read_parquet(OHLCV_PATH)
    tickers = sorted(ohlcv["ticker"].unique())
    print(f"  Fetching dividends for {len(tickers)} tickers...")

    all_divs = []
    failed = 0
    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            divs = t.dividends
            if divs is not None and len(divs) > 0:
                df = divs.reset_index()
                df.columns = ["date", "dividend"]
                df["ticker"] = ticker
                all_divs.append(df)
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(tickers)} done ({len(all_divs)} with data, {failed} failed)")

    if not all_divs:
        return pd.DataFrame()

    result = pd.concat(all_divs, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    print(f"  Total: {len(result)} dividend records for {result['ticker'].nunique()} tickers")
    return result


def build_dividend_intensity(div_data, ohlcv_path=OHLCV_PATH):
    """Build quarterly dividend yield cross-sectional rank panel."""
    ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Quarter-end prices
    price_wide = ohlcv.pivot_table(index="date", columns="ticker", values="close")
    price_quarterly = price_wide.resample("QS").last()

    # Trailing 12-month dividends per share per quarter
    div_data = div_data.copy()
    div_data["quarter"] = div_data["date"].dt.to_period("Q").dt.to_timestamp()

    # Sum dividends per ticker per quarter
    div_quarterly = div_data.groupby(["ticker", "quarter"])["dividend"].sum().unstack("ticker")
    div_quarterly.index = pd.to_datetime(div_quarterly.index)

    # Trailing 4-quarter sum (12-month dividends)
    div_trailing = div_quarterly.rolling(4, min_periods=1).sum()

    # Align dates
    common_dates = div_trailing.index.intersection(price_quarterly.index)
    common_tickers = sorted(set(div_trailing.columns) & set(price_quarterly.columns))

    div_yield = div_trailing.loc[common_dates, common_tickers] / price_quarterly.loc[common_dates, common_tickers]
    div_yield = div_yield.replace([np.inf, -np.inf], np.nan)

    # Cross-sectional percentile rank per quarter
    div_rank = div_yield.rank(axis=1, method="average", pct=True)

    # Filter: need >= 80% non-NaN coverage
    coverage = div_rank.notna().mean()
    good = coverage[coverage > 0.50].index
    div_rank = div_rank[good]

    # Filter date range
    div_rank = div_rank.loc["2005-01-01":"2025-12-31"]

    print(f"  Dividend yield panel: {div_rank.shape[0]} quarters x {div_rank.shape[1]} tickers")
    print(f"  Date range: {div_rank.index.min().date()} to {div_rank.index.max().date()}")
    print(f"  NaN fraction: {div_rank.isna().mean().mean():.4f}")

    return div_rank


# =========================================================================
# Main
# =========================================================================


def main():
    t_total = time.time()
    print("=" * 80)
    print("  ITERATION 5: Dividend Signal Enrichment")
    print("=" * 80)

    # ================================================================
    # Phase 0: Data
    # ================================================================
    print("\n--- Phase 0: Dividend Data Construction ---")

    div_path = OUTPUT_DIR / "iter5_dividends_raw.parquet"
    if div_path.exists():
        print(f"  Loading cached dividends from {div_path}")
        div_data = pd.read_parquet(div_path)
    else:
        div_data = fetch_dividends()
        if div_data.empty:
            print("  FATAL: No dividend data fetched")
            return
        div_data.to_parquet(div_path, index=False)
        print(f"  Saved raw dividends: {div_path}")

    div_panel = build_dividend_intensity(div_data)
    if div_panel.shape[1] < 20:
        print(f"  FATAL: Only {div_panel.shape[1]} tickers with dividend data")
        return

    # Save
    div_long = div_panel.stack().reset_index()
    div_long.columns = ["period", "actor_id", "intensity_value"]
    div_long["normalisation_method"] = "dividend_yield_xsrank"
    div_long.to_parquet(OUTPUT_DIR / "iter5_dividend_intensity.parquet", index=False)

    # Characterise
    print(f"\n  --- Characterisation ---")
    rho_per_actor = div_panel.apply(lambda c: c.dropna().autocorr(1))
    print(f"  AR(1) rho: median={rho_per_actor.median():.4f}, "
          f"mean={rho_per_actor.mean():.4f}, "
          f"range=[{rho_per_actor.min():.4f}, {rho_per_actor.max():.4f}]")

    # Sector composition
    ohlcv = pd.read_parquet(OHLCV_PATH)
    sector_map = ohlcv.drop_duplicates("ticker").set_index("ticker")["sector"]
    div_sectors = sector_map.reindex(div_panel.columns).dropna()
    print(f"  Sectors: {div_sectors.value_counts().to_dict()}")

    # Correlation with CapEx rank
    capex_int = pd.read_parquet(INTENSITY_PATH)
    capex_capex = capex_int[capex_int["normalisation_method"] == "capex_assets_xsrank"]
    if not capex_capex.empty:
        capex_wide = capex_capex.pivot_table(index="period", columns="actor_id", values="intensity_value")
        capex_wide.index = pd.to_datetime(capex_wide.index)
        common_actors = sorted(set(div_panel.columns) & set(capex_wide.columns))
        common_dates = div_panel.index.intersection(capex_wide.index)
        if common_actors and len(common_dates) > 4:
            div_flat = div_panel.loc[common_dates, common_actors].values.ravel()
            cap_flat = capex_wide.loc[common_dates, common_actors].values.ravel()
            mask = ~(np.isnan(div_flat) | np.isnan(cap_flat))
            if mask.sum() > 100:
                from scipy.stats import spearmanr
                rho_s, _ = spearmanr(div_flat[mask], cap_flat[mask])
                print(f"  Spearman(dividend yield rank, CapEx rank): {rho_s:.4f} "
                      f"(n={mask.sum()})")

    # ================================================================
    # Phase 1: SMIM on Dividend Intensity
    # ================================================================
    print(f"\n--- Phase 1: SMIM on Dividend Yield Rank ---")

    for K_use in [3, 5, 8]:
        smim_r2s, ar1_r2s = [], []
        for ty in TEST_YEARS:
            r2s, r2a = run_smim_rolling(div_panel, ty, K_use=K_use)
            if r2s is not None:
                smim_r2s.append(r2s)
                ar1_r2s.append(r2a)
        if smim_r2s:
            deltas = [s - a for s, a in zip(smim_r2s, ar1_r2s)]
            wins = sum(1 for d in deltas if d > 0)
            print(f"  K={K_use}: SMIM={np.mean(smim_r2s):.4f}  AR1={np.mean(ar1_r2s):.4f}  "
                  f"delta={np.mean(deltas):+.4f}  wins={wins}/{len(deltas)}")

    # Detailed per-window for K=8
    print(f"\n  Per-window (K=8):")
    div_rows = []
    for ty in TEST_YEARS:
        r2s, r2a = run_smim_rolling(div_panel, ty, K_use=8)
        if r2s is not None:
            delta = r2s - r2a
            w = "WIN" if delta > 0 else "LOSS"
            print(f"    W{ty}: SMIM={r2s:.4f}  AR1={r2a:.4f}  delta={delta:+.4f} {w}")
            div_rows.append({"year": ty, "smim_r2": r2s, "ar1_r2": r2a, "panel": "dividend"})

    # Basis rotation for dividends
    print(f"\n  --- Basis Rotation (dividend modes) ---")
    q_dates = div_panel.loc["2011-01-01":"2024-12-31"].index
    rotations = []
    for i in range(1, len(q_dates)):
        ts = q_dates[i] - pd.DateOffset(years=5)
        prev_end = q_dates[i - 1]
        curr_end = q_dates[i]

        d_prev = div_panel.loc[ts:prev_end].dropna(axis=1, how="all")
        d_curr = div_panel.loc[ts:curr_end].dropna(axis=1, how="all")
        common = sorted(set(d_prev.columns) & set(d_curr.columns))
        if len(common) < 15:
            continue
        d_prev = d_prev[common].fillna(d_prev[common].mean()).values.astype(np.float64)
        d_curr = d_curr[common].fillna(d_curr[common].mean()).values.astype(np.float64)

        mu_p = ewm_mean(d_prev)
        mu_c = ewm_mean(d_curr)
        U_p = dmd_basis(d_prev - mu_p, 5)
        U_c = dmd_basis(d_curr - mu_c, 5)
        if U_p is not None and U_c is not None:
            angle = subspace_angle(U_p, U_c, min(U_p.shape[1], U_c.shape[1]))
            rotations.append(angle)

    if rotations:
        print(f"  Mean rotation: {np.mean(rotations):.1f} deg/quarter")
        print(f"  Std: {np.std(rotations):.1f}, Range: [{min(rotations):.1f}, {max(rotations):.1f}]")

    # ================================================================
    # Phase 2: Combined CapEx + Dividend Panel
    # ================================================================
    print(f"\n--- Phase 2: Combined CapEx + Dividend Panel ---")

    # Load CapEx intensity for US-LC firms
    capex_all = pd.read_parquet(INTENSITY_PATH)
    capex_wide_all = capex_all.pivot_table(index="period", columns="actor_id", values="intensity_value")
    capex_wide_all.index = pd.to_datetime(capex_wide_all.index)
    capex_wide_all = capex_wide_all.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))

    # Approach 1: Expand actor set (CapEx actors + dividend-only actors)
    capex_actors = set(capex_wide_all.columns)
    div_actors = set(div_panel.columns)
    both_actors = sorted(capex_actors & div_actors)
    capex_only = sorted(capex_actors - div_actors)
    div_only = sorted(div_actors - capex_actors)

    print(f"  Actors with both CapEx + dividend: {len(both_actors)}")
    print(f"  CapEx-only actors: {len(capex_only)}")
    print(f"  Dividend-only actors: {len(div_only)}")

    # Build combined panel: for actors with both, use CapEx. Add div-only actors.
    combined = capex_wide_all.copy()
    for actor in div_only:
        if actor in div_panel.columns:
            combined[actor] = div_panel[actor]
    combined = combined.dropna(how="all")

    print(f"  Combined panel: {combined.shape[0]} quarters x {combined.shape[1]} actors")

    print(f"\n  SMIM on combined panel (K sweep):")
    for K_use in [5, 8, 10]:
        smim_r2s, ar1_r2s = [], []
        for ty in TEST_YEARS:
            r2s, r2a = run_smim_rolling(combined, ty, K_use=K_use)
            if r2s is not None:
                smim_r2s.append(r2s)
                ar1_r2s.append(r2a)
        if smim_r2s:
            deltas = [s - a for s, a in zip(smim_r2s, ar1_r2s)]
            wins = sum(1 for d in deltas if d > 0)
            print(f"    K={K_use}: SMIM={np.mean(smim_r2s):.4f}  AR1={np.mean(ar1_r2s):.4f}  "
                  f"delta={np.mean(deltas):+.4f}  wins={wins}/{len(deltas)}")

    # Approach 2: Dual-intensity for shared actors (CapEx rank + div yield rank)
    if both_actors:
        dual_panel_cols = {}
        for actor in both_actors:
            if actor in capex_wide_all.columns:
                dual_panel_cols[f"{actor}_capex"] = capex_wide_all[actor]
            if actor in div_panel.columns:
                dual_panel_cols[f"{actor}_div"] = div_panel[actor]
        dual_panel = pd.DataFrame(dual_panel_cols)
        dual_panel = dual_panel.dropna(how="all")

        print(f"\n  Dual-intensity panel: {dual_panel.shape[0]} quarters x {dual_panel.shape[1]} columns "
              f"({len(both_actors)} actors x 2 signals)")

        for K_use in [5, 8, 10]:
            smim_r2s, ar1_r2s = [], []
            for ty in TEST_YEARS:
                r2s, r2a = run_smim_rolling(dual_panel, ty, K_use=K_use)
                if r2s is not None:
                    smim_r2s.append(r2s)
                    ar1_r2s.append(r2a)
            if smim_r2s:
                deltas = [s - a for s, a in zip(smim_r2s, ar1_r2s)]
                wins = sum(1 for d in deltas if d > 0)
                print(f"    Dual K={K_use}: SMIM={np.mean(smim_r2s):.4f}  AR1={np.mean(ar1_r2s):.4f}  "
                      f"delta={np.mean(deltas):+.4f}  wins={wins}/{len(deltas)}")

    # ================================================================
    # Phase 3: Cross-signal validation
    # ================================================================
    print(f"\n--- Phase 3: Cross-Signal Validation ---")

    # Subspace angle between CapEx and dividend bases
    if both_actors:
        for ty in [2020, 2022, 2024]:
            ts = pd.Timestamp(f"{ty - 5}-01-01")
            te = pd.Timestamp(f"{ty - 1}-12-31")

            cap_data = capex_wide_all.loc[ts:te, both_actors].fillna(
                capex_wide_all.loc[ts:te, both_actors].mean()
            ).values.astype(np.float64)
            div_data_w = div_panel.loc[ts:te, both_actors].fillna(
                div_panel.loc[ts:te, both_actors].mean()
            ).values.astype(np.float64)

            if len(cap_data) < 8 or len(div_data_w) < 8:
                continue

            mu_c = ewm_mean(cap_data)
            mu_d = ewm_mean(div_data_w)
            U_cap = dmd_basis(cap_data - mu_c, 5)
            U_div = dmd_basis(div_data_w - mu_d, 5)

            if U_cap is not None and U_div is not None:
                angle = subspace_angle(U_cap, U_div,
                                        min(U_cap.shape[1], U_div.shape[1]))
                print(f"  W{ty}: CapEx vs Dividend basis angle = {angle:.1f} degrees")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*80}")
    print(f"#  ITERATION 5 SUMMARY")
    print(f"{'#'*80}")

    if div_rows:
        dr = pd.DataFrame(div_rows)
        print(f"\n  Dividend-only SMIM (K=8):")
        print(f"    Mean SMIM R2: {dr['smim_r2'].mean():.4f}")
        print(f"    Mean AR1 R2:  {dr['ar1_r2'].mean():.4f}")
        delta = dr['smim_r2'].mean() - dr['ar1_r2'].mean()
        wins = (dr['smim_r2'] > dr['ar1_r2']).sum()
        print(f"    Delta:        {delta:+.4f}")
        print(f"    SMIM wins:    {wins}/{len(dr)}")
        print(f"    BRONZE gate (SMIM > AR1): {'PASS' if delta > 0 else 'FAIL'}")
        dr.to_parquet(METRICS_DIR / "iter5_dividend_results.parquet", index=False)

    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
