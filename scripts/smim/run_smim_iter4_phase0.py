#!/usr/bin/env python
"""
SMIM Iteration 4 — Phase 0: Actor Alignment + Projection Characterisation.

Builds the 45-actor bridge between quarterly and daily, estimates basis,
projects daily signals through quarterly modes, and characterises the result.

Usage::
    uv run python scripts/smim/run_smim_iter4_phase0.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
INTENSITY_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
FRED_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "fred_signals.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

FRED_DAILY_MAP = {
    "shock_brent_crude": "DCOILBRENTEU",
    "shock_wti_crude": "DCOILWTICO",
    "shock_fed_funds": "DFF",
    "shock_vix": "VIXCLS",
    "shock_usd_index": "DTWEXBGS",
    "shock_yield_spread": "T10Y2Y",
    "shock_hy_spread": "BAMLH0A0HYM2",
}

TEST_YEARS = list(range(2015, 2025))
EWM_HL = 8  # quarters


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


def main():
    t_total = time.time()
    print("=" * 80)
    print("  ITERATION 4 — Phase 0: Actor Alignment + Projection")
    print("=" * 80)

    # ================================================================
    # A0-1: Build actor bridge
    # ================================================================
    print("\n--- A0-1: Building actor bridge ---")

    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    ohlcv = pd.read_parquet(OHLCV_PATH)
    daily_tickers = set(ohlcv["ticker"].unique())

    # Firm actors whose actor_id IS the ticker
    firm_bridge = {}
    for a in reg["actors"]:
        if a["actor_type"] in ("large_firm", "bank", "sector_leader"):
            if a["actor_id"] in daily_tickers:
                firm_bridge[a["actor_id"]] = a["actor_id"]  # actor_id = ticker

    # FRED shock actors with daily data
    fred_bridge = {}
    for actor_id, fred_id in FRED_DAILY_MAP.items():
        fred_bridge[actor_id] = fred_id

    print(f"  Firm bridge: {len(firm_bridge)} actors")
    print(f"  FRED bridge: {len(fred_bridge)} actors")
    print(f"  Total bridge: {len(firm_bridge) + len(fred_bridge)} actors")
    all_bridge_q = sorted(firm_bridge.keys()) + sorted(fred_bridge.keys())
    print(f"  Sample firm IDs: {sorted(firm_bridge.keys())[:10]}")

    # ================================================================
    # A0-2: Load quarterly intensity + re-estimate basis for bridged actors
    # ================================================================
    print("\n--- A0-2: Quarterly basis for bridged actors ---")

    q_int = pd.read_parquet(INTENSITY_PATH)
    q_wide = q_int.pivot_table(index="period", columns="actor_id", values="intensity_value")
    q_wide.index = pd.to_datetime(q_wide.index)
    q_wide = q_wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))

    # Filter to bridged actors (firms + FRED shocks)
    available_bridge = [c for c in all_bridge_q if c in q_wide.columns]
    q_bridge = q_wide[available_bridge]
    q_bridge = q_bridge.fillna(q_bridge.mean())
    N_bridge = q_bridge.shape[1]
    n_firms = sum(1 for c in available_bridge if c in firm_bridge)
    n_fred = sum(1 for c in available_bridge if c in fred_bridge)
    print(f"  Quarterly bridge panel: {q_bridge.shape[0]} quarters x {N_bridge} actors "
          f"({n_firms} firms + {n_fred} FRED)")

    # K sweep on a representative training window (2010-2018)
    train_q = q_bridge.loc["2010-01-01":"2018-12-31"].values.astype(np.float64)
    mu_q = ewm_mean(train_q)
    dm_q = train_q - mu_q
    print(f"  Training window: {train_q.shape[0]} quarters x {N_bridge} actors")

    for K in [3, 5, 8]:
        U = dmd_basis(dm_q, K)
        if U is not None:
            print(f"  K={K}: basis shape {U.shape}, "
                  f"explained var = {np.sum(np.var(dm_q @ U @ U.T, axis=0)) / np.sum(np.var(dm_q, axis=0)):.3f}")

    # Use K=5 as default (good balance for N=45)
    K_USE = 5
    U_bridge = dmd_basis(dm_q, K_USE)
    print(f"\n  Selected K={K_USE}, basis shape={U_bridge.shape}")

    # ================================================================
    # A0-3: FRED bridge projection (W = U, exact)
    # ================================================================
    print("\n--- A0-3: FRED bridge projection (clean signal) ---")

    fred_raw = pd.read_parquet(FRED_PATH)
    fred_q_actors = sorted(fred_bridge.keys())
    fred_q_idx = [list(q_bridge.columns).index(a) for a in fred_q_actors if a in q_bridge.columns]

    # Load daily FRED
    fred_daily_series = {}
    for actor_id, fred_id in FRED_DAILY_MAP.items():
        if actor_id not in q_bridge.columns:
            continue
        series = fred_raw[fred_raw["signal_id"] == fred_id].copy()
        series = series.set_index("event_date")["value"].sort_index()
        series.index = pd.to_datetime(series.index)
        fred_daily_series[actor_id] = series

    # Build daily FRED panel aligned to trading days
    ohlcv_dates = pd.DatetimeIndex(pd.to_datetime(ohlcv["date"].unique())).sort_values()
    ohlcv_dates = ohlcv_dates[(ohlcv_dates >= "2010-01-01") & (ohlcv_dates <= "2025-12-31")]

    fred_daily_panel = pd.DataFrame(index=ohlcv_dates)
    for actor_id, series in fred_daily_series.items():
        fred_daily_panel[actor_id] = series.reindex(ohlcv_dates).ffill()

    # Normalise FRED daily same way as quarterly (min-max per actor over training)
    for col in fred_daily_panel.columns:
        q_vals = q_bridge[col].dropna()
        vmin, vmax = q_vals.min(), q_vals.max()
        if vmax > vmin:
            fred_daily_panel[col] = (fred_daily_panel[col] - vmin) / (vmax - vmin)
            fred_daily_panel[col] = fred_daily_panel[col].clip(0, 1)

    fred_daily_panel = fred_daily_panel.dropna()
    print(f"  FRED daily panel: {fred_daily_panel.shape}")

    # Project FRED daily onto quarterly basis (U rows for FRED actors only)
    U_fred = U_bridge[fred_q_idx, :]  # (N_fred, K)
    fred_vals = fred_daily_panel[[a for a in fred_q_actors if a in fred_daily_panel.columns]]
    mu_fred_q = mu_q[0, fred_q_idx]  # quarterly mean for FRED actors

    alpha_fred_daily = (fred_vals.values - mu_fred_q) @ U_fred  # (T_daily, K)
    alpha_fred_df = pd.DataFrame(alpha_fred_daily, index=fred_vals.index,
                                  columns=[f"m{i}" for i in range(K_USE)])

    print(f"  FRED daily alpha shape: {alpha_fred_daily.shape}")
    print(f"  FRED daily alpha autocorrelation (mode 0):")
    for lag in [1, 5, 21, 63, 252]:
        rho = alpha_fred_df["m0"].autocorr(lag)
        print(f"    lag={lag:>3d}: rho={rho:+.4f}")

    # Test: does FRED daily alpha at quarter-dates match quarterly alpha?
    q_test_dates = q_bridge.loc["2015-01-01":"2024-12-31"].index
    alpha_q_bridge = (q_bridge.loc["2015-01-01":"2024-12-31"].values.astype(np.float64) - mu_q) @ U_bridge
    alpha_q_fred = (q_bridge.loc["2015-01-01":"2024-12-31", [a for a in fred_q_actors if a in q_bridge.columns]].values.astype(np.float64) - mu_fred_q) @ U_fred

    fred_corrs = []
    for i, qd in enumerate(q_test_dates):
        near = alpha_fred_df.loc[(alpha_fred_df.index >= qd - pd.Timedelta(days=5)) &
                                  (alpha_fred_df.index <= qd + pd.Timedelta(days=5))]
        if len(near) == 0:
            continue
        daily_at_q = near.iloc[0].values
        q_alpha = alpha_q_fred[i] if i < len(alpha_q_fred) else None
        if q_alpha is None:
            continue
        c = np.corrcoef(daily_at_q, q_alpha)[0, 1]
        if not np.isnan(c):
            fred_corrs.append(c)

    fred_corr_mean = np.mean(fred_corrs) if fred_corrs else 0
    print(f"\n  FRED daily α vs quarterly α correlation: {fred_corr_mean:.4f} (n={len(fred_corrs)})")
    print(f"  FRED GATE (>0.50): {'PASS' if fred_corr_mean > 0.50 else 'FAIL'}")

    # ================================================================
    # A0-4: Equity bridge projection
    # ================================================================
    print("\n--- A0-4: Equity bridge projection ---")

    firm_actors = sorted(firm_bridge.keys())
    firm_q_idx = [list(q_bridge.columns).index(a) for a in firm_actors if a in q_bridge.columns]

    # Load daily equity momentum (L=20)
    daily_mom = pd.read_parquet(PROJECT_ROOT / "data" / "smim" / "intensities" / "iter3_panel_a_L20.parquet")
    dm_wide = daily_mom.pivot_table(index="period", columns="actor_id", values="intensity_value")
    dm_wide.index = pd.to_datetime(dm_wide.index)
    dm_wide = dm_wide.sort_index()

    # Align to bridged firm actors
    common_firms = [a for a in firm_actors if a in dm_wide.columns and a in q_bridge.columns]
    firm_q_idx_common = [list(q_bridge.columns).index(a) for a in common_firms]
    print(f"  Common equity actors: {len(common_firms)}")

    U_equity = U_bridge[firm_q_idx_common, :]
    mu_equity_q = mu_q[0, firm_q_idx_common]
    equity_daily = dm_wide[common_firms].dropna()

    alpha_equity_daily = (equity_daily.values - mu_equity_q) @ U_equity
    alpha_eq_df = pd.DataFrame(alpha_equity_daily, index=equity_daily.index,
                                columns=[f"m{i}" for i in range(K_USE)])

    print(f"  Equity daily alpha shape: {alpha_equity_daily.shape}")
    print(f"  Equity daily alpha autocorrelation (mode 0):")
    for lag in [1, 5, 21, 63]:
        rho = alpha_eq_df["m0"].autocorr(lag)
        print(f"    lag={lag:>3d}: rho={rho:+.4f}")

    # Quarterly tracking for equity
    alpha_q_equity = (q_bridge.loc["2015-01-01":"2024-12-31", common_firms].values.astype(np.float64) - mu_equity_q) @ U_equity

    eq_corrs = []
    for i, qd in enumerate(q_test_dates):
        near = alpha_eq_df.loc[(alpha_eq_df.index >= qd - pd.Timedelta(days=5)) &
                                (alpha_eq_df.index <= qd + pd.Timedelta(days=5))]
        if len(near) == 0 or i >= len(alpha_q_equity):
            continue
        c = np.corrcoef(near.iloc[0].values, alpha_q_equity[i])[0, 1]
        if not np.isnan(c):
            eq_corrs.append(c)

    eq_corr_mean = np.mean(eq_corrs) if eq_corrs else 0
    print(f"\n  Equity daily α vs quarterly α correlation: {eq_corr_mean:.4f} (n={len(eq_corrs)})")
    print(f"  Equity GATE (>0.20): {'PASS' if eq_corr_mean > 0.20 else 'FAIL'}")

    # ================================================================
    # A0-5: Estimate W for equity actors + R_D
    # ================================================================
    print("\n--- A0-5: Estimate W for equity actors ---")

    # W estimation: regress daily equity on interpolated quarterly alpha
    # For each training window, interpolate quarterly alpha to daily, then regress
    # Use 2010-2018 training period
    q_train_alpha = (q_bridge.loc["2010-01-01":"2018-12-31", common_firms].values.astype(np.float64) - mu_equity_q) @ U_equity
    q_train_dates = q_bridge.loc["2010-01-01":"2018-12-31"].index

    # Interpolate quarterly alpha to daily using linear interpolation
    alpha_q_interp = pd.DataFrame(q_train_alpha, index=q_train_dates,
                                   columns=[f"m{i}" for i in range(K_USE)])
    daily_dates = equity_daily.loc["2010-01-01":"2018-12-31"].index
    alpha_q_daily_interp = alpha_q_interp.reindex(alpha_q_interp.index.union(daily_dates)).interpolate(method="time").reindex(daily_dates)

    # Daily equity observations for same period
    eq_train_daily = equity_daily.loc["2010-01-01":"2018-12-31"]

    # Align dates
    common_dates = alpha_q_daily_interp.index.intersection(eq_train_daily.index)
    X_interp = alpha_q_daily_interp.loc[common_dates].values  # (T, K)
    Y_daily = (eq_train_daily.loc[common_dates].values - mu_equity_q)  # (T, N_eq)

    # W_hat: Y = W @ X^T => W = Y^T @ X @ (X^T @ X)^-1
    # Ridge regularisation
    lam = 1.0
    XtX = X_interp.T @ X_interp + lam * np.eye(K_USE)
    W_hat = np.linalg.solve(XtX, X_interp.T @ Y_daily).T  # (N_eq, K)

    # R_D from residuals
    Y_pred = X_interp @ W_hat.T
    resid = Y_daily - Y_pred
    r_d_scale = np.mean(resid ** 2)
    r_q_scale_est = np.mean((dm_q[:, firm_q_idx_common] - dm_q[:, firm_q_idx_common] @ U_equity @ U_equity.T) ** 2)

    print(f"  W_hat shape: {W_hat.shape}")
    print(f"  R_D scale (daily): {r_d_scale:.6f}")
    print(f"  R_Q scale (quarterly): {r_q_scale_est:.6f}")
    print(f"  Noise ratio R_D/R_Q: {r_d_scale / r_q_scale_est:.1f}x")
    print(f"  Daily regression R2: {1 - np.sum(resid**2) / np.sum(Y_daily**2):.4f}")

    # ================================================================
    # A0-6: Intra-quarter dynamics
    # ================================================================
    print("\n--- A0-6: Intra-quarter dynamics ---")

    # Does intra-quarter alpha drift predict next-quarter direction?
    # Use FRED daily alpha (cleaner signal)
    q_dates = q_bridge.loc["2015-01-01":"2024-12-31"].index
    alpha_q_full = (q_bridge.loc["2015-01-01":"2024-12-31"].values.astype(np.float64) - mu_q) @ U_bridge

    drift_alignments = []
    for i in range(len(q_dates) - 1):
        q_start = q_dates[i]
        q_end = q_dates[i + 1]
        mask = (alpha_fred_df.index >= q_start) & (alpha_fred_df.index < q_end)
        chunk = alpha_fred_df.loc[mask]
        if len(chunk) < 20:
            continue
        first_half = chunk.iloc[:len(chunk) // 2].mean().values
        second_half = chunk.iloc[len(chunk) // 2:].mean().values
        drift = second_half - first_half
        target = alpha_q_full[i + 1] - alpha_q_full[i]
        norm_d = np.linalg.norm(drift)
        norm_t = np.linalg.norm(target)
        if norm_d > 1e-10 and norm_t > 1e-10:
            cos_angle = np.dot(drift, target) / (norm_d * norm_t)
            drift_alignments.append(cos_angle)

    print(f"  FRED intra-Q drift alignment with Q-to-Q change:")
    print(f"    N quarters: {len(drift_alignments)}")
    print(f"    Mean cosine: {np.mean(drift_alignments):.4f}")
    print(f"    Fraction positive: {np.mean(np.array(drift_alignments) > 0):.2f}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*80}")
    print(f"#  PHASE 0 SUMMARY")
    print(f"{'#'*80}")
    print(f"  Bridged actors:      {len(common_firms)} firms + {len(fred_daily_series)} FRED = {len(common_firms) + len(fred_daily_series)}")
    print(f"  Basis:               K={K_USE} from {N_bridge} bridged quarterly actors")
    print(f"  FRED gate (>0.50):   {fred_corr_mean:.4f} {'PASS' if fred_corr_mean > 0.50 else 'FAIL'}")
    print(f"  Equity gate (>0.20): {eq_corr_mean:.4f} {'PASS' if eq_corr_mean > 0.20 else 'FAIL'}")
    print(f"  Noise ratio R_D/R_Q: {r_d_scale / r_q_scale_est:.1f}x")
    print(f"  Intra-Q drift align: {np.mean(drift_alignments):.4f} (frac+: {np.mean(np.array(drift_alignments) > 0):.2f})")

    # Save bridge data for Phase 1
    bridge_data = {
        "common_firms": common_firms,
        "fred_actors": list(fred_daily_series.keys()),
        "K": K_USE,
        "fred_corr": fred_corr_mean,
        "equity_corr": eq_corr_mean,
        "r_d_scale": r_d_scale,
        "r_q_scale": r_q_scale_est,
    }
    pd.DataFrame([bridge_data]).to_parquet(METRICS_DIR / "iter4_phase0_summary.parquet", index=False)

    # Save key arrays for Phase 1
    np.savez(
        METRICS_DIR / "iter4_phase0_arrays.npz",
        U_bridge=U_bridge,
        U_equity=U_equity,
        U_fred=U_fred,
        W_hat=W_hat,
        mu_q=mu_q,
        mu_equity_q=mu_equity_q,
        mu_fred_q=mu_fred_q,
        r_d_scale=np.array([r_d_scale]),
        r_q_scale=np.array([r_q_scale_est]),
    )
    print(f"\n  Saved phase 0 arrays to results/metrics/iter4_phase0_arrays.npz")
    print(f"  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
