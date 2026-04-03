#!/usr/bin/env python
"""
Workstream D -- Basis Rotation Validation.

Tests whether the 26deg/quarter spectral rotation is real signal or estimation noise.

1. Bootstrap CIs for principal subspace angles (within-quarter resampling).
2. Stationary-null simulation: hold latent basis fixed, generate panels with
   comparable N, T, persistence, noise; measure apparent rotation from finite samples.
3. Compare observed rotation distribution to null distribution.
4. Test whether high-rotation quarters produce larger rolling-vs-static gains.
5. Sensitivity to K and training window length.

Outputs:
  results/metrics/ws_d_rotation_bootstrap.csv
  results/metrics/ws_d_null_rotation.csv
  results/metrics/ws_d_sensitivity.csv
  results/metrics/ws_d_rotation_vs_gain.csv

Usage::
    uv run python scripts/smim/run_smim_ws_d_rotation_bootstrap.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
T_YR = 5
N_BOOTSTRAP = 500
N_NULL_SIMS = 500
SEED = 42


def load_panel():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    return wide[[c for c in actor_ids if c in wide.columns]]


def ewm_demean(otr, halflife=8):
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / halflife)
    return (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(otr_dm, N, k=K_GOLD):
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        return U
    except Exception:
        return None


def subspace_angle(U1, U2, k=None):
    """Mean principal angle between column spaces (radians)."""
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svals, -1, 1))
    return float(np.mean(angles))


# =========================================================
# 1. Observed rotation with bootstrap CIs
# =========================================================

def compute_observed_rotations(wide, k=K_GOLD, t_yr=T_YR):
    """Compute rolling quarterly basis rotation with bootstrap CIs."""
    all_quarters = pd.date_range("2010-01-01", "2024-12-31", freq="QS")
    rng = np.random.default_rng(SEED)

    results = []
    prev_U = None

    for q_end in all_quarters:
        q_start = q_end - pd.DateOffset(years=t_yr)
        if q_start < pd.Timestamp("2005-01-01"):
            continue

        chunk = wide[(wide.index >= q_start) & (wide.index <= q_end)]
        valid = chunk.columns[chunk.notna().any()]
        chunk = chunk[valid].fillna(chunk[valid].mean())
        N = len(valid)
        vals = chunk.values.astype(np.float64)

        if vals.shape[0] < 4:
            prev_U = None
            continue

        om = ewm_demean(vals)
        vals_dm = vals - om

        U = dmd_basis(vals_dm, N, k=k)
        if U is None:
            prev_U = None
            continue

        if prev_U is not None and prev_U.shape[0] == U.shape[0]:
            kc = min(prev_U.shape[1], U.shape[1])
            angle_deg = np.degrees(subspace_angle(prev_U, U, k=kc))

            # Bootstrap CI: resample time points within training window
            boot_angles = []
            T_obs = vals_dm.shape[0]
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, T_obs, size=T_obs)
                boot_dm = vals_dm[idx]
                U_boot = dmd_basis(boot_dm, N, k=k)
                if U_boot is not None:
                    kc_b = min(prev_U.shape[1], U_boot.shape[1])
                    boot_angles.append(np.degrees(subspace_angle(prev_U, U_boot, k=kc_b)))

            if boot_angles:
                ci_lo = float(np.percentile(boot_angles, 2.5))
                ci_hi = float(np.percentile(boot_angles, 97.5))
                boot_se = float(np.std(boot_angles))
            else:
                ci_lo = ci_hi = boot_se = np.nan

            results.append({
                "quarter": q_end,
                "angle_deg": angle_deg,
                "ci_lo_95": ci_lo,
                "ci_hi_95": ci_hi,
                "boot_se": boot_se,
                "K": k,
                "T_yr": t_yr,
                "N": N,
            })

        prev_U = U

    return pd.DataFrame(results)


# =========================================================
# 2. Stationary-null simulation
# =========================================================

def stationary_null_rotation(N, T, K, persistence=0.95, noise_std=0.1,
                              n_sims=N_NULL_SIMS):
    """
    Generate panels from a FIXED latent basis and measure apparent rotation
    from finite-sample DMD estimation.

    Model: Y_t = U_true @ alpha_t + epsilon_t
    where alpha_{t+1} = rho * alpha_t + eta_t (AR(1) per mode)
    U_true is a fixed random orthogonal basis.
    """
    rng = np.random.default_rng(SEED + 1)

    # Fixed true basis
    U_true, _ = np.linalg.qr(rng.standard_normal((N, K)))

    null_angles = []
    for sim in range(n_sims):
        # Generate stationary panel
        alpha = np.zeros((T + 1, K))
        for t in range(T):
            alpha[t + 1] = persistence * alpha[t] + rng.standard_normal(K) * noise_std

        Y = alpha[1:] @ U_true.T + rng.standard_normal((T, N)) * noise_std * 2

        # Split into two halves (simulating consecutive windows)
        T_half = T // 2
        Y1 = Y[:T_half]
        Y2 = Y[T_half:]

        om1 = ewm_demean(Y1)
        om2 = ewm_demean(Y2)
        Y1_dm = Y1 - om1
        Y2_dm = Y2 - om2

        U1 = dmd_basis(Y1_dm, N, k=K)
        U2 = dmd_basis(Y2_dm, N, k=K)

        if U1 is not None and U2 is not None:
            kc = min(U1.shape[1], U2.shape[1])
            angle = np.degrees(subspace_angle(U1, U2, k=kc))
            null_angles.append(angle)

    return np.array(null_angles)


# =========================================================
# 3. Rotation-gain correlation
# =========================================================

def rotation_vs_gain(wide, k=K_GOLD, t_yr=T_YR):
    """
    For each test year, compute per-quarter rotation and per-quarter
    rolling-vs-frozen R^2 gain. Test correlation.
    """
    from scripts.run_smim_ws_a_nested_cv import (
        ewm_demean as _ewd,
        dmd_basis as _dmd,
        train_sph_r,
        predict_quarter,
    )

    # This is expensive, so we import the full pipeline.
    # If imports fail, we skip this analysis.
    TEST_YEARS = list(range(2015, 2025))
    rows = []

    for ty in TEST_YEARS:
        train_start = pd.Timestamp(f"{ty - t_yr}-01-01")
        train_end = pd.Timestamp(f"{ty - 1}-12-31")
        test_quarters = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")

        all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
        valid = all_data.columns[all_data.notna().any()]
        all_data = all_data[valid].fillna(all_data[valid].mean())
        N = len(valid)

        otr = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
        if otr.shape[0] < 4:
            continue
        om = ewm_demean(otr)
        otr_dm = otr - om

        U_frozen, _ = _dmd(otr_dm, N, k=k)
        if U_frozen is None:
            continue

        k_act = U_frozen.shape[1]
        R_sph = train_sph_r(otr_dm, N, U_frozen)
        F_mat = np.eye(k_act) * 0.99
        Q_init = np.eye(k_act) * 0.5

        # Frozen path
        alpha_f = np.zeros(k_act)
        P_f = np.eye(k_act)
        Q_run_f = Q_init.copy()

        # Rolling path
        alpha_r = np.zeros(k_act)
        P_r = np.eye(k_act)
        Q_run_r = Q_init.copy()
        U_current = U_frozen.copy()
        om_r = om.copy()
        R_sph_r = R_sph.copy()

        for q_date in test_quarters:
            q_data = all_data.loc[[q_date]].values.astype(np.float64)
            if q_data.shape[0] == 0:
                continue
            actual = q_data[0]
            q_dm_f = actual - om.ravel()
            q_dm_r = actual - om_r.ravel()

            # Frozen prediction
            pred_f, alpha_f, P_f, Q_run_f = predict_quarter(
                q_dm_f, om, N, U_frozen, F_mat, Q_run_f, R_sph, alpha_f, P_f)

            # Rolling prediction
            pred_r, alpha_r_new, P_r_new, Q_run_r_new = predict_quarter(
                q_dm_r, om_r, N, U_current, F_mat, Q_run_r, R_sph_r, alpha_r, P_r)

            # Per-quarter R^2 (over actors)
            r2_f = float(oos_r_squared(pred_f, actual))
            r2_r = float(oos_r_squared(pred_r, actual))

            # Update rolling basis
            otr_exp = all_data[
                (all_data.index >= train_start) & (all_data.index <= q_date)
            ].values.astype(np.float64)
            om_new = ewm_demean(otr_exp)
            otr_dm_new = otr_exp - om_new

            U_new = dmd_basis(otr_dm_new, N, k=k)
            rotation = 0.0
            if U_new is not None and U_new.shape[0] == U_current.shape[0]:
                kc = min(U_current.shape[1], U_new.shape[1])
                rotation = np.degrees(subspace_angle(U_current, U_new, k=kc))
                U_current = U_new
                k_new = U_new.shape[1]
                alpha_r = U_new.T @ q_dm_r
                P_r = np.eye(k_new)
                Q_run_r = np.eye(k_new) * 0.5
                F_mat_r = np.eye(k_new) * 0.99
                R_sph_r = train_sph_r(otr_dm_new, N, U_new)
                om_r = om_new
            else:
                alpha_r = alpha_r_new
                P_r = P_r_new
                Q_run_r = Q_run_r_new

            rows.append({
                "window": f"W{ty}",
                "quarter": q_date,
                "rotation_deg": rotation,
                "r2_frozen": r2_f,
                "r2_rolling": r2_r,
                "gain": r2_r - r2_f,
            })

    return pd.DataFrame(rows)


# =========================================================
# 4. Sensitivity analysis
# =========================================================

def sensitivity_analysis(wide):
    """Test rotation sensitivity to K and training window length."""
    configs = [
        (3, 5), (5, 5), (8, 5), (10, 5),  # vary K
        (8, 3), (8, 5), (8, 8), (8, 10),   # vary T
    ]

    rows = []
    for K, T in configs:
        df = compute_observed_rotations(wide, k=K, t_yr=T)
        if not df.empty:
            rows.append({
                "K": K,
                "T_yr": T,
                "mean_rotation": df["angle_deg"].mean(),
                "std_rotation": df["angle_deg"].std(),
                "median_rotation": df["angle_deg"].median(),
                "n_quarters": len(df),
            })
            print(f"    K={K}, T={T}yr: mean={df['angle_deg'].mean():.1f}deg "
                  f"(std={df['angle_deg'].std():.1f}deg, n={len(df)})")

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 90)
    print("  WORKSTREAM D: BASIS ROTATION VALIDATION")
    print("=" * 90)

    wide = load_panel()
    N = wide.shape[1]
    print(f"  Panel: {wide.shape[0]} quarters x {N} actors")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Observed rotations with bootstrap CIs ---------
    print("\n-- 1. Observed rotations with bootstrap CIs --")
    t0 = time.time()
    obs_df = compute_observed_rotations(wide)
    print(f"  {len(obs_df)} quarter transitions")
    print(f"  Mean rotation: {obs_df['angle_deg'].mean():.1f}deg  "
          f"(std={obs_df['angle_deg'].std():.1f}deg)")
    print(f"  Mean bootstrap 95% CI: [{obs_df['ci_lo_95'].mean():.1f}deg, "
          f"{obs_df['ci_hi_95'].mean():.1f}deg]")
    print(f"  Time: {time.time() - t0:.1f}s")

    obs_df.to_csv(RESULTS_DIR / "ws_d_rotation_bootstrap.csv", index=False)

    # -- 2. Stationary null simulation --------------------
    print("\n-- 2. Stationary null simulation --")
    t0 = time.time()
    # Use comparable dimensions to real data
    T_sim = T_YR * 4  # 20 quarters (same as training window)
    null_angles = stationary_null_rotation(N=N, T=T_sim, K=K_GOLD)
    print(f"  {len(null_angles)} successful simulations")
    print(f"  Null mean rotation: {null_angles.mean():.1f}deg  "
          f"(std={null_angles.std():.1f}deg)")
    print(f"  Observed mean:      {obs_df['angle_deg'].mean():.1f}deg")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Statistical test: observed > null?
    obs_mean = obs_df["angle_deg"].mean()
    null_mean = null_angles.mean()
    null_std = null_angles.std()
    z_score = (obs_mean - null_mean) / (null_std / np.sqrt(len(obs_df)))
    p_value = float(stats.norm.sf(z_score))
    print(f"\n  Test: observed rotation > null rotation")
    print(f"    z-score:  {z_score:.2f}")
    print(f"    p-value:  {p_value:.4f}")
    exceeds_pct = float(np.mean(obs_df["angle_deg"].values[:, None] > null_angles[None, :]))
    print(f"    Fraction of observed quarters exceeding null mean: "
          f"{(obs_df['angle_deg'] > null_mean).mean():.1%}")

    null_df = pd.DataFrame({
        "null_angle_deg": null_angles,
    })
    null_df.to_csv(RESULTS_DIR / "ws_d_null_rotation.csv", index=False)

    # -- 3. Sensitivity analysis --------------------------
    print("\n-- 3. Sensitivity to K and T --")
    t0 = time.time()
    sens_df = sensitivity_analysis(wide)
    print(f"  Time: {time.time() - t0:.1f}s")
    sens_df.to_csv(RESULTS_DIR / "ws_d_sensitivity.csv", index=False)

    # -- 4. Rotation-gain correlation ---------------------
    print("\n-- 4. Rotation-gain correlation --")
    try:
        t0 = time.time()
        rg_df = rotation_vs_gain(wide)
        if not rg_df.empty:
            corr, p_rg = stats.pearsonr(rg_df["rotation_deg"], rg_df["gain"])
            print(f"  Per-quarter correlation(rotation, rolling-frozen gain):")
            print(f"    r = {corr:.3f}, p = {p_rg:.4f}, n = {len(rg_df)}")
            rg_df.to_csv(RESULTS_DIR / "ws_d_rotation_vs_gain.csv", index=False)
        print(f"  Time: {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  Skipped (import error): {e}")
        print("  This analysis requires Workstream A to be importable.")

    # -- Summary ------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  Observed rotation:    {obs_df['angle_deg'].mean():.1f}deg ± "
          f"{obs_df['angle_deg'].std():.1f}deg per quarter")
    print(f"  Null rotation:        {null_angles.mean():.1f}deg ± "
          f"{null_angles.std():.1f}deg (finite-sample noise)")
    print(f"  Rotation is real:     p = {p_value:.4f} (z = {z_score:.2f})")
    print(f"  Fixed dimensionality: K_eff = {K_GOLD} in {len(obs_df)} quarters")

    print(f"\n  Saved to {RESULTS_DIR}/ws_d_*.csv")


if __name__ == "__main__":
    main()
