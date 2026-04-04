#!/usr/bin/env python
"""
SMIM Iteration 3 — Phase 1: Daily Linear SMIM Baseline.

Runs SMIM (rolling DMD + dual regularisation + online Q) at daily frequency
on Panel A (stock momentum rank) and Panel B (sector/macro).
Compares against daily AR(1) per actor.

Experiments:
  E3-1a: SMIM rolling on Panel A (L=20 momentum, 10 windows)
  E3-1b: SMIM rolling on Panel B (sector/macro, 5 windows)
  E3-1c: SMIM on step-interpolated CapEx (diagnostic, 10 windows)
  E3-1d: AR(1) per actor baselines

Usage::
    uv run python scripts/smim/run_smim_iter3_phase1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

DATA_DIR = PROJECT_ROOT / "data" / "smim" / "intensities"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# DIAMOND config adapted for daily
K_TARGET = 8
K_MAX = 15
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3  # online Q adaptation
EWM_HALFLIFE_DAYS = 60  # ~2 months (analogous to 8Q at quarterly)

TRAIN_YEARS = 5
TEST_DAYS_PER_YEAR = 252
REESTIMATE_INTERVAL = 21  # recompute DMD every ~1 month (21 trading days)


# =========================================================================
# Core pipeline functions
# =========================================================================


def load_panel_wide(path: Path) -> pd.DataFrame:
    """Load intensity panel from parquet (long) to wide format."""
    df = pd.read_parquet(path)
    wide = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def ewm_demean(obs: np.ndarray, halflife: int = EWM_HALFLIFE_DAYS) -> np.ndarray:
    """Compute exponentially-weighted mean for demeaning."""
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(obs_dm: np.ndarray, k: int = K_TARGET) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract DMD basis from demeaned observations. Returns (U, eigenvalues) or (None, None)."""
    N = obs_dm.shape[1]
    k_use = min(k, N - 2, K_MAX)
    if obs_dm.shape[0] < 3:
        return None, None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        eigs = mf.eigenvalues[:min(k_use, mf.K)]
        return U, eigs
    except Exception:
        return None, None


def spherical_r(obs_dm: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Estimate spherical R via single EM pass."""
    N = U.shape[0]
    K = U.shape[1]
    # Quick EM: project onto basis, compute residual variance
    alpha_hat = obs_dm @ U  # (T, K) — least squares projection
    resid = obs_dm - alpha_hat @ U.T  # (T, N)
    r_diag = np.mean(resid ** 2)
    return np.eye(N) * r_diag


def kalman_predict_update(
    obs_day: np.ndarray,  # (N,) observation
    U: np.ndarray,        # (N, K) basis
    F: np.ndarray,        # (K, K) transition
    R_sph: np.ndarray,    # (N, N) observation cov
    alpha: np.ndarray,    # (K,) current state
    P: np.ndarray,        # (K, K) state cov
    Q: np.ndarray,        # (K, K) state noise
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single Kalman predict-update step. Returns (prediction, alpha_filt, P_filt, Q_new, innovation)."""
    N = U.shape[0]
    K = U.shape[1]

    # Predict
    alpha_pred = F @ alpha
    P_pred = F @ P @ F.T + Q

    # Innovation
    pred_obs = U @ alpha_pred
    innov = obs_day - pred_obs

    # Update (Woodbury for efficiency when N >> K)
    S = U @ P_pred @ U.T + R_sph
    try:
        Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        Kg = np.zeros((K, N))

    alpha_filt = alpha_pred + Kg @ innov
    P_filt = (np.eye(K) - Kg @ U) @ P_pred

    # Online Q adaptation
    state_innov = alpha_filt - F @ alpha
    Q_new = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(state_innov, state_innov)
    Q_new = (Q_new + Q_new.T) / 2 + np.eye(K) * 1e-8  # symmetrise + regularise

    return pred_obs, alpha_filt, P_filt, Q_new, innov


def run_smim_window(
    panel_wide: pd.DataFrame,
    test_year: int,
    train_years: int = TRAIN_YEARS,
    reestimate_interval: int = REESTIMATE_INTERVAL,
    label: str = "",
) -> dict | None:
    """Run SMIM rolling DMD on one test window.

    Returns dict with R², predictions, actuals, and diagnostics.
    """
    # Define train/test periods
    train_start = pd.Timestamp(f"{test_year - train_years}-01-01")
    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    # Slice data
    all_data = panel_wide[
        (panel_wide.index >= train_start) & (panel_wide.index <= test_end)
    ].copy()

    # Drop actors with >50% NaN in training period
    train_mask = all_data.index <= train_end
    train_coverage = all_data.loc[train_mask].notna().mean()
    valid_cols = train_coverage[train_coverage > 0.50].index
    all_data = all_data[valid_cols]

    # Fill remaining NaN with column mean
    col_means = all_data.mean()
    all_data = all_data.fillna(col_means)

    N = len(valid_cols)
    if N < 10:
        return None

    # Split train/test
    train_data = all_data[all_data.index <= train_end].values.astype(np.float64)
    test_dates = all_data.index[all_data.index >= test_start]
    test_data = all_data.loc[test_dates].values.astype(np.float64)

    if len(train_data) < 60 or len(test_data) < 10:
        return None

    # Initial DMD on training window
    mu = ewm_demean(train_data)
    train_dm = train_data - mu
    U, eigs = dmd_basis(train_dm)
    if U is None:
        return None

    K = U.shape[1]
    R_sph = spherical_r(train_dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT_SCALE

    # Initialise Kalman state
    alpha = np.zeros(K)
    P = np.eye(K)
    Q_run = Q.copy()

    # Rolling prediction on test period
    predictions = []
    actuals = []
    expanded_data = train_data.copy()
    days_since_reestimate = 0

    for i, t_date in enumerate(test_dates):
        obs_raw = test_data[i]
        obs_dm = obs_raw - mu.ravel()

        # Predict + update
        pred_dm, alpha, P, Q_run, _ = kalman_predict_update(
            obs_dm, U, F, R_sph, alpha, P, Q_run
        )
        pred_raw = pred_dm + mu.ravel()
        predictions.append(pred_raw)
        actuals.append(obs_raw)

        # Rolling basis update
        days_since_reestimate += 1
        if days_since_reestimate >= reestimate_interval:
            # Expand training data to include recent observations
            expanded_data = np.vstack([expanded_data, test_data[max(0, i - reestimate_interval + 1):i + 1]])
            mu_new = ewm_demean(expanded_data)
            dm_new = expanded_data - mu_new
            U_new, eigs_new = dmd_basis(dm_new)

            if U_new is not None:
                K_new = U_new.shape[1]
                # Reproject state onto new basis
                alpha = U_new.T @ obs_dm
                P = np.eye(K_new)
                Q_run = np.eye(K_new) * Q_INIT_SCALE
                F = np.eye(K_new) * F_REG
                R_sph = spherical_r(dm_new, U_new)
                U = U_new
                K = K_new
                mu = mu_new

            days_since_reestimate = 0

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r2 = float(oos_r_squared(predictions.ravel(), actuals.ravel()))

    return {
        "test_year": test_year,
        "r2": r2,
        "N": N,
        "K": K,
        "T_train": len(train_data),
        "T_test": len(test_data),
        "label": label,
    }


def run_ar1_window(
    panel_wide: pd.DataFrame,
    test_year: int,
    train_years: int = TRAIN_YEARS,
) -> dict | None:
    """Run per-actor AR(1) baseline on one test window."""
    train_start = pd.Timestamp(f"{test_year - train_years}-01-01")
    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel_wide[
        (panel_wide.index >= train_start) & (panel_wide.index <= test_end)
    ].copy()

    train_mask = all_data.index <= train_end
    train_coverage = all_data.loc[train_mask].notna().mean()
    valid_cols = train_coverage[train_coverage > 0.50].index
    all_data = all_data[valid_cols].fillna(all_data[valid_cols].mean())

    N = len(valid_cols)
    train_data = all_data[all_data.index <= train_end].values.astype(np.float64)
    test_dates = all_data.index[all_data.index >= test_start]
    test_data = all_data.loc[test_dates].values.astype(np.float64)

    if len(train_data) < 60 or len(test_data) < 10:
        return None

    # Estimate AR(1) per actor on training data
    # y_{t+1} = a + b * y_t
    Y = train_data[1:]
    X = train_data[:-1]
    # OLS per column
    predictions = []
    for i in range(len(test_data)):
        if i == 0:
            prev = train_data[-1]
        else:
            prev = test_data[i - 1]
        # Simple AR(1): predict next = last observed (random walk)
        # Actually estimate AR(1) coefficients
        predictions.append(prev)  # AR(1) with fitted coefficients below

    # Proper AR(1): estimate rho and mu per actor
    mu_train = train_data.mean(axis=0)
    dm = train_data - mu_train
    rho = np.array([
        np.corrcoef(dm[:-1, j], dm[1:, j])[0, 1] if dm[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])

    # Predict test period
    predictions = []
    actuals = []
    prev = train_data[-1]
    for i in range(len(test_data)):
        pred = mu_train + rho * (prev - mu_train)
        predictions.append(pred)
        actuals.append(test_data[i])
        prev = test_data[i]

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r2 = float(oos_r_squared(predictions.ravel(), actuals.ravel()))

    return {"test_year": test_year, "r2": r2, "N": N, "label": "AR1"}


# =========================================================================
# Main experiments
# =========================================================================


def run_experiment(panel_path: Path, label: str, test_years: list[int],
                   reestimate_interval: int = REESTIMATE_INTERVAL) -> pd.DataFrame:
    """Run SMIM + AR(1) on a panel for given test years."""
    panel = load_panel_wide(panel_path)
    print(f"\n  Panel: {panel.shape[0]} periods x {panel.shape[1]} actors")

    rows = []
    for ty in test_years:
        t0 = time.time()

        # SMIM rolling
        res_smim = run_smim_window(panel, ty, reestimate_interval=reestimate_interval, label=label)
        # AR(1)
        res_ar1 = run_ar1_window(panel, ty)

        if res_smim and res_ar1:
            delta = res_smim["r2"] - res_ar1["r2"]
            smim_wins = "WIN" if delta > 0 else "LOSS"
            print(f"    W{ty}: SMIM R2={res_smim['r2']:.4f}  AR1={res_ar1['r2']:.4f}  "
                  f"delta={delta:+.4f} {smim_wins}  (N={res_smim['N']}, K={res_smim['K']}, "
                  f"T_test={res_smim['T_test']}, {time.time()-t0:.1f}s)")
            rows.append({
                "test_year": ty,
                "smim_r2": res_smim["r2"],
                "ar1_r2": res_ar1["r2"],
                "delta_r2": delta,
                "N": res_smim["N"],
                "K": res_smim["K"],
                "T_train": res_smim["T_train"],
                "T_test": res_smim["T_test"],
                "label": label,
            })
        elif res_ar1:
            print(f"    W{ty}: SMIM FAILED, AR1={res_ar1['r2']:.4f}")
        else:
            print(f"    W{ty}: BOTH FAILED")

    return pd.DataFrame(rows)


def main():
    t_total = time.time()
    print("=" * 80)
    print("  SMIM Iteration 3 — Phase 1: Daily Linear SMIM Baseline")
    print("=" * 80)

    all_results = []

    # E3-1a: Panel A (L=20 momentum rank, 10 windows)
    print(f"\n{'='*80}")
    print(f"  E3-1a: SMIM rolling on Panel A (L=20 momentum, daily)")
    print(f"{'='*80}")
    df_a = run_experiment(
        DATA_DIR / "iter3_panel_a_L20.parquet",
        label="panel_a_L20",
        test_years=list(range(2015, 2025)),
        reestimate_interval=21,  # monthly basis update
    )
    all_results.append(df_a)

    # E3-1b: Panel B (sector/macro, 5 windows)
    print(f"\n{'='*80}")
    print(f"  E3-1b: SMIM rolling on Panel B (sector/macro, daily)")
    print(f"{'='*80}")
    df_b = run_experiment(
        DATA_DIR / "iter3_panel_b.parquet",
        label="panel_b",
        test_years=list(range(2020, 2025)),
        reestimate_interval=21,
    )
    all_results.append(df_b)

    # E3-1c: Step-interpolated CapEx diagnostic
    capex_path = DATA_DIR / "iter3_capex_step.parquet"
    if capex_path.exists():
        print(f"\n{'='*80}")
        print(f"  E3-1c: SMIM on step-interpolated CapEx (diagnostic)")
        print(f"{'='*80}")
        df_c = run_experiment(
            capex_path,
            label="capex_step_diag",
            test_years=list(range(2015, 2025)),
            reestimate_interval=63,  # quarterly basis update (matches quarterly data)
        )
        all_results.append(df_c)

    # Combine and save
    results = pd.concat(all_results, ignore_index=True)
    out_path = METRICS_DIR / "iter3_phase1_results.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Summary table
    print(f"\n\n{'#'*80}")
    print(f"#  PHASE 1 SUMMARY")
    print(f"{'#'*80}")

    for label in results["label"].unique():
        subset = results[results["label"] == label]
        mean_smim = subset["smim_r2"].mean()
        mean_ar1 = subset["ar1_r2"].mean()
        mean_delta = subset["delta_r2"].mean()
        wins = (subset["delta_r2"] > 0).sum()
        total = len(subset)
        print(f"\n  {label}:")
        print(f"    Windows: {total}")
        print(f"    SMIM R2:  mean={mean_smim:.4f}, std={subset['smim_r2'].std():.4f}")
        print(f"    AR(1) R2: mean={mean_ar1:.4f}, std={subset['ar1_r2'].std():.4f}")
        print(f"    Delta R2: mean={mean_delta:+.4f}")
        print(f"    SMIM wins: {wins}/{total}")

    # BRONZE gate
    panel_a_results = results[results["label"] == "panel_a_L20"]
    if not panel_a_results.empty:
        bronze_pass = panel_a_results["delta_r2"].mean() > 0
        wins = (panel_a_results["delta_r2"] > 0).sum()
        total = len(panel_a_results)
        print(f"\n  {'='*60}")
        print(f"  BRONZE GATE: Daily SMIM R2 > Daily AR(1)?")
        print(f"    Mean delta R2: {panel_a_results['delta_r2'].mean():+.4f}")
        print(f"    SMIM wins {wins}/{total} windows")
        print(f"    VERDICT: {'PASS' if bronze_pass else 'FAIL'}")
        print(f"  {'='*60}")

    print(f"\n  Total time: {time.time() - t_total:.1f}s")
    print("  DONE")


if __name__ == "__main__":
    main()
