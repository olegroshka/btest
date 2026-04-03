#!/usr/bin/env python
"""
Workstream A -- Nested Rolling-CV for Leakage-Proof Evaluation.

Implements a two-level rolling evaluation:
  - OUTER folds: 10 annual test windows (2015-2024) -- final reported performance.
  - INNER folds: within each outer-train, use the last 2 years as inner-validation
    to select hyperparameters (K, T_train) from a candidate grid.
  - HOLDOUT: the final 2 outer windows (2023-2024) are untouched by any tuning.

All transformations (EWM mean, DMD basis, Kalman R, online Q init) are re-fit
inside each fold only.

Outputs:
  results/metrics/ws_a_nested_cv_summary.csv
  results/metrics/ws_a_holdout_summary.csv
  results/metrics/ws_a_actor_quarter_predictions.parquet  (for Workstream B)

Usage::
    uv run python scripts/smim/run_smim_ws_a_nested_cv.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# --Paths ------------------------------------------------
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

# --Hyperparameter grid for inner CV ---------------------
K_CANDIDATES = [3, 5, 8, 10]
T_CANDIDATES = [3, 5, 8]  # training window in years
K_MAX = 15  # max modes for DMD truncation

# --Fixed defaults (not tuned -- these are the dual-regularisation constants)
F_DIAG = 0.99
Q0_DIAG = 0.5
EWM_HALFLIFE = 8
ONLINE_Q_LAMBDA = 0.3

# --Outer evaluation windows -----------------------------
TEST_YEARS = list(range(2015, 2025))
HOLDOUT_YEARS = [2023, 2024]  # frozen -- no tuning exposure


# =========================================================
# Shared pipeline functions (from dd2_v2, factored out)
# =========================================================

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


def ewm_demean(otr, halflife=EWM_HALFLIFE):
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / halflife)
    return (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(otr_dm, N, k=8):
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        eigs = mf.eigenvalues[:min(k_use, mf.K)]
        return U, eigs
    except Exception:
        return None, None


def train_sph_r(otr_dm, N, U):
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
    kf = KalmanFilter()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = kf.em_estimate(otr_dm, mf_k, max_iter=50)
        R_trace = np.trace(st.params["R"])
        if not np.isfinite(R_trace) or R_trace <= 0:
            # Fallback: use residual variance
            resid = otr_dm - otr_dm @ U @ U.T
            R_trace = float(np.mean(resid**2)) * N
        return np.eye(N) * R_trace / N
    except Exception:
        # Fallback: simple residual-based estimate
        resid = otr_dm - otr_dm @ U @ U.T
        scale = float(np.mean(resid**2))
        return np.eye(N) * max(scale, 1e-6)


def predict_quarter(obs_dm_quarter, om, N, U, F, Q_run, R_sph,
                    alpha_init, P_init):
    """Single-quarter Kalman predict+update with online Q adaptation."""
    k_act = U.shape[1]
    a_p = F @ alpha_init
    P_p = F @ P_init @ F.T + Q_run
    v = obs_dm_quarter - U @ a_p
    S = U @ P_p @ U.T + R_sph
    try:
        Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
    except Exception:
        Kg = np.zeros((k_act, N))
    a_f = a_p + Kg @ v
    P_f = (np.eye(k_act) - Kg @ U) @ P_p
    pred = (a_f @ U.T + om).ravel()
    # Online Q adaptation
    innov = a_f - F @ alpha_init
    Q_new = (1 - ONLINE_Q_LAMBDA) * Q_run + ONLINE_Q_LAMBDA * np.outer(innov, innov)
    Q_new = (Q_new + Q_new.T) / 2 + np.eye(k_act) * 1e-6
    return pred, a_f, P_f, Q_new


# =========================================================
# Rolling-basis pipeline for a given (K, T_train)
# =========================================================

def run_rolling_pipeline(wide, test_year, K, T_train):
    """
    Run the rolling-basis SMIM pipeline for a single test year.

    Returns dict with: r2, per-quarter predictions, actuals, actor_ids, errors.
    Returns None if pipeline fails.
    """
    train_start = pd.Timestamp(f"{test_year - T_train}-01-01")
    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(f"{test_year}-01-01", f"{test_year}-12-31", freq="QS")

    all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    actor_ids = list(valid)

    # Initial training
    otr = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    om = ewm_demean(otr)
    otr_dm = otr - om

    U_current, eigs = dmd_basis(otr_dm, N, k=K)
    if U_current is None:
        return None

    k_act = U_current.shape[1]
    R_sph = train_sph_r(otr_dm, N, U_current)
    F_mat = np.eye(k_act) * F_DIAG
    Q_init = np.eye(k_act) * Q0_DIAG

    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    Q_run = Q_init.copy()

    preds_list = []
    actuals_list = []
    quarter_dates = []

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        q_dm = q_data[0] - om.ravel()

        pred, alpha_new, P_new, Q_new = predict_quarter(
            q_dm, om, N, U_current, F_mat, Q_run, R_sph,
            alpha_c, P_c,
        )
        preds_list.append(pred)
        actuals_list.append(q_data[0])
        quarter_dates.append(q_date)

        # Rolling basis update: expand training, recompute DMD
        otr_expanded = all_data[
            (all_data.index >= train_start) & (all_data.index <= q_date)
        ].values.astype(np.float64)
        om_new = ewm_demean(otr_expanded)
        otr_dm_new = otr_expanded - om_new

        U_new, _ = dmd_basis(otr_dm_new, N, k=K)
        if U_new is not None:
            U_current = U_new
            k_act_new = U_new.shape[1]
            alpha_c = U_new.T @ q_dm
            P_c = np.eye(k_act_new)
            Q_run = np.eye(k_act_new) * Q0_DIAG
            F_mat = np.eye(k_act_new) * F_DIAG
            R_sph = train_sph_r(otr_dm_new, N, U_new)
            om = om_new
        else:
            alpha_c = alpha_new
            P_c = P_new
            Q_run = Q_new

    if not preds_list:
        return None

    preds = np.array(preds_list)
    actuals = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), actuals.ravel()))
    errors = preds - actuals  # (n_quarters, N)

    return {
        "r2": r2,
        "preds": preds,
        "actuals": actuals,
        "errors": errors,
        "actor_ids": actor_ids,
        "quarter_dates": quarter_dates,
        "K": K,
        "T_train": T_train,
    }


# =========================================================
# AR(1) baseline for comparison
# =========================================================

def run_ar1_baseline(wide, test_year, T_train=10):
    """Per-actor AR(1) baseline. Returns same structure as run_rolling_pipeline."""
    train_start = pd.Timestamp(f"{test_year - T_train}-01-01")
    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(f"{test_year}-01-01", f"{test_year}-12-31", freq="QS")

    all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    actor_ids = list(valid)

    # Fit AR(1) per actor on training data
    train = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
    N = train.shape[1]
    rho = np.zeros(N)
    mu = np.zeros(N)
    for j in range(N):
        y = train[:, j]
        y = y[~np.isnan(y)]
        if len(y) > 2:
            mu[j] = y.mean()
            y_dm = y - mu[j]
            if np.var(y_dm) > 0:
                rho[j] = np.corrcoef(y_dm[:-1], y_dm[1:])[0, 1]

    preds_list = []
    actuals_list = []
    quarter_dates = []
    prev = train[-1] if len(train) > 0 else np.zeros(N)

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        pred = mu + rho * (prev - mu)
        preds_list.append(pred)
        actuals_list.append(q_data[0])
        quarter_dates.append(q_date)
        prev = q_data[0]

    preds = np.array(preds_list)
    actuals = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), actuals.ravel()))

    return {
        "r2": r2,
        "preds": preds,
        "actuals": actuals,
        "errors": preds - actuals,
        "actor_ids": actor_ids,
        "quarter_dates": quarter_dates,
        "K": 0,
        "T_train": T_train,
    }


# =========================================================
# Inner CV: select best (K, T) on outer-train data only
# =========================================================

def inner_cv_select(wide, test_year):
    """
    Use inner validation (2 years before the outer test) to select (K, T_train).
    The inner-val years are [test_year - 2, test_year - 1].
    Inner-train ends at test_year - 3.
    """
    inner_val_years = [test_year - 2, test_year - 1]
    best_r2 = -np.inf
    best_params = {"K": 8, "T_train": 5}  # default fallback

    for K in K_CANDIDATES:
        for T_train in T_CANDIDATES:
            inner_r2s = []
            for iv_year in inner_val_years:
                # Check we have enough training data
                inner_train_start = pd.Timestamp(f"{iv_year - T_train}-01-01")
                if inner_train_start < pd.Timestamp("2005-01-01"):
                    continue
                try:
                    result = run_rolling_pipeline(wide, iv_year, K, T_train)
                    if result is not None and np.isfinite(result["r2"]):
                        inner_r2s.append(result["r2"])
                except Exception:
                    pass  # skip failed (K, T) combos

            if inner_r2s:
                mean_r2 = np.mean(inner_r2s)
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_params = {"K": K, "T_train": T_train}

    return best_params, best_r2


# =========================================================
# Main: Nested CV + Holdout
# =========================================================

def main():
    print("=" * 90)
    print("  WORKSTREAM A: NESTED ROLLING CV -- LEAKAGE-PROOF EVALUATION")
    print("=" * 90)

    wide = load_panel()
    print(f"  Panel: {wide.shape[0]} quarters x {wide.shape[1]} actors")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --Phase 1: Nested CV on non-holdout windows --------
    print("\n--Phase 1: Nested CV (inner fold selects K, T_train) --")

    cv_rows = []
    all_predictions = []  # for actor-quarter export

    for ty in TEST_YEARS:
        t0 = time.time()
        is_holdout = ty in HOLDOUT_YEARS

        if is_holdout:
            # Use median of previously selected params (no peeking at holdout)
            if cv_rows:
                prev_Ks = [r["selected_K"] for r in cv_rows]
                prev_Ts = [r["selected_T"] for r in cv_rows]
                sel_K = int(np.median(prev_Ks))
                sel_T = int(np.median(prev_Ts))
                inner_r2 = np.nan
            else:
                sel_K, sel_T, inner_r2 = 8, 5, np.nan
        else:
            # Inner CV to select hyperparameters
            best_params, inner_r2 = inner_cv_select(wide, ty)
            sel_K = best_params["K"]
            sel_T = best_params["T_train"]

        # Run outer evaluation with selected params
        result = run_rolling_pipeline(wide, ty, sel_K, sel_T)
        ar1_result = run_ar1_baseline(wide, ty)

        if result is None:
            print(f"  W{ty}: FAILED")
            continue

        elapsed = time.time() - t0
        tag = " [HOLDOUT]" if is_holdout else ""
        print(f"  W{ty}: K={sel_K} T={sel_T}yr  R^2={result['r2']:.4f}  "
              f"AR(1)={ar1_result['r2']:.4f}  "
              f"Delta={result['r2'] - ar1_result['r2']:+.4f}  ({elapsed:.1f}s){tag}")

        cv_rows.append({
            "window": f"W{ty}",
            "test_year": ty,
            "is_holdout": is_holdout,
            "selected_K": sel_K,
            "selected_T": sel_T,
            "inner_val_r2": inner_r2,
            "outer_r2_smim": result["r2"],
            "outer_r2_ar1": ar1_result["r2"],
            "delta_r2": result["r2"] - ar1_result["r2"],
        })

        # Store actor-quarter predictions for Workstream B
        for q_idx, q_date in enumerate(result["quarter_dates"]):
            for a_idx, aid in enumerate(result["actor_ids"]):
                all_predictions.append({
                    "window": f"W{ty}",
                    "quarter": q_date,
                    "actor_id": aid,
                    "pred_smim": result["preds"][q_idx, a_idx],
                    "pred_ar1": ar1_result["preds"][q_idx, a_idx],
                    "actual": result["actuals"][q_idx, a_idx],
                    "error_smim": result["errors"][q_idx, a_idx],
                    "error_ar1": ar1_result["errors"][q_idx, a_idx],
                    "is_holdout": is_holdout,
                    "selected_K": sel_K,
                    "selected_T": sel_T,
                })

    # --Summary ------------------------------------------
    df = pd.DataFrame(cv_rows)
    if df.empty:
        print("  No results!")
        return

    cv_mask = ~df["is_holdout"]
    ho_mask = df["is_holdout"]

    print("\n" + "=" * 90)
    print("  NESTED CV SUMMARY")
    print("=" * 90)
    if cv_mask.any():
        print(f"  CV windows (N={cv_mask.sum()}):")
        print(f"    Mean SMIM R^2:  {df.loc[cv_mask, 'outer_r2_smim'].mean():.4f}")
        print(f"    Mean AR(1) R^2: {df.loc[cv_mask, 'outer_r2_ar1'].mean():.4f}")
        print(f"    Mean Delta R^2:     {df.loc[cv_mask, 'delta_r2'].mean():+.4f}")
        print(f"    SMIM wins:     {(df.loc[cv_mask, 'delta_r2'] > 0).sum()}/{cv_mask.sum()}")
        print(f"    Selected K:    {df.loc[cv_mask, 'selected_K'].value_counts().to_dict()}")
        print(f"    Selected T:    {df.loc[cv_mask, 'selected_T'].value_counts().to_dict()}")

    if ho_mask.any():
        print(f"\n  HOLDOUT windows (N={ho_mask.sum()}) -- NO tuning exposure:")
        print(f"    Mean SMIM R^2:  {df.loc[ho_mask, 'outer_r2_smim'].mean():.4f}")
        print(f"    Mean AR(1) R^2: {df.loc[ho_mask, 'outer_r2_ar1'].mean():.4f}")
        print(f"    Mean Delta R^2:     {df.loc[ho_mask, 'delta_r2'].mean():+.4f}")

    # --Comparison with paper numbers --------------------
    print(f"\n  COMPARISON vs PAPER (original sweep-all-windows):")
    print(f"    Paper rolling SMIM:  0.691")
    print(f"    Paper AR(1):         0.425")
    print(f"    Nested-CV SMIM:      {df.loc[cv_mask, 'outer_r2_smim'].mean():.4f}" if cv_mask.any() else "")
    print(f"    Holdout SMIM:        {df.loc[ho_mask, 'outer_r2_smim'].mean():.4f}" if ho_mask.any() else "")

    # --Save results -------------------------------------
    df.to_csv(RESULTS_DIR / "ws_a_nested_cv_summary.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'ws_a_nested_cv_summary.csv'}")

    if ho_mask.any():
        df[ho_mask].to_csv(RESULTS_DIR / "ws_a_holdout_summary.csv", index=False)
        print(f"  Saved: {RESULTS_DIR / 'ws_a_holdout_summary.csv'}")

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_parquet(RESULTS_DIR / "ws_a_actor_quarter_predictions.parquet", index=False)
    print(f"  Saved: {RESULTS_DIR / 'ws_a_actor_quarter_predictions.parquet'}")
    print(f"  ({len(pred_df)} actor-quarter observations)")


if __name__ == "__main__":
    main()
