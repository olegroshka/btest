#!/usr/bin/env python
"""
Iteration 5.1 -- Phases D-E: Operator Improvements + Ablation & Economic Validation
for CapEx/Revenue intensity.

Phase D: Operator learning improvements (stop after first non-improvement)
  D-1: Ridge penalty on operator weights
  D-2: Increase Nelder-Mead budget (100 -> 300 iters)
  D-3: Intensity-proximity operator
  D-5: Differential evolution optimiser

Phase E: Ablation + Economic validation
  E-1: Full ablation ladder (7 steps)
  E-5: CapEx revision prediction (gap -> future CapEx change)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1_de.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize, differential_evolution

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# Import reusable functions from iter5_1
from run_smim_iter5_1 import (
    build_capex_revenue_panel,
    ewm_demean,
    dmd_basis,
    train_sph_r,
    build_operator_library,
    optimize_operator,
    run_diamond_window,
    run_ar1_window,
    EDGAR_PATH,
    METRICS_DIR,
    F_REG,
    Q_INIT_SCALE,
    LAMBDA_Q,
    K_MAX,
    TEST_YEARS,
)

# Base config
BASE_K = 3
BASE_EWM = 12
BASE_T_TRAIN = 3


# =========================================================================
# Phase D: Modified operator optimisers
# =========================================================================


def optimize_operator_ridge(
    obs_dm: np.ndarray,
    library: list,
    N: int,
    K_opt: int = 3,
    ridge_lambda: float = 0.1,
) -> tuple:
    """D-1: optimize_operator with L2 ridge penalty on weights."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st = obs_dm[:t_split]
    obs_sv = obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = np.zeros((N, N))
            for w, (_, B) in zip(weights, library):
                A += w * B
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0

            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6

            K = U.shape[1]
            R_sph = train_sph_r(obs_st, U)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT_SCALE

            alpha = np.zeros(K)
            P = np.eye(K)
            Q_run = Q.copy()
            preds = []
            for t in range(len(obs_sv)):
                alpha_pred = F @ alpha
                P_pred = F @ P @ F.T + Q_run
                preds.append(U @ alpha_pred)

                obs = obs_sv[t]
                S = U @ P_pred @ U.T + R_sph
                try:
                    Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
                except np.linalg.LinAlgError:
                    Kg = np.zeros((K, N))
                alpha = alpha_pred + Kg @ (obs - U @ alpha_pred)
                P = (np.eye(K) - Kg @ U) @ P_pred
                si = alpha - alpha_pred
                Q_run = 0.7 * Q_run + 0.3 * np.outer(si, si)
                Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8

            preds = np.array(preds)
            r2 = float(oos_r_squared(preds.ravel(), obs_sv.ravel()))
            # Add ridge penalty
            penalty = ridge_lambda * float(np.sum(weights ** 2))
            return (-r2 + penalty) if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    opt = sp_minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": 100, "xatol": 0.02, "fatol": 0.002},
    )

    A_opt = np.zeros((N, N))
    for w, (_, B) in zip(opt.x, library):
        A_opt += w * B

    return A_opt, opt.x, -opt.fun


def optimize_operator_budget300(
    obs_dm: np.ndarray,
    library: list,
    N: int,
    K_opt: int = 3,
) -> tuple:
    """D-2: optimize_operator with 300 Nelder-Mead iterations."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st = obs_dm[:t_split]
    obs_sv = obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = np.zeros((N, N))
            for w, (_, B) in zip(weights, library):
                A += w * B
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0

            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6

            K = U.shape[1]
            R_sph = train_sph_r(obs_st, U)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT_SCALE

            alpha = np.zeros(K)
            P = np.eye(K)
            Q_run = Q.copy()
            preds = []
            for t in range(len(obs_sv)):
                alpha_pred = F @ alpha
                P_pred = F @ P @ F.T + Q_run
                preds.append(U @ alpha_pred)

                obs = obs_sv[t]
                S = U @ P_pred @ U.T + R_sph
                try:
                    Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
                except np.linalg.LinAlgError:
                    Kg = np.zeros((K, N))
                alpha = alpha_pred + Kg @ (obs - U @ alpha_pred)
                P = (np.eye(K) - Kg @ U) @ P_pred
                si = alpha - alpha_pred
                Q_run = 0.7 * Q_run + 0.3 * np.outer(si, si)
                Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8

            preds = np.array(preds)
            r2 = float(oos_r_squared(preds.ravel(), obs_sv.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    opt = sp_minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 0.01, "fatol": 0.001},
    )

    A_opt = np.zeros((N, N))
    for w, (_, B) in zip(opt.x, library):
        A_opt += w * B

    return A_opt, opt.x, -opt.fun


def build_operator_library_with_proximity(
    obs_dm: np.ndarray, N: int,
) -> list:
    """D-3: Standard library + intensity-proximity operator."""
    library = build_operator_library(obs_dm, N)

    # Intensity proximity: actors with similar mean intensity levels co-move
    with np.errstate(invalid="ignore"):
        mean_levels = np.nanmean(obs_dm, axis=0)
    mean_levels = np.nan_to_num(mean_levels, nan=0.0)
    # Gaussian kernel on level differences
    diffs = mean_levels[:, None] - mean_levels[None, :]
    sigma = max(np.std(mean_levels), 1e-8)
    proximity = np.exp(-(diffs ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(proximity, 0.0)
    # Row-normalise
    row_sums = proximity.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    proximity = proximity / row_sums
    library.append(("intensity_proximity", proximity))

    return library


def optimize_operator_diffevo(
    obs_dm: np.ndarray,
    library: list,
    N: int,
    K_opt: int = 3,
) -> tuple:
    """D-5: Replace Nelder-Mead with differential evolution."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st = obs_dm[:t_split]
    obs_sv = obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = np.zeros((N, N))
            for w, (_, B) in zip(weights, library):
                A += w * B
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0

            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6

            K = U.shape[1]
            R_sph = train_sph_r(obs_st, U)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT_SCALE

            alpha = np.zeros(K)
            P = np.eye(K)
            Q_run = Q.copy()
            preds = []
            for t in range(len(obs_sv)):
                alpha_pred = F @ alpha
                P_pred = F @ P @ F.T + Q_run
                preds.append(U @ alpha_pred)

                obs = obs_sv[t]
                S = U @ P_pred @ U.T + R_sph
                try:
                    Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
                except np.linalg.LinAlgError:
                    Kg = np.zeros((K, N))
                alpha = alpha_pred + Kg @ (obs - U @ alpha_pred)
                P = (np.eye(K) - Kg @ U) @ P_pred
                si = alpha - alpha_pred
                Q_run = 0.7 * Q_run + 0.3 * np.outer(si, si)
                Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8

            preds = np.array(preds)
            r2 = float(oos_r_squared(preds.ravel(), obs_sv.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    bounds = [(-2.0, 2.0)] * n_bases
    opt = differential_evolution(
        objective, bounds,
        seed=42, maxiter=100, tol=0.002, polish=False,
        popsize=10,
    )

    A_opt = np.zeros((N, N))
    for w, (_, B) in zip(opt.x, library):
        A_opt += w * B

    return A_opt, opt.x, -opt.fun


# =========================================================================
# Phase D: Run a single window with a custom operator optimiser
# =========================================================================


def run_diamond_window_custom_op(
    panel: pd.DataFrame,
    test_year: int,
    K: int = BASE_K,
    ewm_hl: int = BASE_EWM,
    T_train_yr: int = BASE_T_TRAIN,
    op_func: str = "base",
) -> dict | None:
    """Run DIAMOND pipeline with a custom operator optimiser variant.

    op_func: "base", "ridge", "budget300", "proximity", "diffevo"
    """
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    # Build library (possibly with proximity operator)
    if op_func == "proximity":
        library = build_operator_library_with_proximity(otr_dm, N)
    else:
        library = build_operator_library(otr_dm, N)

    # Operator learning with selected variant
    op_shape = np.eye(N)
    if library:
        if op_func == "ridge":
            A_opt, weights, sv_r2 = optimize_operator_ridge(
                otr_dm, library, N, K_opt=K,
            )
        elif op_func == "budget300":
            A_opt, weights, sv_r2 = optimize_operator_budget300(
                otr_dm, library, N, K_opt=K,
            )
        elif op_func == "proximity":
            A_opt, weights, sv_r2 = optimize_operator(
                otr_dm, library, N, K_opt=K,
            )
        elif op_func == "diffevo":
            A_opt, weights, sv_r2 = optimize_operator_diffevo(
                otr_dm, library, N, K_opt=K,
            )
        else:
            A_opt, weights, sv_r2 = optimize_operator(
                otr_dm, library, N, K_opt=K,
            )
        op_shape = np.eye(N) + 0.1 * A_opt

    # Initial DMD
    U, eigs = dmd_basis(otr_dm @ op_shape, K)
    if U is None:
        U, eigs = dmd_basis(otr_dm, K)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
    F_plat = np.eye(k_act) * F_REG
    Q_init = np.eye(k_act) * Q_INIT_SCALE

    alpha = np.zeros(k_act)
    P = np.eye(k_act)
    Q_run = Q_init.copy()

    preds_smim = []
    actuals = []

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        obs_dm = obs_raw - om.ravel()

        # PREDICTIVE alpha
        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_run
        pred_dm = U @ alpha_pred
        pred_raw = pred_dm + om.ravel()
        preds_smim.append(pred_raw)
        actuals.append(obs_raw)

        # Kalman update
        S = U @ P_pred @ U.T + R_sph
        try:
            Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
        except np.linalg.LinAlgError:
            Kg = np.zeros((k_act, N))
        alpha = alpha_pred + Kg @ (obs_dm - U @ alpha_pred)
        P = (np.eye(k_act) - Kg @ U) @ P_pred

        innov = alpha - alpha_pred
        Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(k_act) * 1e-6

        # Rolling basis update
        expanded = np.vstack([otr, q_data])
        otr = expanded
        om_new = ewm_demean(expanded, ewm_hl)
        dm_new = expanded - om_new

        if op_func == "proximity":
            lib_new = build_operator_library_with_proximity(dm_new, N)
        else:
            lib_new = build_operator_library(dm_new, N)

        # Re-learn operator on rolling window (use same variant)
        op_shape_new = np.eye(N)
        if lib_new:
            try:
                if op_func == "ridge":
                    A_new, _, _ = optimize_operator_ridge(
                        dm_new, lib_new, N, K_opt=K,
                    )
                elif op_func == "budget300":
                    A_new, _, _ = optimize_operator_budget300(
                        dm_new, lib_new, N, K_opt=K,
                    )
                elif op_func == "diffevo":
                    A_new, _, _ = optimize_operator_diffevo(
                        dm_new, lib_new, N, K_opt=K,
                    )
                else:
                    A_new, _, _ = optimize_operator(
                        dm_new, lib_new, N, K_opt=K,
                    )
                op_shape_new = np.eye(N) + 0.1 * A_new
            except Exception:
                op_shape_new = op_shape

        U_new, _ = dmd_basis(dm_new @ op_shape_new, K)
        if U_new is None:
            U_new, _ = dmd_basis(dm_new, K)
        if U_new is not None:
            k_new = U_new.shape[1]
            alpha = U_new.T @ obs_dm
            P = np.eye(k_new)
            Q_run = np.eye(k_new) * Q_INIT_SCALE
            F_plat = np.eye(k_new) * F_REG
            R_sph = train_sph_r(dm_new, U_new)
            U = U_new
            k_act = k_new
            om = om_new
            op_shape = op_shape_new

    if not preds_smim:
        return None

    preds_s = np.array(preds_smim)
    acts = np.array(actuals)
    r2_smim = float(oos_r_squared(preds_s.ravel(), acts.ravel()))

    return {
        "year": test_year, "smim_r2": r2_smim, "N": N, "K": k_act,
        "preds": preds_s, "actuals": acts,
    }


# =========================================================================
# Phase E-1: Ablation ladder functions
# =========================================================================


def run_ablation_step0(panel: pd.DataFrame, test_year: int,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 0: Constant mean only -- predict cross-sectional mean of train."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)

    # Constant mean prediction
    mu = np.nanmean(otr, axis=0)

    preds_list = []
    actuals_list = []
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        preds_list.append(mu)
        actuals_list.append(q_data[0])

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))
    return {"year": test_year, "r2": r2, "N": N}


def run_ablation_step1(panel: pd.DataFrame, test_year: int,
                       ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 1: Constant mean + EWM demeaning -- predict EWM mean."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)

    preds_list = []
    actuals_list = []
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        # EWM mean is the prediction (no spectral component)
        preds_list.append(om.ravel())
        actuals_list.append(q_data[0])
        # Update EWM with new observation
        otr = np.vstack([otr, q_data])
        om = ewm_demean(otr, ewm_hl)

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))
    return {"year": test_year, "r2": r2, "N": N}


def run_ablation_step2(panel: pd.DataFrame, test_year: int,
                       K: int = BASE_K, ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 2: + DMD basis (OLS projection, no Kalman)."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    U, _ = dmd_basis(otr_dm, K)
    if U is None:
        return None

    # OLS projection of last train observation
    alpha_ols = U.T @ otr_dm[-1]

    preds_list = []
    actuals_list = []
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        # Predict: project last known demeaned onto basis + mean
        pred_dm = U @ alpha_ols
        pred_raw = pred_dm + om.ravel()
        preds_list.append(pred_raw)
        actuals_list.append(q_data[0])

        # Update: OLS with new observation
        obs_dm = q_data[0] - om.ravel()
        alpha_ols = U.T @ obs_dm

        # Rolling update
        otr = np.vstack([otr, q_data])
        om = ewm_demean(otr, ewm_hl)

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))
    return {"year": test_year, "r2": r2, "N": N}


def run_ablation_step3(panel: pd.DataFrame, test_year: int,
                       K: int = BASE_K, ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 3: + Kalman with spherical R (no Q adaptation, no rolling basis)."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    U, _ = dmd_basis(otr_dm, K)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
    F_plat = np.eye(k_act) * F_REG
    Q_fixed = np.eye(k_act) * Q_INIT_SCALE

    alpha = np.zeros(k_act)
    P = np.eye(k_act)

    preds_list = []
    actuals_list = []
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        obs_dm = obs_raw - om.ravel()

        # PREDICTIVE
        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_fixed
        pred_dm = U @ alpha_pred
        pred_raw = pred_dm + om.ravel()
        preds_list.append(pred_raw)
        actuals_list.append(obs_raw)

        # Kalman update
        S = U @ P_pred @ U.T + R_sph
        try:
            Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
        except np.linalg.LinAlgError:
            Kg = np.zeros((k_act, N))
        alpha = alpha_pred + Kg @ (obs_dm - U @ alpha_pred)
        P = (np.eye(k_act) - Kg @ U) @ P_pred

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))
    return {"year": test_year, "r2": r2, "N": N}


def run_ablation_step4(panel: pd.DataFrame, test_year: int,
                       K: int = BASE_K, ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 4: + Online Q adaptation (no rolling basis, no operator learning)."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(
        f"{test_year}-01-01", f"{test_year}-12-31", freq="QS",
    )

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    U, _ = dmd_basis(otr_dm, K)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
    F_plat = np.eye(k_act) * F_REG
    Q_init = np.eye(k_act) * Q_INIT_SCALE

    alpha = np.zeros(k_act)
    P = np.eye(k_act)
    Q_run = Q_init.copy()

    preds_list = []
    actuals_list = []
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        obs_dm = obs_raw - om.ravel()

        # PREDICTIVE
        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_run
        pred_dm = U @ alpha_pred
        pred_raw = pred_dm + om.ravel()
        preds_list.append(pred_raw)
        actuals_list.append(obs_raw)

        # Kalman update
        S = U @ P_pred @ U.T + R_sph
        try:
            Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
        except np.linalg.LinAlgError:
            Kg = np.zeros((k_act, N))
        alpha = alpha_pred + Kg @ (obs_dm - U @ alpha_pred)
        P = (np.eye(k_act) - Kg @ U) @ P_pred

        # Online Q adaptation
        innov = alpha - alpha_pred
        Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(k_act) * 1e-6

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))
    return {"year": test_year, "r2": r2, "N": N}


def run_ablation_step5(panel: pd.DataFrame, test_year: int,
                       K: int = BASE_K, ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 5: + Rolling basis (no operator learning).

    Same as run_diamond_window with use_oplearn=False.
    """
    return run_diamond_window(
        panel, test_year, K=K, ewm_hl=ewm_hl,
        T_train_yr=T_train_yr, use_oplearn=False,
    )


def run_ablation_step6(panel: pd.DataFrame, test_year: int,
                       K: int = BASE_K, ewm_hl: int = BASE_EWM,
                       T_train_yr: int = BASE_T_TRAIN) -> dict | None:
    """Step 6: + Operator learning (full pipeline).

    Same as run_diamond_window with use_oplearn=True.
    """
    return run_diamond_window(
        panel, test_year, K=K, ewm_hl=ewm_hl,
        T_train_yr=T_train_yr, use_oplearn=True,
    )


# =========================================================================
# Phase E-5: CapEx revision prediction
# =========================================================================


def run_capex_revision_prediction(
    panel: pd.DataFrame,
) -> dict:
    """E-5: Test if SMIM gaps predict future CapEx changes.

    Build gaps: gap_{i,t} = actual_{i,t} - predicted_{i,t} (from SMIM)
    Regression: Delta_y_{i,t->t+4} = b0 + b1*gap_{i,t} + b2*y_{i,t} + eps
    Report: b1, t-stat with actor-clustered standard errors.
    """
    # Run full pipeline on all windows to collect gaps
    all_gaps = []  # (ticker_idx, quarter_idx, gap, y_current)
    all_actual_changes = []  # Delta_y_{t->t+4}

    # Get raw data for computing future changes
    raw_vals = panel.values.astype(np.float64)
    quarter_dates = panel.index
    tickers = panel.columns

    for ty in TEST_YEARS:
        res = run_diamond_window(
            panel, ty, K=BASE_K, ewm_hl=BASE_EWM,
            T_train_yr=BASE_T_TRAIN, use_oplearn=True,
        )
        if res is None:
            continue

        test_quarters = pd.date_range(
            f"{ty}-01-01", f"{ty}-12-31", freq="QS",
        )

        preds = res["preds"]
        actuals = res["actuals"]

        # Match test quarters to panel
        train_start = pd.Timestamp(f"{ty - BASE_T_TRAIN}-01-01")
        test_end = pd.Timestamp(f"{ty}-12-31")
        all_data = panel[
            (panel.index >= train_start) & (panel.index <= test_end)
        ].copy()
        valid = all_data.columns[all_data.notna().any()]

        for q_idx, q_date in enumerate(test_quarters):
            if q_idx >= len(preds):
                break

            # Find this quarter in the panel
            panel_q_idx = np.searchsorted(quarter_dates, q_date)
            if panel_q_idx >= len(quarter_dates):
                continue

            # 4-quarter-ahead change
            future_q_idx = panel_q_idx + 4
            if future_q_idx >= len(quarter_dates):
                continue

            gap = actuals[q_idx] - preds[q_idx]
            y_current = actuals[q_idx]
            y_future = raw_vals[future_q_idx]

            # Valid tickers in this window
            valid_idx = np.array([
                i for i, t in enumerate(valid)
                if t in tickers
            ])

            # Map back to panel column indices
            for local_j, ticker in enumerate(valid):
                if local_j >= len(gap):
                    break
                panel_col_idx = list(tickers).index(ticker)
                y_fut = y_future[panel_col_idx]
                y_cur = y_current[local_j]

                if np.isfinite(y_fut) and np.isfinite(y_cur) and np.isfinite(gap[local_j]):
                    delta_y = y_fut - y_cur
                    all_gaps.append({
                        "ticker_idx": panel_col_idx,
                        "quarter_idx": panel_q_idx,
                        "gap": gap[local_j],
                        "y_current": y_cur,
                        "delta_y": delta_y,
                    })

    if not all_gaps:
        return {"beta1": np.nan, "t_stat": np.nan, "n_obs": 0, "n_actors": 0}

    df = pd.DataFrame(all_gaps)

    # OLS regression: delta_y = b0 + b1*gap + b2*y_current + eps
    n = len(df)
    X = np.column_stack([
        np.ones(n),
        df["gap"].values,
        df["y_current"].values,
    ])
    y = df["delta_y"].values

    with np.errstate(invalid="ignore"):
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"beta1": np.nan, "t_stat": np.nan, "n_obs": n, "n_actors": 0}

    residuals = y - X @ beta

    # Actor-clustered standard errors
    n_actors = df["ticker_idx"].nunique()
    actors = df["ticker_idx"].values
    unique_actors = np.unique(actors)
    p = X.shape[1]

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {"beta1": float(beta[1]), "t_stat": np.nan, "n_obs": n, "n_actors": n_actors}

    # Cluster-robust sandwich estimator
    meat = np.zeros((p, p))
    for actor in unique_actors:
        mask = actors == actor
        X_g = X[mask]
        e_g = residuals[mask]
        score_g = X_g.T @ e_g  # (p,)
        meat += np.outer(score_g, score_g)

    # Small-sample adjustment: G/(G-1) * (n-1)/(n-p)
    G = len(unique_actors)
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    V_cluster = adj * XtX_inv @ meat @ XtX_inv

    se_cluster = np.sqrt(np.maximum(np.diag(V_cluster), 1e-16))
    t_stats = beta / se_cluster

    return {
        "beta0": float(beta[0]),
        "beta1_gap": float(beta[1]),
        "beta2_y": float(beta[2]),
        "se_beta1": float(se_cluster[1]),
        "t_stat_beta1": float(t_stats[1]),
        "n_obs": n,
        "n_actors": n_actors,
        "n_clusters": G,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    t_total = time.time()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  ITERATION 5.1 PHASES D-E: Operator Improvements + Ablation + Economic Validation")
    print("=" * 90)

    # Build panel once
    panel = build_capex_revenue_panel()
    print(f"\n  Panel: {panel.shape[0]} quarters x {panel.shape[1]} actors")

    # ==================================================================
    # First, get the base result for comparison
    # ==================================================================
    print("\n" + "-" * 90)
    print("  BASE RESULT (K=3, EWM=12, T=3yr, standard OpLearn)")
    print("-" * 90)

    base_rows = []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_window(
            panel, ty, K=BASE_K, ewm_hl=BASE_EWM,
            T_train_yr=BASE_T_TRAIN, use_oplearn=True,
        )
        ar1 = run_ar1_window(panel, ty, T_train_yr=BASE_T_TRAIN)
        if res is None:
            print(f"  W{ty}: FAILED")
            continue
        delta = res["smim_r2"] - ar1["ar1_r2"]
        print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1={ar1['ar1_r2']:.4f}  "
              f"D={delta:+.4f}  ({time.time()-t0:.1f}s)")
        base_rows.append({
            "year": ty, "smim_r2": res["smim_r2"],
            "ar1_r2": ar1["ar1_r2"], "delta": delta,
        })

    df_base = pd.DataFrame(base_rows)
    base_mean_r2 = df_base["smim_r2"].mean()
    base_mean_delta = df_base["delta"].mean()
    base_wins = int((df_base["delta"] > 0).sum())
    print(f"\n  Base: mean SMIM R²={base_mean_r2:.4f}  "
          f"mean delta={base_mean_delta:+.4f}  wins={base_wins}/{len(df_base)}")

    # ==================================================================
    # PHASE D: OPERATOR LEARNING IMPROVEMENTS
    # ==================================================================
    print("\n" + "=" * 90)
    print("  PHASE D: OPERATOR LEARNING IMPROVEMENTS")
    print("  (stop after first non-improvement)")
    print("=" * 90)

    phase_d_results = []
    experiments = [
        ("D-1", "ridge", "Ridge penalty (lambda=0.1)"),
        ("D-2", "budget300", "Nelder-Mead budget 300 iters"),
        ("D-3", "proximity", "Intensity-proximity operator"),
        ("D-5", "diffevo", "Differential evolution optimiser"),
    ]

    stop_flag = False
    for exp_id, op_func, description in experiments:
        if stop_flag:
            print(f"\n  {exp_id}: SKIPPED (stopping after non-improvement)")
            phase_d_results.append({
                "experiment": exp_id, "op_func": op_func,
                "description": description, "status": "skipped",
                "mean_smim_r2": np.nan, "mean_delta": np.nan,
                "wins": np.nan, "improvement_pp": np.nan,
            })
            continue

        print(f"\n  {exp_id}: {description}")
        print("  " + "-" * 60)

        exp_rows = []
        for ty in TEST_YEARS:
            t0 = time.time()
            res = run_diamond_window_custom_op(
                panel, ty, K=BASE_K, ewm_hl=BASE_EWM,
                T_train_yr=BASE_T_TRAIN, op_func=op_func,
            )
            ar1 = run_ar1_window(panel, ty, T_train_yr=BASE_T_TRAIN)
            if res is None:
                print(f"    W{ty}: FAILED")
                continue
            delta = res["smim_r2"] - ar1["ar1_r2"]
            print(f"    W{ty}: SMIM={res['smim_r2']:.4f}  "
                  f"D={delta:+.4f}  ({time.time()-t0:.1f}s)")
            exp_rows.append({
                "year": ty, "smim_r2": res["smim_r2"],
                "ar1_r2": ar1["ar1_r2"], "delta": delta,
            })

        if not exp_rows:
            print(f"  {exp_id}: ALL WINDOWS FAILED")
            phase_d_results.append({
                "experiment": exp_id, "op_func": op_func,
                "description": description, "status": "failed",
                "mean_smim_r2": np.nan, "mean_delta": np.nan,
                "wins": np.nan, "improvement_pp": np.nan,
            })
            stop_flag = True
            continue

        df_exp = pd.DataFrame(exp_rows)
        exp_mean_r2 = df_exp["smim_r2"].mean()
        exp_mean_delta = df_exp["delta"].mean()
        exp_wins = int((df_exp["delta"] > 0).sum())
        improvement_pp = (exp_mean_r2 - base_mean_r2) * 100

        improved = exp_mean_r2 > base_mean_r2
        status = "improved" if improved else "not_improved"

        print(f"\n  {exp_id} Summary: mean R²={exp_mean_r2:.4f}  "
              f"delta={exp_mean_delta:+.4f}  wins={exp_wins}/{len(df_exp)}")
        print(f"  vs Base: {improvement_pp:+.1f}pp  "
              f"{'IMPROVEMENT' if improved else 'NO IMPROVEMENT'}")

        phase_d_results.append({
            "experiment": exp_id, "op_func": op_func,
            "description": description, "status": status,
            "mean_smim_r2": exp_mean_r2, "mean_delta": exp_mean_delta,
            "wins": exp_wins, "improvement_pp": improvement_pp,
        })

        if not improved:
            print(f"\n  >>> STOPPING: {exp_id} did not improve over base")
            stop_flag = True

    # Phase D summary
    df_d = pd.DataFrame(phase_d_results)
    print("\n" + "-" * 90)
    print("  PHASE D SUMMARY")
    print("-" * 90)
    for _, row in df_d.iterrows():
        if row["status"] == "skipped":
            print(f"  {row['experiment']}: {row['description']} -- SKIPPED")
        elif row["status"] == "failed":
            print(f"  {row['experiment']}: {row['description']} -- FAILED")
        else:
            print(f"  {row['experiment']}: {row['description']} -- "
                  f"R²={row['mean_smim_r2']:.4f}  "
                  f"({row['improvement_pp']:+.1f}pp vs base)  "
                  f"{row['status'].upper()}")

    df_d.to_parquet(METRICS_DIR / "iter5_1_phase_d.parquet", index=False)
    print(f"\n  Saved: iter5_1_phase_d.parquet")

    # ==================================================================
    # PHASE E: ABLATION + ECONOMIC VALIDATION
    # ==================================================================
    print("\n" + "=" * 90)
    print("  PHASE E: ABLATION LADDER + ECONOMIC VALIDATION")
    print("=" * 90)

    # E-1: Full ablation ladder
    print("\n  E-1: Full Ablation Ladder")
    print("  " + "-" * 60)

    ablation_steps = [
        ("Step 0", "Constant mean", run_ablation_step0),
        ("Step 1", "+ EWM demeaning", run_ablation_step1),
        ("Step 2", "+ DMD basis (OLS)", run_ablation_step2),
        ("Step 3", "+ Kalman (spherical R)", run_ablation_step3),
        ("Step 4", "+ Online Q adaptation", run_ablation_step4),
        ("Step 5", "+ Rolling basis", run_ablation_step5),
        ("Step 6", "+ Operator learning", run_ablation_step6),
    ]

    ablation_rows = []
    for step_id, step_desc, step_func in ablation_steps:
        t0 = time.time()
        step_r2s = []
        for ty in TEST_YEARS:
            if step_id in ("Step 5", "Step 6"):
                # These return dict with "smim_r2" key
                result = step_func(panel, ty)
                if result is not None:
                    step_r2s.append(result["smim_r2"])
            else:
                result = step_func(panel, ty)
                if result is not None:
                    step_r2s.append(result["r2"])

        if step_r2s:
            mean_r2 = np.mean(step_r2s)
            elapsed = time.time() - t0
            print(f"  {step_id} ({step_desc}): R²={mean_r2:.4f}  "
                  f"({len(step_r2s)} windows, {elapsed:.1f}s)")
            ablation_rows.append({
                "step": step_id, "description": step_desc,
                "mean_r2": mean_r2, "n_windows": len(step_r2s),
            })
        else:
            print(f"  {step_id} ({step_desc}): ALL FAILED")
            ablation_rows.append({
                "step": step_id, "description": step_desc,
                "mean_r2": np.nan, "n_windows": 0,
            })

    # Print ablation table
    df_ablation = pd.DataFrame(ablation_rows)
    print("\n  Ablation Ladder:")
    print(f"  {'Step':<10} {'Description':<30} {'R^2':>8} {'Marginal':>10}")
    print("  " + "-" * 62)
    prev_r2 = 0.0
    for _, row in df_ablation.iterrows():
        r2 = row["mean_r2"]
        if np.isfinite(r2):
            marginal = r2 - prev_r2
            print(f"  {row['step']:<10} {row['description']:<30} "
                  f"{r2:>8.4f} {marginal:>+10.4f}")
            prev_r2 = r2
        else:
            print(f"  {row['step']:<10} {row['description']:<30} "
                  f"{'FAIL':>8} {'':>10}")

    # Also add AR(1) baseline for reference
    ar1_r2s = []
    for ty in TEST_YEARS:
        ar1 = run_ar1_window(panel, ty, T_train_yr=BASE_T_TRAIN)
        if ar1 is not None:
            ar1_r2s.append(ar1["ar1_r2"])
    if ar1_r2s:
        ar1_mean = np.mean(ar1_r2s)
        print(f"\n  Reference: AR(1) T=3yr mean R² = {ar1_mean:.4f}")

    # E-5: CapEx revision prediction
    print("\n  E-5: CapEx Revision Prediction")
    print("  " + "-" * 60)
    print("  Running: gap_{i,t} -> Delta_y_{i,t->t+4}")
    t0 = time.time()
    revision = run_capex_revision_prediction(panel)
    elapsed = time.time() - t0

    print(f"\n  Regression: Delta_y = b0 + b1*gap + b2*y_current + eps")
    print(f"    b1 (gap coeff):        {revision['beta1_gap']:.4f}")
    print(f"    SE (actor-clustered):   {revision.get('se_beta1', np.nan):.4f}")
    print(f"    t-stat:                 {revision['t_stat_beta1']:.2f}")
    if abs(revision['t_stat_beta1']) > 1.96:
        sig = "YES (|t| > 1.96)"
    else:
        sig = "NO (|t| < 1.96)"
    print(f"    Significant at 5%:      {sig}")
    print(f"    N obs:                  {revision['n_obs']}")
    print(f"    N clusters (actors):    {revision.get('n_clusters', 'N/A')}")
    print(f"    ({elapsed:.1f}s)")

    # Interpretation
    if revision["beta1_gap"] < 0 and abs(revision["t_stat_beta1"]) > 1.96:
        print("\n  Interpretation: NEGATIVE b1 is expected -- positive gaps")
        print("  (actual > predicted = over-investment) predict LOWER future intensity.")
        print("  This supports mean-reversion of investment gaps.")
    elif revision["beta1_gap"] > 0 and abs(revision["t_stat_beta1"]) > 1.96:
        print("\n  Interpretation: POSITIVE b1 -- gaps are persistent.")
        print("  Over-investing firms continue to over-invest.")

    # ==================================================================
    # Save Phase E results
    # ==================================================================
    phase_e_data = {
        "ablation": ablation_rows,
        "revision": revision,
    }

    # Save as flat parquet
    e_rows = []
    for row in ablation_rows:
        e_rows.append({
            "experiment": "E-1",
            "sub_id": row["step"],
            "description": row["description"],
            "metric": "mean_r2",
            "value": row["mean_r2"],
            "n_windows": row["n_windows"],
        })
    e_rows.append({
        "experiment": "E-5",
        "sub_id": "revision",
        "description": "CapEx revision prediction",
        "metric": "beta1_gap",
        "value": revision["beta1_gap"],
        "n_windows": revision["n_obs"],
    })
    e_rows.append({
        "experiment": "E-5",
        "sub_id": "revision",
        "description": "CapEx revision prediction",
        "metric": "t_stat_beta1",
        "value": revision["t_stat_beta1"],
        "n_windows": revision["n_obs"],
    })
    e_rows.append({
        "experiment": "E-5",
        "sub_id": "revision",
        "description": "CapEx revision prediction",
        "metric": "se_beta1",
        "value": revision.get("se_beta1", np.nan),
        "n_windows": revision["n_obs"],
    })
    e_rows.append({
        "experiment": "E-5",
        "sub_id": "revision",
        "description": "CapEx revision prediction",
        "metric": "n_clusters",
        "value": float(revision.get("n_clusters", 0)),
        "n_windows": revision["n_obs"],
    })

    df_e = pd.DataFrame(e_rows)
    df_e.to_parquet(METRICS_DIR / "iter5_1_phase_e.parquet", index=False)
    print(f"\n  Saved: iter5_1_phase_e.parquet")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print(f"\n  Base: SMIM R²={base_mean_r2:.4f}  delta={base_mean_delta:+.4f}  "
          f"wins={base_wins}/{len(df_base)}")

    # Best Phase D result
    ran = df_d[df_d["status"].isin(["improved", "not_improved"])]
    if not ran.empty:
        best_d = ran.loc[ran["mean_smim_r2"].idxmax()]
        print(f"\n  Best D: {best_d['experiment']} ({best_d['description']})")
        print(f"    R²={best_d['mean_smim_r2']:.4f}  "
              f"({best_d['improvement_pp']:+.1f}pp)")

    # Ablation key findings
    if not df_ablation.empty and df_ablation["mean_r2"].notna().any():
        last_valid = df_ablation.dropna(subset=["mean_r2"]).iloc[-1]
        first_valid = df_ablation.dropna(subset=["mean_r2"]).iloc[0]
        total_lift = last_valid["mean_r2"] - first_valid["mean_r2"]
        print(f"\n  Ablation: total lift from {first_valid['step']} to "
              f"{last_valid['step']} = {total_lift:+.4f}")

    # Economic validation
    print(f"\n  CapEx revision: b1={revision['beta1_gap']:.4f}  "
          f"t={revision['t_stat_beta1']:.2f}  "
          f"(n={revision['n_obs']})")

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print(f"  Saved: iter5_1_phase_d.parquet, iter5_1_phase_e.parquet")


if __name__ == "__main__":
    main()
