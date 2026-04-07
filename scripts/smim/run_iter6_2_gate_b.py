#!/usr/bin/env python
"""
Iteration 6.2 Gate B — Regime-Specific Tests (93-actor panel only).

Run after Gate A regardless of outcome.  Even if DMD loses at h=1/T=5yr,
it may win under stress conditions.

Tests:
  B1: Training-window sweep  T ∈ {2, 3, 5, 8} years
  B2: Multi-horizon sweep    h ∈ {1, 2, 4} quarters
  B3: Refit-frequency robustness  {quarterly, annual, no-refit}

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_2_gate_b.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_DEFAULT = 8
K_MAX = 15


# ══════════════════════════════════════════════════════════════════════
#  Shared infrastructure (from Gate A)
# ══════════════════════════════════════════════════════════════════════

def load_93_actor_panel():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    layer_map = {a["actor_id"]: a["layer"] for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def ar1_baseline(otr, N):
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c
    return mu, rho


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_t_test(deltas):
    d = np.array([x for x in deltas if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    mean_d = d.mean()
    se_d = d.std(ddof=1) / np.sqrt(len(d))
    if se_d < 1e-15:
        return np.nan, np.nan
    t_stat = mean_d / se_d
    p_val = 2 * t_dist.sf(abs(t_stat), df=len(d) - 1)
    return float(t_stat), float(p_val)


def _clip_sr(F, max_sr=0.99):
    eigvals = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(eigvals)))
    return F * (max_sr / mx) if mx > max_sr else F


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 10:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def dmd_full(dm, k_svd=K_MAX):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def _mean_valid(lst):
    vals = [x for x in lst if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def _pca_basis(dm, K):
    N = dm.shape[1]
    C = dm.T @ dm / dm.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    ka = min(K, N - 2)
    return eigvecs[:, idx][:, :ka], ka


# ══════════════════════════════════════════════════════════════════════
#  B1: Training-Window Sweep — T ∈ {2, 3, 5, 8}
# ══════════════════════════════════════════════════════════════════════

def run_window_5models(panel, ty, T_yr=5, K=K_DEFAULT, ewm=12, refit="quarterly"):
    """Run 5 key models for one window. Supports variable T_yr and refit mode.

    Models: AR(1), PCA+diag, DMD+diag, Ridge, DMD+full Ã
    Refit modes: quarterly (default), annual (no rolling in test year),
                 none (fixed from initial training).

    Returns dict of model→R², or None.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # ── Stage 1 & baselines ──
    rho_pool, bar_y = estimate_pooled_ar1(otr)
    mu_ar1, rho_ar1 = ar1_baseline(otr, N)
    predicted_train = bar_y + rho_pool * (otr[:-1] - bar_y)
    residuals = otr[1:] - predicted_train

    # ── Residual bases ──
    om_r = ewm_demean(residuals, ewm)
    dm_r = residuals - om_r

    # PCA
    U_pca, ka_pca = _pca_basis(dm_r, K)
    fac_pca = dm_r @ U_pca
    phi_pca = np.zeros(ka_pca)
    for k in range(ka_pca):
        f = fac_pca[:, k]
        if len(f) >= 3 and np.std(f[:-1]) > 1e-10:
            c = np.corrcoef(f[:-1], f[1:])[0, 1]
            if np.isfinite(c):
                phi_pca[k] = np.clip(c, -0.99, 0.99)

    # DMD
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None
    ka_dmd = min(K, mf_r.basis.shape[0] - 2, mf_r.K)
    U_dmd = mf_r.metadata["U"][:, :ka_dmd]
    A_dmd_full = mf_r.metadata["Atilde"][:ka_dmd, :ka_dmd].real.copy()
    A_dmd_diag = np.diag(np.diag(A_dmd_full))

    # Ridge on raw residuals
    N_r = dm_r.shape[1]
    X_rr, Y_rr = dm_r[:-1], dm_r[1:]
    ridge_alpha = max(1.0, N_r * 0.1)
    try:
        C_ridge = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha * np.eye(N_r),
                                    X_rr.T @ Y_rr)).T
    except np.linalg.LinAlgError:
        C_ridge = np.eye(N_r) * 0.5

    # Kalman states for DMD models
    a_diag, P_diag = np.zeros(ka_dmd), np.eye(ka_dmd)
    Q_diag = np.eye(ka_dmd) * Q_INIT_SCALE
    a_full, P_full = np.zeros(ka_dmd), np.eye(ka_dmd)
    Q_full = np.eye(ka_dmd) * Q_INIT_SCALE
    F_diag = _clip_sr(A_dmd_diag)
    F_full = _clip_sr(A_dmd_full.copy())
    R_dmd = sph_r(dm_r, U_dmd)

    # PCA Kalman state
    a_pca, P_pca = np.zeros(ka_pca), np.eye(ka_pca)
    Q_pca = np.eye(ka_pca) * Q_INIT_SCALE
    F_pca = np.diag(phi_pca)
    R_pca = sph_r(dm_r, U_pca)

    # Accumulators
    models = ["ar1", "pca_diag", "dmd_diag", "dmd_full", "ridge"]
    preds_map = {m: [] for m in models}
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    prev_resid = np.zeros(N)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_ar = bar_y + rho_pool * (prev - bar_y)

        # AR(1)
        preds_map["ar1"].append(mu_ar1 + rho_ar1 * (prev - mu_ar1))

        # PCA+diag Kalman
        ap_pca = F_pca @ a_pca
        Pp_pca = F_pca @ P_pca @ F_pca.T + Q_pca
        pred_pca = U_pca @ ap_pca + om_r.ravel()
        if not np.all(np.isfinite(pred_pca)):
            pred_pca = np.zeros(N)
        preds_map["pca_diag"].append(y_ar + pred_pca)

        # DMD+diag Kalman
        ap_diag = F_diag @ a_diag
        Pp_diag = F_diag @ P_diag @ F_diag.T + Q_diag
        pred_diag = U_dmd @ ap_diag + om_r.ravel()
        if not np.all(np.isfinite(pred_diag)):
            pred_diag = np.zeros(N)
        preds_map["dmd_diag"].append(y_ar + pred_diag)

        # DMD+full Ã Kalman
        ap_full = F_full @ a_full
        Pp_full = F_full @ P_full @ F_full.T + Q_full
        pred_full = U_dmd @ ap_full + om_r.ravel()
        if not np.all(np.isfinite(pred_full)):
            pred_full = np.zeros(N)
        preds_map["dmd_full"].append(y_ar + pred_full)

        # Ridge on residuals
        pred_ridge = C_ridge @ (prev_resid - om_r.ravel()) + om_r.ravel()
        if not np.all(np.isfinite(pred_ridge)):
            pred_ridge = np.zeros(N)
        preds_map["ridge"].append(y_ar + pred_ridge)

        actuals.append(obs)
        actual_resid = obs - y_ar
        odm_r = actual_resid - om_r.ravel()

        # Kalman updates
        for a_ref, P_ref, Q_ref, F_ref, U_ref, R_ref, ka_ref, label in [
            (a_pca, P_pca, Q_pca, F_pca, U_pca, R_pca, ka_pca, "pca"),
            (a_diag, P_diag, Q_diag, F_diag, U_dmd, R_dmd, ka_dmd, "diag"),
            (a_full, P_full, Q_full, F_full, U_dmd, R_dmd, ka_dmd, "full"),
        ]:
            ap = F_ref @ a_ref
            Pp = F_ref @ P_ref @ F_ref.T + Q_ref
            S = U_ref @ Pp @ U_ref.T + R_ref
            try:
                Kg = Pp @ U_ref.T @ np.linalg.solve(S, np.eye(N))
            except Exception:
                Kg = np.zeros((ka_ref, N))
            a_new = ap + Kg @ (odm_r - U_ref @ ap)
            P_new = (np.eye(ka_ref) - Kg @ U_ref) @ Pp
            inn = a_new - ap
            Q_new = (1 - LAMBDA_Q) * Q_ref + LAMBDA_Q * np.outer(inn, inn)
            Q_new = (Q_new + Q_new.T) / 2 + np.eye(ka_ref) * 1e-6
            if label == "pca":
                a_pca, P_pca, Q_pca = a_new, P_new, Q_new
            elif label == "diag":
                a_diag, P_diag, Q_diag = a_new, P_new, Q_new
            else:
                a_full, P_full, Q_full = a_new, P_new, Q_new

        prev_resid = actual_resid
        prev = obs

        # ── Rolling update (skip for non-quarterly refit) ──
        if refit == "quarterly":
            otr = np.vstack([otr, qv])
            rho_pool, bar_y = estimate_pooled_ar1(otr)
            mu_ar1, rho_ar1 = ar1_baseline(otr, N)
            residuals_new = otr[1:] - (bar_y + rho_pool * (otr[:-1] - bar_y))
            om_r = ewm_demean(residuals_new, ewm)
            dm_r = residuals_new - om_r

            # Re-estimate PCA
            U_pca_new, ka_pca_new = _pca_basis(dm_r, K)
            fac_pca_new = dm_r @ U_pca_new
            phi_pca_new = np.zeros(ka_pca_new)
            for k in range(ka_pca_new):
                f = fac_pca_new[:, k]
                if len(f) >= 3 and np.std(f[:-1]) > 1e-10:
                    c = np.corrcoef(f[:-1], f[1:])[0, 1]
                    if np.isfinite(c):
                        phi_pca_new[k] = np.clip(c, -0.99, 0.99)
            U_pca = U_pca_new; ka_pca = ka_pca_new; phi_pca = phi_pca_new
            R_pca = sph_r(dm_r, U_pca)
            a_pca = U_pca.T @ (actual_resid - om_r.ravel())
            P_pca = np.eye(ka_pca); Q_pca = np.eye(ka_pca) * Q_INIT_SCALE
            F_pca = np.diag(phi_pca)

            # Re-estimate DMD
            mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
            if mf_r2 is not None:
                ka_dmd_new = min(K, mf_r2.basis.shape[0] - 2, mf_r2.K)
                U_dmd = mf_r2.metadata["U"][:, :ka_dmd_new]
                A_dmd_full = mf_r2.metadata["Atilde"][:ka_dmd_new, :ka_dmd_new].real.copy()
                A_dmd_diag = np.diag(np.diag(A_dmd_full))
                ka_dmd = ka_dmd_new; R_dmd = sph_r(dm_r, U_dmd)
                F_diag = _clip_sr(A_dmd_diag)
                F_full = _clip_sr(A_dmd_full.copy())
                a_diag = U_dmd.T @ (actual_resid - om_r.ravel())
                P_diag = np.eye(ka_dmd); Q_diag = np.eye(ka_dmd) * Q_INIT_SCALE
                a_full = U_dmd.T @ (actual_resid - om_r.ravel())
                P_full = np.eye(ka_dmd); Q_full = np.eye(ka_dmd) * Q_INIT_SCALE

            # Re-estimate Ridge
            N_r = dm_r.shape[1]
            X_rr, Y_rr = dm_r[:-1], dm_r[1:]
            ridge_alpha = max(1.0, N_r * 0.1)
            try:
                C_ridge = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha * np.eye(N_r),
                                            X_rr.T @ Y_rr)).T
            except np.linalg.LinAlgError:
                pass
        elif refit == "annual":
            # Add data but do NOT refit models
            otr = np.vstack([otr, qv])
        # refit == "none": don't even add data

    if not actuals:
        return None
    act_a = np.array(actuals)
    r2s = {}
    for m in models:
        pa = np.array(preds_map[m])
        if pa.shape[0] == 0 or not np.all(np.isfinite(pa)):
            r2s[m] = None
        else:
            r2s[m] = float(oos_r_squared(pa.ravel(), act_a.ravel()))
    return r2s


# ══════════════════════════════════════════════════════════════════════
#  B2: Multi-Horizon Sweep — h ∈ {1, 2, 4}
# ══════════════════════════════════════════════════════════════════════

def run_window_multihorizon(panel, ty, h=1, K=K_DEFAULT, ewm=12, T_yr=5):
    """Run multi-step forecast at horizon h quarters.

    Models with h-step methods:
      DMD+diag: F^h · α (eigenvalue extrapolation)
      PCA+diag: Φ^h · α (persistence extrapolation)
      Ridge iterated: C^h · r (iterated one-step)
      Ridge direct h: separate β_h estimated on (x_t, y_{t+h}) pairs
      Lag-augmented Ridge: β · [r_t; r_{t-1}] → y_{t+h}

    For h=1, this reduces to the standard models.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # Stage 1
    rho_pool, bar_y = estimate_pooled_ar1(otr)
    mu_ar1, rho_ar1 = ar1_baseline(otr, N)
    predicted_train = bar_y + rho_pool * (otr[:-1] - bar_y)
    residuals = otr[1:] - predicted_train

    om_r = ewm_demean(residuals, ewm)
    dm_r = residuals - om_r

    # PCA
    U_pca, ka_pca = _pca_basis(dm_r, K)
    fac_pca = dm_r @ U_pca
    phi_pca = np.zeros(ka_pca)
    for k in range(ka_pca):
        f = fac_pca[:, k]
        if len(f) >= 3 and np.std(f[:-1]) > 1e-10:
            c = np.corrcoef(f[:-1], f[1:])[0, 1]
            if np.isfinite(c):
                phi_pca[k] = np.clip(c, -0.99, 0.99)

    # DMD
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None
    ka_dmd = min(K, mf_r.basis.shape[0] - 2, mf_r.K)
    U_dmd = mf_r.metadata["U"][:, :ka_dmd]
    A_dmd_full = mf_r.metadata["Atilde"][:ka_dmd, :ka_dmd].real.copy()
    A_dmd_diag = np.diag(np.diag(A_dmd_full))
    F_diag = _clip_sr(A_dmd_diag)

    # h-step DMD: F^h
    F_diag_h = np.linalg.matrix_power(F_diag, h)
    # h-step PCA: Φ^h
    Phi_pca_h = np.diag(phi_pca ** h)

    # Ridge on raw residuals: one-step C
    N_r = dm_r.shape[1]
    X_rr, Y_rr = dm_r[:-1], dm_r[1:]
    ridge_alpha = max(1.0, N_r * 0.1)
    try:
        C_ridge_1 = (np.linalg.solve(X_rr.T @ X_rr + ridge_alpha * np.eye(N_r),
                                      X_rr.T @ Y_rr)).T
    except np.linalg.LinAlgError:
        C_ridge_1 = np.eye(N_r) * 0.5

    # Ridge iterated: C^h
    C_ridge_h_iter = np.linalg.matrix_power(C_ridge_1, h)

    # Ridge direct h: estimate β_h on (x_t, y_{t+h}) pairs
    if dm_r.shape[0] > h + 1:
        X_h = dm_r[:-h]
        Y_h = dm_r[h:]
        try:
            C_ridge_h_direct = (np.linalg.solve(X_h.T @ X_h + ridge_alpha * np.eye(N_r),
                                                 X_h.T @ Y_h)).T
        except np.linalg.LinAlgError:
            C_ridge_h_direct = C_ridge_h_iter
    else:
        C_ridge_h_direct = C_ridge_h_iter

    # Lag-augmented Ridge: β · [r_t; r_{t-1}] → y_{t+h}
    if dm_r.shape[0] > h + 2:
        X_lag = np.hstack([dm_r[1:-h], dm_r[:-h-1]])  # [r_t; r_{t-1}]
        Y_lag = dm_r[1+h:]                             # y_{t+h}
        N2 = X_lag.shape[1]
        try:
            C_lag_aug = (np.linalg.solve(X_lag.T @ X_lag + ridge_alpha * np.eye(N2),
                                          X_lag.T @ Y_lag)).T
        except np.linalg.LinAlgError:
            C_lag_aug = None
    else:
        C_lag_aug = None

    # ── Rolling test (standard quarterly, h-step forecast) ──
    # For h>1: predict h quarters ahead from current position
    # We still iterate through quarters, but the target is h quarters ahead
    models = ["ar1", "pca_diag_h", "dmd_diag_h",
              "ridge_iter_h", "ridge_direct_h", "ridge_lag_aug_h"]
    preds_map = {m: [] for m in models}
    actuals_h = []

    # Collect all test-year data as a sequence
    test_data = []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] > 0:
            test_data.append(qv[0])

    if len(test_data) < h + 1:
        return None

    # For h-step: predict from quarter t to quarter t+h
    prev = np.nan_to_num(otr[-1], nan=0.5)
    prev_prev = np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5)

    for t_idx in range(len(test_data) - h):
        obs_future = test_data[t_idx + h]  # actual h steps ahead

        # Use prev as the latest known observation
        if t_idx == 0:
            current = prev
            prev_lag = prev_prev
        else:
            current = test_data[t_idx - 1]
            prev_lag = prev if t_idx == 1 else test_data[t_idx - 2]

        y_ar = bar_y + rho_pool * (current - bar_y)
        current_resid = current - (bar_y + rho_pool * (prev_lag - bar_y)) if t_idx > 0 else np.zeros(N)

        # AR(1): rho^h extrapolation
        preds_map["ar1"].append(mu_ar1 + (rho_ar1 ** h) * (current - mu_ar1))

        # PCA+diag h-step: Φ^h · α
        f_pca = U_pca.T @ (current_resid - om_r.ravel())
        pred_pca_h = U_pca @ (Phi_pca_h @ f_pca) + om_r.ravel()
        if not np.all(np.isfinite(pred_pca_h)):
            pred_pca_h = np.zeros(N)
        preds_map["pca_diag_h"].append(y_ar + pred_pca_h)

        # DMD+diag h-step: F^h · α
        f_dmd = U_dmd.T @ (current_resid - om_r.ravel())
        pred_dmd_h = U_dmd @ (F_diag_h @ f_dmd) + om_r.ravel()
        if not np.all(np.isfinite(pred_dmd_h)):
            pred_dmd_h = np.zeros(N)
        preds_map["dmd_diag_h"].append(y_ar + pred_dmd_h)

        # Ridge iterated h-step: C^h · r
        pred_ridge_iter = C_ridge_h_iter @ (current_resid - om_r.ravel()) + om_r.ravel()
        if not np.all(np.isfinite(pred_ridge_iter)):
            pred_ridge_iter = np.zeros(N)
        preds_map["ridge_iter_h"].append(y_ar + pred_ridge_iter)

        # Ridge direct h-step: β_h · r
        pred_ridge_direct = C_ridge_h_direct @ (current_resid - om_r.ravel()) + om_r.ravel()
        if not np.all(np.isfinite(pred_ridge_direct)):
            pred_ridge_direct = np.zeros(N)
        preds_map["ridge_direct_h"].append(y_ar + pred_ridge_direct)

        # Lag-augmented Ridge
        if C_lag_aug is not None:
            prev_resid = prev_lag - bar_y  # approximate previous residual
            lag_input = np.concatenate([current_resid - om_r.ravel(),
                                        prev_resid - om_r.ravel()])
            pred_lag = C_lag_aug @ lag_input + om_r.ravel()
            if not np.all(np.isfinite(pred_lag)):
                pred_lag = np.zeros(N)
        else:
            pred_lag = pred_ridge_direct  # fallback
        preds_map["ridge_lag_aug_h"].append(y_ar + pred_lag)

        actuals_h.append(obs_future)

    if not actuals_h:
        return None
    act_a = np.array(actuals_h)
    r2s = {}
    for m in models:
        pa = np.array(preds_map[m])
        if pa.shape[0] == 0 or not np.all(np.isfinite(pa)):
            r2s[m] = None
        else:
            r2s[m] = float(oos_r_squared(pa.ravel(), act_a.ravel()))
    return r2s


# ══════════════════════════════════════════════════════════════════════
#  B1 Runner
# ══════════════════════════════════════════════════════════════════════

def run_b1_t_sweep(panel):
    """B1: Training-window sweep T ∈ {2, 3, 5, 8}."""
    T_values = [2, 3, 5, 8]
    results = {}

    for T_yr in T_values:
        print(f"\n  T={T_yr}yr:")
        year_results = []
        # For shorter windows, skip early years that don't have enough data
        valid_years = [ty for ty in TEST_YEARS if ty - T_yr >= 2005]
        for ty in valid_years:
            t0 = time.time()
            r2s = run_window_5models(panel, ty, T_yr=T_yr, refit="quarterly")
            if r2s:
                year_results.append({"year": ty, **r2s})
                print(f"    W{ty}: AR1={r2s['ar1']:.4f}  PCA={r2s['pca_diag']:.4f}"
                      f"  DMD={r2s['dmd_diag']:.4f}  Ridge={r2s['ridge']:.4f}  ({time.time()-t0:.1f}s)")
            else:
                print(f"    W{ty}: FAILED  ({time.time()-t0:.1f}s)")
        results[T_yr] = year_results

    return results


# ══════════════════════════════════════════════════════════════════════
#  B2 Runner
# ══════════════════════════════════════════════════════════════════════

def run_b2_h_sweep(panel):
    """B2: Multi-horizon sweep h ∈ {1, 2, 4}."""
    H_values = [1, 2, 4]
    results = {}

    for h in H_values:
        print(f"\n  h={h}:")
        year_results = []
        for ty in TEST_YEARS:
            t0 = time.time()
            r2s = run_window_multihorizon(panel, ty, h=h, T_yr=5)
            if r2s:
                year_results.append({"year": ty, **r2s})
                dmd = r2s.get("dmd_diag_h", 0)
                pca = r2s.get("pca_diag_h", 0)
                ridge_d = r2s.get("ridge_direct_h", 0)
                print(f"    W{ty}: AR1={r2s.get('ar1',0):.4f}  PCA_h={pca:.4f}"
                      f"  DMD_h={dmd:.4f}  Ridge_d={ridge_d:.4f}  ({time.time()-t0:.1f}s)")
            else:
                print(f"    W{ty}: FAILED  ({time.time()-t0:.1f}s)")
        results[h] = year_results

    return results


# ══════════════════════════════════════════════════════════════════════
#  B3 Runner
# ══════════════════════════════════════════════════════════════════════

def run_b3_refit(panel):
    """B3: Refit-frequency robustness {quarterly, annual, none}."""
    refit_modes = ["quarterly", "annual", "none"]
    results = {}

    for mode in refit_modes:
        print(f"\n  Refit={mode}:")
        year_results = []
        for ty in TEST_YEARS:
            t0 = time.time()
            r2s = run_window_5models(panel, ty, T_yr=5, refit=mode)
            if r2s:
                year_results.append({"year": ty, **r2s})
                print(f"    W{ty}: DMD_diag={r2s['dmd_diag']:.4f}  PCA_diag={r2s['pca_diag']:.4f}"
                      f"  Ridge={r2s['ridge']:.4f}  ({time.time()-t0:.1f}s)")
            else:
                print(f"    W{ty}: FAILED  ({time.time()-t0:.1f}s)")
        results[mode] = year_results

    return results


# ══════════════════════════════════════════════════════════════════════
#  Analysis & Reporting
# ══════════════════════════════════════════════════════════════════════

def _extract_model(year_results, model):
    return [r.get(model) for r in year_results if r.get(model) is not None]


def print_b1_results(b1_results):
    print("\n" + "=" * 80)
    print("  B1 — TRAINING-WINDOW SWEEP: T ∈ {2, 3, 5, 8} years")
    print("=" * 80)

    print(f"\n  {'T':>4s} {'AR(1)':>7s} {'PCA+d':>7s} {'DMD+d':>7s} {'DMD full':>8s} {'Ridge':>7s}"
          f" {'DMD−PCA':>8s} {'DMD−Ridge':>10s}")
    print(f"  {'-'*64}")

    for T_yr, yr in sorted(b1_results.items()):
        ar1 = _mean_valid(_extract_model(yr, "ar1"))
        pca = _mean_valid(_extract_model(yr, "pca_diag"))
        dmd = _mean_valid(_extract_model(yr, "dmd_diag"))
        dmd_f = _mean_valid(_extract_model(yr, "dmd_full"))
        ridge = _mean_valid(_extract_model(yr, "ridge"))
        d_pca = dmd - pca if np.isfinite(dmd) and np.isfinite(pca) else np.nan
        d_ridge = dmd - ridge if np.isfinite(dmd) and np.isfinite(ridge) else np.nan
        print(f"  {T_yr:>4d} {ar1:7.4f} {pca:7.4f} {dmd:7.4f} {dmd_f:8.4f} {ridge:7.4f}"
              f" {d_pca:+8.4f} {d_ridge:+10.4f}")

    # Key comparison: DMD vs PCA at each T (paired)
    print(f"\n  DMD vs PCA at matched complexity (paired):")
    for T_yr, yr in sorted(b1_results.items()):
        dmd_vals = _extract_model(yr, "dmd_diag")
        pca_vals = _extract_model(yr, "pca_diag")
        if len(dmd_vals) >= 3 and len(pca_vals) >= 3:
            deltas = [d - p for d, p in zip(dmd_vals, pca_vals)]
            t_stat, p_val = paired_t_test(deltas)
            ci = bootstrap_ci(np.array(deltas))
            print(f"    T={T_yr}: Δ={np.mean(deltas):+.4f}  t={t_stat:.2f}  p={p_val:.4f}"
                  f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")


def print_b2_results(b2_results):
    print("\n" + "=" * 80)
    print("  B2 — MULTI-HORIZON SWEEP: h ∈ {1, 2, 4} quarters")
    print("=" * 80)

    for h, yr in sorted(b2_results.items()):
        ar1 = _mean_valid(_extract_model(yr, "ar1"))
        pca = _mean_valid(_extract_model(yr, "pca_diag_h"))
        dmd = _mean_valid(_extract_model(yr, "dmd_diag_h"))
        ridge_i = _mean_valid(_extract_model(yr, "ridge_iter_h"))
        ridge_d = _mean_valid(_extract_model(yr, "ridge_direct_h"))
        ridge_l = _mean_valid(_extract_model(yr, "ridge_lag_aug_h"))

        print(f"\n  h={h}:")
        print(f"    AR(1):            {ar1:.4f}")
        print(f"    PCA+diag h-step:  {pca:.4f}")
        print(f"    DMD+diag h-step:  {dmd:.4f}")
        print(f"    Ridge iterated:   {ridge_i:.4f}")
        print(f"    Ridge direct h:   {ridge_d:.4f}")
        print(f"    Ridge lag-aug h:  {ridge_l:.4f}")

        # DMD vs PCA at each h
        dmd_vals = _extract_model(yr, "dmd_diag_h")
        pca_vals = _extract_model(yr, "pca_diag_h")
        if len(dmd_vals) >= 3 and len(pca_vals) >= 3:
            deltas = [d - p for d, p in zip(dmd_vals, pca_vals)]
            t_stat, p_val = paired_t_test(deltas)
            ci = bootstrap_ci(np.array(deltas))
            print(f"    DMD−PCA: Δ={np.mean(deltas):+.4f}  t={t_stat:.2f}  p={p_val:.4f}"
                  f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

        # DMD vs best Ridge
        ridge_max_vals = [max(ri, rd, rl)
                          for ri, rd, rl in zip(
                              _extract_model(yr, "ridge_iter_h"),
                              _extract_model(yr, "ridge_direct_h"),
                              _extract_model(yr, "ridge_lag_aug_h"))]
        if len(dmd_vals) >= 3 and len(ridge_max_vals) >= 3:
            d2 = [d - r for d, r in zip(dmd_vals, ridge_max_vals[:len(dmd_vals)])]
            t2, p2 = paired_t_test(d2)
            ci2 = bootstrap_ci(np.array(d2))
            print(f"    DMD−Ridge(best): Δ={np.mean(d2):+.4f}  t={t2:.2f}  p={p2:.4f}"
                  f"  CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}]")


def print_b3_results(b3_results):
    print("\n" + "=" * 80)
    print("  B3 — REFIT-FREQUENCY ROBUSTNESS: {quarterly, annual, no-refit}")
    print("=" * 80)

    modes = ["quarterly", "annual", "none"]
    models = ["ar1", "pca_diag", "dmd_diag", "dmd_full", "ridge"]

    print(f"\n  {'Model':<12s}", end="")
    for mode in modes:
        print(f" {mode:>12s}", end="")
    print(f" {'Q−A degrad':>11s} {'Q−N degrad':>11s}")
    print(f"  {'-'*70}")

    for model in models:
        vals = {}
        for mode in modes:
            vals[mode] = _mean_valid(_extract_model(b3_results[mode], model))
        q = vals["quarterly"]
        a = vals["annual"]
        n = vals["none"]
        degrad_a = q - a if np.isfinite(q) and np.isfinite(a) else np.nan
        degrad_n = q - n if np.isfinite(q) and np.isfinite(n) else np.nan
        print(f"  {model:<12s}", end="")
        for mode in modes:
            v = vals[mode]
            print(f" {v:12.4f}" if np.isfinite(v) else "         N/A", end="")
        d_a = f"{degrad_a:+11.4f}" if np.isfinite(degrad_a) else "        N/A"
        d_n = f"{degrad_n:+11.4f}" if np.isfinite(degrad_n) else "        N/A"
        print(f" {d_a} {d_n}")

    # DMD degrades less than others?
    print(f"\n  Degradation comparison (quarterly − annual):")
    for model in models:
        q_vals = _extract_model(b3_results["quarterly"], model)
        a_vals = _extract_model(b3_results["annual"], model)
        if len(q_vals) >= 3 and len(a_vals) >= 3:
            deltas = [q - a for q, a in zip(q_vals, a_vals)]
            print(f"    {model:<12s}: mean degrad = {np.mean(deltas):+.4f}")


def print_error_correlations(b1_results):
    """Cross-condition error correlations."""
    print("\n" + "=" * 80)
    print("  FORECAST-ERROR CORRELATION BY REGIME")
    print("=" * 80)

    for T_yr, yr in sorted(b1_results.items()):
        dmd_vals = _extract_model(yr, "dmd_diag")
        pca_vals = _extract_model(yr, "pca_diag")
        ridge_vals = _extract_model(yr, "ridge")
        if len(dmd_vals) >= 3:
            # Use R² as proxy for error correlation (imperfect but informative)
            print(f"  T={T_yr}: DMD={_mean_valid(dmd_vals):.4f}  PCA={_mean_valid(pca_vals):.4f}"
                  f"  Ridge={_mean_valid(ridge_vals):.4f}")


def evaluate_kill_rule_b(b1_results, b2_results, b3_results):
    """Evaluate Kill Rule B."""
    print("\n" + "=" * 80)
    print("  KILL RULE B EVALUATION")
    print("=" * 80)

    any_dmd_win = False

    # B1: Check DMD vs PCA at short T
    for T_yr in [2, 3]:
        yr = b1_results.get(T_yr, [])
        dmd_vals = _extract_model(yr, "dmd_diag")
        pca_vals = _extract_model(yr, "pca_diag")
        if len(dmd_vals) >= 3 and len(pca_vals) >= 3:
            deltas = [d - p for d, p in zip(dmd_vals, pca_vals)]
            ci = bootstrap_ci(np.array(deltas))
            wins = ci[0] > 0
            any_dmd_win = any_dmd_win or wins
            print(f"  B1 T={T_yr}: DMD−PCA CI [{ci[0]:+.4f}, {ci[1]:+.4f}] {'WIN' if wins else 'no'}")

    # B2: Check DMD vs PCA at h > 1
    for h in [2, 4]:
        yr = b2_results.get(h, [])
        dmd_vals = _extract_model(yr, "dmd_diag_h")
        pca_vals = _extract_model(yr, "pca_diag_h")
        if len(dmd_vals) >= 3 and len(pca_vals) >= 3:
            deltas = [d - p for d, p in zip(dmd_vals, pca_vals)]
            ci = bootstrap_ci(np.array(deltas))
            wins = ci[0] > 0
            any_dmd_win = any_dmd_win or wins
            print(f"  B2 h={h}: DMD−PCA CI [{ci[0]:+.4f}, {ci[1]:+.4f}] {'WIN' if wins else 'no'}")

    # B3: Check DMD degrades less
    q_dmd = _extract_model(b3_results.get("quarterly", []), "dmd_diag")
    a_dmd = _extract_model(b3_results.get("annual", []), "dmd_diag")
    q_pca = _extract_model(b3_results.get("quarterly", []), "pca_diag")
    a_pca = _extract_model(b3_results.get("annual", []), "pca_diag")
    if len(q_dmd) >= 3 and len(a_dmd) >= 3 and len(q_pca) >= 3 and len(a_pca) >= 3:
        dmd_degrad = [q - a for q, a in zip(q_dmd, a_dmd)]
        pca_degrad = [q - a for q, a in zip(q_pca, a_pca)]
        diff_degrad = [d - p for d, p in zip(dmd_degrad, pca_degrad)]
        ci_d = bootstrap_ci(np.array(diff_degrad))
        dmd_more_robust = ci_d[1] < 0  # DMD degrades less
        any_dmd_win = any_dmd_win or dmd_more_robust
        print(f"  B3 annual: DMD degrad − PCA degrad CI [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]"
              f" {'DMD more robust' if dmd_more_robust else 'no robustness advantage'}")

    print(f"\n  ═══════════════════════════════════════════════")
    if any_dmd_win:
        print(f"  KILL RULE B: NOT TRIGGERED — DMD shows regime-specific advantage")
    else:
        print(f"  KILL RULE B: TRIGGERED")
        print(f"  DMD-specific forecasting claim is DEAD across all tested regimes.")
    print(f"  ═══════════════════════════════════════════════")

    return not any_dmd_win


# ══════════════════════════════════════════════════════════════════════
#  Save
# ══════════════════════════════════════════════════════════════════════

def save_results(b1_results, b2_results, b3_results):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # B1
    rows = []
    for T_yr, yr in b1_results.items():
        for r in yr:
            rows.append({"T_yr": T_yr, **r})
    if rows:
        pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_2_gate_b_b1.parquet", index=False)

    # B2
    rows = []
    for h, yr in b2_results.items():
        for r in yr:
            rows.append({"horizon": h, **r})
    if rows:
        pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_2_gate_b_b2.parquet", index=False)

    # B3
    rows = []
    for mode, yr in b3_results.items():
        for r in yr:
            rows.append({"refit": mode, **r})
    if rows:
        pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_2_gate_b_b3.parquet", index=False)

    print(f"\n  Saved: iter6_2_gate_b_{{b1,b2,b3}}.parquet")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 80)
    print("  ITERATION 6.2 GATE B — REGIME-SPECIFIC TESTS")
    print("  Training-window sweep, multi-horizon, refit robustness")
    print("=" * 80)

    panel = load_93_actor_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # B1: Training-window sweep
    print("\n" + "─" * 80)
    print("  B1: TRAINING-WINDOW SWEEP T ∈ {2, 3, 5, 8}")
    print("─" * 80)
    b1_results = run_b1_t_sweep(panel)

    # B2: Multi-horizon sweep
    print("\n" + "─" * 80)
    print("  B2: MULTI-HORIZON SWEEP h ∈ {1, 2, 4}")
    print("─" * 80)
    b2_results = run_b2_h_sweep(panel)

    # B3: Refit-frequency robustness
    print("\n" + "─" * 80)
    print("  B3: REFIT-FREQUENCY ROBUSTNESS")
    print("─" * 80)
    b3_results = run_b3_refit(panel)

    # ── Analysis ──
    print_b1_results(b1_results)
    print_b2_results(b2_results)
    print_b3_results(b3_results)
    print_error_correlations(b1_results)

    # ── Kill Rule B ──
    killed = evaluate_kill_rule_b(b1_results, b2_results, b3_results)

    # ── Save ──
    save_results(b1_results, b2_results, b3_results)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    return b1_results, b2_results, b3_results, killed


if __name__ == "__main__":
    main()
