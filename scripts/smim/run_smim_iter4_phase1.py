#!/usr/bin/env python
"""
SMIM Iteration 4 — Phase 1: Multi-Frequency Kalman Experiments.

Tests whether inter-quarter daily equity observations (projected through
quarterly spectral basis) improve quarterly prediction over DIAMOND.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter4_phase1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITY_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
MOMENTUM_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "iter3_panel_a_L20.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_USE = 5
F_Q = 0.99              # quarterly transition
F_D = 0.99 ** (1 / 63)  # daily transition (~0.99984)
Q_INIT = 0.5
LAMBDA_Q = 0.3
EWM_HL = 8
TEST_YEARS = list(range(2015, 2025))

# Firm actors that bridge quarterly <-> daily
with open(REGISTRY_PATH) as f:
    REG = json.load(f)
OHLCV_TICKERS = set(pd.read_parquet(OHLCV_PATH)["ticker"].unique())
FIRM_BRIDGE = sorted(
    a["actor_id"] for a in REG["actors"]
    if a["actor_type"] in ("large_firm", "bank", "sector_leader")
    and a["actor_id"] in OHLCV_TICKERS
)


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


def kf_update(obs, U, R, alpha_pred, P_pred):
    """Kalman measurement update. Returns alpha_filt, P_filt."""
    N, K = U.shape
    S = U @ P_pred @ U.T + R
    try:
        Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        return alpha_pred, P_pred
    innov = obs - U @ alpha_pred
    alpha_filt = alpha_pred + Kg @ innov
    P_filt = (np.eye(K) - Kg @ U) @ P_pred
    return alpha_filt, P_filt


def run_quarterly_only(q_panel, test_year, train_years=5):
    """DIAMOND-equivalent: quarterly-only rolling SMIM on bridged actors."""
    ts = pd.Timestamp(f"{test_year - train_years}-01-01")
    te_tr = pd.Timestamp(f"{test_year - 1}-12-31")
    te_te = pd.Timestamp(f"{test_year}-12-31")

    all_d = q_panel[(q_panel.index >= ts) & (q_panel.index <= te_te)].copy()
    all_d = all_d.fillna(all_d.mean())
    N = all_d.shape[1]

    train = all_d[all_d.index <= te_tr].values.astype(np.float64)
    test_dates = all_d.index[all_d.index > te_tr]
    test = all_d.loc[test_dates].values.astype(np.float64)
    if len(train) < 8 or len(test) == 0:
        return None

    mu = ewm_mean(train)
    dm = train - mu
    U = dmd_basis(dm, K_USE)
    if U is None:
        return None
    K = U.shape[1]
    R_Q = sph_r(dm, U)
    F = np.eye(K) * F_Q
    Q = np.eye(K) * Q_INIT

    alpha = np.zeros(K)
    P = np.eye(K)
    Q_run = Q.copy()

    preds = []
    actuals = []
    exp_data = train.copy()

    for i, td in enumerate(test_dates):
        obs = test[i]
        obs_dm = obs - mu.ravel()

        # Predict
        alpha_pred = F @ alpha
        P_pred = F @ P @ F.T + Q_run

        # Quarterly update
        alpha, P = kf_update(obs_dm, U, R_Q, alpha_pred, P_pred)

        # Online Q
        innov = alpha - F @ (alpha - (P @ U.T @ np.linalg.solve(U @ P @ U.T + R_Q, np.eye(N))) @ (obs_dm - U @ alpha_pred))
        si = alpha - alpha_pred
        Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(si, si)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8

        pred = U @ alpha_pred + mu.ravel()
        preds.append(pred)
        actuals.append(obs)

        # Rolling basis (expand each quarter)
        exp_data = np.vstack([exp_data, obs.reshape(1, -1)])
        mu = ewm_mean(exp_data)
        dm_new = exp_data - mu
        U_new = dmd_basis(dm_new, K_USE)
        if U_new is not None:
            K_new = U_new.shape[1]
            alpha = U_new.T @ obs_dm
            P = np.eye(K_new)
            Q_run = np.eye(K_new) * Q_INIT
            F = np.eye(K_new) * F_Q
            R_Q = sph_r(dm_new, U_new)
            U, K = U_new, K_new

    return float(oos_r_squared(np.array(preds).ravel(), np.array(actuals).ravel()))


def run_mf_kalman(q_panel, d_panel, test_year, train_years=5, use_daily=True,
                  daily_noise_mult=1.0):
    """Multi-frequency Kalman: quarterly anchors + daily equity updates."""
    ts = pd.Timestamp(f"{test_year - train_years}-01-01")
    te_tr = pd.Timestamp(f"{test_year - 1}-12-31")
    te_te = pd.Timestamp(f"{test_year}-12-31")

    # Quarterly data
    q_all = q_panel[(q_panel.index >= ts) & (q_panel.index <= te_te)].copy()
    q_all = q_all.fillna(q_all.mean())
    N_q = q_all.shape[1]

    q_train = q_all[q_all.index <= te_tr].values.astype(np.float64)
    q_test_dates = q_all.index[q_all.index > te_tr]
    q_test = q_all.loc[q_test_dates].values.astype(np.float64)
    if len(q_train) < 8 or len(q_test) == 0:
        return None

    # Quarterly basis + params
    mu_q = ewm_mean(q_train)
    dm_q = q_train - mu_q
    U = dmd_basis(dm_q, K_USE)
    if U is None:
        return None
    K = U.shape[1]
    R_Q = sph_r(dm_q, U)

    # Estimate W from training data (daily equity -> quarterly modes)
    if use_daily and d_panel is not None:
        d_train = d_panel[(d_panel.index >= ts) & (d_panel.index <= te_tr)]
        d_train = d_train.fillna(d_train.mean())
        if len(d_train) < 60:
            use_daily = False

    W = None
    R_D = None
    mu_d = None
    if use_daily and d_panel is not None:
        # Interpolate quarterly alpha to daily for W estimation
        alpha_q_train = dm_q @ U  # (T_q, K)
        alpha_q_df = pd.DataFrame(alpha_q_train, index=q_all[q_all.index <= te_tr].index)
        d_train_dates = d_train.index
        alpha_interp = alpha_q_df.reindex(
            alpha_q_df.index.union(d_train_dates)
        ).interpolate(method="time").reindex(d_train_dates).dropna()

        common_dates = alpha_interp.index.intersection(d_train.index)
        if len(common_dates) < 60:
            use_daily = False
        else:
            X = alpha_interp.loc[common_dates].values
            mu_d = d_train.loc[common_dates].mean().values
            Y = d_train.loc[common_dates].values - mu_d
            N_d = Y.shape[1]

            # W = Y^T X (X^T X + lam I)^-1
            lam = 1.0
            W = np.linalg.solve(X.T @ X + lam * np.eye(K), X.T @ Y).T  # (N_d, K)

            # R_D from residuals
            resid = Y - X @ W.T
            r_d = np.mean(resid ** 2) * daily_noise_mult
            R_D = np.eye(N_d) * max(r_d, 1e-8)

    # State init
    F = np.eye(K) * F_D  # DAILY transition
    Q_daily = np.eye(K) * (Q_INIT / 63)  # scale Q for daily
    alpha = np.zeros(K)
    P = np.eye(K)
    Q_run = Q_daily.copy()

    preds = []
    actuals = []
    exp_q = q_train.copy()
    q_idx = 0

    # Get daily dates for test year
    if use_daily and d_panel is not None:
        d_test = d_panel[(d_panel.index > te_tr) & (d_panel.index <= te_te)]
        d_test = d_test.fillna(d_test.mean())
        all_dates = sorted(set(q_test_dates) | set(d_test.index))
    else:
        all_dates = list(q_test_dates)

    for date in all_dates:
        # Daily predict step
        alpha_pred = F @ alpha
        P_pred = F @ P @ F.T + Q_run

        # Daily equity update (if available and enabled)
        if use_daily and d_panel is not None and date in d_test.index and W is not None:
            d_obs = d_test.loc[date].values - mu_d
            alpha_pred, P_pred = kf_update(d_obs, W, R_D, alpha_pred, P_pred)

            # Online Q at daily scale
            si = alpha_pred - F @ alpha
            Q_run = (1 - LAMBDA_Q / 10) * Q_run + (LAMBDA_Q / 10) * np.outer(si, si)
            Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-10

        # Quarterly update (if this is a quarterly date)
        if date in q_test_dates and q_idx < len(q_test):
            obs_q = q_test[q_idx]
            obs_q_dm = obs_q - mu_q.ravel()

            # Record prediction BEFORE quarterly update
            pred = U @ alpha_pred + mu_q.ravel()
            preds.append(pred)
            actuals.append(obs_q)

            # Large quarterly update
            alpha, P = kf_update(obs_q_dm, U, R_Q, alpha_pred, P_pred)

            # Online Q (quarterly scale)
            si = alpha - alpha_pred
            Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(si, si) / 63
            Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-10

            # Rolling basis update
            exp_q = np.vstack([exp_q, obs_q.reshape(1, -1)])
            mu_q_new = ewm_mean(exp_q)
            dm_new = exp_q - mu_q_new
            U_new = dmd_basis(dm_new, K_USE)
            if U_new is not None:
                K_new = U_new.shape[1]
                alpha = U_new.T @ obs_q_dm
                P = np.eye(K_new)
                Q_run = np.eye(K_new) * (Q_INIT / 63)
                F = np.eye(K_new) * F_D
                R_Q = sph_r(dm_new, U_new)
                U, K = U_new, K_new
                mu_q = mu_q_new

                # Re-estimate W with new basis if daily is active
                if use_daily and d_panel is not None:
                    alpha_q_new = dm_new @ U
                    aq_df = pd.DataFrame(alpha_q_new, index=q_all.index[:len(exp_q)])
                    d_all = d_panel[(d_panel.index >= ts) & (d_panel.index <= date)]
                    d_all = d_all.fillna(d_all.mean())
                    common = aq_df.index.union(d_all.index)
                    aq_interp = aq_df.reindex(common).interpolate("time").reindex(d_all.index).dropna()
                    cd = aq_interp.index.intersection(d_all.index)
                    if len(cd) > 60:
                        Xn = aq_interp.loc[cd].values
                        mu_d = d_all.loc[cd].mean().values
                        Yn = d_all.loc[cd].values - mu_d
                        N_d = Yn.shape[1]
                        W = np.linalg.solve(Xn.T @ Xn + np.eye(K_new), Xn.T @ Yn).T
                        resid = Yn - Xn @ W.T
                        R_D = np.eye(N_d) * max(np.mean(resid ** 2) * daily_noise_mult, 1e-8)

            q_idx += 1
        else:
            alpha = alpha_pred
            P = P_pred

    if not preds:
        return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(actuals).ravel()))


def main():
    t_total = time.time()
    print("=" * 80)
    print("  ITERATION 4 - Phase 1: Multi-Frequency Kalman")
    print("=" * 80)

    # Load quarterly panel for bridged actors
    q_int = pd.read_parquet(INTENSITY_PATH)
    q_wide = q_int.pivot_table(index="period", columns="actor_id", values="intensity_value")
    q_wide.index = pd.to_datetime(q_wide.index)
    q_wide = q_wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))

    available = [c for c in FIRM_BRIDGE if c in q_wide.columns]
    q_panel = q_wide[available]
    print(f"  Quarterly panel: {q_panel.shape[1]} bridged firm actors")

    # Load daily momentum panel
    d_int = pd.read_parquet(MOMENTUM_PATH)
    d_wide = d_int.pivot_table(index="period", columns="actor_id", values="intensity_value")
    d_wide.index = pd.to_datetime(d_wide.index)
    d_wide = d_wide.sort_index()
    common = sorted(set(available) & set(d_wide.columns))
    d_panel = d_wide[common]
    print(f"  Daily panel: {d_panel.shape[1]} common actors, {d_panel.shape[0]} days")

    rows = []

    # ---- MF-1: Quarterly-only baseline (DIAMOND equivalent) ----
    print(f"\n--- Quarterly-only (DIAMOND baseline) ---")
    for ty in TEST_YEARS:
        r2 = run_quarterly_only(q_panel[common], ty)
        if r2 is not None:
            print(f"  W{ty}: R2={r2:.4f}")
            rows.append({"year": ty, "model": "Q_only", "r2": r2})

    # ---- MF-1a: Q + daily equity ----
    print(f"\n--- MF: Quarterly + daily equity ---")
    for ty in TEST_YEARS:
        t0 = time.time()
        r2 = run_mf_kalman(q_panel[common], d_panel, ty, use_daily=True)
        if r2 is not None:
            print(f"  W{ty}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
            rows.append({"year": ty, "model": "MF_Q_daily", "r2": r2})

    # ---- MF with different noise multipliers ----
    print(f"\n--- MF: noise sensitivity (sigma_D multiplier) ---")
    for mult in [0.5, 2.0, 5.0]:
        r2s = []
        for ty in TEST_YEARS:
            r2 = run_mf_kalman(q_panel[common], d_panel, ty, use_daily=True, daily_noise_mult=mult)
            if r2 is not None:
                r2s.append(r2)
        if r2s:
            print(f"  mult={mult:.1f}: mean R2={np.mean(r2s):.4f}")
            for ty, r2 in zip(TEST_YEARS, r2s):
                rows.append({"year": ty, "model": f"MF_noise_{mult}", "r2": r2})

    # ---- MF-1f: Interpolation baseline ----
    print(f"\n--- Interpolation baseline ---")
    for ty in TEST_YEARS:
        r2 = run_mf_kalman(q_panel[common], d_panel, ty, use_daily=False)
        if r2 is not None:
            rows.append({"year": ty, "model": "Q_only_check", "r2": r2})

    # ---- Summary ----
    results = pd.DataFrame(rows)
    results.to_parquet(METRICS_DIR / "iter4_phase1_results.parquet", index=False)

    print(f"\n{'#'*80}")
    print(f"#  PHASE 1 SUMMARY")
    print(f"{'#'*80}")

    for model in results["model"].unique():
        sub = results[results["model"] == model]
        print(f"\n  {model:20s}: mean R2={sub['r2'].mean():.4f}  "
              f"std={sub['r2'].std():.4f}  n={len(sub)}")

    # Compare Q_only vs MF
    q_only = results[results["model"] == "Q_only"].set_index("year")["r2"]
    mf_daily = results[results["model"] == "MF_Q_daily"].set_index("year")["r2"]
    common_years = q_only.index.intersection(mf_daily.index)

    if len(common_years) > 0:
        delta = mf_daily.loc[common_years] - q_only.loc[common_years]
        wins = (delta > 0).sum()
        print(f"\n  MF vs Q-only:")
        print(f"    Mean delta R2: {delta.mean():+.5f}")
        print(f"    MF wins: {wins}/{len(delta)}")
        print(f"    Per window:")
        for y in common_years:
            d = delta.loc[y]
            w = "WIN" if d > 0 else "LOSS"
            print(f"      W{y}: Q={q_only.loc[y]:.4f}  MF={mf_daily.loc[y]:.4f}  "
                  f"delta={d:+.5f} {w}")

    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
