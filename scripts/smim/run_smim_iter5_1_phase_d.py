#!/usr/bin/env python
"""
Phase D: Quick operator improvement experiments.

Tests (in priority order, stops after first non-improvement):
D-1: Ridge penalty on operator weights
D-2: Increase Nelder-Mead budget (100 → 300)
D-3: Differential evolution optimizer

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1_phase_d.py
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

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))


# ── Shared functions (from iter5_1.py) ───────────────────────

def build_capex_revenue_panel() -> pd.DataFrame:
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    capex = edgar[edgar["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"][
        ["ticker", "event_date", "value"]
    ].copy()
    revenue = edgar[edgar["tag"] == "Revenues"][
        ["ticker", "event_date", "value"]
    ].copy()
    capex["q"] = capex["event_date"].dt.to_period("Q").dt.to_timestamp()
    revenue["q"] = revenue["event_date"].dt.to_period("Q").dt.to_timestamp()
    capex = capex.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    revenue = revenue.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    merged = capex.merge(revenue, on=["ticker", "q"], suffixes=("_capex", "_rev"))
    merged["ratio"] = merged["value_capex"] / merged["value_rev"]
    merged = merged.replace([np.inf, -np.inf], np.nan)
    panel = merged.pivot_table(index="q", columns="ticker", values="ratio")
    panel.index = pd.to_datetime(panel.index)
    ranked = panel.rank(axis=1, method="average", pct=True)
    coverage = ranked.notna().mean()
    good = coverage[coverage > 0.50].index
    return ranked[good].loc["2005-01-01":"2025-12-31"]


def ewm_demean(obs, halflife=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(obs_dm, k=3):
    N = obs_dm.shape[1]
    k_use = min(k, N - 2)
    if obs_dm.shape[0] < 3:
        return None, None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        eigs = mf.eigenvalues[:min(k_use, mf.K)]
        return U, eigs
    except Exception:
        return None, None


def train_sph_r(obs_dm, U):
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * max(np.mean(resid ** 2), 1e-8)


def build_operator_library(obs_dm, N):
    T = obs_dm.shape[0]
    library = []
    with np.errstate(invalid="ignore"):
        corr0 = np.corrcoef(obs_dm.T)
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 0.0)
    library.append(("corr_inst", corr0))
    if T > 2:
        lag1 = np.zeros((N, N))
        for lo in [1, 2]:
            if T > lo:
                xp, xn = obs_dm[:-lo], obs_dm[lo:]
                xp_dm, xn_dm = xp - xp.mean(0), xn - xn.mean(0)
                num = (xp_dm.T @ xn_dm) / max(len(xp) - 1, 1)
                sp_std = np.std(xp, 0, keepdims=True).T
                sn_std = np.std(xn, 0, keepdims=True)
                denom = np.maximum(sp_std @ sn_std, 1e-10)
                lag1 += np.nan_to_num(num / denom, nan=0.0)
        np.fill_diagonal(lag1, 0.0)
        library.append(("corr_lag", lag1))
    for scale in [4, 8]:
        if T >= scale + 1:
            kernel = np.ones(scale) / scale
            sm = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, obs_dm)
            if sm.shape[0] >= 2:
                norms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                normed = sm / norms
                sim = normed.T @ normed
                np.fill_diagonal(sim, 0.0)
                library.append((f"cosine_{scale}Q", sim))
    return library


# ── Operator optimizers (variants) ───────────────────────────

def _kalman_sv_r2(obs_st, obs_sv, U, N):
    """Quick Kalman sub-validation R²."""
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


def optimize_operator_base(obs_dm, library, N, K_opt=3, maxiter=100):
    """Baseline: same as iter5_1."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st, obs_sv = obs_dm[:t_split], obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = sum(w * B for w, (_, B) in zip(weights, library))
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0
            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6
            return _kalman_sv_r2(obs_st, obs_sv, U, N)
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.02, "fatol": 0.002})
    A_opt = sum(w * B for w, (_, B) in zip(opt.x, library))
    return np.eye(N) + 0.1 * A_opt, -opt.fun


def optimize_operator_ridge(obs_dm, library, N, K_opt=3, ridge_alpha=0.1, maxiter=100):
    """D-1: Ridge penalty on weights."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st, obs_sv = obs_dm[:t_split], obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = sum(w * B for w, (_, B) in zip(weights, library))
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0
            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6
            r2_loss = _kalman_sv_r2(obs_st, obs_sv, U, N)
            penalty = ridge_alpha * np.sum(weights ** 2)
            return r2_loss + penalty
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.02, "fatol": 0.002})
    A_opt = sum(w * B for w, (_, B) in zip(opt.x, library))
    return np.eye(N) + 0.1 * A_opt, -opt.fun


def optimize_operator_de(obs_dm, library, N, K_opt=3, maxiter=100):
    """D-3: Differential evolution."""
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st, obs_sv = obs_dm[:t_split], obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = sum(w * B for w, (_, B) in zip(weights, library))
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0
            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6
            return _kalman_sv_r2(obs_st, obs_sv, U, N)
        except Exception:
            return 1e6

    bounds = [(-2, 2)] * n_bases
    opt = differential_evolution(objective, bounds, maxiter=maxiter, seed=42, tol=0.002)
    A_opt = sum(w * B for w, (_, B) in zip(opt.x, library))
    return np.eye(N) + 0.1 * A_opt, -opt.fun


# ── Pipeline ─────────────────────────────────────────────────

def run_window(panel, test_year, op_func, **op_kwargs):
    """Run one test window with a given operator optimization function."""
    T_train_yr = 3
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)
    if N < 15:
        return None

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(f"{test_year}-01-01", f"{test_year}-12-31", freq="QS")

    otr = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
    om = ewm_demean(otr)
    otr_dm = otr - om

    library = build_operator_library(otr_dm, N)
    if library:
        op_shape, sv_r2 = op_func(otr_dm, library, N, **op_kwargs)
    else:
        op_shape = np.eye(N)

    U, eigs = dmd_basis(otr_dm @ op_shape, 3)
    if U is None:
        U, eigs = dmd_basis(otr_dm, 3)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
    F_plat = np.eye(k_act) * F_REG
    Q_init = np.eye(k_act) * Q_INIT_SCALE

    alpha = np.zeros(k_act)
    P = np.eye(k_act)
    Q_run = Q_init.copy()

    preds_smim, actuals = [], []

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        obs_dm = obs_raw - om.ravel()

        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_run
        pred_raw = U @ alpha_pred + om.ravel()
        preds_smim.append(pred_raw)
        actuals.append(obs_raw)

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

        expanded = np.vstack([otr, q_data])
        otr = expanded
        om_new = ewm_demean(expanded)
        dm_new = expanded - om_new
        U_new, _ = dmd_basis(dm_new @ op_shape, 3)
        if U_new is None:
            U_new, _ = dmd_basis(dm_new, 3)
        if U_new is not None:
            k_new = U_new.shape[1]
            alpha = U_new.T @ obs_dm
            P = np.eye(k_new)
            Q_run = np.eye(k_new) * Q_INIT_SCALE
            F_plat = np.eye(k_new) * F_REG
            R_sph = train_sph_r(dm_new, U_new)
            U, k_act, om = U_new, k_new, om_new

    if not preds_smim:
        return None
    return float(oos_r_squared(np.array(preds_smim).ravel(), np.array(actuals).ravel()))


def run_experiment(panel, label, op_func, **op_kwargs):
    """Run all 10 windows, print per-window R², return mean."""
    results = []
    for ty in TEST_YEARS:
        t0 = time.time()
        r2 = run_window(panel, ty, op_func, **op_kwargs)
        if r2 is not None:
            results.append((ty, r2))
            print(f"    W{ty}: {r2:.4f}  ({time.time()-t0:.1f}s)")
        else:
            print(f"    W{ty}: FAILED")
    mean_r2 = np.mean([r for _, r in results])
    return mean_r2, results


def main():
    t0 = time.time()
    panel = build_capex_revenue_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    print("\n--- BASELINE (Nelder-Mead, 100 iter, no ridge) ---")
    base_r2, base_res = run_experiment(panel, "baseline", optimize_operator_base, K_opt=3)
    print(f"  Mean R²: {base_r2:.4f}")

    # D-1: Ridge penalty
    print("\n--- D-1: RIDGE PENALTY (alpha=0.1) ---")
    ridge_r2, ridge_res = run_experiment(panel, "ridge", optimize_operator_ridge,
                                          K_opt=3, ridge_alpha=0.1)
    d1_delta = ridge_r2 - base_r2
    print(f"  Mean R²: {ridge_r2:.4f}  Δ vs base: {d1_delta:+.4f}")
    if d1_delta <= 0:
        print("  → No improvement. Trying D-2.")

    # D-2: Increased budget
    print("\n--- D-2: INCREASED BUDGET (300 iter) ---")
    budget_r2, budget_res = run_experiment(panel, "budget300", optimize_operator_base,
                                            K_opt=3, maxiter=300)
    d2_delta = budget_r2 - base_r2
    print(f"  Mean R²: {budget_r2:.4f}  Δ vs base: {d2_delta:+.4f}")

    # D-3: Differential evolution (only if previous improved)
    best_delta = max(d1_delta, d2_delta)
    if best_delta > 0.001:
        print("\n--- D-3: DIFFERENTIAL EVOLUTION ---")
        de_r2, de_res = run_experiment(panel, "diff_evol", optimize_operator_de,
                                        K_opt=3, maxiter=50)
        d3_delta = de_r2 - base_r2
        print(f"  Mean R²: {de_r2:.4f}  Δ vs base: {d3_delta:+.4f}")
    else:
        print("\n  Skipping D-3 (no improvement from D-1/D-2)")
        de_r2 = base_r2

    # Summary
    print(f"\n{'='*70}")
    print(f"  PHASE D SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline:         {base_r2:.4f}")
    print(f"  D-1 Ridge:        {ridge_r2:.4f}  Δ={ridge_r2-base_r2:+.4f}")
    print(f"  D-2 Budget(300):  {budget_r2:.4f}  Δ={budget_r2-base_r2:+.4f}")
    best = max(base_r2, ridge_r2, budget_r2, de_r2)
    print(f"  Best:             {best:.4f}")
    print(f"\n  Total: {time.time()-t0:.1f}s")

    # Save
    rows = []
    for ty, r2 in base_res:
        rows.append({"year": ty, "config": "baseline", "smim_r2": r2})
    for ty, r2 in ridge_res:
        rows.append({"year": ty, "config": "ridge_0.1", "smim_r2": r2})
    for ty, r2 in budget_res:
        rows.append({"year": ty, "config": "budget_300", "smim_r2": r2})
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter5_1_phase_d.parquet", index=False)


if __name__ == "__main__":
    main()
