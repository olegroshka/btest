#!/usr/bin/env python
"""
DIAMOND + Operator Learning: the combined pipeline.

Based on dd2_v2's proven DIAMOND rolling DMD structure, augmented with
A1-style operator learning. The operator pre-shapes data before DMD,
optimised end-to-end through the actual DIAMOND prediction pipeline.

Tests on experiment_a1 (CapEx/Assets) to compare with published R²=0.691.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_diamond_oplearn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
T_YR = 5
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
TEST_YEARS = list(range(2015, 2025))


# =========================================================================
# Core functions (from dd2_v2 — proven DIAMOND infrastructure)
# =========================================================================


def ewm_demean(otr, halflife=8):
    T = otr.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(otr_dm, N, k=K_GOLD):
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        return U
    except Exception:
        return None


def train_sph_r(otr_dm, N, U):
    """Spherical R from projection residuals (stable, no EM needed)."""
    resid = otr_dm - (otr_dm @ U) @ U.T
    return np.eye(N) * max(np.mean(resid ** 2), 1e-8)


def platinum_predict_quarter(obs_dm_quarter, om, N, U, F, Q_init, R_sph,
                              alpha_init, P_init, Q_run):
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
    pred = (a_p @ U.T + om).ravel()  # prediction uses alpha_pred (before update)
    innov = a_f - F @ alpha_init
    Q_new = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(innov, innov)
    Q_new = (Q_new + Q_new.T) / 2 + np.eye(k_act) * 1e-6
    return pred, a_f, P_f, Q_new


# =========================================================================
# Operator library
# =========================================================================


def build_operator_library(obs_dm, N, sector_map=None):
    """Build basis operator library for Nelder-Mead optimisation."""
    T = obs_dm.shape[0]
    library = []
    names = []

    # B0: instantaneous cross-correlation
    corr0 = np.corrcoef(obs_dm.T)
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 0.0)
    library.append(corr0)
    names.append("corr_inst")

    # B1: lag-1+2 cross-correlation (directed)
    if T > 2:
        lag = np.zeros((N, N))
        for lo in [1, 2]:
            if T > lo:
                xp = obs_dm[:-lo]
                xn = obs_dm[lo:]
                xp_dm = xp - xp.mean(0)
                xn_dm = xn - xn.mean(0)
                num = (xp_dm.T @ xn_dm) / max(len(xp) - 1, 1)
                sp = np.std(xp, 0, keepdims=True).T
                sn = np.std(xn, 0, keepdims=True)
                lag += np.nan_to_num(num / np.maximum(sp @ sn, 1e-10), nan=0.0)
        np.fill_diagonal(lag, 0.0)
        library.append(lag)
        names.append("corr_lag")

    # B2-B3: multi-scale cosine similarity
    for scale in [4, 8]:
        if T >= scale + 1:
            kernel = np.ones(scale) / scale
            sm = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="valid"), axis=0, arr=obs_dm
            )
            if sm.shape[0] >= 2:
                norms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                normed = sm / norms
                sim = normed.T @ normed
                np.fill_diagonal(sim, 0.0)
                library.append(sim)
                names.append(f"cosine_{scale}Q")

    # B4: sector membership (if available)
    if sector_map is not None:
        sec_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if sector_map.get(i) == sector_map.get(j) and sector_map.get(i) is not None:
                    sec_mat[i, j] = 1.0
                    sec_mat[j, i] = 1.0
        if sec_mat.sum() > 0:
            sec_mat /= max(sec_mat.max(), 1e-10)
            library.append(sec_mat)
            names.append("sector")

    return library, names


def optimize_operator_diamond(obs_dm, N, library, K_use=K_GOLD):
    """Optimise operator weights through the DIAMOND pipeline (not Schur+EM)."""
    T = obs_dm.shape[0]
    t_split = max(int(T * 0.70), T - 4)  # at least 4 quarters for sub-val
    obs_st = obs_dm[:t_split]
    obs_sv = obs_dm[t_split:]
    n_bases = len(library)

    if len(obs_sv) < 2:
        return np.ones(n_bases) / n_bases, 0.0

    def objective(params):
        try:
            weights = params[:n_bases]
            alpha_shape = max(params[n_bases], 0.01) if len(params) > n_bases else 0.1

            A = sum(w * B for w, B in zip(weights, library))
            shaped = obs_st @ (np.eye(N) + alpha_shape * A)

            U = dmd_basis(shaped, N, K_use)
            if U is None:
                return 1e6
            K = U.shape[1]
            R_sph = np.eye(N) * max(np.mean((obs_st - (obs_st @ U) @ U.T) ** 2), 1e-8)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT_SCALE

            alpha = np.zeros(K)
            P = np.eye(K)
            Q_run = Q.copy()
            preds = []
            om_st = ewm_demean(obs_st)

            for t in range(len(obs_sv)):
                a_p = F @ alpha
                P_p = F @ P @ F.T + Q_run
                preds.append(U @ a_p)

                obs = obs_sv[t]
                S = U @ P_p @ U.T + R_sph
                try:
                    Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
                except Exception:
                    Kg = np.zeros((K, N))
                alpha = a_p + Kg @ (obs - U @ a_p)
                P = (np.eye(K) - Kg @ U) @ P_p
                si = alpha - a_p
                Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(si, si)
                Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8

            r2 = float(oos_r_squared(np.array(preds).ravel(), obs_sv.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    # Include shaping factor alpha as optimised parameter
    x0 = np.append(np.ones(n_bases) / n_bases, 0.1)
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 120, "xatol": 0.01, "fatol": 0.001})

    weights = opt.x[:n_bases]
    alpha_shape = max(opt.x[n_bases], 0.01)
    A_opt = sum(w * B for w, B in zip(weights, library))
    return weights, alpha_shape, -opt.fun, A_opt


# =========================================================================
# Main experiment
# =========================================================================


def load_panel():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]

    # Sector map for operator library
    sector_map_raw = {a["actor_id"]: a.get("sector", None) for a in reg["actors"]}

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], sector_map_raw


def run_diamond_oplearn(wide, sector_map_raw, ty, use_oplearn=True):
    """Run DIAMOND + operator learning for one test window."""
    train_start = pd.Timestamp(f"{ty - T_YR}-01-01")
    test_quarters = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")

    all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)

    # Sector map indexed by position
    col_list = list(valid)
    sector_pos = {i: sector_map_raw.get(col_list[i]) for i in range(N)}

    train_end = pd.Timestamp(f"{ty - 1}-12-31")
    otr = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
    om = ewm_demean(otr)
    otr_dm = otr - om

    # AR(1) baseline
    mu_a = otr.mean(0)
    dm_a = otr - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10 and np.isfinite(np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1])
        else 0
        for j in range(N)
    ])

    # Operator learning
    op_shaping = np.eye(N)
    alpha_shape = 0.0
    sv_r2 = 0.0
    op_weights_str = ""

    if use_oplearn:
        library, names = build_operator_library(otr_dm, N, sector_pos)
        weights, alpha_shape, sv_r2, A_opt = optimize_operator_diamond(otr_dm, N, library)
        op_shaping = np.eye(N) + alpha_shape * A_opt
        op_weights_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
        op_weights_str += f", alpha={alpha_shape:.3f}"

    # Initial DMD with operator shaping
    U = dmd_basis(otr_dm @ op_shaping, N)
    if U is None:
        U = dmd_basis(otr_dm, N)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, N, U)
    F_plat = np.eye(k_act) * F_REG
    Q_init = np.eye(k_act) * Q_INIT_SCALE

    # Rolling prediction
    alpha = np.zeros(k_act)
    P = np.eye(k_act)
    Q_run = Q_init.copy()
    preds_rolling = []
    preds_ar1 = []
    actuals = []
    prev_obs = otr[-1]

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        q_dm = obs_raw - om.ravel()

        pred, alpha_new, P_new, Q_new = platinum_predict_quarter(
            q_dm, om, N, U, F_plat, Q_init, R_sph, alpha, P, Q_run
        )
        preds_rolling.append(pred)
        preds_ar1.append(mu_a + rho * (prev_obs - mu_a))
        actuals.append(obs_raw)
        prev_obs = obs_raw

        # Expand training and recompute basis
        expanded_end = q_date
        otr_expanded = all_data[
            (all_data.index >= train_start) & (all_data.index <= expanded_end)
        ].values.astype(np.float64)
        om_new = ewm_demean(otr_expanded)
        otr_dm_new = otr_expanded - om_new

        U_new = dmd_basis(otr_dm_new @ op_shaping, N)
        if U_new is None:
            U_new = dmd_basis(otr_dm_new, N)
        if U_new is not None:
            # Reproject alpha onto new basis
            alpha = U_new.T @ q_dm
            P = np.eye(U_new.shape[1])
            Q_run = np.eye(U_new.shape[1]) * Q_INIT_SCALE
            F_plat = np.eye(U_new.shape[1]) * F_REG
            R_sph = train_sph_r(otr_dm_new, N, U_new)
            U = U_new
            k_act = U_new.shape[1]
            om = om_new
        else:
            alpha = alpha_new
            P = P_new
            Q_run = Q_new

    if not preds_rolling:
        return None

    r2_smim = float(oos_r_squared(np.array(preds_rolling).ravel(), np.array(actuals).ravel()))
    r2_ar1 = float(oos_r_squared(np.array(preds_ar1).ravel(), np.array(actuals).ravel()))

    return {
        "year": ty, "smim_r2": r2_smim, "ar1_r2": r2_ar1,
        "delta": r2_smim - r2_ar1, "N": N, "K": k_act,
        "sv_r2": sv_r2, "op_weights": op_weights_str,
    }


def main():
    t_total = time.time()
    print("=" * 80)
    print("  DIAMOND + Operator Learning on experiment_a1 (CapEx/Assets)")
    print("=" * 80)

    wide, sector_map = load_panel()
    print(f"  Panel: {wide.shape[1]} actors, {wide.shape[0]} quarters")

    # --- DIAMOND baseline (no operator learning) ---
    print("\n--- DIAMOND baseline (no operator learning) ---")
    rows_base = []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_oplearn(wide, sector_map, ty, use_oplearn=False)
        if res:
            w = "WIN" if res["delta"] > 0 else "LOSS"
            print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1={res['ar1_r2']:.4f}  "
                  f"delta={res['delta']:+.4f} {w}  ({time.time()-t0:.1f}s)")
            rows_base.append(res)

    # --- DIAMOND + operator learning ---
    print("\n--- DIAMOND + Operator Learning ---")
    rows_opl = []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_oplearn(wide, sector_map, ty, use_oplearn=True)
        if res:
            w = "WIN" if res["delta"] > 0 else "LOSS"
            print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1={res['ar1_r2']:.4f}  "
                  f"delta={res['delta']:+.4f} {w}  "
                  f"(sv={res['sv_r2']:.3f}, {time.time()-t0:.1f}s)")
            if res["op_weights"]:
                print(f"         weights: {res['op_weights']}")
            rows_opl.append(res)

    # --- Summary ---
    print(f"\n{'#'*80}")
    print(f"#  SUMMARY")
    print(f"{'#'*80}")

    for label, rows in [("DIAMOND (baseline)", rows_base), ("DIAMOND+OpLearn", rows_opl)]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        wins = (df["delta"] > 0).sum()
        print(f"\n  {label}:")
        print(f"    Mean SMIM R2: {df['smim_r2'].mean():.4f}")
        print(f"    Mean AR1 R2:  {df['ar1_r2'].mean():.4f}")
        print(f"    Mean delta:   {df['delta'].mean():+.4f}")
        print(f"    SMIM wins:    {wins}/{len(df)}")

    # Improvement from operator learning
    if rows_base and rows_opl:
        base_df = pd.DataFrame(rows_base).set_index("year")
        opl_df = pd.DataFrame(rows_opl).set_index("year")
        common = base_df.index.intersection(opl_df.index)
        improvement = opl_df.loc[common, "smim_r2"] - base_df.loc[common, "smim_r2"]
        print(f"\n  Operator learning effect:")
        print(f"    Mean R2 improvement: {improvement.mean():+.4f}")
        print(f"    Per window:")
        for y in common:
            print(f"      W{y}: base={base_df.loc[y, 'smim_r2']:.4f} "
                  f"oplearn={opl_df.loc[y, 'smim_r2']:.4f} "
                  f"improvement={improvement.loc[y]:+.4f}")

    # Save
    all_rows = []
    for r in rows_base:
        r["config"] = "diamond_base"
        all_rows.append(r)
    for r in rows_opl:
        r["config"] = "diamond_oplearn"
        all_rows.append(r)
    pd.DataFrame(all_rows).to_parquet(METRICS_DIR / "iter5_diamond_oplearn.parquet")
    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
