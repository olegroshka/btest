#!/usr/bin/env python
"""
SMIM Iteration 5 — Path A: DIAMOND pipeline on CapEx/Revenue intensity.

Constructs CapEx/Revenue quarterly intensity from EDGAR, then runs the
full DIAMOND config (rolling DMD, dual regularisation, online Q adaptation)
plus operator learning (from A1). Compares with AR(1) baseline.

This is a NEW pipeline on a NEW signal — no risk to existing scripts.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_a1i5.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
K_MIN = 3
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
EWM_HL = 8
T_TRAIN_YR = 5
TEST_YEARS = list(range(2015, 2025))


# =========================================================================
# Data construction
# =========================================================================


def build_capex_revenue_panel() -> pd.DataFrame:
    """Build quarterly CapEx/Revenue cross-sectional rank panel from EDGAR."""
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

    # Cross-sectional percentile rank per quarter
    ranked = panel.rank(axis=1, method="average", pct=True)

    # Filter coverage
    coverage = ranked.notna().mean()
    good = coverage[coverage > 0.50].index
    ranked = ranked[good].loc["2005-01-01":"2025-12-31"]

    return ranked


# =========================================================================
# Core pipeline (DIAMOND config)
# =========================================================================


def ewm_demean(obs: np.ndarray, halflife: int = EWM_HL) -> np.ndarray:
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(obs_dm: np.ndarray, k: int = K_GOLD) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def train_sph_r(obs_dm: np.ndarray, U: np.ndarray) -> np.ndarray:
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * max(np.mean(resid ** 2), 1e-8)


def build_operator_library(obs_dm: np.ndarray, N: int) -> list[tuple[str, np.ndarray]]:
    """Build basis operator library for Nelder-Mead optimization."""
    T = obs_dm.shape[0]
    library = []

    # B0: instantaneous cross-correlation
    corr0 = np.corrcoef(obs_dm.T)
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 0.0)
    library.append(("corr_inst", corr0))

    # B1: lag-1 cross-correlation
    if T > 2:
        lag1 = np.zeros((N, N))
        for lag_offset in [1, 2]:
            if T > lag_offset:
                xp = obs_dm[:-lag_offset]
                xn = obs_dm[lag_offset:]
                xp_dm = xp - xp.mean(axis=0)
                xn_dm = xn - xn.mean(axis=0)
                num = (xp_dm.T @ xn_dm) / max(len(xp) - 1, 1)
                sp_std = np.std(xp, axis=0, keepdims=True).T
                sn_std = np.std(xn, axis=0, keepdims=True)
                denom = np.maximum(sp_std @ sn_std, 1e-10)
                lag1 += np.nan_to_num(num / denom, nan=0.0)
        np.fill_diagonal(lag1, 0.0)
        library.append(("corr_lag", lag1))

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
                library.append((f"cosine_{scale}Q", sim))

    return library


def optimize_operator(obs_dm: np.ndarray, library: list[tuple[str, np.ndarray]],
                      N: int, K_opt: int = K_MIN) -> np.ndarray:
    """Nelder-Mead optimization of operator weights for sub-validation R²."""
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

            # DMD on the operator-shaped data (use operator as pre-filter)
            shaped = obs_st @ (np.eye(N) + 0.1 * A)  # light operator shaping
            U, _ = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6

            K = U.shape[1]
            R_sph = train_sph_r(obs_st, U)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT_SCALE

            # Quick Kalman on sub-val
            alpha = np.zeros(K)
            P = np.eye(K)
            Q_run = Q.copy()
            preds = []
            mu_st = ewm_demean(obs_st)

            for t in range(len(obs_sv)):
                alpha_pred = F @ alpha
                P_pred = F @ P @ F.T + Q_run
                pred = U @ alpha_pred
                preds.append(pred)

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
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 0.02, "fatol": 0.002})

    A_opt = np.zeros((N, N))
    for w, (_, B) in zip(opt.x, library):
        A_opt += w * B

    return A_opt, opt.x, -opt.fun


def run_diamond_window(panel: pd.DataFrame, test_year: int,
                       use_operator_learning: bool = True) -> dict | None:
    """Run DIAMOND pipeline on one test window. Returns result dict."""
    train_start = pd.Timestamp(f"{test_year - T_TRAIN_YR}-01-01")
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

    # AR(1) baseline
    mu_a = otr.mean(axis=0)
    dm_a = otr - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])

    # Operator learning (if enabled)
    op_shape = np.eye(N)
    if use_operator_learning:
        library = build_operator_library(otr_dm, N)
        if library:
            A_opt, weights, sv_r2 = optimize_operator(otr_dm, library, N, K_opt=K_MIN)
            op_shape = np.eye(N) + 0.1 * A_opt

    # Initial DMD (on operator-shaped data)
    U, eigs = dmd_basis(otr_dm @ op_shape, K_GOLD)
    if U is None:
        # Fallback: DMD without operator shaping
        U, eigs = dmd_basis(otr_dm, K_GOLD)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
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
        obs_dm = obs_raw - om.ravel()

        # DIAMOND prediction
        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_run
        pred_dm = U @ alpha_pred
        pred_raw = pred_dm + om.ravel()
        preds_rolling.append(pred_raw)

        # AR(1) prediction
        ar1_pred = mu_a + rho * (prev_obs - mu_a)
        preds_ar1.append(ar1_pred)
        actuals.append(obs_raw)
        prev_obs = obs_raw

        # Kalman update
        S = U @ P_pred @ U.T + R_sph
        try:
            Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
        except np.linalg.LinAlgError:
            Kg = np.zeros((k_act, N))
        alpha = alpha_pred + Kg @ (obs_dm - U @ alpha_pred)
        P = (np.eye(k_act) - Kg @ U) @ P_pred

        # Online Q
        innov = alpha - alpha_pred
        Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(k_act) * 1e-6

        # Rolling basis: expand training to include this quarter
        expanded = np.vstack([otr, q_data])
        otr = expanded
        om_new = ewm_demean(expanded)
        dm_new = expanded - om_new

        U_new, _ = dmd_basis(dm_new @ op_shape, K_GOLD)
        if U_new is None:
            U_new, _ = dmd_basis(dm_new, K_GOLD)
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

    if not preds_rolling:
        return None

    preds_r = np.array(preds_rolling)
    preds_a = np.array(preds_ar1)
    acts = np.array(actuals)

    r2_smim = float(oos_r_squared(preds_r.ravel(), acts.ravel()))
    r2_ar1 = float(oos_r_squared(preds_a.ravel(), acts.ravel()))

    return {
        "year": test_year, "smim_r2": r2_smim, "ar1_r2": r2_ar1,
        "delta": r2_smim - r2_ar1, "N": N, "K": k_act,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    t_total = time.time()
    print("=" * 80)
    print("  ITERATION 5 — Path A: DIAMOND + Operator Learning on CapEx/Revenue")
    print("=" * 80)

    # Build intensity panel
    panel = build_capex_revenue_panel()
    print(f"\n  CapEx/Revenue panel: {panel.shape[0]} quarters x {panel.shape[1]} actors")
    rho_med = panel.apply(lambda c: c.dropna().autocorr(1)).median()
    print(f"  Median AR(1) rho: {rho_med:.4f}")

    # Run with operator learning
    print(f"\n--- DIAMOND + Operator Learning ---")
    rows_full = []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_window(panel, ty, use_operator_learning=True)
        if res:
            w = "WIN" if res["delta"] > 0 else "LOSS"
            print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1={res['ar1_r2']:.4f}  "
                  f"delta={res['delta']:+.4f} {w}  (N={res['N']}, K={res['K']}, "
                  f"{time.time()-t0:.1f}s)")
            rows_full.append(res)

    # Run without operator learning (pure DMD)
    print(f"\n--- DIAMOND (pure DMD, no operator learning) ---")
    rows_dmd = []
    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_window(panel, ty, use_operator_learning=False)
        if res:
            w = "WIN" if res["delta"] > 0 else "LOSS"
            print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1={res['ar1_r2']:.4f}  "
                  f"delta={res['delta']:+.4f} {w}  ({time.time()-t0:.1f}s)")
            rows_dmd.append(res)

    # Summary
    print(f"\n{'#'*80}")
    print(f"#  SUMMARY")
    print(f"{'#'*80}")

    for label, rows in [("DIAMOND+OpLearn", rows_full), ("DIAMOND pure DMD", rows_dmd)]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        wins = (df["delta"] > 0).sum()
        print(f"\n  {label}:")
        print(f"    Mean SMIM R2: {df['smim_r2'].mean():.4f}")
        print(f"    Mean AR1 R2:  {df['ar1_r2'].mean():.4f}")
        print(f"    Mean delta:   {df['delta'].mean():+.4f}")
        print(f"    SMIM wins:    {wins}/{len(df)}")

    # Save
    all_rows = []
    for r in rows_full:
        r["config"] = "diamond_oplearn"
        all_rows.append(r)
    for r in rows_dmd:
        r["config"] = "diamond_dmd"
        all_rows.append(r)
    pd.DataFrame(all_rows).to_parquet(METRICS_DIR / "iter5_path_a_capex_revenue.parquet", index=False)

    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
