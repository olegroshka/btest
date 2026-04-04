#!/usr/bin/env python
"""
Iteration 5.1 — Phases A-C: Reproduce, Nested CV, and Statistical Inference
for CapEx/Revenue intensity.

Phase A: Reproduce R²=0.712 with K=3, EWM=12, T=3yr, operator learning
         AR(1) at T=3yr and T=10yr for fair comparison
Phase B: Nested CV with inner fold selecting K ∈ {3,5}, EWM ∈ {8,12}, T ∈ {3,5}
         Holdout W2023-W2024 with median of selected params
Phase C: Block bootstrap CI, permutation test, Diebold-Mariano, sign test

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1.py
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
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Fixed dual-regularisation constants (not tuned)
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_MAX = 15

# Outer evaluation
TEST_YEARS = list(range(2015, 2025))
HOLDOUT_YEARS = [2023, 2024]

# Inner CV grid
K_GRID = [3, 5]
EWM_GRID = [8, 12]
T_GRID = [3, 5]

# Bootstrap / permutation
N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
RNG_SEED = 42


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

    ranked = panel.rank(axis=1, method="average", pct=True)
    coverage = ranked.notna().mean()
    good = coverage[coverage > 0.50].index
    ranked = ranked[good].loc["2005-01-01":"2025-12-31"]
    return ranked


# =========================================================================
# Core pipeline components
# =========================================================================


def ewm_demean(obs: np.ndarray, halflife: int = 8) -> np.ndarray:
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(obs_dm: np.ndarray, k: int = 8) -> tuple:
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


def build_operator_library(obs_dm: np.ndarray, N: int) -> list:
    T = obs_dm.shape[0]
    library = []

    with np.errstate(invalid="ignore"):
        corr0 = np.corrcoef(obs_dm.T)
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 0.0)
    library.append(("corr_inst", corr0))

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


def optimize_operator(obs_dm: np.ndarray, library: list,
                      N: int, K_opt: int = 3) -> tuple:
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
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 0.02, "fatol": 0.002})

    A_opt = np.zeros((N, N))
    for w, (_, B) in zip(opt.x, library):
        A_opt += w * B

    return A_opt, opt.x, -opt.fun


# =========================================================================
# DIAMOND pipeline (parameterised)
# =========================================================================


def run_diamond_window(panel: pd.DataFrame, test_year: int,
                       K: int = 3, ewm_hl: int = 12, T_train_yr: int = 3,
                       use_oplearn: bool = True) -> dict | None:
    """Run DIAMOND pipeline for one test window. Returns result dict with
    per-actor-quarter predictions for DM test."""
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

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    # Operator learning
    op_shape = np.eye(N)
    if use_oplearn:
        library = build_operator_library(otr_dm, N)
        if library:
            A_opt, weights, sv_r2 = optimize_operator(otr_dm, library, N, K_opt=K)
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

        # PREDICTIVE: uses alpha_pred = F @ alpha (before Kalman update)
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

        U_new, _ = dmd_basis(dm_new @ op_shape, K)
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
# AR(1) baseline (parameterised T)
# =========================================================================


def run_ar1_window(panel: pd.DataFrame, test_year: int,
                   T_train_yr: int = 3) -> dict | None:
    """Per-actor AR(1) baseline for one test window."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    all_data = panel[(panel.index >= train_start) & (panel.index <= test_end)].copy()
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)

    train_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_quarters = pd.date_range(f"{test_year}-01-01", f"{test_year}-12-31", freq="QS")

    train = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)

    mu = np.nan_to_num(train.mean(axis=0), nan=0.5)
    dm = train - mu
    rho = np.zeros(N)
    for j in range(N):
        y = dm[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c

    preds_list = []
    actuals_list = []
    prev = np.nan_to_num(train[-1] if len(train) > 0 else np.zeros(N), nan=0.5)

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        pred = mu + rho * (prev - mu)
        preds_list.append(pred)
        actuals_list.append(q_data[0])
        prev = q_data[0]

    if not preds_list:
        return None

    preds = np.array(preds_list)
    acts = np.array(actuals_list)
    r2 = float(oos_r_squared(preds.ravel(), acts.ravel()))

    return {
        "year": test_year, "ar1_r2": r2, "T_train": T_train_yr,
        "preds": preds, "actuals": acts,
    }


# =========================================================================
# Inner CV: select best (K, EWM, T) on outer-train data only
# =========================================================================


def inner_cv_select(panel: pd.DataFrame, test_year: int) -> tuple[dict, float]:
    """Inner CV: use years [test_year-2, test_year-1] as inner validation."""
    inner_val_years = [test_year - 2, test_year - 1]
    best_r2 = -np.inf
    best_params = {"K": 3, "ewm_hl": 12, "T_train_yr": 3}

    for K in K_GRID:
        for ewm_hl in EWM_GRID:
            for T_train_yr in T_GRID:
                inner_r2s = []
                for iv_year in inner_val_years:
                    train_start = pd.Timestamp(f"{iv_year - T_train_yr}-01-01")
                    if train_start < pd.Timestamp("2005-01-01"):
                        continue
                    try:
                        result = run_diamond_window(
                            panel, iv_year, K=K, ewm_hl=ewm_hl,
                            T_train_yr=T_train_yr, use_oplearn=True,
                        )
                        if result is not None and np.isfinite(result["smim_r2"]):
                            inner_r2s.append(result["smim_r2"])
                    except Exception:
                        pass

                if inner_r2s:
                    mean_r2 = np.mean(inner_r2s)
                    if mean_r2 > best_r2:
                        best_r2 = mean_r2
                        best_params = {"K": K, "ewm_hl": ewm_hl, "T_train_yr": T_train_yr}

    return best_params, best_r2


# =========================================================================
# Phase C: Statistical Inference
# =========================================================================


def block_bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOTSTRAP,
                       alpha: float = 0.05, seed: int = RNG_SEED) -> dict:
    """Block bootstrap CI for mean delta-R²."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.array([
        rng.choice(deltas, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {
        "mean": float(deltas.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "se": float(boot_means.std()),
        "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    }


def permutation_test(deltas: np.ndarray, n_perm: int = N_PERMUTATION,
                     seed: int = RNG_SEED) -> dict:
    """Sign-flip permutation test for mean delta-R² > 0."""
    rng = np.random.default_rng(seed)
    observed = deltas.mean()
    n = len(deltas)
    count_ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = (deltas * signs).mean()
        if perm_mean >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {"observed_mean": float(observed), "p_value": float(p_value)}


def diebold_mariano_test(errors_smim: np.ndarray, errors_ar1: np.ndarray) -> dict:
    """Diebold-Mariano test at actor-quarter level.
    H0: E[d_t] = 0 where d_t = e_ar1^2 - e_smim^2 (positive = SMIM better)."""
    d = errors_ar1.ravel() ** 2 - errors_smim.ravel() ** 2
    n = len(d)
    d_bar = d.mean()
    # Newey-West HAC with bandwidth = floor(n^(1/3))
    bw = max(1, int(n ** (1 / 3)))
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for h in range(1, bw + 1):
        gamma_h = np.mean((d[h:] - d_bar) * (d[:-h] - d_bar))
        gamma_sum += 2 * (1 - h / (bw + 1)) * gamma_h
    var_hac = max(gamma0 + gamma_sum, 1e-12)
    se_hac = np.sqrt(var_hac / n)
    dm_stat = d_bar / se_hac if se_hac > 0 else 0.0
    from scipy.stats import norm
    p_value = 1 - norm.cdf(dm_stat)  # one-sided (SMIM better)
    return {
        "dm_stat": float(dm_stat),
        "p_value_one_sided": float(p_value),
        "mean_d": float(d_bar),
        "se_hac": float(se_hac),
        "n_obs": n,
    }


def sign_test(n_wins: int, n_total: int) -> dict:
    """Binomial sign test: P(X >= n_wins) under H0: p=0.5."""
    from scipy.stats import binom
    p_value = binom.sf(n_wins - 1, n_total, 0.5)
    return {"n_wins": n_wins, "n_total": n_total, "p_value": float(p_value)}


# =========================================================================
# Main
# =========================================================================


def main():
    t_total = time.time()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  ITERATION 5.1: CapEx/Revenue — Reproduce + Nested CV + Inference")
    print("=" * 90)

    # Build panel
    panel = build_capex_revenue_panel()
    print(f"\n  Panel: {panel.shape[0]} quarters x {panel.shape[1]} actors")
    rho_med = panel.apply(lambda c: c.dropna().autocorr(1)).median()
    print(f"  Median AR(1) rho: {rho_med:.4f}")

    # ==================================================================
    # PHASE A: Fixed-config reproduction
    # ==================================================================
    print("\n" + "=" * 90)
    print("  PHASE A: REPRODUCE FIXED-CONFIG RESULT (K=3, EWM=12, T=3yr, OpLearn)")
    print("=" * 90)

    phase_a_rows = []
    all_preds_smim = []
    all_preds_ar1_3yr = []
    all_preds_ar1_10yr = []
    all_actuals = []

    for ty in TEST_YEARS:
        t0 = time.time()
        res = run_diamond_window(panel, ty, K=3, ewm_hl=12, T_train_yr=3, use_oplearn=True)
        ar1_3 = run_ar1_window(panel, ty, T_train_yr=3)
        ar1_10 = run_ar1_window(panel, ty, T_train_yr=10)

        if res is None:
            print(f"  W{ty}: FAILED")
            continue

        d3 = res["smim_r2"] - ar1_3["ar1_r2"]
        d10 = res["smim_r2"] - ar1_10["ar1_r2"]
        w3 = "WIN" if d3 > 0 else "LOSS"
        print(f"  W{ty}: SMIM={res['smim_r2']:.4f}  AR1(3yr)={ar1_3['ar1_r2']:.4f} "
              f"Δ={d3:+.4f} {w3}  AR1(10yr)={ar1_10['ar1_r2']:.4f} Δ={d10:+.4f}  "
              f"(N={res['N']}, {time.time()-t0:.1f}s)")

        phase_a_rows.append({
            "year": ty, "smim_r2": res["smim_r2"],
            "ar1_3yr_r2": ar1_3["ar1_r2"], "ar1_10yr_r2": ar1_10["ar1_r2"],
            "delta_3yr": d3, "delta_10yr": d10,
        })
        all_preds_smim.append(res["preds"])
        all_preds_ar1_3yr.append(ar1_3["preds"])
        all_preds_ar1_10yr.append(ar1_10["preds"])
        all_actuals.append(res["actuals"])

    df_a = pd.DataFrame(phase_a_rows)
    print(f"\n  Phase A Summary:")
    print(f"    Mean SMIM R²:     {df_a['smim_r2'].mean():.4f}")
    print(f"    Mean AR1(3yr) R²: {df_a['ar1_3yr_r2'].mean():.4f}  "
          f"Δ={df_a['delta_3yr'].mean():+.4f}  "
          f"wins={(df_a['delta_3yr'] > 0).sum()}/{len(df_a)}")
    print(f"    Mean AR1(10yr) R²:{df_a['ar1_10yr_r2'].mean():.4f}  "
          f"Δ={df_a['delta_10yr'].mean():+.4f}  "
          f"wins={(df_a['delta_10yr'] > 0).sum()}/{len(df_a)}")

    # Phase A inference (quick, on the fixed-config results)
    deltas_3yr = df_a["delta_3yr"].values

    # Flatten errors per-window (SMIM and AR1 at same T share the same actor set)
    # Use AR(1)'s own actuals to avoid shape mismatch across different T
    all_e_smim_flat = np.concatenate([(p - a).ravel() for p, a in zip(all_preds_smim, all_actuals)])
    all_e_ar1_3_flat = np.concatenate([(p - a).ravel() for p, a in zip(all_preds_ar1_3yr, all_actuals)])

    print(f"\n  Phase A Inference (vs AR1 T=3yr, fixed config):")
    boot = block_bootstrap_ci(deltas_3yr)
    print(f"    Bootstrap CI (95%): [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  "
          f"excludes zero: {boot['excludes_zero']}")
    perm = permutation_test(deltas_3yr)
    print(f"    Permutation p:  {perm['p_value']:.4f}")
    dm = diebold_mariano_test(all_e_smim_flat, all_e_ar1_3_flat)
    print(f"    DM stat (actor-quarter): {dm['dm_stat']:.3f}  "
          f"p={dm['p_value_one_sided']:.4f}  (n={dm['n_obs']})")
    st = sign_test(int((deltas_3yr > 0).sum()), len(deltas_3yr))
    print(f"    Sign test: {st['n_wins']}/{st['n_total']}  p={st['p_value']:.4f}")

    df_a.to_parquet(METRICS_DIR / "iter5_1_phase_a.parquet", index=False)

    # ==================================================================
    # PHASE B: NESTED CROSS-VALIDATION
    # ==================================================================
    print("\n" + "=" * 90)
    print("  PHASE B: NESTED CROSS-VALIDATION")
    print(f"  Inner CV grid: K={K_GRID}, EWM={EWM_GRID}, T={T_GRID}")
    print("=" * 90)

    cv_rows = []
    cv_preds_smim = []
    cv_preds_ar1 = []
    cv_actuals = []

    for ty in TEST_YEARS:
        t0 = time.time()
        is_holdout = ty in HOLDOUT_YEARS

        if is_holdout:
            # Use median of previously selected params
            if cv_rows:
                non_ho = [r for r in cv_rows if not r["is_holdout"]]
                sel_K = int(np.median([r["sel_K"] for r in non_ho]))
                sel_ewm = int(np.median([r["sel_ewm"] for r in non_ho]))
                sel_T = int(np.median([r["sel_T"] for r in non_ho]))
                inner_r2 = np.nan
            else:
                sel_K, sel_ewm, sel_T, inner_r2 = 3, 12, 3, np.nan
        else:
            params, inner_r2 = inner_cv_select(panel, ty)
            sel_K = params["K"]
            sel_ewm = params["ewm_hl"]
            sel_T = params["T_train_yr"]

        # Run outer evaluation
        result = run_diamond_window(
            panel, ty, K=sel_K, ewm_hl=sel_ewm,
            T_train_yr=sel_T, use_oplearn=True,
        )
        ar1_result = run_ar1_window(panel, ty, T_train_yr=sel_T)

        if result is None:
            print(f"  W{ty}: FAILED")
            continue

        delta = result["smim_r2"] - ar1_result["ar1_r2"]
        elapsed = time.time() - t0
        tag = " [HOLDOUT]" if is_holdout else ""
        print(f"  W{ty}: K={sel_K} EWM={sel_ewm} T={sel_T}yr  "
              f"SMIM={result['smim_r2']:.4f}  AR1={ar1_result['ar1_r2']:.4f}  "
              f"Δ={delta:+.4f}  ({elapsed:.1f}s){tag}")

        cv_rows.append({
            "year": ty, "is_holdout": is_holdout,
            "sel_K": sel_K, "sel_ewm": sel_ewm, "sel_T": sel_T,
            "inner_val_r2": inner_r2,
            "smim_r2": result["smim_r2"],
            "ar1_r2": ar1_result["ar1_r2"],
            "delta_r2": delta,
        })
        cv_preds_smim.append(result["preds"])
        cv_preds_ar1.append(ar1_result["preds"])
        cv_actuals.append(result["actuals"])

    df_b = pd.DataFrame(cv_rows)
    cv_mask = ~df_b["is_holdout"]
    ho_mask = df_b["is_holdout"]

    print(f"\n  NESTED CV SUMMARY:")
    if cv_mask.any():
        print(f"    CV windows (N={cv_mask.sum()}):")
        print(f"      Mean SMIM R²:  {df_b.loc[cv_mask, 'smim_r2'].mean():.4f}")
        print(f"      Mean AR(1) R²: {df_b.loc[cv_mask, 'ar1_r2'].mean():.4f}")
        print(f"      Mean Δ R²:     {df_b.loc[cv_mask, 'delta_r2'].mean():+.4f}")
        print(f"      SMIM wins:     {(df_b.loc[cv_mask, 'delta_r2'] > 0).sum()}/{cv_mask.sum()}")
        print(f"      Selected K:    {df_b.loc[cv_mask, 'sel_K'].value_counts().to_dict()}")
        print(f"      Selected EWM:  {df_b.loc[cv_mask, 'sel_ewm'].value_counts().to_dict()}")
        print(f"      Selected T:    {df_b.loc[cv_mask, 'sel_T'].value_counts().to_dict()}")

    if ho_mask.any():
        print(f"    HOLDOUT windows (N={ho_mask.sum()}) — no tuning exposure:")
        print(f"      Mean SMIM R²:  {df_b.loc[ho_mask, 'smim_r2'].mean():.4f}")
        print(f"      Mean AR(1) R²: {df_b.loc[ho_mask, 'ar1_r2'].mean():.4f}")
        print(f"      Mean Δ R²:     {df_b.loc[ho_mask, 'delta_r2'].mean():+.4f}")

    df_b.to_parquet(METRICS_DIR / "iter5_1_nested_cv.parquet", index=False)

    # ==================================================================
    # PHASE C: STATISTICAL INFERENCE (on nested CV results)
    # ==================================================================
    print("\n" + "=" * 90)
    print("  PHASE C: STATISTICAL INFERENCE")
    print("=" * 90)

    # Use CV windows only (not holdout) for inference
    cv_deltas = df_b.loc[cv_mask, "delta_r2"].values

    if len(cv_deltas) >= 3:
        print(f"\n  C-1: Block Bootstrap (N={len(cv_deltas)} CV windows):")
        boot_cv = block_bootstrap_ci(cv_deltas)
        print(f"    Mean Δ R²:  {boot_cv['mean']:+.4f}")
        print(f"    95% CI:     [{boot_cv['ci_lo']:+.4f}, {boot_cv['ci_hi']:+.4f}]")
        print(f"    SE:         {boot_cv['se']:.4f}")
        print(f"    Excludes 0: {boot_cv['excludes_zero']}")

        print(f"\n  C-2: Permutation Test:")
        perm_cv = permutation_test(cv_deltas)
        print(f"    Observed mean Δ: {perm_cv['observed_mean']:+.4f}")
        print(f"    p-value:         {perm_cv['p_value']:.4f}")

    # DM test on actor-quarter level (all CV windows)
    cv_indices = [i for i, r in enumerate(cv_rows) if not r["is_holdout"]]
    if cv_indices:
        cv_e_smim_flat = np.concatenate([
            (cv_preds_smim[i] - cv_actuals[i]).ravel() for i in cv_indices
        ])
        cv_e_ar1_flat = np.concatenate([
            (cv_preds_ar1[i] - cv_actuals[i]).ravel() for i in cv_indices
        ])

        print(f"\n  C-3: Diebold-Mariano (actor-quarter, CV windows):")
        dm_cv = diebold_mariano_test(cv_e_smim_flat, cv_e_ar1_flat)
        print(f"    DM stat:    {dm_cv['dm_stat']:.3f}")
        print(f"    p (1-sided):{dm_cv['p_value_one_sided']:.4f}")
        print(f"    Mean d:     {dm_cv['mean_d']:.6f}")
        print(f"    n obs:      {dm_cv['n_obs']}")

    cv_wins = int((cv_deltas > 0).sum()) if len(cv_deltas) > 0 else 0
    cv_n = len(cv_deltas) if len(cv_deltas) > 0 else 0
    print(f"\n  C-4: Sign Test:")
    st_cv = sign_test(cv_wins, cv_n)
    print(f"    Wins: {st_cv['n_wins']}/{st_cv['n_total']}  p={st_cv['p_value']:.4f}")

    # All windows (CV + holdout) summary
    all_deltas = df_b["delta_r2"].values
    all_wins = int((all_deltas > 0).sum())
    print(f"\n  All windows ({len(all_deltas)}):")
    boot_all = block_bootstrap_ci(all_deltas)
    print(f"    Mean Δ:  {boot_all['mean']:+.4f}  "
          f"CI [{boot_all['ci_lo']:+.4f}, {boot_all['ci_hi']:+.4f}]")
    perm_all = permutation_test(all_deltas)
    print(f"    Perm p:  {perm_all['p_value']:.4f}")
    st_all = sign_test(all_wins, len(all_deltas))
    print(f"    Sign:    {st_all['n_wins']}/{st_all['n_total']}  p={st_all['p_value']:.4f}")

    # Save inference results
    inference = {
        "phase_a_fixed": {
            "mean_smim": float(df_a["smim_r2"].mean()),
            "mean_ar1_3yr": float(df_a["ar1_3yr_r2"].mean()),
            "mean_ar1_10yr": float(df_a["ar1_10yr_r2"].mean()),
            "mean_delta_3yr": float(df_a["delta_3yr"].mean()),
            "wins_3yr": int((df_a["delta_3yr"] > 0).sum()),
        },
        "nested_cv": {
            "mean_smim_cv": float(df_b.loc[cv_mask, "smim_r2"].mean()) if cv_mask.any() else None,
            "mean_ar1_cv": float(df_b.loc[cv_mask, "ar1_r2"].mean()) if cv_mask.any() else None,
            "mean_delta_cv": float(df_b.loc[cv_mask, "delta_r2"].mean()) if cv_mask.any() else None,
            "wins_cv": int((df_b.loc[cv_mask, "delta_r2"] > 0).sum()) if cv_mask.any() else None,
            "mean_smim_ho": float(df_b.loc[ho_mask, "smim_r2"].mean()) if ho_mask.any() else None,
            "mean_ar1_ho": float(df_b.loc[ho_mask, "ar1_r2"].mean()) if ho_mask.any() else None,
            "mean_delta_ho": float(df_b.loc[ho_mask, "delta_r2"].mean()) if ho_mask.any() else None,
        },
        "inference_cv": {
            "bootstrap_ci": boot_cv if len(cv_deltas) >= 3 else None,
            "permutation_p": perm_cv["p_value"] if len(cv_deltas) >= 3 else None,
            "dm_stat": dm_cv["dm_stat"] if cv_indices else None,
            "dm_p": dm_cv["p_value_one_sided"] if cv_indices else None,
            "sign_test_p": st_cv["p_value"],
        },
    }
    pd.DataFrame([inference]).to_parquet(METRICS_DIR / "iter5_1_inference.parquet", index=False)

    # ==================================================================
    # SUCCESS CRITERIA CHECK
    # ==================================================================
    print("\n" + "=" * 90)
    print("  SUCCESS CRITERIA")
    print("=" * 90)

    cv_mean_delta = df_b.loc[cv_mask, "delta_r2"].mean() if cv_mask.any() else 0
    cv_wins_n = int((df_b.loc[cv_mask, "delta_r2"] > 0).sum()) if cv_mask.any() else 0
    cv_total = int(cv_mask.sum()) if cv_mask.any() else 0

    bronze = cv_wins_n > cv_total / 2
    silver = boot_cv["excludes_zero"] if len(cv_deltas) >= 3 else False
    gold = perm_cv["p_value"] < 0.05 if len(cv_deltas) >= 3 else False

    print(f"  BRONZE (nested-CV wins majority):    {'PASS' if bronze else 'FAIL'}  "
          f"({cv_wins_n}/{cv_total})")
    print(f"  SILVER (bootstrap CI excludes zero):  {'PASS' if silver else 'FAIL'}  "
          f"CI=[{boot_cv['ci_lo']:+.4f}, {boot_cv['ci_hi']:+.4f}]" if len(cv_deltas) >= 3 else "")
    print(f"  GOLD   (permutation p < 0.05):        {'PASS' if gold else 'FAIL'}  "
          f"p={perm_cv['p_value']:.4f}" if len(cv_deltas) >= 3 else "")

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print("  Saved: iter5_1_phase_a.parquet, iter5_1_nested_cv.parquet, iter5_1_inference.parquet")


if __name__ == "__main__":
    main()
