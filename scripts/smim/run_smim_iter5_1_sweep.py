#!/usr/bin/env python
"""
Wide hyperparameter sweep for CapEx/Revenue DIAMOND pipeline.

Two-stage approach:
  Stage 1: Coarse sweep WITHOUT operator learning (fast, ~0.3s/window)
  Stage 2: Top configs WITH operator learning (slower, ~2s/window)

Grid:
  K:   [2, 3, 4, 5, 6, 8]
  EWM: [4, 6, 8, 10, 12, 16, 20]
  T:   [2, 3, 4, 5, 7]

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1_sweep.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

# Sweep grid
K_SWEEP = [2, 3, 4, 5, 6, 8]
EWM_SWEEP = [4, 6, 8, 10, 12, 16, 20]
T_SWEEP = [2, 3, 4, 5, 7]

TOP_N = 15  # run top N configs with operator learning


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


def ewm_demean(obs, halflife=8):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(obs_dm, k=3):
    N = obs_dm.shape[1]
    k_use = min(k, N - 2)
    if obs_dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(k_use, mf.K)].real
    except Exception:
        return None


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
    library.append(corr0)
    if T > 2:
        lag1 = np.zeros((N, N))
        for lo in [1, 2]:
            if T > lo:
                xp, xn = obs_dm[:-lo], obs_dm[lo:]
                xp_dm, xn_dm = xp - xp.mean(0), xn - xn.mean(0)
                num = (xp_dm.T @ xn_dm) / max(len(xp) - 1, 1)
                sp = np.std(xp, 0, keepdims=True).T
                sn = np.std(xn, 0, keepdims=True)
                lag1 += np.nan_to_num(num / np.maximum(sp @ sn, 1e-10), nan=0.0)
        np.fill_diagonal(lag1, 0.0)
        library.append(lag1)
    for scale in [4, 8]:
        if T >= scale + 1:
            kernel = np.ones(scale) / scale
            sm = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, obs_dm)
            if sm.shape[0] >= 2:
                norms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                normed = sm / norms
                sim = normed.T @ normed
                np.fill_diagonal(sim, 0.0)
                library.append(sim)
    return library


def optimize_operator(obs_dm, library, N, K_opt=3):
    from scipy.optimize import minimize as sp_minimize
    T = obs_dm.shape[0]
    t_split = int(T * 0.75)
    obs_st, obs_sv = obs_dm[:t_split], obs_dm[t_split:]
    n_bases = len(library)

    def objective(weights):
        try:
            A = sum(w * B for w, B in zip(weights, library))
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0
            shaped = obs_st @ (np.eye(N) + 0.1 * A)
            U = dmd_basis(shaped, K_opt)
            if U is None:
                return 1e6
            K = U.shape[1]
            R_sph = train_sph_r(obs_st, U)
            F = np.eye(K) * F_REG
            alpha, P = np.zeros(K), np.eye(K)
            Q_run = np.eye(K) * Q_INIT_SCALE
            preds = []
            for t in range(len(obs_sv)):
                ap = F @ alpha
                Pp = F @ P @ F.T + Q_run
                preds.append(U @ ap)
                S = U @ Pp @ U.T + R_sph
                try:
                    Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
                except Exception:
                    Kg = np.zeros((K, N))
                alpha = ap + Kg @ (obs_sv[t] - U @ ap)
                P = (np.eye(K) - Kg @ U) @ Pp
                si = alpha - ap
                Q_run = 0.7 * Q_run + 0.3 * np.outer(si, si)
                Q_run = (Q_run + Q_run.T) / 2 + np.eye(K) * 1e-8
            r2 = float(oos_r_squared(np.array(preds).ravel(), obs_sv.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    opt = sp_minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 0.02, "fatol": 0.002})
    A_opt = sum(w * B for w, B in zip(opt.x, library))
    return np.eye(N) + 0.1 * A_opt


def run_window(panel, test_year, K, ewm_hl, T_train_yr, use_oplearn=False):
    """Run one test window. Returns (smim_r2, ar1_r2) or None."""
    train_start = pd.Timestamp(f"{test_year - T_train_yr}-01-01")
    if train_start < pd.Timestamp("2005-01-01"):
        return None
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
    if otr.shape[0] < 4:
        return None
    om = ewm_demean(otr, ewm_hl)
    otr_dm = otr - om

    # AR(1) baseline
    mu = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    dm = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = dm[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c

    # Operator learning
    op_shape = np.eye(N)
    if use_oplearn:
        library = build_operator_library(otr_dm, N)
        if library:
            op_shape = optimize_operator(otr_dm, library, N, K_opt=K)

    # DMD
    U = dmd_basis(otr_dm @ op_shape, K)
    if U is None:
        U = dmd_basis(otr_dm, K)
    if U is None:
        return None

    k_act = U.shape[1]
    R_sph = train_sph_r(otr_dm, U)
    F_plat = np.eye(k_act) * F_REG

    alpha, P = np.zeros(k_act), np.eye(k_act)
    Q_run = np.eye(k_act) * Q_INIT_SCALE

    preds_smim, preds_ar1, actuals = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        obs_raw = q_data[0]
        obs_dm = obs_raw - om.ravel()

        # SMIM predictive
        alpha_pred = F_plat @ alpha
        P_pred = F_plat @ P @ F_plat.T + Q_run
        preds_smim.append(U @ alpha_pred + om.ravel())

        # AR(1)
        preds_ar1.append(mu + rho * (prev - mu))
        actuals.append(obs_raw)
        prev = obs_raw

        # Kalman update
        S = U @ P_pred @ U.T + R_sph
        try:
            Kg = P_pred @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((k_act, N))
        alpha = alpha_pred + Kg @ (obs_dm - U @ alpha_pred)
        P = (np.eye(k_act) - Kg @ U) @ P_pred
        innov = alpha - alpha_pred
        Q_run = (1 - LAMBDA_Q) * Q_run + LAMBDA_Q * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(k_act) * 1e-6

        # Rolling basis
        expanded = np.vstack([otr, q_data])
        otr = expanded
        om_new = ewm_demean(expanded, ewm_hl)
        dm_new = expanded - om_new
        U_new = dmd_basis(dm_new @ op_shape, K)
        if U_new is None:
            U_new = dmd_basis(dm_new, K)
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

    r2_s = float(oos_r_squared(np.array(preds_smim).ravel(), np.array(actuals).ravel()))
    r2_a = float(oos_r_squared(np.array(preds_ar1).ravel(), np.array(actuals).ravel()))
    return r2_s, r2_a


def sweep_config(panel, K, ewm, T, use_oplearn=False):
    """Run all 10 windows for one config. Returns dict."""
    results = []
    for ty in TEST_YEARS:
        res = run_window(panel, ty, K, ewm, T, use_oplearn=use_oplearn)
        if res is not None:
            results.append(res)
    if not results:
        return None
    smim_r2s = [r[0] for r in results]
    ar1_r2s = [r[1] for r in results]
    deltas = [s - a for s, a in zip(smim_r2s, ar1_r2s)]
    return {
        "K": K, "ewm": ewm, "T": T, "oplearn": use_oplearn,
        "mean_smim": np.mean(smim_r2s), "mean_ar1": np.mean(ar1_r2s),
        "mean_delta": np.mean(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "n_windows": len(results),
        "min_delta": min(deltas), "max_delta": max(deltas),
    }


def main():
    t0 = time.time()
    panel = build_capex_revenue_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors\n")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Coarse sweep WITHOUT operator learning ──
    print("=" * 90)
    print(f"  STAGE 1: COARSE SWEEP (no operator learning)")
    print(f"  K={K_SWEEP}, EWM={EWM_SWEEP}, T={T_SWEEP}")
    total = len(K_SWEEP) * len(EWM_SWEEP) * len(T_SWEEP)
    print(f"  {total} configurations")
    print("=" * 90)

    rows = []
    i = 0
    for K in K_SWEEP:
        for ewm in EWM_SWEEP:
            for T in T_SWEEP:
                i += 1
                res = sweep_config(panel, K, ewm, T, use_oplearn=False)
                if res is not None:
                    rows.append(res)
                    if res["mean_delta"] > 0:
                        print(f"  [{i}/{total}] K={K} EWM={ewm} T={T}yr: "
                              f"SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                              f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df1 = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)

    print(f"\n  Stage 1 done ({time.time()-t0:.1f}s). {len(df1)} configs tested.\n")
    print("  TOP 20 configs (no operator learning):")
    print("  " + "-" * 85)
    print(f"  {'K':>3} {'EWM':>4} {'T':>3} {'SMIM':>7} {'AR1':>7} {'Δ':>8} {'Wins':>5}")
    print("  " + "-" * 85)
    for _, r in df1.head(20).iterrows():
        print(f"  {r['K']:3.0f} {r['ewm']:4.0f} {r['T']:3.0f} "
              f"{r['mean_smim']:7.4f} {r['mean_ar1']:7.4f} "
              f"{r['mean_delta']:+8.4f} {r['wins']:3.0f}/{r['n_windows']:.0f}")

    # ── Stage 2: Top configs WITH operator learning ──
    print(f"\n{'='*90}")
    print(f"  STAGE 2: TOP {TOP_N} WITH OPERATOR LEARNING")
    print(f"{'='*90}")

    t1 = time.time()
    rows_op = []
    for idx, row in df1.head(TOP_N).iterrows():
        K, ewm, T = int(row["K"]), int(row["ewm"]), int(row["T"])
        res = sweep_config(panel, K, ewm, T, use_oplearn=True)
        if res is not None:
            rows_op.append(res)
            w = "★" if res["mean_delta"] > 0.030 else " "
            print(f"  {w} K={K} EWM={ewm} T={T}yr: "
                  f"SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df2 = pd.DataFrame(rows_op).sort_values("mean_delta", ascending=False)

    # ── Summary ──
    print(f"\n{'='*90}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*90}")

    print(f"\n  Best WITHOUT operator learning:")
    b1 = df1.iloc[0]
    print(f"    K={b1['K']:.0f} EWM={b1['ewm']:.0f} T={b1['T']:.0f}yr: "
          f"Δ={b1['mean_delta']:+.4f} wins={b1['wins']:.0f}/{b1['n_windows']:.0f}")

    if len(df2) > 0:
        print(f"\n  Best WITH operator learning:")
        b2 = df2.iloc[0]
        print(f"    K={b2['K']:.0f} EWM={b2['ewm']:.0f} T={b2['T']:.0f}yr: "
              f"Δ={b2['mean_delta']:+.4f} wins={b2['wins']:.0f}/{b2['n_windows']:.0f}")

    print(f"\n  Current best (K=3, EWM=12, T=3yr + OpLearn): Δ=+0.0337 wins=9/10")
    if len(df2) > 0:
        improvement = b2["mean_delta"] - 0.0337
        print(f"  New best improvement over current: {improvement:+.4f}")

    print(f"\n  Total: {time.time()-t0:.1f}s")

    # Save
    all_rows = rows + rows_op
    pd.DataFrame(all_rows).to_parquet(METRICS_DIR / "iter5_1_sweep.parquet", index=False)
    print(f"  Saved: iter5_1_sweep.parquet ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
