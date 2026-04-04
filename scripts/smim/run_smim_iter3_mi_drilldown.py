#!/usr/bin/env python
"""
SMIM Iteration 3 — MI Drilldown.

Tests whether the MI-derived spectral basis (which is 68 degrees from
the correlation basis) produces better SMIM predictions than correlation/DMD.

Experiments:
  MI-1a: MI basis -> Kalman pipeline (replace DMD basis with MI eigenvectors)
  MI-1b: MI-weighted DMD (weight snapshot matrix by MI row sums)
  MI-1c: Blended basis (alpha * U_MI + (1-alpha) * U_corr)
  MI-1d: MI basis on pre-whitened Panel B (AR(1) residuals)
  MI-1e: Test on Panel A weekly for robustness

All compared against: correlation-DMD baseline and AR(1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.spatial import KDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

DATA_DIR = PROJECT_ROOT / "data" / "smim" / "intensities"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

F_REG = 0.99
Q_INIT = 0.5
REEST = 21


def ewm_mean(obs, hl=60):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def sph_r(obs_dm, U):
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * np.mean(resid ** 2)


def kf_step(obs, U, F, R, alpha, P, Q):
    N, K = U.shape
    ap = F @ alpha
    Pp = F @ P @ F.T + Q
    pred = U @ ap
    S = U @ Pp @ U.T + R
    try:
        Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        Kg = np.zeros((K, N))
    af = ap + Kg @ (obs - pred)
    Pf = (np.eye(K) - Kg @ U) @ Pp
    si = af - F @ alpha
    Qn = 0.7 * Q + 0.3 * np.outer(si, si)
    Qn = (Qn + Qn.T) / 2 + np.eye(K) * 1e-8
    return pred, af, Pf, Qn


def ksg_mi(x, y, k=5):
    """KSG mutual information estimator."""
    n = len(x)
    if n < k + 1:
        return 0.0
    xy = np.column_stack([x, y])
    tree = KDTree(xy, leafsize=16)
    dists, _ = tree.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]
    nx = np.array([np.sum(np.abs(x - x[i]) < eps[i]) - 1 for i in range(n)])
    ny = np.array([np.sum(np.abs(y - y[i]) < eps[i]) - 1 for i in range(n)])
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    return max(digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny)), 0.0)


def compute_mi_operator(data):
    """Compute pairwise MI matrix. data: (T, N)."""
    N = data.shape[1]
    mi_op = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            mi_val = ksg_mi(data[:, i], data[:, j], k=5)
            mi_op[i, j] = mi_val
            mi_op[j, i] = mi_val
    np.fill_diagonal(mi_op, np.max(mi_op) * 1.1 if np.max(mi_op) > 0 else 1.0)
    return mi_op


def mi_basis(mi_op, k):
    """Top-k eigenvectors of MI operator."""
    evals, evecs = np.linalg.eigh(mi_op)
    idx = np.argsort(-evals)
    return evecs[:, idx[:k]]


def corr_basis(data, k):
    """Top-k eigenvectors of correlation matrix."""
    C = np.corrcoef(data.T)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(-evals)
    return evecs[:, idx[:k]]


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


def run_pipeline_with_basis(train, test_d, U, N, reest_fn=None):
    """Run SMIM Kalman pipeline with a given basis U. Returns R²."""
    K = U.shape[1]
    mu = ewm_mean(train)
    train_dm = train - mu
    R_s = sph_r(train_dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT
    a = np.zeros(K)
    P = np.eye(K)
    Qr = Q.copy()

    preds = []
    acts = []
    exp = train.copy()
    ds = 0

    for i in range(len(test_d)):
        obs = test_d[i]
        obs_dm = obs - mu.ravel()
        sp, a, P, Qr = kf_step(obs_dm, U, F, R_s, a, P, Qr)
        preds.append(sp + mu.ravel())
        acts.append(obs)

        ds += 1
        if ds >= REEST and reest_fn is not None:
            exp = np.vstack([exp, test_d[max(0, i - REEST + 1):i + 1]])
            mu_n = ewm_mean(exp)
            dm_n = exp - mu_n
            U_n = reest_fn(dm_n, exp, K)
            if U_n is not None:
                Kn = U_n.shape[1]
                a = U_n.T @ obs_dm
                P = np.eye(Kn)
                Qr = np.eye(Kn) * Q_INIT
                F = np.eye(Kn) * F_REG
                R_s = sph_r(dm_n, U_n)
                U, K = U_n, Kn
                mu = mu_n
            ds = 0

    ps = np.array(preds)
    ac = np.array(acts)
    return float(oos_r_squared(ps.ravel(), ac.ravel()))


def ar1_r2(train, test_d, N):
    """AR(1) baseline R²."""
    mu_a = train.mean(0)
    dm = train - mu_a
    rho = np.array([
        np.corrcoef(dm[:-1, j], dm[1:, j])[0, 1]
        if dm[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])
    preds = []
    prev = train[-1]
    for i in range(len(test_d)):
        preds.append(mu_a + rho * (prev - mu_a))
        prev = test_d[i]
    return float(oos_r_squared(np.array(preds).ravel(), test_d.ravel()))


def get_train_test(panel, ty, train_years=5):
    ts = pd.Timestamp(f"{ty - train_years}-01-01")
    te_tr = pd.Timestamp(f"{ty - 1}-12-31")
    te_te = pd.Timestamp(f"{ty}-12-31")
    all_d = panel[(panel.index >= ts) & (panel.index <= te_te)].copy()
    cov = all_d.loc[all_d.index <= te_tr].notna().mean()
    valid = cov[cov > 0.5].index
    all_d = all_d[valid].ffill().fillna(all_d[valid].mean())
    N = len(valid)
    train = all_d[all_d.index <= te_tr].values.astype(np.float64)
    test_dates = all_d.index[all_d.index > te_tr]
    test_d = all_d.loc[test_dates].values.astype(np.float64)
    return train, test_d, N


def main():
    t_total = time.time()
    df = pd.read_parquet(DATA_DIR / "iter3_panel_b.parquet")
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    TEST_YEARS = list(range(2020, 2025))
    K_TGT = 5

    print("=" * 80)
    print("  MI DRILLDOWN: Testing MI basis in the SMIM pipeline")
    print("=" * 80)

    all_rows = []

    for ty in TEST_YEARS:
        train, test_d, N = get_train_test(panel, ty)
        if len(train) < 60 or len(test_d) < 10:
            continue

        t0 = time.time()
        mu = ewm_mean(train)
        train_dm = train - mu

        # ---- Compute bases ----
        # DMD basis (standard)
        U_dmd = dmd_basis(train_dm, K_TGT)

        # Correlation basis (PCA)
        U_corr = corr_basis(train, K_TGT)

        # MI basis
        mi_op = compute_mi_operator(train)
        U_mi = mi_basis(mi_op, K_TGT)

        # MI-weighted DMD: weight rows by MI connectivity
        mi_strength = mi_op.sum(axis=1)
        mi_strength = mi_strength / mi_strength.mean()  # normalise to mean=1
        weighted_dm = train_dm * mi_strength[np.newaxis, :]  # upweight connected actors
        U_mi_dmd = dmd_basis(weighted_dm, K_TGT)

        # ---- Run pipelines ----
        # DMD rolling basis (recompute DMD)
        def reest_dmd(dm_new, raw_new, k):
            return dmd_basis(dm_new, k)

        # MI rolling basis (recompute MI operator + eigenvectors)
        def reest_mi(dm_new, raw_new, k):
            mi_new = compute_mi_operator(raw_new)
            return mi_basis(mi_new, k)

        # Correlation rolling basis
        def reest_corr(dm_new, raw_new, k):
            return corr_basis(raw_new, k)

        r2_ar1 = ar1_r2(train, test_d, N)

        # Static bases (no rolling update)
        r2_dmd_static = run_pipeline_with_basis(train, test_d, U_dmd, N) if U_dmd is not None else None
        r2_corr_static = run_pipeline_with_basis(train, test_d, U_corr, N)
        r2_mi_static = run_pipeline_with_basis(train, test_d, U_mi, N)

        # Rolling bases
        r2_dmd_roll = run_pipeline_with_basis(train, test_d, U_dmd, N, reest_dmd) if U_dmd is not None else None
        r2_mi_roll = run_pipeline_with_basis(train, test_d, U_mi, N, reest_mi)

        # MI-weighted DMD
        r2_mi_dmd = run_pipeline_with_basis(train, test_d, U_mi_dmd, N, reest_dmd) if U_mi_dmd is not None else None

        elapsed = time.time() - t0

        row = {
            "year": ty, "N": N,
            "ar1": r2_ar1,
            "dmd_static": r2_dmd_static,
            "corr_static": r2_corr_static,
            "mi_static": r2_mi_static,
            "dmd_roll": r2_dmd_roll,
            "mi_roll": r2_mi_roll,
            "mi_dmd": r2_mi_dmd,
        }
        all_rows.append(row)

        # Print comparison
        print(f"\n  W{ty} (N={N}, {elapsed:.1f}s):")
        print(f"    AR(1):               {r2_ar1:.4f}")
        print(f"    --- Static basis ---")
        print(f"    Correlation (PCA):   {r2_corr_static:.4f}  delta vs AR1: {r2_corr_static - r2_ar1:+.4f}")
        if r2_dmd_static is not None:
            print(f"    DMD:                 {r2_dmd_static:.4f}  delta vs AR1: {r2_dmd_static - r2_ar1:+.4f}")
        print(f"    MI:                  {r2_mi_static:.4f}  delta vs AR1: {r2_mi_static - r2_ar1:+.4f}")
        print(f"    --- Rolling basis ---")
        if r2_dmd_roll is not None:
            print(f"    DMD rolling:         {r2_dmd_roll:.4f}  delta vs AR1: {r2_dmd_roll - r2_ar1:+.4f}")
        print(f"    MI rolling:          {r2_mi_roll:.4f}  delta vs AR1: {r2_mi_roll - r2_ar1:+.4f}")
        if r2_mi_dmd is not None:
            print(f"    MI-weighted DMD:     {r2_mi_dmd:.4f}  delta vs AR1: {r2_mi_dmd - r2_ar1:+.4f}")

        # Key comparison: MI vs DMD
        if r2_dmd_roll is not None:
            mi_vs_dmd = r2_mi_roll - r2_dmd_roll
            winner = "MI" if mi_vs_dmd > 0 else "DMD"
            print(f"    >>> MI roll vs DMD roll: {mi_vs_dmd:+.4f} ({winner} wins)")

    # ---- Summary ----
    results = pd.DataFrame(all_rows)
    results.to_parquet(METRICS_DIR / "iter3_mi_drilldown.parquet", index=False)

    print(f"\n\n{'#'*80}")
    print(f"#  MI DRILLDOWN SUMMARY")
    print(f"{'#'*80}")

    for col in ["ar1", "corr_static", "dmd_static", "mi_static", "dmd_roll", "mi_roll", "mi_dmd"]:
        vals = results[col].dropna()
        if not vals.empty:
            label = col.replace("_", " ").title()
            delta_ar1 = (vals - results.loc[vals.index, "ar1"]).mean()
            print(f"  {label:25s}: mean R2={vals.mean():.4f}  delta vs AR1={delta_ar1:+.4f}")

    # H_MI verdict
    mi_roll_vals = results["mi_roll"].dropna()
    dmd_roll_vals = results["dmd_roll"].dropna()
    if not mi_roll_vals.empty and not dmd_roll_vals.empty:
        common = results.dropna(subset=["mi_roll", "dmd_roll"])
        mi_wins = (common["mi_roll"] > common["dmd_roll"]).sum()
        delta = (common["mi_roll"] - common["dmd_roll"]).mean()
        print(f"\n  H_MI (MI roll vs DMD roll): delta={delta:+.4f}  MI wins {mi_wins}/{len(common)}")
        print(f"  Verdict: {'MI basis OUTPERFORMS' if delta > 0 else 'MI basis does NOT outperform'} DMD")

    mi_static = results["mi_static"].dropna()
    corr_static = results["corr_static"].dropna()
    if not mi_static.empty and not corr_static.empty:
        delta_sc = (mi_static - corr_static).mean()
        wins_sc = (mi_static.values > corr_static.values).sum()
        print(f"  MI static vs Corr static: delta={delta_sc:+.4f}  MI wins {wins_sc}/{len(mi_static)}")

    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
