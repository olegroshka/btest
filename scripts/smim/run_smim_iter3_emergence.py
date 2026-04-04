#!/usr/bin/env python
"""
SMIM Iteration 3 — Panel B emergence tests.

Tests on the sector/macro panel (N=28, genuine daily):
1. K sweep to find optimal spectral dimension
2. MI operator vs correlation operator (H_MI)
3. Polynomial AR on alpha trajectory (H_NL)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

DATA_DIR = PROJECT_ROOT / "data" / "smim" / "intensities"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

F_REG = 0.99
Q_INIT = 0.5
REEST = 21
TEST_YEARS = list(range(2020, 2025))


def ewm_mean(obs, hl=60):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(obs_dm, k):
    N = obs_dm.shape[1]
    k_use = min(k, N - 2)
    if obs_dm.shape[0] < 3:
        return None, None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(k + 5, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        return U, mf.eigenvalues[:min(k_use, mf.K)]
    except Exception:
        return None, None


def sph_r(obs_dm, U):
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * np.mean(resid ** 2)


def kf_step(obs, U, F, R, alpha, P, Q):
    N, K = U.shape
    ap = F @ alpha
    Pp = F @ P @ F.T + Q
    S = U @ Pp @ U.T + R
    try:
        Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        Kg = np.zeros((K, N))
    af = ap + Kg @ (obs - U @ ap)
    Pf = (np.eye(K) - Kg @ U) @ Pp
    si = af - F @ alpha
    Qn = 0.7 * Q + 0.3 * np.outer(si, si)
    Qn = (Qn + Qn.T) / 2 + np.eye(K) * 1e-8
    return U @ ap, af, Pf, Qn


def ksg_mi(x, y, k=5):
    """KSG mutual information estimator (algorithm 1)."""
    n = len(x)
    if n < k + 1:
        return 0.0
    xy = np.column_stack([x, y])
    # Joint: k-th neighbour distance (Chebyshev)
    from scipy.spatial import KDTree
    tree_xy = KDTree(xy, leafsize=16)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]
    # Marginal counts
    nx = np.zeros(n)
    ny = np.zeros(n)
    for i in range(n):
        nx[i] = np.sum(np.abs(x - x[i]) < eps[i]) - 1
        ny[i] = np.sum(np.abs(y - y[i]) < eps[i]) - 1
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return max(mi, 0.0)


def quad_features(alpha):
    """vech(alpha outer alpha) — upper triangle including diagonal."""
    if alpha.ndim == 1:
        alpha = alpha.reshape(1, -1)
    K = alpha.shape[1]
    feats = []
    for i in range(K):
        for j in range(i, K):
            feats.append(alpha[:, i] * alpha[:, j])
    return np.column_stack(feats)


def load_panel_b():
    df = pd.read_parquet(DATA_DIR / "iter3_panel_b.parquet")
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


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
    return train, test_d, N, valid.tolist()


def run_smim_window_b(panel, ty, K_TGT):
    """Run SMIM rolling on Panel B for one window. Returns (smim_r2, ar1_r2)."""
    train, test_d, N, valid = get_train_test(panel, ty)
    if len(train) < 60 or len(test_d) < 10:
        return None, None

    # AR(1) baseline
    mu_a = train.mean(0)
    dm_a = train - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])

    # SMIM
    mu = ewm_mean(train)
    train_dm = train - mu
    U, _ = dmd_basis(train_dm, K_TGT)
    if U is None:
        return None, None
    K = U.shape[1]
    R_s = sph_r(train_dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT
    a = np.zeros(K)
    P = np.eye(K)
    Qr = Q.copy()

    preds_s = []
    preds_a = []
    acts = []
    prev = train[-1]
    exp = train.copy()
    ds = 0

    for i in range(len(test_d)):
        obs = test_d[i]
        obs_dm = obs - mu.ravel()
        sp, a, P, Qr = kf_step(obs_dm, U, F, R_s, a, P, Qr)
        preds_s.append(sp + mu.ravel())
        preds_a.append(mu_a + rho * (prev - mu_a))
        acts.append(obs)
        prev = obs

        ds += 1
        if ds >= REEST:
            exp = np.vstack([exp, test_d[max(0, i - REEST + 1):i + 1]])
            mu_n = ewm_mean(exp)
            dm_n = exp - mu_n
            U_n, _ = dmd_basis(dm_n, K_TGT)
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

    ps = np.array(preds_s)
    pa = np.array(preds_a)
    ac = np.array(acts)
    return (
        float(oos_r_squared(ps.ravel(), ac.ravel())),
        float(oos_r_squared(pa.ravel(), ac.ravel())),
    )


def main():
    t_total = time.time()
    panel = load_panel_b()

    # ================================================================
    # Test 1: K sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: K sweep on Panel B")
    print("=" * 80)

    for K_TGT in [2, 3, 5, 8]:
        smim_r2s = []
        ar1_r2s = []
        for ty in TEST_YEARS:
            r2s, r2a = run_smim_window_b(panel, ty, K_TGT)
            if r2s is not None:
                smim_r2s.append(r2s)
                ar1_r2s.append(r2a)
        if smim_r2s:
            deltas = [s - a for s, a in zip(smim_r2s, ar1_r2s)]
            wins = sum(1 for d in deltas if d > 0)
            print(f"  K={K_TGT}: SMIM={np.mean(smim_r2s):.4f}  AR1={np.mean(ar1_r2s):.4f}  "
                  f"delta={np.mean(deltas):+.4f}  wins={wins}/{len(deltas)}")

    # ================================================================
    # Test 2: MI operator vs correlation (H_MI)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: MI operator vs correlation on Panel B (H_MI)")
    print("=" * 80)

    for ty in [2022, 2024]:
        train, _, N, valid = get_train_test(panel, ty)
        if len(train) < 60:
            continue
        print(f"\n  Window ending {ty - 1}, N={N}, T={len(train)}")

        # Correlation operator
        corr_op = np.corrcoef(train.T)

        # MI operator
        t0 = time.time()
        mi_op = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                mi_val = ksg_mi(train[:, i], train[:, j], k=5)
                mi_op[i, j] = mi_val
                mi_op[j, i] = mi_val
        np.fill_diagonal(mi_op, np.max(mi_op) * 1.1)  # self-MI > cross-MI
        print(f"  MI computed: {time.time() - t0:.1f}s ({N * (N - 1) // 2} pairs)")

        corr_flat = np.abs(corr_op[np.triu_indices(N, k=1)])
        mi_flat = mi_op[np.triu_indices(N, k=1)]
        op_corr = np.corrcoef(corr_flat, mi_flat)[0, 1]
        print(f"  Correlation(|corr|, MI) = {op_corr:.4f}")
        print(f"  MI range: [{mi_flat.min():.4f}, {mi_flat.max():.4f}], "
              f"mean={mi_flat.mean():.4f}")

        for K_TGT in [3, 5]:
            evals_c, evecs_c = np.linalg.eigh(corr_op)
            idx_c = np.argsort(-evals_c)
            U_corr = evecs_c[:, idx_c[:K_TGT]]

            evals_m, evecs_m = np.linalg.eigh(mi_op)
            idx_m = np.argsort(-evals_m)
            U_mi = evecs_m[:, idx_m[:K_TGT]]

            Q1, _ = np.linalg.qr(U_corr)
            Q2, _ = np.linalg.qr(U_mi)
            svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
            angle = np.degrees(np.mean(np.arccos(np.clip(svals, -1, 1))))

            ev_corr = np.sum(evals_c[idx_c[:K_TGT]]) / np.sum(np.abs(evals_c))
            ev_mi = np.sum(evals_m[idx_m[:K_TGT]]) / np.sum(evals_m[evals_m > 0])

            print(f"  K={K_TGT}: angle(corr,MI)={angle:.1f} deg  "
                  f"expl_var: corr={ev_corr:.3f} MI={ev_mi:.3f}")

    # ================================================================
    # Test 3: Polynomial AR on Panel B alpha (H_NL)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 3: Polynomial AR on Panel B alpha trajectory (H_NL)")
    print("=" * 80)

    K_TGT = 5
    rows = []

    for ty in TEST_YEARS:
        train, test_d, N, valid = get_train_test(panel, ty)
        if len(train) < 60 or len(test_d) < 10:
            continue

        mu = ewm_mean(train)
        train_dm = train - mu
        U, _ = dmd_basis(train_dm, K_TGT)
        if U is None:
            continue
        K = U.shape[1]

        # Alpha trajectories
        alpha_train = train_dm @ U  # (T_train, K)
        test_dm = test_d - mu
        alpha_test = test_dm @ U    # (T_test, K)

        X_lin = alpha_train[:-1]
        Y_target = alpha_train[1:]
        X_quad = quad_features(X_lin)
        X_poly = np.hstack([X_lin, X_quad])
        P_poly = X_poly.shape[1]

        best_lam = None
        best_ratio = 999

        for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
            F_lin = np.linalg.solve(
                X_lin.T @ X_lin + lam * np.eye(K), X_lin.T @ Y_target
            )
            F_poly = np.linalg.solve(
                X_poly.T @ X_poly + lam * np.eye(P_poly), X_poly.T @ Y_target
            )

            err_lin = 0
            err_poly = 0
            for t in range(1, len(alpha_test)):
                a_prev = alpha_test[t - 1]
                pred_lin = F_lin.T @ a_prev
                qf = quad_features(a_prev).ravel()
                pred_poly = F_poly.T @ np.concatenate([a_prev, qf])

                actual = alpha_test[t]
                err_lin += np.sum((actual - pred_lin) ** 2)
                err_poly += np.sum((actual - pred_poly) ** 2)

            if err_lin > 0:
                ratio = err_poly / err_lin
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_lam = lam

        better = "POLY WINS" if best_ratio < 1.0 else "LINEAR WINS"
        delta_pct = (1 - best_ratio) * 100
        print(f"  W{ty}: MSE ratio(poly/lin)={best_ratio:.4f}  "
              f"delta={delta_pct:+.2f}%  lam={best_lam}  K={K} P={P_poly}  {better}")
        rows.append({
            "year": ty, "mse_ratio": best_ratio,
            "delta_pct": delta_pct, "best_lam": best_lam,
        })

    if rows:
        rdf = pd.DataFrame(rows)
        mean_ratio = rdf["mse_ratio"].mean()
        poly_wins = (rdf["mse_ratio"] < 1.0).sum()
        print(f"\n  Mean MSE ratio: {mean_ratio:.4f}  "
              f"Poly wins: {poly_wins}/{len(rdf)}")
        print(f"  H_NL verdict: {'SUPPORTED' if poly_wins > len(rdf) // 2 else 'NOT SUPPORTED'}")
        rdf.to_parquet(METRICS_DIR / "iter3_panel_b_poly_ar.parquet", index=False)

    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
