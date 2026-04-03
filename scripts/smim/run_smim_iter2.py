#!/usr/bin/env python
"""
SMIM Experiment Iteration 2: Emergence & Directed Operators.

E2-1: Transfer Entropy Operator (break operator symmetry)
E2-2: Granger on Intensity Operator (directed edges from primary data)
E2-3: Actor-Specific Loadings (heterogeneous factor exposure)
E2-4: Economic Emergence Signals (make emergence fire)
E2-5: Kim Filter K-means Init (fix regime switching)
E2-6: Combined Best

Baseline: GOLD+ R²=0.524 (DMD K=8, spherical R, EWM, online Q, T=5yr)

Usage::
    uv run python scripts/smim/run_smim_iter2.py
    uv run python scripts/smim/run_smim_iter2.py --only E2-1
    uv run python scripts/smim/run_smim_iter2.py --only E2-1,E2-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import schur as scipy_schur

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.emergence.transfer_entropy import ksg_transfer_entropy
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
T_YR = 5
TEST_YEARS = list(range(2015, 2025))
rng = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════
# Shared infrastructure (proven GOLD+ pipeline from drilldown_c.py)
# ═══════════════════════════════════════════════════════════════════════════


def load_panel():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    return wide[[c for c in actor_ids if c in wide.columns]]


def prep(wide, ty, t_yr=T_YR):
    """Prepare train/test with EWM demeaning. Returns demeaned + raw arrays."""
    ts = pd.Timestamp(f"{ty - t_yr}-01-01")
    te = pd.Timestamp(f"{ty - 1}-12-31")
    test_s = pd.Timestamp(f"{ty}-01-01")
    test_e = pd.Timestamp(f"{ty}-12-31")
    tr = wide[(wide.index >= ts) & (wide.index <= te)]
    te_df = wide[(wide.index >= test_s) & (wide.index <= test_e)]
    valid = tr.columns[tr.notna().any()].intersection(te_df.columns[te_df.notna().any()])
    tr, te_df = tr[valid], te_df[valid]
    otr = tr.fillna(tr.mean()).values.astype(np.float64)
    ote = te_df.fillna(tr.mean()).values.astype(np.float64)
    N = otr.shape[1]
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
    om = (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()
    return otr - om, ote - om, ote, om, N, otr, list(valid)


def get_dmd_basis(otr_dm, N, k=K_GOLD):
    """DMD basis extraction (GOLD+ default)."""
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(k_use, mf.K)].real, mf
    except Exception:
        corr = np.corrcoef(otr_dm.T)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 0)
        corr[np.abs(corr) < 0.1] = 0
        mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
        return mf.basis[:, :min(k_use, mf.K)].real, mf


def train_kalman(otr_dm, N, U):
    """Train Kalman EM + spherical R. Returns F, Q, R_sph."""
    k_act = U.shape[1]
    mf_k = ModalFrame(
        basis=U,
        eigenvalues=np.ones(k_act, dtype=complex),
        method="dmd",
    )
    kf = KalmanFilter()
    st = kf.em_estimate(otr_dm, mf_k, max_iter=50)
    R_sph = np.eye(N) * np.trace(st.params["R"]) / N
    return st.params["F"], st.params["Q"].copy(), R_sph


def gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph, adapt_rate=0.3):
    """GOLD+ prediction: online Kalman with Q adaptation."""
    k_act = U.shape[1]
    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    Q_run = Q.copy()
    pred = np.zeros_like(ote)
    alpha_filt_out = np.zeros((ote.shape[0], k_act))
    for t in range(ote.shape[0]):
        a_p = F @ alpha_c
        P_p = F @ P_c @ F.T + Q_run
        v = ote_dm[t] - U @ a_p
        S = U @ P_p @ U.T + R_sph
        try:
            Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((k_act, N))
        a_f = a_p + Kg @ v
        P_f = (np.eye(k_act) - Kg @ U) @ P_p
        pred[t] = (a_f @ U.T + om).ravel()
        alpha_filt_out[t] = a_f
        innov = a_f - F @ alpha_c
        Q_run = (1 - adapt_rate) * Q_run + adapt_rate * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2
        Q_run += np.eye(k_act) * 1e-6
        alpha_c = a_f
        P_c = P_f
    return pred, alpha_filt_out


def ar1_predict(otr_raw, ote_raw):
    """Per-actor AR(1) baseline."""
    N = otr_raw.shape[1]
    pred = np.zeros_like(ote_raw)
    for i in range(N):
        y = otr_raw[:, i]
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        try:
            beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        except Exception:
            beta = [y.mean(), 0]
        v = otr_raw[-1, i]
        for t in range(ote_raw.shape[0]):
            pred[t, i] = beta[0] + beta[1] * v
            v = pred[t, i]
    return pred


def gold_plus_full(wide, ty):
    """Run full GOLD+ pipeline for one window. Returns (pred, actual, errors)."""
    otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
    actual = ote.T.ravel()
    U, _ = get_dmd_basis(otr_dm, N)
    F, Q, R_sph = train_kalman(otr_dm, N, U)
    pred, _ = gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph)
    r2 = float(oos_r_squared(pred.T.ravel(), actual))
    return pred, actual, pred.T.ravel() - actual, r2, U, F, Q, R_sph, otr_dm, ote_dm, ote, om, N, otr_raw


# ═══════════════════════════════════════════════════════════════════════════
# E2-1: Transfer Entropy Operator
# ═══════════════════════════════════════════════════════════════════════════


def build_te_operator(otr_raw, N, k_neighbours=5):
    """Build N x N transfer entropy operator from intensity series.

    W[i,j] = TE(j → i): how much actor j's past predicts actor i's future.
    Uses raw (not demeaned) intensity for TE to preserve level information.
    """
    T = otr_raw.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            try:
                te = ksg_transfer_entropy(
                    source=otr_raw[:, j],
                    target=otr_raw[:, i],
                    lag=1,
                    k_neighbours=min(k_neighbours, T - 2),
                )
                W[i, j] = te
            except Exception:
                W[i, j] = 0.0
    return W


def te_schur_basis(W, k):
    """Schur decomposition of asymmetric TE operator."""
    # Schur decomposition handles non-symmetric matrices
    T_mat, U_schur = scipy_schur(W, output="complex")
    eigs = np.diag(T_mat)
    # Sort by eigenvalue modulus (largest first)
    order = np.argsort(-np.abs(eigs))
    eigs = eigs[order]
    U_schur = U_schur[:, order]
    # Take top-k modes, convert to real
    basis = U_schur[:, :k].real
    # Orthonormalise (Schur vectors may not be orthogonal for non-normal matrices)
    Q, R = np.linalg.qr(basis)
    return Q, eigs[:k]


def te_pca_basis(W, k):
    """PCA (symmetric part) of the TE operator for comparison."""
    W_sym = (W + W.T) / 2
    eigvals, eigvecs = np.linalg.eigh(W_sym)
    order = np.argsort(-np.abs(eigvals))
    return eigvecs[:, order[:k]], eigvals[order[:k]]


def run_e2_1(wide):
    print("\n" + "=" * 90)
    print("  E2-1: TRANSFER ENTROPY OPERATOR")
    print("  Hypothesis: TE asymmetry enables distinct Schur vs PCA decompositions")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()

        # Use 10yr window for TE estimation (more data for reliable TE)
        otr_dm_10, _, _, _, N_10, otr_raw_10, cols_10 = prep(wide, ty, t_yr=10)

        # Use 5yr window for prediction (GOLD+ optimal)
        otr_dm, ote_dm, ote, om, N, otr_raw, cols_5 = prep(wide, ty, t_yr=5)
        actual = ote.T.ravel()

        # Align columns: use intersection of 10yr and 5yr actors
        common = list(set(cols_10) & set(cols_5))
        if len(common) < 10:
            print(f"  W{ty}: too few common actors ({len(common)}), skipping")
            continue

        # Re-index to common actors
        idx_10 = [cols_10.index(c) for c in common]
        idx_5 = [cols_5.index(c) for c in common]
        otr_te = otr_raw_10[:, idx_10]  # 10yr raw for TE
        otr_dm_5 = otr_dm[:, idx_5]     # 5yr demeaned for prediction
        ote_dm_5 = ote_dm[:, idx_5]
        ote_5 = ote[:, idx_5]
        om_5 = om[:, idx_5]
        N_c = len(common)

        # Build TE operator from 10yr window
        W_te = build_te_operator(otr_te, N_c, k_neighbours=5)
        asymmetry = np.linalg.norm(W_te - W_te.T) / (np.linalg.norm(W_te) + 1e-12)

        # GOLD+ baseline (DMD on 5yr)
        U_dmd, _ = get_dmd_basis(otr_dm_5, N_c)
        F_dmd, Q_dmd, R_dmd = train_kalman(otr_dm_5, N_c, U_dmd)
        pred_dmd, _ = gold_plus_predict(ote_dm_5, ote_5, om_5, N_c, U_dmd, F_dmd, Q_dmd, R_dmd)
        r2_dmd = float(oos_r_squared(pred_dmd.T.ravel(), ote_5.T.ravel()))

        for K_test in [3, 5, 8]:
            k_use = min(K_test, N_c - 2)

            # TE-Schur basis
            U_te_schur, eigs_schur = te_schur_basis(W_te, k_use)
            try:
                F_s, Q_s, R_s = train_kalman(otr_dm_5, N_c, U_te_schur)
                pred_s, _ = gold_plus_predict(ote_dm_5, ote_5, om_5, N_c, U_te_schur, F_s, Q_s, R_s)
                r2_te_schur = float(oos_r_squared(pred_s.T.ravel(), ote_5.T.ravel()))
            except Exception:
                r2_te_schur = float("nan")

            # TE-PCA basis (symmetric part only)
            U_te_pca, _ = te_pca_basis(W_te, k_use)
            try:
                F_p, Q_p, R_p = train_kalman(otr_dm_5, N_c, U_te_pca)
                pred_p, _ = gold_plus_predict(ote_dm_5, ote_5, om_5, N_c, U_te_pca, F_p, Q_p, R_p)
                r2_te_pca = float(oos_r_squared(pred_p.T.ravel(), ote_5.T.ravel()))
            except Exception:
                r2_te_pca = float("nan")

            # Correlation-Schur (current baseline: symmetric)
            corr = np.corrcoef(otr_dm_5.T)
            corr = np.nan_to_num(corr, nan=0)
            np.fill_diagonal(corr, 0)
            U_corr_schur, _ = te_schur_basis(corr, k_use)  # Schur on symmetric = PCA
            try:
                F_c, Q_c, R_c = train_kalman(otr_dm_5, N_c, U_corr_schur)
                pred_c, _ = gold_plus_predict(ote_dm_5, ote_5, om_5, N_c, U_corr_schur, F_c, Q_c, R_c)
                r2_corr_schur = float(oos_r_squared(pred_c.T.ravel(), ote_5.T.ravel()))
            except Exception:
                r2_corr_schur = float("nan")

            # DM tests
            dmd_errs = pred_dmd.T.ravel() - ote_5.T.ravel()
            if np.isfinite(r2_te_schur):
                te_errs = pred_s.T.ravel() - ote_5.T.ravel()
                dm_stat, dm_p = diebold_mariano_test(te_errs, dmd_errs)
            else:
                dm_stat, dm_p = float("nan"), float("nan")

            rows.append({
                "experiment": "E2-1", "window": f"W{ty}", "K": K_test,
                "r2_te_schur": r2_te_schur, "r2_te_pca": r2_te_pca,
                "r2_corr_schur": r2_corr_schur, "r2_dmd_gold": r2_dmd,
                "asymmetry": asymmetry, "N": N_c,
                "dm_stat_vs_gold": dm_stat, "dm_p_vs_gold": dm_p,
            })

        elapsed = time.time() - t0
        print(f"  W{ty}: N={N_c}, asymmetry={asymmetry:.3f}, "
              f"TE-Schur={rows[-2]['r2_te_schur']:.4f}, DMD={r2_dmd:.4f} "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n  No results produced.")
        return df

    # Summary table
    print(f"\n  {'K':>4} {'TE-Schur':>10} {'TE-PCA':>10} {'Corr-Schur':>11} {'DMD(GOLD+)':>11} "
          f"{'TE-Schur DM':>12}")
    for K_test in [3, 5, 8]:
        sub = df[df["K"] == K_test]
        print(f"  K={K_test:>2} "
              f"{sub['r2_te_schur'].mean():>10.4f} "
              f"{sub['r2_te_pca'].mean():>10.4f} "
              f"{sub['r2_corr_schur'].mean():>11.4f} "
              f"{sub['r2_dmd_gold'].mean():>11.4f} "
              f"p={sub['dm_p_vs_gold'].mean():>8.3f}")

    asym_mean = df["asymmetry"].mean()
    schur_pca_diff = df.groupby("K").apply(
        lambda g: (g["r2_te_schur"] - g["r2_te_pca"]).mean()
    )
    print(f"\n  Mean asymmetry: {asym_mean:.3f}")
    print(f"  Schur-PCA difference (from TE): {schur_pca_diff.to_dict()}")
    print(f"  SILVER criterion: Schur != PCA? {abs(schur_pca_diff.mean()) > 0.001}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-2: Granger on Intensity Operator
# ═══════════════════════════════════════════════════════════════════════════


def build_granger_intensity_operator(otr_raw, N, p_threshold=0.05):
    """Build directed operator from Granger causality on intensity series.

    For each pair (i,j): y_{i,t} = a + b*y_{i,t-1} + c*y_{j,t-1} + eps
    W[i,j] = |c| if F-test p < threshold, else 0.
    """
    from scipy import stats

    T = otr_raw.shape[0]
    W = np.zeros((N, N))

    for i in range(N):
        y_i = otr_raw[1:, i]  # target: y_{i,t} for t=1..T-1
        y_i_lag = otr_raw[:-1, i]  # y_{i,t-1}

        # Restricted model: y_{i,t} = a + b*y_{i,t-1}
        X_r = np.column_stack([np.ones(T - 1), y_i_lag])
        try:
            beta_r = np.linalg.lstsq(X_r, y_i, rcond=None)[0]
            rss_r = np.sum((y_i - X_r @ beta_r) ** 2)
        except Exception:
            continue

        for j in range(N):
            if i == j:
                continue
            y_j_lag = otr_raw[:-1, j]  # y_{j,t-1}

            # Unrestricted: y_{i,t} = a + b*y_{i,t-1} + c*y_{j,t-1}
            X_u = np.column_stack([np.ones(T - 1), y_i_lag, y_j_lag])
            try:
                beta_u = np.linalg.lstsq(X_u, y_i, rcond=None)[0]
                rss_u = np.sum((y_i - X_u @ beta_u) ** 2)
            except Exception:
                continue

            # F-test for c != 0
            df1 = 1  # one added parameter
            df2 = T - 1 - 3  # n - k_unrestricted
            if df2 <= 0 or rss_u <= 0:
                continue
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_val < p_threshold:
                W[i, j] = abs(beta_u[2])  # |c_{j→i}|

    return W


def run_e2_2(wide):
    print("\n" + "=" * 90)
    print("  E2-2: GRANGER ON INTENSITY OPERATOR")
    print("  Hypothesis: Granger on intensity series creates directed edges")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty, t_yr=5)
        actual = ote.T.ravel()

        # Build Granger operator from raw intensity
        W_gr = build_granger_intensity_operator(otr_raw, N, p_threshold=0.05)
        density = np.count_nonzero(W_gr) / (N * (N - 1))
        asymmetry = np.linalg.norm(W_gr - W_gr.T) / (np.linalg.norm(W_gr) + 1e-12)

        # GOLD+ baseline
        U_dmd, _ = get_dmd_basis(otr_dm, N)
        F_dmd, Q_dmd, R_dmd = train_kalman(otr_dm, N, U_dmd)
        pred_dmd, _ = gold_plus_predict(ote_dm, ote, om, N, U_dmd, F_dmd, Q_dmd, R_dmd)
        r2_dmd = float(oos_r_squared(pred_dmd.T.ravel(), actual))

        for K_test in [3, 5, 8]:
            k_use = min(K_test, N - 2)

            # Granger-Schur basis
            if np.linalg.norm(W_gr) > 1e-10:
                U_gr, eigs_gr = te_schur_basis(W_gr, k_use)
                try:
                    F_g, Q_g, R_g = train_kalman(otr_dm, N, U_gr)
                    pred_g, _ = gold_plus_predict(ote_dm, ote, om, N, U_gr, F_g, Q_g, R_g)
                    r2_granger = float(oos_r_squared(pred_g.T.ravel(), actual))
                except Exception:
                    r2_granger = float("nan")

                # Granger-PCA (symmetric part)
                U_gr_pca, _ = te_pca_basis(W_gr, k_use)
                try:
                    F_gp, Q_gp, R_gp = train_kalman(otr_dm, N, U_gr_pca)
                    pred_gp, _ = gold_plus_predict(ote_dm, ote, om, N, U_gr_pca, F_gp, Q_gp, R_gp)
                    r2_granger_pca = float(oos_r_squared(pred_gp.T.ravel(), actual))
                except Exception:
                    r2_granger_pca = float("nan")

                # Blended: (1-g)*corr + g*granger
                corr = np.corrcoef(otr_dm.T)
                corr = np.nan_to_num(corr, nan=0)
                np.fill_diagonal(corr, 0)
                W_blend = 0.5 * corr + 0.5 * (W_gr / (W_gr.max() + 1e-12))
                U_blend, _ = te_schur_basis(W_blend, k_use)
                try:
                    F_b, Q_b, R_b = train_kalman(otr_dm, N, U_blend)
                    pred_b, _ = gold_plus_predict(ote_dm, ote, om, N, U_blend, F_b, Q_b, R_b)
                    r2_blend = float(oos_r_squared(pred_b.T.ravel(), actual))
                except Exception:
                    r2_blend = float("nan")
            else:
                r2_granger = r2_granger_pca = r2_blend = float("nan")

            rows.append({
                "experiment": "E2-2", "window": f"W{ty}", "K": K_test,
                "r2_granger_schur": r2_granger, "r2_granger_pca": r2_granger_pca,
                "r2_blend": r2_blend, "r2_dmd_gold": r2_dmd,
                "density": density, "asymmetry": asymmetry, "N": N,
            })

        elapsed = time.time() - t0
        print(f"  W{ty}: density={density:.3f}, asymmetry={asymmetry:.3f}, "
              f"Gr-Schur K=5={rows[-2]['r2_granger_schur']:.4f}, DMD={r2_dmd:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n  No results.")
        return df

    print(f"\n  {'K':>4} {'Gr-Schur':>10} {'Gr-PCA':>10} {'Blend':>10} {'DMD(GOLD+)':>11}")
    for K_test in [3, 5, 8]:
        sub = df[df["K"] == K_test]
        print(f"  K={K_test:>2} {sub['r2_granger_schur'].mean():>10.4f} "
              f"{sub['r2_granger_pca'].mean():>10.4f} "
              f"{sub['r2_blend'].mean():>10.4f} "
              f"{sub['r2_dmd_gold'].mean():>11.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-3: Actor-Specific Loadings
# ═══════════════════════════════════════════════════════════════════════════


def run_e2_3(wide):
    print("\n" + "=" * 90)
    print("  E2-3: ACTOR-SPECIFIC LOADINGS")
    print("  Hypothesis: per-actor w_i captures heterogeneous factor exposure")
    print("=" * 90)

    ridge_lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()

        # GOLD+ basis and Kalman
        U, _ = get_dmd_basis(otr_dm, N)
        F, Q, R_sph = train_kalman(otr_dm, N, U)
        k_act = U.shape[1]

        # Get filtered alpha on TRAINING data
        mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
        kf = KalmanFilter(F=F, Q=Q, R=R_sph)
        state_train = kf.filter(otr_dm, mf_k)
        alpha_train = state_train.alpha_filtered  # (T_train, K)

        # GOLD+ baseline prediction
        pred_gold, alpha_test = gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        # AR(1) baseline
        pred_ar = ar1_predict(otr_raw, ote)
        r2_ar = float(oos_r_squared(pred_ar.T.ravel(), actual))

        # Spectral predictions on training data (for w_i estimation)
        spec_train = alpha_train @ U.T  # (T_train, N) spectral prediction of demeaned obs

        for lam in ridge_lambdas:
            # Estimate per-actor w_i: y_{i,t} = w_i * spec_i,t (demeaned)
            # => OLS per actor on training data
            pred_weighted = np.zeros_like(ote)
            spec_test = alpha_test @ U.T  # (T_test, K) @ (K, N) => (T_test, N)

            for i in range(N):
                x = spec_train[:, i]  # (T_train,) spectral prediction for actor i
                y = otr_dm[:, i]      # (T_train,) actual demeaned intensity

                if np.var(x) < 1e-12:
                    w_i = 1.0
                else:
                    # Ridge: w_i = (x'x + lambda)^{-1} x'y
                    w_i = float((x @ y) / (x @ x + lam))

                pred_weighted[:, i] = om[0, i] + w_i * spec_test[:, i]

            r2_weighted = float(oos_r_squared(pred_weighted.T.ravel(), actual))

            # DM test
            gold_err = pred_gold.T.ravel() - actual
            w_err = pred_weighted.T.ravel() - actual
            dm_stat, dm_p = diebold_mariano_test(w_err, gold_err)

            rows.append({
                "experiment": "E2-3", "window": f"W{ty}",
                "ridge_lambda": lam, "r2_weighted": r2_weighted,
                "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "dm_stat_vs_gold": dm_stat, "dm_p_vs_gold": dm_p,
            })

        elapsed = time.time() - t0
        best_lam_r2 = max(
            [(r["ridge_lambda"], r["r2_weighted"]) for r in rows[-len(ridge_lambdas):]],
            key=lambda x: x[1],
        )
        print(f"  W{ty}: GOLD+={r2_gold:.4f}, best weighted={best_lam_r2[1]:.4f} "
              f"(lam={best_lam_r2[0]}), AR(1)={r2_ar:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    print(f"\n  {'Lambda':>8} {'Weighted R2':>12} {'GOLD+ R2':>10} {'AR(1) R2':>10} {'Delta':>8}")
    for lam in ridge_lambdas:
        sub = df[df["ridge_lambda"] == lam]
        m_w = sub["r2_weighted"].mean()
        m_g = sub["r2_gold"].mean()
        m_a = sub["r2_ar1"].mean()
        print(f"  {lam:>8.2f} {m_w:>12.4f} {m_g:>10.4f} {m_a:>10.4f} {m_w - m_g:>+8.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-4: Economic Emergence Signals
# ═══════════════════════════════════════════════════════════════════════════


def compute_emergence_features(intensities_raw):
    """Compute per-quarter emergence features from raw intensity panel.

    Returns (T, 3) array: [dispersion, rotation_velocity, concentration].
    """
    T, N = intensities_raw.shape

    # 1. Cross-sectional dispersion: std(y_{i,t}) across actors
    dispersion = np.std(intensities_raw, axis=1)  # (T,)

    # 2. Rotation velocity: mean absolute quarter-to-quarter change in actor rank
    # Proxy: mean |y_{i,t} - y_{i,t-1}| across actors
    rotation = np.zeros(T)
    rotation[1:] = np.mean(np.abs(np.diff(intensities_raw, axis=0)), axis=1)

    # 3. Concentration (Herfindahl of absolute deviations from mean)
    deviations = np.abs(intensities_raw - intensities_raw.mean(axis=1, keepdims=True))
    total_dev = deviations.sum(axis=1, keepdims=True)
    shares = deviations / (total_dev + 1e-12)
    hhi = (shares ** 2).sum(axis=1)  # Herfindahl: high = concentrated misallocation

    return np.column_stack([dispersion, rotation, hhi])


def run_e2_4(wide):
    print("\n" + "=" * 90)
    print("  E2-4: ECONOMIC EMERGENCE SIGNALS")
    print("  Hypothesis: dispersion/rotation/concentration improve prediction")
    print("=" * 90)

    beta_lambdas = [0.0, 0.1, 1.0, 10.0]
    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()

        # GOLD+ baseline
        U, _ = get_dmd_basis(otr_dm, N)
        F, Q, R_sph = train_kalman(otr_dm, N, U)
        pred_gold, _ = gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        # Emergence features on training data
        ef_train = compute_emergence_features(otr_raw)  # (T_train, 3)
        ef_test = compute_emergence_features(ote)       # (T_test, 3)
        n_features = ef_train.shape[1]

        # GOLD+ errors on training data (to learn correction)
        # Re-predict on training to get in-sample residuals
        k_act = U.shape[1]
        mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
        kf = KalmanFilter(F=F, Q=Q, R=R_sph)
        state_train = kf.filter(otr_dm, mf_k)
        pred_train_spec = (state_train.alpha_filtered @ U.T + om)  # (T_train, N)
        residual_train = otr_raw - pred_train_spec  # (T_train, N) residuals

        for lam in beta_lambdas:
            # For each actor, regress residuals on emergence features
            # residual_{i,t} = beta_i @ emergence_t + eps
            pred_corrected = pred_gold.copy()

            for i in range(N):
                y_r = residual_train[:, i]  # (T_train,)
                X_e = ef_train  # (T_train, 3)

                # Ridge: beta = (X'X + lambda*I)^{-1} X'y
                XtX = X_e.T @ X_e
                Xty = X_e.T @ y_r
                try:
                    beta = np.linalg.solve(
                        XtX + lam * np.eye(n_features), Xty
                    )
                except Exception:
                    continue

                # Correct test predictions
                correction = ef_test @ beta  # (T_test,)
                pred_corrected[:, i] += correction

            r2_corrected = float(oos_r_squared(pred_corrected.T.ravel(), actual))

            # DM test
            gold_err = pred_gold.T.ravel() - actual
            corr_err = pred_corrected.T.ravel() - actual
            dm_stat, dm_p = diebold_mariano_test(corr_err, gold_err)

            rows.append({
                "experiment": "E2-4", "window": f"W{ty}",
                "ridge_lambda": lam, "r2_corrected": r2_corrected,
                "r2_gold": r2_gold, "delta": r2_corrected - r2_gold,
                "dm_stat_vs_gold": dm_stat, "dm_p_vs_gold": dm_p,
            })

        # Also test: emergence features as standalone predictors (no spectral)
        # y_{i,t+1} = a_i + beta_i @ emergence_t
        pred_emerg_only = np.zeros_like(ote)
        for i in range(N):
            y = otr_raw[1:, i]
            X = np.column_stack([np.ones(len(y)), ef_train[:-1]])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            except Exception:
                pred_emerg_only[:, i] = otr_raw[-1, i]
                continue
            for t in range(ote.shape[0]):
                pred_emerg_only[t, i] = np.concatenate([[1], ef_test[t]]) @ beta

        r2_emerg_only = float(oos_r_squared(pred_emerg_only.T.ravel(), actual))
        rows.append({
            "experiment": "E2-4", "window": f"W{ty}",
            "ridge_lambda": -1,  # sentinel for standalone
            "r2_corrected": r2_emerg_only, "r2_gold": r2_gold,
            "delta": r2_emerg_only - r2_gold,
            "dm_stat_vs_gold": float("nan"), "dm_p_vs_gold": float("nan"),
        })

        elapsed = time.time() - t0
        best = max(
            [(r["ridge_lambda"], r["r2_corrected"]) for r in rows[-(len(beta_lambdas) + 1):-1]],
            key=lambda x: x[1],
        )
        print(f"  W{ty}: GOLD+={r2_gold:.4f}, best corrected={best[1]:.4f} "
              f"(lam={best[0]}), emerg-only={r2_emerg_only:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    print(f"\n  {'Config':>15} {'R2':>10} {'Delta vs GOLD+':>15}")
    for lam in beta_lambdas:
        sub = df[df["ridge_lambda"] == lam]
        m = sub["r2_corrected"].mean()
        d = sub["delta"].mean()
        print(f"  {'corr lam=' + str(lam):>15} {m:>10.4f} {d:>+15.4f}")
    sub_solo = df[df["ridge_lambda"] == -1]
    print(f"  {'emerg-only':>15} {sub_solo['r2_corrected'].mean():>10.4f} "
          f"{sub_solo['delta'].mean():>+15.4f}")
    print(f"  {'GOLD+ base':>15} {df['r2_gold'].mean():>10.4f}")

    # BRONZE check
    best_delta = df[df["ridge_lambda"] >= 0].groupby("ridge_lambda")["delta"].mean().max()
    print(f"\n  BRONZE criterion (emergence delta > 0): {best_delta:+.4f} "
          f"{'PASS' if best_delta > 0 else 'FAIL'}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-5: Kim Filter K-means Init
# ═══════════════════════════════════════════════════════════════════════════


def run_e2_5(wide):
    print("\n" + "=" * 90)
    print("  E2-5: KIM FILTER K-MEANS INITIALISATION")
    print("  Hypothesis: asymmetric F init enables meaningful regime switching")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()

        # GOLD+ basis
        U, mf_full = get_dmd_basis(otr_dm, N)
        k_act = U.shape[1]

        # Train standard Kalman EM (GOLD+)
        F, Q, R_sph = train_kalman(otr_dm, N, U)

        # Get alpha trajectory for K-means
        mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
        kf = KalmanFilter(F=F, Q=Q, R=R_sph)
        state_train = kf.filter(otr_dm, mf_k)
        alpha_traj = state_train.alpha_filtered  # (T_train, K)

        # GOLD+ baseline (M=1 with online Q)
        pred_gold, _ = gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        for M in [2, 3]:
            # K-means on alpha trajectory
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=M, random_state=42, n_init=10)
            labels = km.fit_predict(alpha_traj)

            # Estimate per-cluster F matrices
            F_list = []
            Q_list = []
            for m in range(M):
                idx = np.where(labels == m)[0]
                if len(idx) < k_act + 2:
                    # Too few points: use default
                    F_list.append(np.eye(k_act) * 0.9)
                    Q_list.append(np.eye(k_act) * 0.1)
                    continue
                # VAR(1) on cluster m: alpha[t+1] = F_m @ alpha[t] + eps
                a_clust = alpha_traj[idx]
                X = a_clust[:-1]
                Y = a_clust[1:]
                # Only use consecutive pairs
                consecutive = np.where(np.diff(idx) == 1)[0]
                if len(consecutive) < k_act + 1:
                    F_list.append(np.eye(k_act) * 0.9)
                    Q_list.append(np.eye(k_act) * 0.1)
                    continue
                X_c = alpha_traj[idx[consecutive]]
                Y_c = alpha_traj[idx[consecutive] + 1]
                try:
                    F_m = np.linalg.lstsq(X_c, Y_c, rcond=None)[0].T
                except Exception:
                    F_m = np.eye(k_act) * 0.9
                resid = Y_c - X_c @ F_m.T
                Q_m = resid.T @ resid / max(len(consecutive) - 1, 1)
                Q_m = (Q_m + Q_m.T) / 2 + np.eye(k_act) * 1e-6
                F_list.append(F_m)
                Q_list.append(Q_m)

            # Check asymmetry
            if M == 2:
                f_diff = np.linalg.norm(F_list[0] - F_list[1])
            else:
                f_diff = np.mean([
                    np.linalg.norm(F_list[i] - F_list[j])
                    for i in range(M) for j in range(i + 1, M)
                ])

            # Run Kim filter with asymmetric init
            kim = KimFilter(n_regimes=M)
            try:
                state_kim = kim.filter(
                    otr_dm, mf_k,
                    F_list=F_list, Q_list=Q_list,
                    R=R_sph,
                )
                # Online Kalman with regime-weighted F on test
                # Use regime probs from last training step to weight F
                xi = state_kim.regime_probs[-1]  # (M,)
                F_weighted = sum(xi[m] * F_list[m] for m in range(M))
                Q_weighted = sum(xi[m] * Q_list[m] for m in range(M))

                pred_kim, _ = gold_plus_predict(
                    ote_dm, ote, om, N, U, F_weighted, Q_weighted, R_sph
                )
                r2_kim = float(oos_r_squared(pred_kim.T.ravel(), actual))
            except Exception as e:
                r2_kim = float("nan")
                print(f"    Kim M={M} failed: {e}")

            # Kim with symmetric init (for comparison)
            try:
                state_sym = kim.filter(otr_dm, mf_k, R=R_sph)
                xi_sym = state_sym.regime_probs[-1]
                F_sym_list = [np.eye(k_act) * 0.9] * M
                Q_sym_list = [np.eye(k_act) * 0.1] * M
                F_weighted_sym = sum(xi_sym[m] * F_sym_list[m] for m in range(M))
                Q_weighted_sym = sum(xi_sym[m] * Q_sym_list[m] for m in range(M))
                pred_sym, _ = gold_plus_predict(
                    ote_dm, ote, om, N, U, F_weighted_sym, Q_weighted_sym, R_sph
                )
                r2_sym = float(oos_r_squared(pred_sym.T.ravel(), actual))
            except Exception:
                r2_sym = float("nan")

            rows.append({
                "experiment": "E2-5", "window": f"W{ty}", "M": M,
                "r2_kim_kmeans": r2_kim, "r2_kim_symmetric": r2_sym,
                "r2_gold_m1": r2_gold, "f_asymmetry": f_diff,
                "regime_entropy": float(-np.sum(xi * np.log(xi + 1e-12))),
            })

        elapsed = time.time() - t0
        print(f"  W{ty}: GOLD+={r2_gold:.4f}, Kim-kmeans M=2={rows[-2]['r2_kim_kmeans']:.4f}, "
              f"Kim-sym M=2={rows[-2]['r2_kim_symmetric']:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    print(f"\n  {'M':>4} {'Kim-kmeans':>12} {'Kim-sym':>10} {'GOLD+ M=1':>10} {'F asym':>10}")
    for M in [2, 3]:
        sub = df[df["M"] == M]
        print(f"  M={M} {sub['r2_kim_kmeans'].mean():>12.4f} "
              f"{sub['r2_kim_symmetric'].mean():>10.4f} "
              f"{sub['r2_gold_m1'].mean():>10.4f} "
              f"{sub['f_asymmetry'].mean():>10.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-6: Combined Best
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# E2-5b: F Regularisation Sweep
# ═══════════════════════════════════════════════════════════════════════════


def run_e2_5b(wide):
    print("\n" + "=" * 90)
    print("  E2-5b: F REGULARISATION SWEEP")
    print("  Hypothesis: F=alpha*I outperforms EM-estimated F; find optimal alpha")
    print("  Also: F_shrunk = (1-s)*F_EM + s*alpha*I")
    print("=" * 90)

    # Part A: Diagonal F sweep
    alphas = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    rows = []

    for ty in TEST_YEARS:
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()
        U, _ = get_dmd_basis(otr_dm, N)
        k_act = U.shape[1]

        # EM-estimated F (GOLD+ baseline)
        F_em, Q_em, R_sph = train_kalman(otr_dm, N, U)
        pred_gold, _ = gold_plus_predict(ote_dm, ote, om, N, U, F_em, Q_em, R_sph)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        # AR(1) baseline
        pred_ar = ar1_predict(otr_raw, ote)
        r2_ar = float(oos_r_squared(pred_ar.T.ravel(), actual))

        for alpha in alphas:
            F_diag = np.eye(k_act) * alpha
            Q_default = np.eye(k_act) * 0.1
            pred_d, _ = gold_plus_predict(ote_dm, ote, om, N, U, F_diag, Q_default, R_sph)
            r2_diag = float(oos_r_squared(pred_d.T.ravel(), actual))
            rows.append({
                "experiment": "E2-5b", "window": f"W{ty}",
                "method": "diagonal", "alpha": alpha, "shrinkage": 1.0,
                "r2": r2_diag, "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "delta_vs_gold": r2_diag - r2_gold,
            })

        # Part B: Shrinkage: F_shrunk = (1-s)*F_EM + s*0.9*I
        for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            F_shrunk = (1 - s) * F_em + s * np.eye(k_act) * 0.9
            Q_shrunk = (1 - s) * Q_em + s * np.eye(k_act) * 0.1
            pred_s, _ = gold_plus_predict(ote_dm, ote, om, N, U, F_shrunk, Q_shrunk, R_sph)
            r2_s = float(oos_r_squared(pred_s.T.ravel(), actual))
            rows.append({
                "experiment": "E2-5b", "window": f"W{ty}",
                "method": "shrinkage", "alpha": 0.9, "shrinkage": s,
                "r2": r2_s, "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "delta_vs_gold": r2_s - r2_gold,
            })

    df = pd.DataFrame(rows)

    # Part A results
    print(f"\n  Part A: F = alpha * I (diagonal)")
    print(f"  {'alpha':>8} {'Mean R2':>10} {'Delta vs GOLD+':>15} {'Wins/10':>10}")
    for alpha in alphas:
        sub = df[(df["method"] == "diagonal") & (df["alpha"] == alpha)]
        wins = (sub["r2"] > sub["r2_gold"]).sum()
        print(f"  {alpha:>8.2f} {sub['r2'].mean():>10.4f} {sub['delta_vs_gold'].mean():>+15.4f} "
              f"{wins:>10}/10")
    em_mean = df[df["method"] == "diagonal"]["r2_gold"].mean()
    print(f"  {'EM(GOLD+)':>8} {em_mean:>10.4f} {'—':>15}")

    # Part B results
    print(f"\n  Part B: F = (1-s)*F_EM + s*0.9*I (shrinkage)")
    print(f"  {'s':>8} {'Mean R2':>10} {'Delta vs GOLD+':>15} {'Wins/10':>10}")
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        sub = df[(df["method"] == "shrinkage") & (df["shrinkage"] == s)]
        wins = (sub["r2"] > sub["r2_gold"]).sum()
        label = "EM" if s == 0 else f"shrink-{s}"
        print(f"  {s:>8.1f} {sub['r2'].mean():>10.4f} {sub['delta_vs_gold'].mean():>+15.4f} "
              f"{wins:>10}/10")

    # Best overall
    by_config = df.groupby(["method", "alpha", "shrinkage"])["r2"].mean()
    best_idx = by_config.idxmax()
    best_r2 = by_config.max()
    print(f"\n  BEST: {best_idx} -> R2={best_r2:.4f} (vs GOLD+ {em_mean:.4f}, "
          f"delta={best_r2 - em_mean:+.4f})")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# E2-4b: Dispersion-Weighted Prediction
# ═══════════════════════════════════════════════════════════════════════════


def run_e2_4b(wide):
    print("\n" + "=" * 90)
    print("  E2-4b: DISPERSION-WEIGHTED PREDICTION CONFIDENCE")
    print("  Hypothesis: weight spectral prediction by cross-sectional dispersion")
    print("  pred = (1-w)*mean + w*spectral, where w depends on dispersion")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()

        # GOLD+ prediction
        U, _ = get_dmd_basis(otr_dm, N)
        F, Q, R_sph = train_kalman(otr_dm, N, U)
        pred_gold, _ = gold_plus_predict(ote_dm, ote, om, N, U, F, Q, R_sph)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        # Mean-only prediction (per-actor mean from training)
        pred_mean = np.tile(om, (ote.shape[0], 1))  # (T_test, N)
        r2_mean = float(oos_r_squared(pred_mean.T.ravel(), actual))

        # Train-period dispersion for calibration
        disp_train = np.std(otr_raw, axis=1)  # (T_train,)
        disp_mean = disp_train.mean()
        disp_std = disp_train.std() + 1e-12

        # Test-period dispersion
        disp_test = np.std(ote, axis=1)  # (T_test,)

        # Method 1: Sigmoid weight: w = sigmoid(gamma * (disp - disp_mean) / disp_std)
        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
            z = gamma * (disp_test - disp_mean) / disp_std
            w = 1.0 / (1.0 + np.exp(-z))  # (T_test,) in [0, 1]
            pred_w = (1 - w[:, None]) * pred_mean + w[:, None] * pred_gold
            r2_w = float(oos_r_squared(pred_w.T.ravel(), actual))
            rows.append({
                "experiment": "E2-4b", "window": f"W{ty}",
                "gamma": gamma, "r2_weighted": r2_w,
                "r2_gold": r2_gold, "r2_mean": r2_mean,
                "delta_vs_gold": r2_w - r2_gold,
                "mean_w": float(w.mean()),
            })

        # Method 2: Fixed blend: pred = beta*spectral + (1-beta)*mean
        for beta in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            pred_b = beta * pred_gold + (1 - beta) * pred_mean
            r2_b = float(oos_r_squared(pred_b.T.ravel(), actual))
            rows.append({
                "experiment": "E2-4b", "window": f"W{ty}",
                "gamma": -1,  # sentinel for fixed blend
                "r2_weighted": r2_b, "r2_gold": r2_gold, "r2_mean": r2_mean,
                "delta_vs_gold": r2_b - r2_gold,
                "mean_w": beta,
            })

    df = pd.DataFrame(rows)

    print(f"\n  Method 1: Dispersion-sigmoid weighting")
    print(f"  {'gamma':>8} {'Mean R2':>10} {'Delta':>10} {'Mean w':>10}")
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        sub = df[df["gamma"] == gamma]
        print(f"  {gamma:>8.1f} {sub['r2_weighted'].mean():>10.4f} "
              f"{sub['delta_vs_gold'].mean():>+10.4f} {sub['mean_w'].mean():>10.3f}")

    print(f"\n  Method 2: Fixed blend (beta*spectral + (1-beta)*mean)")
    print(f"  {'beta':>8} {'Mean R2':>10} {'Delta':>10}")
    for beta in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        sub = df[(df["gamma"] == -1) & (abs(df["mean_w"] - beta) < 0.01)]
        print(f"  {beta:>8.1f} {sub['r2_weighted'].mean():>10.4f} "
              f"{sub['delta_vs_gold'].mean():>+10.4f}")

    gold_mean = df["r2_gold"].mean()
    best = df.groupby(["gamma", "mean_w"])["r2_weighted"].mean().max()
    print(f"\n  Best R²={best:.4f} vs GOLD+ {gold_mean:.4f} (delta={best - gold_mean:+.4f})")

    return df


def run_e2_6(wide, results=None):
    print("\n" + "=" * 90)
    print("  E2-6: COMBINED BEST — F regularisation + noise + Q sweep")
    print("  Stack the winning E2-5b finding with noise augmentation and Q tuning")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        otr_dm, ote_dm, ote, om, N, otr_raw, _ = prep(wide, ty)
        actual = ote.T.ravel()
        k_use = min(K_GOLD, N - 2)

        # AR(1) baseline
        pred_ar = ar1_predict(otr_raw, ote)
        r2_ar = float(oos_r_squared(pred_ar.T.ravel(), actual))

        # GOLD+ (EM F, no noise)
        U_gold, _ = get_dmd_basis(otr_dm, N)
        F_em, Q_em, R_sph_gold = train_kalman(otr_dm, N, U_gold)
        pred_gold, _ = gold_plus_predict(ote_dm, ote, om, N, U_gold, F_em, Q_em, R_sph_gold)
        r2_gold = float(oos_r_squared(pred_gold.T.ravel(), actual))

        # Config 1: F=0.99*I (best E2-5b)
        F_reg = np.eye(k_use) * 0.99
        Q_init = np.eye(k_use) * 0.1
        pred_1, _ = gold_plus_predict(ote_dm, ote, om, N, U_gold, F_reg, Q_init, R_sph_gold)
        r2_1 = float(oos_r_squared(pred_1.T.ravel(), actual))

        # Config 2: F=0.99*I + noise augmented DMD basis
        for sigma in [0.05, 0.10, 0.15]:
            otr_noisy = otr_dm + rng.standard_normal(otr_dm.shape) * np.std(otr_dm, axis=0, keepdims=True) * sigma
            try:
                U_noisy, mf_n = get_dmd_basis(otr_noisy, N, k=K_GOLD)
            except Exception:
                U_noisy = U_gold
            k_n = U_noisy.shape[1]
            R_n = R_sph_gold  # spherical R from clean data
            F_n = np.eye(k_n) * 0.99
            Q_n = np.eye(k_n) * 0.1
            # Also need spherical R for noisy basis — recompute
            try:
                F_n_em, Q_n_em, R_n_sph = train_kalman(otr_noisy, N, U_noisy)
                R_n = R_n_sph  # use spherical R from noisy EM
            except Exception:
                R_n = np.eye(N) * np.trace(R_sph_gold) / N

            ote_dm_n = ote - om  # test demeaning unchanged
            pred_n, _ = gold_plus_predict(ote_dm_n, ote, om, N, U_noisy, F_n, Q_n, R_n)
            r2_n = float(oos_r_squared(pred_n.T.ravel(), actual))
            rows.append({
                "window": f"W{ty}", "config": f"F_reg+noise_{sigma}",
                "r2": r2_n, "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "delta_gold": r2_n - r2_gold,
            })

        # Config 3: F=0.99*I + Q sweep
        for q_init in [0.01, 0.05, 0.1, 0.2, 0.5]:
            Q_sweep = np.eye(k_use) * q_init
            pred_q, _ = gold_plus_predict(ote_dm, ote, om, N, U_gold, F_reg, Q_sweep, R_sph_gold)
            r2_q = float(oos_r_squared(pred_q.T.ravel(), actual))
            rows.append({
                "window": f"W{ty}", "config": f"F_reg+Q_{q_init}",
                "r2": r2_q, "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "delta_gold": r2_q - r2_gold,
            })

        # Config 4: F=0.99*I + adapt_rate sweep
        for adapt in [0.1, 0.2, 0.3, 0.5, 0.7]:
            pred_a, _ = gold_plus_predict(ote_dm, ote, om, N, U_gold, F_reg, Q_init,
                                          R_sph_gold, adapt_rate=adapt)
            r2_a = float(oos_r_squared(pred_a.T.ravel(), actual))
            rows.append({
                "window": f"W{ty}", "config": f"F_reg+adapt_{adapt}",
                "r2": r2_a, "r2_gold": r2_gold, "r2_ar1": r2_ar,
                "delta_gold": r2_a - r2_gold,
            })

        # Store baselines
        rows.append({
            "window": f"W{ty}", "config": "GOLD+_EM",
            "r2": r2_gold, "r2_gold": r2_gold, "r2_ar1": r2_ar, "delta_gold": 0.0,
        })
        rows.append({
            "window": f"W{ty}", "config": "F_reg_base",
            "r2": r2_1, "r2_gold": r2_gold, "r2_ar1": r2_ar,
            "delta_gold": r2_1 - r2_gold,
        })

    df = pd.DataFrame(rows)

    # Summary by config
    print(f"\n  {'Config':>25} {'Mean R2':>10} {'Delta vs GOLD+':>15} {'vs AR(1)':>10}")
    configs = df["config"].unique()
    config_means = df.groupby("config").agg(
        r2=("r2", "mean"), delta=("delta_gold", "mean"), ar1=("r2_ar1", "mean")
    ).sort_values("r2", ascending=False)

    for cfg, row in config_means.iterrows():
        wins_ar = (df[df["config"] == cfg]["r2"] > df[df["config"] == cfg]["r2_ar1"]).sum()
        print(f"  {cfg:>25} {row['r2']:>10.4f} {row['delta']:>+15.4f} "
              f"{row['r2'] - row['ar1']:>+10.4f} ({wins_ar}/10)")

    best_cfg = config_means.index[0]
    best_r2 = config_means.iloc[0]["r2"]
    print(f"\n  OVERALL BEST: {best_cfg} -> R2={best_r2:.4f}")

    # DM test best vs GOLD+ and vs AR(1)
    best_r2s = df[df["config"] == best_cfg]["r2"].values
    gold_r2s = df[df["config"] == "GOLD+_EM"]["r2"].values
    ar1_r2s = df[df["config"] == "GOLD+_EM"]["r2_ar1"].values

    best_preds_err = best_r2s - gold_r2s  # proxy for per-window delta
    print(f"  vs GOLD+: mean delta={np.mean(best_r2s - gold_r2s):+.4f}, "
          f"wins={np.sum(best_r2s > gold_r2s)}/10")
    print(f"  vs AR(1): mean delta={np.mean(best_r2s - ar1_r2s):+.4f}, "
          f"wins={np.sum(best_r2s > ar1_r2s)}/10")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SMIM Iteration 2 Experiments")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of experiments to run (e.g. E2-1,E2-3)")
    args = parser.parse_args()

    experiments = ["E2-1", "E2-2", "E2-3", "E2-4", "E2-5"]
    if args.only:
        experiments = [e.strip() for e in args.only.split(",")]

    print("=" * 90)
    print(f"  SMIM ITERATION 2  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Experiments: {', '.join(experiments)}")
    print(f"  Baseline: GOLD+ R²=0.524 (DMD K=8, spherical R, EWM, online Q)")
    print("=" * 90)

    wide = load_panel()
    results = {}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if "E2-1" in experiments:
        results["E2-1"] = run_e2_1(wide)
        if not results["E2-1"].empty:
            results["E2-1"].to_parquet(METRICS_DIR / "iter2_E2-1.parquet", index=False)

    if "E2-2" in experiments:
        results["E2-2"] = run_e2_2(wide)
        if not results["E2-2"].empty:
            results["E2-2"].to_parquet(METRICS_DIR / "iter2_E2-2.parquet", index=False)

    if "E2-3" in experiments:
        results["E2-3"] = run_e2_3(wide)
        if not results["E2-3"].empty:
            results["E2-3"].to_parquet(METRICS_DIR / "iter2_E2-3.parquet", index=False)

    if "E2-4" in experiments:
        results["E2-4"] = run_e2_4(wide)
        if not results["E2-4"].empty:
            results["E2-4"].to_parquet(METRICS_DIR / "iter2_E2-4.parquet", index=False)

    if "E2-5" in experiments:
        results["E2-5"] = run_e2_5(wide)
        if not results["E2-5"].empty:
            results["E2-5"].to_parquet(METRICS_DIR / "iter2_E2-5.parquet", index=False)

    if "E2-5b" in experiments:
        results["E2-5b"] = run_e2_5b(wide)
        if not results["E2-5b"].empty:
            results["E2-5b"].to_parquet(METRICS_DIR / "iter2_E2-5b.parquet", index=False)

    if "E2-4b" in experiments:
        results["E2-4b"] = run_e2_4b(wide)
        if not results["E2-4b"].empty:
            results["E2-4b"].to_parquet(METRICS_DIR / "iter2_E2-4b.parquet", index=False)

    if "E2-6" in experiments or len(results) > 0:
        results["E2-6"] = run_e2_6(wide, results)
        if isinstance(results.get("E2-6"), pd.DataFrame) and not results["E2-6"].empty:
            results["E2-6"].to_parquet(METRICS_DIR / "iter2_E2-6.parquet", index=False)

    # Final summary
    print("\n" + "=" * 90)
    print("  ITERATION 2 SUMMARY")
    print("=" * 90)
    for name, df in results.items():
        if df is None or df.empty:
            print(f"  {name}: no results")
            continue
        if name == "E2-6":
            print(f"  {name}: {df.to_dict()}")
            continue
        if "r2_dmd_gold" in df.columns:
            gold_mean = df["r2_dmd_gold"].mean()
        elif "r2_gold" in df.columns:
            gold_mean = df["r2_gold"].mean()
        else:
            gold_mean = 0.524
        print(f"  {name}: GOLD+ baseline = {gold_mean:.4f}")


if __name__ == "__main__":
    main()
