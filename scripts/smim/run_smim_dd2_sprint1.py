#!/usr/bin/env python
"""
SMIM Drilldown #2 Sprint 1: Venues 1 & 3 on existing quarterly data.

V1-1: Extended DMD degree 2 on alpha trajectory (P=44)
V1-3: Diagonal-only quadratic (P=16)
V1-4: EDMD basis + PLATINUM Kalman
V3-1: Subspace angle between fast (2yr) and slow (5yr) DMD
V3-2: Angle predicts next-window R^2
V3-3: Angle-weighted prediction

All run on the EXISTING experiment_a1 quarterly intensity panel.

Usage::
    uv run python scripts/smim/run_smim_dd2_sprint1.py
    uv run python scripts/smim/run_smim_dd2_sprint1.py --only V1
    uv run python scripts/smim/run_smim_dd2_sprint1.py --only V3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
T_YR = 5
TEST_YEARS = list(range(2015, 2025))


# =========================================================================
# Shared infrastructure (PLATINUM pipeline from iteration 2)
# =========================================================================


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
    return otr - om, ote - om, ote, om, N, otr


def get_dmd_basis(otr_dm, N, k=K_GOLD):
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(k_use, mf.K)].real, mf
    except Exception:
        return np.eye(N)[:, :k_use], None


def train_kalman_sph_r(otr_dm, N, U):
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
    kf = KalmanFilter()
    st = kf.em_estimate(otr_dm, mf_k, max_iter=50)
    R_sph = np.eye(N) * np.trace(st.params["R"]) / N
    return st.params["F"], st.params["Q"].copy(), R_sph


def platinum_predict(ote_dm, ote, om, N, U, F, Q, R_sph, adapt_rate=0.3):
    """PLATINUM: F=0.99*I, Q=0.5*I, online Q adapt."""
    k_act = U.shape[1]
    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    Q_run = Q.copy()
    pred = np.zeros_like(ote)
    alpha_filt = np.zeros((ote.shape[0], k_act))
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
        alpha_filt[t] = a_f
        innov = a_f - F @ alpha_c
        Q_run = (1 - adapt_rate) * Q_run + adapt_rate * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2 + np.eye(k_act) * 1e-6
        alpha_c = a_f
        P_c = P_f
    return pred, alpha_filt


def get_alpha_trajectory(otr_dm, N, U, R_sph):
    """Run Kalman filter on training data to get alpha trajectory."""
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
    F_reg = np.eye(k_act) * 0.99
    Q_init = np.eye(k_act) * 0.5
    kf = KalmanFilter(F=F_reg, Q=Q_init, R=R_sph)
    state = kf.filter(otr_dm, mf_k)
    return state.alpha_filtered  # (T_train, K)


# =========================================================================
# VENUE 1: Extended DMD on modal alpha
# =========================================================================


def lift_alpha_polynomial(alpha, degree=2, diagonal_only=False):
    """Lift (T, K) alpha trajectory to polynomial features.

    Full degree 2 with K=8: P = K + K*(K+1)/2 = 8 + 36 = 44
    Diagonal only:           P = K + K = 16
    """
    T, K = alpha.shape
    if diagonal_only:
        # [alpha_1, ..., alpha_K, alpha_1^2, ..., alpha_K^2]
        return np.column_stack([alpha, alpha ** 2])
    else:
        pf = PolynomialFeatures(degree=degree, include_bias=False)
        return pf.fit_transform(alpha)  # (T, P)


def edmd_predict(alpha_train_lifted, alpha_test_lifted, K):
    """Fit linear dynamics on lifted alpha, predict, extract first K components.

    F_theta = argmin ||Theta[1:] - Theta[:-1] @ F||
    theta_{t+1|t} = F_theta @ theta_t
    alpha_{t+1|t} = theta_{t+1|t}[:K]
    """
    T_train, P = alpha_train_lifted.shape
    X = alpha_train_lifted[:-1]  # (T-1, P)
    Y = alpha_train_lifted[1:]   # (T-1, P)

    # Ridge regression for F_theta (regularise to avoid overfitting)
    lam = 0.1
    try:
        F_theta = np.linalg.solve(
            X.T @ X + lam * np.eye(P), X.T @ Y
        ).T  # (P, P)
    except np.linalg.LinAlgError:
        F_theta = np.eye(P) * 0.9

    # Multi-step prediction on test
    T_test = alpha_test_lifted.shape[0]
    pred_lifted = np.zeros((T_test, P))
    theta_curr = alpha_train_lifted[-1]  # last training state

    for t in range(T_test):
        theta_next = F_theta @ theta_curr
        pred_lifted[t] = theta_next
        theta_curr = theta_next

    return pred_lifted[:, :K]  # extract linear alpha components


def run_v1(wide):
    print("\n" + "=" * 90)
    print("  VENUE 1: EXTENDED DMD ON MODAL ALPHA")
    print("  Test nonlinear mode coupling via Koopman lifting")
    print("=" * 90)

    configs = [
        ("EDMD_deg2_full", 2, False),     # V1-1: P=44
        ("EDMD_deg2_diag", 2, True),       # V1-3: P=16
        ("EDMD_deg3_full", 3, False),      # V1-2: P=164
    ]

    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        otr_dm, ote_dm, ote, om, N, otr_raw = prep(wide, ty)
        actual = ote.T.ravel()

        # Get DMD basis (standard linear)
        U, mf = get_dmd_basis(otr_dm, N)
        k_act = U.shape[1]

        # Spherical R (need EM just for R estimation)
        _, _, R_sph = train_kalman_sph_r(otr_dm, N, U)

        # Get alpha trajectory on training data
        alpha_train = get_alpha_trajectory(otr_dm, N, U, R_sph)  # (T_train, K)

        # PLATINUM baseline (linear)
        F_plat = np.eye(k_act) * 0.99
        Q_plat = np.eye(k_act) * 0.5
        pred_plat, alpha_test = platinum_predict(
            ote_dm, ote, om, N, U, F_plat, Q_plat, R_sph
        )
        r2_plat = float(oos_r_squared(pred_plat.T.ravel(), actual))

        for name, degree, diag_only in configs:
            # Lift alpha trajectories
            alpha_train_lifted = lift_alpha_polynomial(alpha_train, degree, diag_only)
            alpha_test_lifted = lift_alpha_polynomial(alpha_test, degree, diag_only)
            P = alpha_train_lifted.shape[1]

            # EDMD predict: fit dynamics on lifted training, predict test
            alpha_pred_edmd = edmd_predict(alpha_train_lifted, alpha_test_lifted, k_act)

            # V1-1/V1-3: EDMD prediction -> reconstruction
            pred_edmd = (alpha_pred_edmd @ U.T + om)
            r2_edmd = float(oos_r_squared(pred_edmd.T.ravel(), actual))

            # V1-4: Use EDMD as correction to PLATINUM
            # Blend: pred = (1-w)*PLATINUM + w*EDMD
            for w in [0.0, 0.1, 0.3, 0.5, 1.0]:
                pred_blend = (1 - w) * pred_plat + w * pred_edmd
                r2_blend = float(oos_r_squared(pred_blend.T.ravel(), actual))
                rows.append({
                    "experiment": name, "window": f"W{ty}",
                    "P": P, "T_train": alpha_train.shape[0],
                    "T_over_P": alpha_train.shape[0] / P,
                    "blend_w": w, "r2": r2_blend,
                    "r2_platinum": r2_plat, "r2_edmd_pure": r2_edmd,
                    "delta_vs_plat": r2_blend - r2_plat,
                })

        elapsed = time.time() - t0
        # Print summary for this window
        r2_v1_1 = [r for r in rows if r["window"] == f"W{ty}" and r["experiment"] == "EDMD_deg2_full" and r["blend_w"] == 1.0]
        r2_v1_3 = [r for r in rows if r["window"] == f"W{ty}" and r["experiment"] == "EDMD_deg2_diag" and r["blend_w"] == 1.0]
        print(f"  W{ty}: PLAT={r2_plat:.4f}  "
              f"EDMD_full={r2_v1_1[0]['r2']:.4f}  "
              f"EDMD_diag={r2_v1_3[0]['r2']:.4f}  "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(rows)

    # Summary by config
    print(f"\n  === VENUE 1 SUMMARY ===")
    print(f"  {'Config':>20} {'blend_w':>8} {'Mean R2':>10} {'Delta':>10} {'P':>5} {'T/P':>6}")
    for name, _, _ in configs:
        for w in [0.0, 0.3, 0.5, 1.0]:
            sub = df[(df["experiment"] == name) & (df["blend_w"] == w)]
            if sub.empty:
                continue
            tag = f"{name}" if w == 1.0 else f"{name}+plat({1-w:.0%})"
            if w == 0.0:
                tag = "PLATINUM (baseline)"
            print(f"  {tag:>35} {w:>8.1f} {sub['r2'].mean():>10.4f} "
                  f"{sub['delta_vs_plat'].mean():>+10.4f} "
                  f"{sub['P'].iloc[0]:>5} {sub['T_over_P'].mean():>6.1f}")

    # Key question: does ANY EDMD config beat PLATINUM?
    best_edmd = df[df["blend_w"] == 1.0].groupby("experiment")["r2"].mean()
    plat_mean = df["r2_platinum"].mean()
    print(f"\n  PLATINUM baseline:  {plat_mean:.4f}")
    for name, r2 in best_edmd.items():
        delta = r2 - plat_mean
        verdict = "BETTER" if delta > 0.001 else "WORSE" if delta < -0.001 else "SAME"
        print(f"  {name:>20}: {r2:.4f} ({delta:+.4f}) [{verdict}]")

    return df


# =========================================================================
# VENUE 3: Multi-Resolution DMD Divergence
# =========================================================================


def subspace_angle(U1, U2, k=None):
    """Mean principal angle between column spaces of U1 and U2 (radians)."""
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    U1k = U1[:, :k]
    U2k = U2[:, :k]
    Q1, _ = np.linalg.qr(U1k)
    Q2, _ = np.linalg.qr(U2k)
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svals, -1, 1))
    return float(np.mean(angles))  # mean principal angle


def get_dmd_basis_from_window(wide, start, end, k=K_GOLD):
    """Extract DMD basis from a specific date window."""
    tr = wide[(wide.index >= start) & (wide.index <= end)]
    valid = tr.columns[tr.notna().any()]
    tr = tr[valid]
    otr = tr.fillna(tr.mean()).values.astype(np.float64)
    N = otr.shape[1]
    # EWM demeaning
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
    om = (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()
    otr_dm = otr - om
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(k_use, mf.K)].real, list(valid), otr_dm, om
    except Exception:
        return None, list(valid), otr_dm, om


def run_v3(wide):
    print("\n" + "=" * 90)
    print("  VENUE 3: MULTI-RESOLUTION DMD DIVERGENCE")
    print("  Test emergence as disagreement between fast and slow spectral bases")
    print("=" * 90)

    # V3-1: Compute subspace angles
    rows_angle = []
    rows_pred = []

    for ty in TEST_YEARS:
        t0 = time.time()

        # Slow basis: full 5yr training window (standard)
        slow_start = pd.Timestamp(f"{ty - 5}-01-01")
        slow_end = pd.Timestamp(f"{ty - 1}-12-31")
        U_slow, cols_slow, _, _ = get_dmd_basis_from_window(wide, slow_start, slow_end)

        if U_slow is None:
            print(f"  W{ty}: slow DMD failed, skipping")
            continue

        # Fast bases at different timescales
        for fast_yr in [1, 2, 3]:
            fast_start = pd.Timestamp(f"{ty - 1 - fast_yr}-01-01")
            fast_end = pd.Timestamp(f"{ty - 1}-12-31")
            U_fast, cols_fast, _, _ = get_dmd_basis_from_window(wide, fast_start, fast_end)

            if U_fast is None:
                continue

            # Align columns
            common = list(set(cols_slow) & set(cols_fast))
            if len(common) < 20:
                continue
            idx_slow = [cols_slow.index(c) for c in common]
            idx_fast = [cols_fast.index(c) for c in common]
            U_s = U_slow[idx_slow]
            U_f = U_fast[idx_fast]

            # Compute angles at different K
            for k_test in [3, 5, 8]:
                k_use = min(k_test, U_s.shape[1], U_f.shape[1])
                angle = subspace_angle(U_s, U_f, k=k_use)
                angle_deg = np.degrees(angle)
                rows_angle.append({
                    "window": f"W{ty}", "fast_yr": fast_yr,
                    "K": k_test, "angle_rad": angle, "angle_deg": angle_deg,
                })

        # V3-2: PLATINUM R^2 for this window (for correlation with angle)
        otr_dm, ote_dm, ote, om, N, otr_raw = prep(wide, ty)
        actual = ote.T.ravel()
        U, _ = get_dmd_basis(otr_dm, N)
        k_act = U.shape[1]
        _, _, R_sph = train_kalman_sph_r(otr_dm, N, U)
        F_plat = np.eye(k_act) * 0.99
        Q_plat = np.eye(k_act) * 0.5
        pred_plat, _ = platinum_predict(ote_dm, ote, om, N, U, F_plat, Q_plat, R_sph)
        r2_plat = float(oos_r_squared(pred_plat.T.ravel(), actual))

        # V3-3: Angle-weighted prediction
        # When angle is large -> less trust in spectral, more in mean
        pred_mean = np.tile(om, (ote.shape[0], 1))
        # Use the 2yr fast vs 5yr slow angle at K=5
        angle_2yr = [r for r in rows_angle
                     if r["window"] == f"W{ty}" and r["fast_yr"] == 2 and r["K"] == 5]
        if angle_2yr:
            angle_val = angle_2yr[0]["angle_deg"]
        else:
            angle_val = 0.0

        for gamma in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            # Sigmoid: w_spectral = 1/(1 + exp(gamma * (angle - 45)/45))
            # When angle=0: w=high (trust spectral). When angle=90: w=low
            z = gamma * (angle_val - 45) / 45
            w_spectral = 1.0 / (1.0 + np.exp(z))
            pred_w = w_spectral * pred_plat + (1 - w_spectral) * pred_mean
            r2_w = float(oos_r_squared(pred_w.T.ravel(), actual))
            rows_pred.append({
                "window": f"W{ty}", "gamma": gamma,
                "angle_deg": angle_val, "w_spectral": w_spectral,
                "r2": r2_w, "r2_platinum": r2_plat,
                "delta": r2_w - r2_plat,
            })

        elapsed = time.time() - t0
        angles_2yr = [r["angle_deg"] for r in rows_angle
                      if r["window"] == f"W{ty}" and r["fast_yr"] == 2 and r["K"] == 5]
        a_str = f"{angles_2yr[0]:.1f}deg" if angles_2yr else "N/A"
        print(f"  W{ty}: PLAT={r2_plat:.4f}  angle(2yr/5yr,K=5)={a_str}  ({elapsed:.1f}s)")

    df_angle = pd.DataFrame(rows_angle)
    df_pred = pd.DataFrame(rows_pred)

    if df_angle.empty:
        print("\n  No angle results.")
        return df_angle, df_pred

    # V3-1 Summary: angle distribution
    print(f"\n  === V3-1: SUBSPACE ANGLES (degrees) ===")
    print(f"  {'Fast(yr)':>10} {'K':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for fast_yr in [1, 2, 3]:
        for k_test in [3, 5, 8]:
            sub = df_angle[(df_angle["fast_yr"] == fast_yr) & (df_angle["K"] == k_test)]
            if sub.empty:
                continue
            print(f"  {fast_yr:>10} {k_test:>4} {sub['angle_deg'].mean():>8.1f} "
                  f"{sub['angle_deg'].std():>8.1f} "
                  f"{sub['angle_deg'].min():>8.1f} {sub['angle_deg'].max():>8.1f}")

    # V3-2: Correlation between angle and R^2
    print(f"\n  === V3-2: ANGLE vs R^2 CORRELATION ===")
    # Get per-window PLATINUM R^2
    plat_r2_by_window = df_pred.groupby("window")["r2_platinum"].first()
    angle_by_window = df_angle[(df_angle["fast_yr"] == 2) & (df_angle["K"] == 5)].set_index("window")["angle_deg"]
    common_wins = plat_r2_by_window.index.intersection(angle_by_window.index)
    if len(common_wins) >= 5:
        from scipy.stats import spearmanr, pearsonr
        r2_vals = plat_r2_by_window[common_wins].values
        angle_vals = angle_by_window[common_wins].values
        sp_corr, sp_p = spearmanr(angle_vals, r2_vals)
        pe_corr, pe_p = pearsonr(angle_vals, r2_vals)
        print(f"  Angle(2yr,K=5) vs PLATINUM R^2:")
        print(f"    Spearman: r={sp_corr:.3f}, p={sp_p:.3f}")
        print(f"    Pearson:  r={pe_corr:.3f}, p={pe_p:.3f}")
        print(f"    Interpretation: {'angle PREDICTS R^2' if abs(sp_corr) > 0.3 else 'no relationship'}")

        # Also test: angle predicts NEXT window R^2
        if len(common_wins) >= 6:
            wins_sorted = sorted(common_wins)
            angle_prev = [angle_by_window[w] for w in wins_sorted[:-1]]
            r2_next = [plat_r2_by_window[w] for w in wins_sorted[1:]]
            sp_lag, sp_lag_p = spearmanr(angle_prev, r2_next)
            print(f"  Angle(t) vs R^2(t+1) [predictive]:")
            print(f"    Spearman: r={sp_lag:.3f}, p={sp_lag_p:.3f}")
    else:
        print(f"  Not enough common windows ({len(common_wins)}) for correlation")

    # V3-3: Angle-weighted prediction
    print(f"\n  === V3-3: ANGLE-WEIGHTED PREDICTION ===")
    print(f"  {'gamma':>8} {'Mean R2':>10} {'Delta':>10} {'Mean w_spec':>12}")
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sub = df_pred[df_pred["gamma"] == gamma]
        print(f"  {gamma:>8.1f} {sub['r2'].mean():>10.4f} "
              f"{sub['delta'].mean():>+10.4f} {sub['w_spectral'].mean():>12.3f}")

    return df_angle, df_pred


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="SMIM DD2 Sprint 1")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only V1 or V3 (e.g. --only V1)")
    args = parser.parse_args()

    run_v1_flag = args.only is None or "V1" in args.only.upper()
    run_v3_flag = args.only is None or "V3" in args.only.upper()

    print("=" * 90)
    print(f"  SMIM DRILLDOWN #2 SPRINT 1  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Baseline: PLATINUM R^2=0.543")
    print(f"  Venues: {'V1 ' if run_v1_flag else ''}{'V3' if run_v3_flag else ''}")
    print("=" * 90)

    wide = load_panel()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if run_v1_flag:
        df_v1 = run_v1(wide)
        df_v1.to_parquet(METRICS_DIR / "dd2_V1_edmd.parquet", index=False)

    if run_v3_flag:
        df_angle, df_pred = run_v3(wide)
        if not df_angle.empty:
            df_angle.to_parquet(METRICS_DIR / "dd2_V3_angles.parquet", index=False)
        if not df_pred.empty:
            df_pred.to_parquet(METRICS_DIR / "dd2_V3_pred.parquet", index=False)

    # Final summary
    print("\n" + "=" * 90)
    print("  SPRINT 1 FINDINGS")
    print("=" * 90)

    if run_v1_flag:
        best_edmd = df_v1[df_v1["blend_w"] == 1.0].groupby("experiment")["r2"].mean()
        plat = df_v1["r2_platinum"].mean()
        print(f"\n  V1 (Nonlinear mode coupling):")
        print(f"    PLATINUM baseline: {plat:.4f}")
        for name, r2 in best_edmd.items():
            print(f"    {name}: {r2:.4f} ({r2 - plat:+.4f})")
        any_better = any(r2 > plat + 0.001 for r2 in best_edmd.values)
        print(f"    VERDICT: {'NONLINEAR STRUCTURE DETECTED' if any_better else 'No nonlinear improvement'}")

    if run_v3_flag and not df_angle.empty:
        mean_angle = df_angle[
            (df_angle["fast_yr"] == 2) & (df_angle["K"] == 5)
        ]["angle_deg"].mean()
        print(f"\n  V3 (Multi-resolution emergence):")
        print(f"    Mean subspace angle (2yr/5yr, K=5): {mean_angle:.1f} degrees")
        print(f"    {'EMERGENCE SIGNAL' if mean_angle > 15 else 'Bases are aligned (no emergence)'}")


if __name__ == "__main__":
    main()
