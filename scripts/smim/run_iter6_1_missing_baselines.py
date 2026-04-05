"""
Iteration 6.1 — Missing Baselines (Reviewer Response)

Two new residual-stage ablations requested by reviewers:

  1. PCA + diagonal AR(1) on residual PC scores
     Tests whether the gain is DMD-specific or any-basis + per-mode dynamics

  2. Ridge regression on lagged cross-sectional residuals
     Tests whether a simple regularised regression achieves similar gains

Runs on the 93-actor multilayer panel with the same 10 rolling OOS windows
and identical Stage 1 (pooled AR(1)+FE) as the main ablation ladder.

Usage:
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_missing_baselines.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# ── Paths ─────────────────────────────────────────────────
INTENSITIES_93 = ROOT / "data/smim/intensities/experiment_a1_intensities.parquet"
REGISTRY_93 = ROOT / "data/smim/registries/experiment_a1_registry.json"
METRICS_DIR = ROOT / "results/metrics"

# ── Constants (match main ablation exactly) ───────────────
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_DEFAULT = 8
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))


# ── Helpers (copied from validation script for self-containment) ──

def load_93_actor_panel():
    df = pd.read_parquet(INTENSITIES_93)
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    otr = ad[ad.index < pd.Timestamp(f"{ty}-01-01")].values
    tq = panel[
        (panel.index >= pd.Timestamp(f"{ty}-01-01"))
        & (panel.index <= pd.Timestamp(f"{ty}-12-31"))
    ].index.unique()
    N = ad.shape[1]
    return ad, otr, tq, N


def _clip_sr(F, max_sr=0.99):
    eigvals = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(eigvals)))
    return F * (max_sr / mx) if mx > max_sr else F


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def dmd_full(dm, k_svd=K_MAX):
    from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


# ── PCA diagonal AR(1) on residual PC scores ─────────────

def _fit_pca_diag_ar1(dm_r, K):
    """Fit PCA basis + per-component AR(1) on demeaned residuals."""
    dm_r = np.nan_to_num(dm_r, nan=0.0)
    N = dm_r.shape[1]
    C_r = dm_r.T @ dm_r / dm_r.shape[0]
    ev_r, ec_r = np.linalg.eigh(C_r)
    idx = np.argsort(ev_r)[::-1]
    U_pca = ec_r[:, idx][:, :min(K, N - 2)]

    # Project to PC scores
    fac = dm_r @ U_pca  # (T-1, K)

    # Per-component AR(1)
    k_act = U_pca.shape[1]
    mu_pc = np.nan_to_num(fac.mean(0), nan=0.0)
    d_pc = fac - mu_pc
    rho_pc = np.zeros(k_act)
    for k in range(k_act):
        y = d_pc[:, k]
        if len(y) >= 3 and np.std(y[:-1]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho_pc[k] = np.clip(c, -0.99, 0.99)

    return U_pca, mu_pc, rho_pc


def _predict_pca_diag_ar1(U_pca, mu_pc, rho_pc, prev_resid, om_r):
    """Predict residual using PCA + diagonal AR(1)."""
    f_prev = U_pca.T @ (prev_resid - om_r)
    f_pred = mu_pc + rho_pc * (f_prev - mu_pc)
    return om_r + U_pca @ f_pred


# ── Ridge regression on lagged cross-sectional residuals ──

def _fit_ridge(dm_r, alpha=1.0):
    """Fit Ridge regression: r_{t} = B @ r_{t-1} with L2 penalty."""
    X = np.nan_to_num(dm_r[:-1], nan=0.0)
    Y = np.nan_to_num(dm_r[1:], nan=0.0)
    N = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(N)
    try:
        B = np.linalg.solve(XtX, X.T @ Y).T  # (N, N)
    except np.linalg.LinAlgError:
        B = np.eye(N) * 0.5
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return _clip_sr(B, 0.99)


def _predict_ridge(B, prev_resid, om_r):
    """Predict residual using Ridge regression."""
    return om_r + B @ (prev_resid - om_r)


# ── Main ablation runner ──────────────────────────────────

def run_window_new_baselines(panel, ty, K=K_DEFAULT, ewm_hl=12, T_yr=5):
    """Run the two new residual-stage baselines for one test year."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N = prep

    # Drop actors (columns) with >20% NaN in training window
    otr = otr.astype(np.float64)
    nan_frac = np.isnan(otr).mean(axis=0)
    good_cols = nan_frac < 0.2
    if good_cols.sum() < 10:
        print(f"    Too few actors with data for {ty}: {good_cols.sum()}")
        return None
    otr = otr[:, good_cols]
    ad = ad.iloc[:, good_cols]
    N = otr.shape[1]
    # Forward-fill then backfill remaining NaN within each actor
    for j in range(N):
        col = otr[:, j]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            # Forward fill
            last_val = np.nan
            for i in range(len(col)):
                if np.isnan(col[i]):
                    if not np.isnan(last_val):
                        col[i] = last_val
                else:
                    last_val = col[i]
            # Backfill remaining
            next_val = np.nan
            for i in range(len(col) - 1, -1, -1):
                if np.isnan(col[i]):
                    if not np.isnan(next_val):
                        col[i] = next_val
                else:
                    next_val = col[i]
            otr[:, j] = col

    # Stage 1: pooled+FE (identical to main ablation)
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Residual basis estimations
    om_r = ewm_demean(residuals, ewm_hl)
    dm_r = residuals - om_r

    # Fit new baselines
    U_pca, mu_pc, rho_pc = _fit_pca_diag_ar1(dm_r, K)

    # Ridge: try multiple alphas
    ridge_alphas = [0.1, 1.0, 10.0]
    B_ridges = {a: _fit_ridge(dm_r, alpha=a) for a in ridge_alphas}

    # Also re-run the existing DMD/Kalman diag for comparison
    dm_r_clean = np.nan_to_num(dm_r, nan=0.0)
    mf_r = dmd_full(dm_r_clean, k_svd=K_MAX)
    if mf_r is None:
        print(f"    DMD failed for {ty}: dm_r shape={dm_r.shape}, nans={np.isnan(dm_r).sum()}")
        return None
    ka = min(K, mf_r.basis.shape[0] - 2, mf_r.K)
    U_dmd = mf_r.metadata["U"][:, :ka]
    A_full = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
    F_diag = _clip_sr(np.diag(np.diag(A_full)))
    F_full = _clip_sr(A_full)
    R_dmd = sph_r(dm_r, U_dmd)

    # Kalman states
    a_diag = np.zeros(ka)
    P_diag = np.eye(ka)
    Q_diag = np.eye(ka) * Q_INIT_SCALE
    a_full = np.zeros(ka)
    P_full_st = np.eye(ka)
    Q_full_st = np.eye(ka) * Q_INIT_SCALE

    # Accumulators
    models = [
        "pooled_only", "resid_pca_diag_ar1",
        "resid_ridge_0.1", "resid_ridge_1.0", "resid_ridge_10.0",
        "resid_dmd_diag", "resid_dmd_full",
    ]
    results = {k: ([], []) for k in models}
    prev = np.nan_to_num(otr[-1], nan=0.5)
    prev_resid = np.zeros(N)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_ar = bar_y + rho * (prev - bar_y)
        actual_resid = obs - y_ar

        # 1. Pooled only (reference)
        results["pooled_only"][0].append(y_ar)

        # 2. PCA + diagonal AR(1)
        pred_pca_ar1 = _predict_pca_diag_ar1(
            U_pca, mu_pc, rho_pc, prev_resid, om_r.ravel()
        )
        if np.all(np.isfinite(pred_pca_ar1)):
            results["resid_pca_diag_ar1"][0].append(y_ar + pred_pca_ar1)
        else:
            results["resid_pca_diag_ar1"][0].append(y_ar)

        # 3. Ridge regression variants
        for alpha in ridge_alphas:
            pred_ridge = _predict_ridge(B_ridges[alpha], prev_resid, om_r.ravel())
            key = f"resid_ridge_{alpha}"
            if np.all(np.isfinite(pred_ridge)):
                results[key][0].append(y_ar + pred_ridge)
            else:
                results[key][0].append(y_ar)

        # 4-5. DMD/Kalman diag and full (reference, for comparison)
        for label, F, a, P, Q in [
            ("resid_dmd_diag", F_diag, a_diag, P_diag, Q_diag),
            ("resid_dmd_full", F_full, a_full, P_full_st, Q_full_st),
        ]:
            ap = F @ a
            pred_r = U_dmd @ ap + om_r.ravel()
            if not np.all(np.isfinite(pred_r)):
                pred_r = np.zeros(N)
            results[label][0].append(y_ar + pred_r)

        for k in results:
            results[k][1].append(obs)

        # Kalman updates for DMD variants
        odm_r = actual_resid - om_r.ravel()
        for F, a_ref, P_ref, Q_ref, out_vars in [
            (F_diag, a_diag, P_diag, Q_diag, "diag"),
            (F_full, a_full, P_full_st, Q_full_st, "full"),
        ]:
            ap = F @ a_ref
            Pp = F @ P_ref @ F.T + Q_ref
            S = U_dmd @ Pp @ U_dmd.T + R_dmd
            try:
                Kg = Pp @ U_dmd.T @ np.linalg.solve(S, np.eye(N))
            except np.linalg.LinAlgError:
                Kg = np.zeros((ka, N))
            a_new = ap + Kg @ (odm_r - U_dmd @ ap)
            P_new = (np.eye(ka) - Kg @ U_dmd) @ Pp
            inn = a_new - ap
            Q_new = (1 - LAMBDA_Q) * Q_ref + LAMBDA_Q * np.outer(inn, inn)
            Q_new = (Q_new + Q_new.T) / 2 + np.eye(ka) * 1e-6
            if out_vars == "diag":
                a_diag, P_diag, Q_diag = a_new, P_new, Q_new
            else:
                a_full, P_full_st, Q_full_st = a_new, P_new, Q_new

        prev_resid = actual_resid
        prev = obs

        # Rolling update
        otr = np.vstack([otr, qv])
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, ewm_hl)
        dm_r = residuals_new - om_r

        # Re-fit PCA + diag AR(1)
        U_pca, mu_pc, rho_pc = _fit_pca_diag_ar1(dm_r, K)

        # Re-fit Ridge
        B_ridges = {a: _fit_ridge(dm_r, alpha=a) for a in ridge_alphas}

        # Re-fit DMD
        dm_r_clean = np.nan_to_num(dm_r, nan=0.0)
        mf_r2 = dmd_full(dm_r_clean, k_svd=K_MAX)
        if mf_r2 is not None:
            ka2 = min(K, mf_r2.basis.shape[0] - 2, mf_r2.K)
            U_dmd = mf_r2.metadata["U"][:, :ka2]
            A_full = mf_r2.metadata["Atilde"][:ka2, :ka2].real.copy()
            F_diag = _clip_sr(np.diag(np.diag(A_full)))
            F_full = _clip_sr(A_full)
            R_dmd = sph_r(dm_r, U_dmd)
            a_new_st = U_dmd.T @ (actual_resid - om_r.ravel())
            a_diag = a_new_st
            P_diag = np.eye(ka2)
            Q_diag = np.eye(ka2) * Q_INIT_SCALE
            a_full = a_new_st
            P_full_st = np.eye(ka2)
            Q_full_st = np.eye(ka2) * Q_INIT_SCALE
            ka = ka2

    # Compute R² for each model
    out = {}
    for k in results:
        ps, ac = results[k]
        if not ps:
            return None
        psa, aca = np.array(ps), np.array(ac)
        if not np.all(np.isfinite(psa)):
            out[k] = None
        else:
            out[k] = float(oos_r_squared(psa.ravel(), aca.ravel()))
    return out


# ── Main ──────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ITERATION 6.1 — MISSING BASELINES (REVIEWER RESPONSE)")
    print("=" * 70)

    panel = load_93_actor_panel()
    print(f"Panel: {panel.shape[1]} actors, {panel.shape[0]} quarters")

    all_results = {k: [] for k in [
        "pooled_only", "resid_pca_diag_ar1",
        "resid_ridge_0.1", "resid_ridge_1.0", "resid_ridge_10.0",
        "resid_dmd_diag", "resid_dmd_full",
    ]}

    for ty in TEST_YEARS:
        res = run_window_new_baselines(panel, ty)
        if res is None:
            print(f"  W{ty}: SKIP")
            for k in all_results:
                all_results[k].append(None)
            continue
        line = f"  W{ty}:"
        for k in sorted(res):
            v = res[k]
            all_results[k].append(v)
            if v is not None:
                line += f"  {k}={v:.3f}"
        print(line)

    print("\n" + "=" * 70)
    print("SUMMARY: Mean R² across 10 windows")
    print("=" * 70)

    ar1_r2 = 0.594  # from main results
    for k in [
        "pooled_only", "resid_pca_diag_ar1",
        "resid_ridge_0.1", "resid_ridge_1.0", "resid_ridge_10.0",
        "resid_dmd_diag", "resid_dmd_full",
    ]:
        vals = [v for v in all_results[k] if v is not None]
        if vals:
            m = np.mean(vals)
            delta_pool = m - 0.591
            delta_ar1 = m - ar1_r2
            print(f"  {k:30s}  R²={m:.3f}  Δ(pool)={delta_pool:+.3f}  Δ(AR1)={delta_ar1:+.3f}")

    print("\n" + "=" * 70)
    print("KEY COMPARISON")
    print("=" * 70)

    pca_vals = [v for v in all_results["resid_pca_diag_ar1"] if v is not None]
    dmd_diag_vals = [v for v in all_results["resid_dmd_diag"] if v is not None]
    dmd_full_vals = [v for v in all_results["resid_dmd_full"] if v is not None]

    if pca_vals and dmd_diag_vals:
        pca_m = np.mean(pca_vals)
        dmd_d_m = np.mean(dmd_diag_vals)
        dmd_f_m = np.mean(dmd_full_vals)
        print(f"  PCA + diag AR(1):     R² = {pca_m:.3f}")
        print(f"  DMD/Kalman diag(Ã):   R² = {dmd_d_m:.3f}")
        print(f"  DMD/Kalman full Ã:    R² = {dmd_f_m:.3f}")
        print(f"  Δ(DMD diag - PCA diag AR1) = {dmd_d_m - pca_m:+.3f}")
        print()

        # Per-window comparison
        pairs = [(p, d) for p, d in zip(pca_vals, dmd_diag_vals)
                 if p is not None and d is not None]
        dmd_wins = sum(1 for p, d in pairs if d > p)
        print(f"  DMD diag wins {dmd_wins}/{len(pairs)} windows vs PCA diag AR(1)")

        # Paired t-test
        diffs = np.array([d - p for p, d in pairs])
        if len(diffs) > 1:
            t_stat = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            print(f"  Paired t = {t_stat:.2f}  (mean Δ = {np.mean(diffs):+.4f})")

    # Best Ridge
    best_ridge_k, best_ridge_m = None, -999
    for alpha in [0.1, 1.0, 10.0]:
        rv = [v for v in all_results[f"resid_ridge_{alpha}"] if v is not None]
        if rv:
            m = np.mean(rv)
            if m > best_ridge_m:
                best_ridge_m, best_ridge_k = m, f"resid_ridge_{alpha}"
    if best_ridge_k:
        print(f"\n  Best Ridge: {best_ridge_k}  R² = {best_ridge_m:.3f}")
        print(f"  Δ(DMD diag - best Ridge) = {np.mean(dmd_diag_vals) - best_ridge_m:+.3f}")

    # Save results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ty in enumerate(TEST_YEARS):
        row = {"year": ty}
        for k in all_results:
            row[k] = all_results[k][i]
        rows.append(row)
    out_path = METRICS_DIR / "iter6_1_missing_baselines.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
