#!/usr/bin/env python
"""
Phase A-B Drill-Down Phase 2: DD-7, DD-8, DD-9.

DD-7: Factor-augmented AR(1) with ridge regularisation
DD-8: Ensemble weight optimisation (alpha*AR1 + (1-alpha)*L1)
DD-9: Shrinkage Kalman (Ledoit-Wolf on R)

All use the Phase 1 best config as baseline: T=5yr, K=5, EWM_8Q demeaning.

Output:
  results/metrics/drilldown_DD-{7,8,9}.parquet

Usage::
    uv run python scripts/smim/run_smim_drilldown_p2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"

K_MAX = 15
K_BEST = 5
T_BEST = 5  # years
TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_panel():
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols]


def prepare_window(wide, ty, t_yr=T_BEST):
    ts = pd.Timestamp(f"{ty - t_yr}-01-01")
    te = pd.Timestamp(f"{ty - 1}-12-31")
    test_s = pd.Timestamp(f"{ty}-01-01")
    test_e = pd.Timestamp(f"{ty}-12-31")

    tr = wide[(wide.index >= ts) & (wide.index <= te)]
    te_df = wide[(wide.index >= test_s) & (wide.index <= test_e)]
    valid = tr.columns[tr.notna().any()]
    tr, te_df = tr[valid], te_df[valid]
    cm = tr.mean()
    otr = tr.fillna(cm).values.astype(np.float64)
    ote = te_df.fillna(cm).values.astype(np.float64)

    # EWM demeaning
    weights = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
    om = (otr * weights[:, None]).sum(axis=0, keepdims=True) / weights.sum()

    return otr - om, ote - om, ote, om, list(valid), otr.shape[1], otr


def build_basis(obs_train, N, K):
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0
    mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    return mf.basis[:, :min(K, mf.K)].real, mf


def l1_predict(obs_train, obs_test, obs_mean, U):
    """L1 graph-factor OLS prediction. Returns (T_test, N) restored predictions."""
    K = U.shape[1]
    factors = obs_train @ U
    try:
        A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.9
    f_last = factors[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        fn = A.T @ f_last
        pred[t] = fn @ U.T
        f_last = fn
    return pred + obs_mean


def ar1_predict(obs_train_raw, obs_test_raw):
    """Per-actor AR(1). Returns (T_test, N) predictions."""
    N = obs_train_raw.shape[1]
    pred = np.zeros_like(obs_test_raw)
    for i in range(N):
        y = obs_train_raw[:, i]
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        try:
            beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = [y.mean(), 0]
        v = obs_train_raw[-1, i]
        for t in range(obs_test_raw.shape[0]):
            pred[t, i] = beta[0] + beta[1] * v
            v = pred[t, i]
    return pred


# --- DD-7: Factor-Augmented AR(1) with Ridge --------------------------------


def run_dd7(wide):
    print("\n" + "=" * 70)
    print("  DD-7: FACTOR-AUGMENTED AR(1) WITH RIDGE REGULARISATION")
    print("  y_{i,t+1} = a_i + b_i*y_{i,t} + Sum_k w_{ik}*f_{k,t} + eps")
    print("  Ridge penalty prevents overfitting from added factor regressors")
    print("=" * 70)

    ridge_lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    K_values = [1, 3, 5]
    rows = []

    for ty in TEST_YEARS:
        otr, ote, ote_raw, om, _, N, otr_raw = prepare_window(wide, ty)
        actual = ote_raw.T.ravel()
        U, _ = build_basis(otr, N, max(K_values))

        # Pure AR(1) baseline
        pred_ar = ar1_predict(otr_raw, ote_raw)
        r2_ar = float(oos_r_squared(pred_ar.T.ravel(), actual))

        for K in K_values:
            Uk = U[:, :K]
            factors = otr @ Uk  # (T, K) demeaned factors

            for lam in ridge_lambdas:
                pred_fa = np.zeros_like(ote_raw)
                for i in range(N):
                    y = otr_raw[:, i]
                    # Regressors: intercept + lagged y + lagged factors
                    y_lag = y[1:-1]
                    f_lag = factors[1:-1]
                    y_curr = y[2:]
                    X = np.column_stack([np.ones(len(y_lag)), y_lag, f_lag])

                    # Ridge regression: (X'X + lambda*I)^{-1} X'y
                    XtX = X.T @ X
                    Xty = X.T @ y_curr
                    n_params = XtX.shape[0]
                    reg_mat = lam * np.eye(n_params)
                    reg_mat[0, 0] = 0  # don't regularise intercept
                    try:
                        beta = np.linalg.solve(XtX + reg_mat, Xty)
                    except np.linalg.LinAlgError:
                        pred_fa[:, i] = otr_raw[-1, i]
                        continue

                    last_y = otr_raw[-1, i]
                    last_f = factors[-1, :K]
                    for t in range(ote_raw.shape[0]):
                        x = np.concatenate([[1, last_y], last_f])
                        pred_fa[t, i] = x @ beta
                        last_y = pred_fa[t, i]

                r2_fa = float(oos_r_squared(pred_fa.T.ravel(), actual))
                rows.append({"dd": "DD-7", "window": f"W{ty}", "K": K,
                              "ridge_lambda": lam, "r2_fa_ar1": r2_fa, "r2_ar1": r2_ar})

    df = pd.DataFrame(rows)

    # Find best lambda per K
    print(f"\n  Best config per K (mean across windows):")
    print(f"  {'K':>4} {'Lambda':>8} {'FA-AR1 R2':>10} {'AR(1) R2':>10} {'Delta':>10}")
    for K in K_values:
        sub = df[df["K"] == K]
        by_lam = sub.groupby("ridge_lambda")["r2_fa_ar1"].mean()
        best_lam = by_lam.idxmax()
        best_r2 = by_lam.max()
        ar1_mean = sub["r2_ar1"].mean()
        print(f"  {K:>4} {best_lam:>8.2f} {best_r2:>10.4f} {ar1_mean:>10.4f} {best_r2 - ar1_mean:>+10.4f}")

    # Detailed: best overall
    best_overall = df.groupby(["K", "ridge_lambda"])["r2_fa_ar1"].mean()
    best_idx = best_overall.idxmax()
    print(f"\n  Overall best: K={best_idx[0]}, lambda={best_idx[1]}, R2={best_overall.max():.4f}")
    return df


# --- DD-8: Ensemble Weight Optimisation --------------------------------------


def run_dd8(wide):
    print("\n" + "=" * 70)
    print("  DD-8: ENSEMBLE WEIGHT OPTIMISATION")
    print("  pred = alpha * AR(1) + (1-alpha) * L1")
    print("=" * 70)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []

    for ty in TEST_YEARS:
        otr, ote, ote_raw, om, _, N, otr_raw = prepare_window(wide, ty)
        actual = ote_raw.T.ravel()
        U, _ = build_basis(otr, N, K_BEST)

        pred_l1 = l1_predict(otr, ote, om, U)
        pred_ar = ar1_predict(otr_raw, ote_raw)

        for alpha in alphas:
            ensemble = alpha * pred_ar + (1 - alpha) * pred_l1
            r2 = float(oos_r_squared(ensemble.T.ravel(), actual))
            rows.append({"dd": "DD-8", "window": f"W{ty}", "alpha": alpha, "oos_r2": r2})

    df = pd.DataFrame(rows)
    by_alpha = df.groupby("alpha")["oos_r2"].mean()

    print(f"\n  {'Alpha':>8} {'Mean R2':>10}  Description")
    for alpha in alphas:
        r2 = by_alpha[alpha]
        desc = ""
        if alpha == 0.0:
            desc = "(pure L1)"
        elif alpha == 1.0:
            desc = "(pure AR1)"
        elif alpha == by_alpha.idxmax():
            desc = "<-- BEST"
        print(f"  {alpha:>8.1f} {r2:>10.4f}  {desc}")

    best_alpha = by_alpha.idxmax()
    best_r2 = by_alpha.max()
    print(f"\n  Optimal alpha = {best_alpha:.1f}  (R2 = {best_r2:.4f})")
    print(f"  Interpretation: {'AR(1) dominates' if best_alpha > 0.7 else 'Balanced mix' if best_alpha > 0.3 else 'L1 dominates'}")
    return df


# --- DD-9: Shrinkage Kalman --------------------------------------------------


def run_dd9(wide):
    print("\n" + "=" * 70)
    print("  DD-9: SHRINKAGE KALMAN (LEDOIT-WOLF ON R)")
    print("  R_shrunk = (1-s)*R_sample + s*trace(R)/N * I")
    print("=" * 70)

    shrinkage_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    rows = []

    for ty in TEST_YEARS:
        otr, ote, ote_raw, om, _, N, otr_raw = prepare_window(wide, ty)
        actual = ote_raw.T.ravel()
        U, mf_full = build_basis(otr, N, K_BEST)
        mf_k = ModalFrame(basis=U, eigenvalues=mf_full.eigenvalues[:K_BEST], method=mf_full.method)

        # Pure AR(1) for comparison
        pred_ar = ar1_predict(otr_raw, ote_raw)
        r2_ar = float(oos_r_squared(pred_ar.T.ravel(), actual))

        # L1 baseline (no Kalman)
        pred_l1 = l1_predict(otr, ote, om, U)
        r2_l1 = float(oos_r_squared(pred_l1.T.ravel(), actual))

        for shrinkage in shrinkage_levels:
            try:
                # Run Kalman EM
                kf = KalmanFilter()
                state_train = kf.em_estimate(otr, mf_k, max_iter=50)

                # Apply shrinkage to R
                R_sample = state_train.params["R"]
                target = np.eye(N) * np.trace(R_sample) / N
                R_shrunk = (1 - shrinkage) * R_sample + shrinkage * target

                # Filter test with shrunk R
                kf_test = KalmanFilter(
                    F=state_train.params["F"],
                    Q=state_train.params["Q"],
                    R=R_shrunk,
                )
                state_test = kf_test.filter(ote, mf_k)

                # Predict: use alpha_predicted (predictive benchmark)
                pred_kalman = (state_test.alpha_predicted @ U.T + om)
                r2_kalman = float(oos_r_squared(pred_kalman.T.ravel(), actual))

                # Modal: use alpha_filtered
                pred_modal = (state_test.alpha_filtered @ U.T + om)
                r2_modal = float(oos_r_squared(pred_modal.T.ravel(), actual))
            except Exception:
                r2_kalman = float("nan")
                r2_modal = float("nan")

            rows.append({"dd": "DD-9", "window": f"W{ty}", "shrinkage": shrinkage,
                          "r2_kalman_pred": r2_kalman, "r2_kalman_modal": r2_modal,
                          "r2_l1": r2_l1, "r2_ar1": r2_ar})

    df = pd.DataFrame(rows)
    by_s = df.groupby("shrinkage").agg(
        kalman_pred=("r2_kalman_pred", "mean"),
        kalman_modal=("r2_kalman_modal", "mean"),
        l1=("r2_l1", "mean"),
        ar1=("r2_ar1", "mean"),
    )

    print(f"\n  {'Shrinkage':>10} {'Kalman pred':>12} {'Kalman modal':>13} {'L1 OLS':>10} {'AR(1)':>10}")
    for s, row in by_s.iterrows():
        marker = ""
        if row["kalman_modal"] == by_s["kalman_modal"].max():
            marker = " <-- BEST modal"
        print(f"  {s:>10.1f} {row['kalman_pred']:>12.4f} {row['kalman_modal']:>13.4f} "
              f"{row['l1']:>10.4f} {row['ar1']:>10.4f}{marker}")

    best_s = by_s["kalman_modal"].idxmax()
    best_modal = by_s.loc[best_s, "kalman_modal"]
    l1_ref = by_s["l1"].mean()
    print(f"\n  Best shrinkage = {best_s:.1f}: modal R2 = {best_modal:.4f} "
          f"(vs L1 = {l1_ref:.4f}, vs AR1 = {by_s['ar1'].mean():.4f})")
    kalman_helps = best_modal > l1_ref
    print(f"  Kalman {'HELPS' if kalman_helps else 'still hurts'} vs L1 with shrinkage = {best_s:.1f}")
    return df


# --- main --------------------------------------------------------------------


def main():
    print("=" * 70)
    print(f"  DRILL-DOWN PHASE 2  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DD-7 (FA-AR1), DD-8 (Ensemble), DD-9 (Shrinkage Kalman)")
    print(f"  Baseline: T={T_BEST}yr, K={K_BEST}, EWM_8Q demeaning")
    print("=" * 70)

    wide = load_panel()

    df7 = run_dd7(wide)
    df8 = run_dd8(wide)
    df9 = run_dd9(wide)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [("DD-7", df7), ("DD-8", df8), ("DD-9", df9)]:
        df.to_parquet(METRICS_DIR / f"drilldown_{name}.parquet", index=False)

    print(f"\n  Saved DD-7, DD-8, DD-9 results to results/metrics/")


if __name__ == "__main__":
    main()
