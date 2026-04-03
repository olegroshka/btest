#!/usr/bin/env python
"""
Workstream E supplement -- CapEx-revision regression with panel FE and clustered SEs.

Re-runs the D2 panel regression:
  delta_y_{i,t->t+4} = beta_0 + beta_1 * gap_{i,t} + beta_2 * y_{i,t} + epsilon

With upgrades:
  1. Pooled OLS (original)
  2. Actor fixed effects
  3. Time fixed effects
  4. Two-way FE (actor + time)
  5. Clustered SEs (by actor, by time, two-way)

Uses gaps from the Workstream A actor-quarter predictions (SMIM pipeline gaps).

Output: results/metrics/ws_e_capex_regression.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
PRED_PATH = RESULTS_DIR / "ws_a_actor_quarter_predictions.parquet"


def demean_by(df, col, group_col):
    """Demean a column by group (for FE absorption)."""
    return df[col] - df.groupby(group_col)[col].transform("mean")


def ols_with_se(y, X, cluster_ids=None):
    """OLS with optional clustered SEs."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    if cluster_ids is None:
        # Heteroskedasticity-robust (HC1)
        sigma2 = resid**2
        bread = np.linalg.inv(X.T @ X)
        meat = X.T @ np.diag(sigma2) @ X
        V = bread @ meat @ bread * n / (n - k)
    else:
        # Clustered SEs
        bread = np.linalg.inv(X.T @ X)
        clusters = np.unique(cluster_ids)
        G = len(clusters)
        meat = np.zeros((k, k))
        for c in clusters:
            mask = cluster_ids == c
            Xc = X[mask]
            ec = resid[mask]
            score = Xc.T @ ec
            meat += np.outer(score, score)
        # Small-sample correction
        correction = G / (G - 1) * (n - 1) / (n - k)
        V = bread @ meat @ bread * correction

    se = np.sqrt(np.diag(V))
    t_stat = beta / se
    return beta, se, t_stat


def main():
    print("=" * 90)
    print("  WORKSTREAM E: CAPEX-REVISION REGRESSION WITH FE AND CLUSTERED SEs")
    print("=" * 90)

    if not PRED_PATH.exists():
        print(f"  ERROR: {PRED_PATH} not found. Run Workstream A first.")
        return

    pred_df = pd.read_parquet(PRED_PATH)
    print(f"  Loaded {len(pred_df)} actor-quarter observations")

    # Compute gaps: gap = actual - pred_smim (model-implied gap)
    pred_df["gap"] = pred_df["actual"] - pred_df["pred_smim"]
    pred_df["quarter"] = pd.to_datetime(pred_df["quarter"])

    # Build panel: need next-4-quarter intensity change
    # Sort and compute forward change per actor
    panel = pred_df[["actor_id", "quarter", "actual", "gap"]].copy()
    panel = panel.sort_values(["actor_id", "quarter"])
    panel["delta_y_4q"] = panel.groupby("actor_id")["actual"].diff(4).shift(-4)

    # Drop NaN outcomes
    reg_df = panel.dropna(subset=["delta_y_4q", "gap", "actual"]).copy()
    reg_df["actor_num"] = pd.Categorical(reg_df["actor_id"]).codes
    reg_df["quarter_num"] = pd.Categorical(reg_df["quarter"]).codes

    n = len(reg_df)
    print(f"  Regression sample: {n} actor-quarter observations")
    print(f"  Actors: {reg_df['actor_id'].nunique()}, Quarters: {reg_df['quarter'].nunique()}")

    if n < 20:
        print("  Too few observations for regression. Need more test windows.")
        return

    y = reg_df["delta_y_4q"].values
    gap = reg_df["gap"].values
    intensity = reg_df["actual"].values
    ones = np.ones(n)

    results = []

    # --- Spec 1: Pooled OLS ---
    X = np.column_stack([ones, gap, intensity])
    beta, se, t = ols_with_se(y, X)
    results.append({
        "spec": "Pooled OLS (HC1)",
        "beta_gap": beta[1], "se_gap": se[1], "t_gap": t[1],
        "beta_level": beta[2], "se_level": se[2], "t_level": t[2],
        "n": n,
    })
    print(f"\n  Pooled OLS (HC1): beta_gap={beta[1]:.4f}, t={t[1]:.2f}, n={n}")

    # --- Spec 2: Actor-clustered SEs ---
    beta, se, t = ols_with_se(y, X, cluster_ids=reg_df["actor_num"].values)
    results.append({
        "spec": "Pooled OLS (actor-clustered)",
        "beta_gap": beta[1], "se_gap": se[1], "t_gap": t[1],
        "beta_level": beta[2], "se_level": se[2], "t_level": t[2],
        "n": n,
    })
    print(f"  Actor-clustered:   beta_gap={beta[1]:.4f}, t={t[1]:.2f}")

    # --- Spec 3: Time-clustered SEs ---
    beta, se, t = ols_with_se(y, X, cluster_ids=reg_df["quarter_num"].values)
    results.append({
        "spec": "Pooled OLS (time-clustered)",
        "beta_gap": beta[1], "se_gap": se[1], "t_gap": t[1],
        "beta_level": beta[2], "se_level": se[2], "t_level": t[2],
        "n": n,
    })
    print(f"  Time-clustered:    beta_gap={beta[1]:.4f}, t={t[1]:.2f}")

    # --- Spec 4: Actor FE (within-actor demeaning) ---
    y_dm = demean_by(reg_df.assign(y=y), "y", "actor_id").values
    gap_dm = demean_by(reg_df.assign(g=gap), "g", "actor_id").values
    int_dm = demean_by(reg_df.assign(i=intensity), "i", "actor_id").values
    X_fe = np.column_stack([gap_dm, int_dm])
    beta_fe, se_fe, t_fe = ols_with_se(y_dm, X_fe, cluster_ids=reg_df["actor_num"].values)
    results.append({
        "spec": "Actor FE (actor-clustered)",
        "beta_gap": beta_fe[0], "se_gap": se_fe[0], "t_gap": t_fe[0],
        "beta_level": beta_fe[1], "se_level": se_fe[1], "t_level": t_fe[1],
        "n": n,
    })
    print(f"  Actor FE:          beta_gap={beta_fe[0]:.4f}, t={t_fe[0]:.2f}")

    # --- Spec 5: Time FE (within-quarter demeaning) ---
    y_dm_t = demean_by(reg_df.assign(y=y), "y", "quarter").values
    gap_dm_t = demean_by(reg_df.assign(g=gap), "g", "quarter").values
    int_dm_t = demean_by(reg_df.assign(i=intensity), "i", "quarter").values
    X_fe_t = np.column_stack([gap_dm_t, int_dm_t])
    beta_ft, se_ft, t_ft = ols_with_se(y_dm_t, X_fe_t, cluster_ids=reg_df["quarter_num"].values)
    results.append({
        "spec": "Time FE (time-clustered)",
        "beta_gap": beta_ft[0], "se_gap": se_ft[0], "t_gap": t_ft[0],
        "beta_level": beta_ft[1], "se_level": se_ft[1], "t_level": t_ft[1],
        "n": n,
    })
    print(f"  Time FE:           beta_gap={beta_ft[0]:.4f}, t={t_ft[0]:.2f}")

    # --- Spec 6: Two-way FE ---
    y_2w = demean_by(reg_df.assign(y=y_dm), "y", "quarter").values
    gap_2w = demean_by(reg_df.assign(g=gap_dm), "g", "quarter").values
    int_2w = demean_by(reg_df.assign(i=int_dm), "i", "quarter").values
    X_2w = np.column_stack([gap_2w, int_2w])
    beta_2w, se_2w, t_2w = ols_with_se(y_2w, X_2w, cluster_ids=reg_df["actor_num"].values)
    results.append({
        "spec": "Two-way FE (actor-clustered)",
        "beta_gap": beta_2w[0], "se_gap": se_2w[0], "t_gap": t_2w[0],
        "beta_level": beta_2w[1], "se_level": se_2w[1], "t_level": t_2w[1],
        "n": n,
    })
    print(f"  Two-way FE:        beta_gap={beta_2w[0]:.4f}, t={t_2w[0]:.2f}")

    # --- Summary ---
    rdf = pd.DataFrame(results)
    print(f"\n  Summary:")
    print(rdf[["spec", "beta_gap", "t_gap", "n"]].to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(RESULTS_DIR / "ws_e_capex_regression.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'ws_e_capex_regression.csv'}")


if __name__ == "__main__":
    main()
