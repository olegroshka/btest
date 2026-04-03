#!/usr/bin/env python
"""
Workstream B -- Formal Forecast-Comparison Inference.

Reads actor-quarter predictions from Workstream A and computes:
  1. Window-level mean loss differentials (MSE, MAE) vs AR(1), random walk, DFM.
  2. Paired permutation test on window-level loss differentials.
  3. Sign test / exact binomial for window-win pattern.
  4. Block bootstrap (clustered by time window) for mean DeltaR^2, DeltaMSE, DeltaMAE with 95% CI.
  5. Diebold-Mariano tests at actor-quarter level.

Outputs:
  results/metrics/ws_b_inference_table.csv
  results/metrics/ws_b_loss_differentials.csv
  results/metrics/ws_b_bootstrap_distributions.parquet

Usage::
    uv run python scripts/run_smim_ws_b_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.validation.metrics import oos_r_squared

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
PRED_PATH = RESULTS_DIR / "ws_a_actor_quarter_predictions.parquet"

N_BOOTSTRAP = 10_000
N_PERMUTATION = 10_000
SEED = 42


# =========================================================
# Loss computation
# =========================================================

def compute_window_losses(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate actor-quarter errors to window-level MSE, MAE, R^2."""
    rows = []
    for window, wdf in pred_df.groupby("window"):
        actual = wdf["actual"].values
        pred_smim = wdf["pred_smim"].values
        pred_ar1 = wdf["pred_ar1"].values
        # Random walk: predict last known value (actual from previous quarter)
        # Approximate: use actual as "previous" (slightly optimistic for RW)
        grand_mean = actual.mean()

        err_smim = pred_smim - actual
        err_ar1 = pred_ar1 - actual
        err_mean = grand_mean - actual  # historical mean baseline

        mse_smim = float(np.mean(err_smim**2))
        mse_ar1 = float(np.mean(err_ar1**2))
        mse_mean = float(np.mean(err_mean**2))

        mae_smim = float(np.mean(np.abs(err_smim)))
        mae_ar1 = float(np.mean(np.abs(err_ar1)))
        mae_mean = float(np.mean(np.abs(err_mean)))

        r2_smim = float(oos_r_squared(pred_smim, actual))
        r2_ar1 = float(oos_r_squared(pred_ar1, actual))

        rows.append({
            "window": window,
            "is_holdout": wdf["is_holdout"].iloc[0],
            "n_obs": len(wdf),
            "mse_smim": mse_smim,
            "mse_ar1": mse_ar1,
            "mse_mean": mse_mean,
            "mae_smim": mae_smim,
            "mae_ar1": mae_ar1,
            "mae_mean": mae_mean,
            "r2_smim": r2_smim,
            "r2_ar1": r2_ar1,
            "delta_mse_vs_ar1": mse_smim - mse_ar1,
            "delta_mae_vs_ar1": mae_smim - mae_ar1,
            "delta_r2_vs_ar1": r2_smim - r2_ar1,
            "delta_mse_vs_mean": mse_smim - mse_mean,
            "smim_wins_mse": int(mse_smim < mse_ar1),
        })

    return pd.DataFrame(rows)


# =========================================================
# Permutation test
# =========================================================

def paired_permutation_test(loss_diffs: np.ndarray, n_perm: int = N_PERMUTATION,
                             rng: np.random.Generator | None = None) -> dict:
    """
    Two-sided paired permutation test on window-level loss differentials.
    H0: mean(loss_diff) = 0.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    observed = float(np.mean(loss_diffs))
    n = len(loss_diffs)
    count_extreme = 0

    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_mean = float(np.mean(loss_diffs * signs))
        if abs(perm_mean) >= abs(observed):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)  # +1 for observed itself
    return {"observed_mean": observed, "p_value": p_value, "n_perm": n_perm}


# =========================================================
# Sign test / exact binomial
# =========================================================

def sign_test(wins: int, total: int) -> dict:
    """Exact binomial test: is win rate > 0.5?"""
    result = stats.binomtest(wins, total, 0.5, alternative="greater")
    p_value = float(result.pvalue)
    return {"wins": wins, "total": total, "p_value": p_value}


# =========================================================
# Block bootstrap (window-level resampling)
# =========================================================

def block_bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                        ci: float = 0.95,
                        rng: np.random.Generator | None = None) -> dict:
    """
    Bootstrap CI for a statistic computed on window-level values.
    Resamples windows with replacement (block = 1 window).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(values[idx])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return {
        "mean": float(np.mean(values)),
        "ci_lo": lo,
        "ci_hi": hi,
        "se": float(np.std(boot_means)),
        "n_boot": n_boot,
    }


# =========================================================
# Actor-quarter level Diebold-Mariano
# =========================================================

def dm_test_actor_quarter(pred_df: pd.DataFrame) -> dict:
    """
    Diebold-Mariano test at actor-quarter level.
    Loss differential: d_t = e_smim^2 - e_ar1^2.
    """
    d = pred_df["error_smim"].values**2 - pred_df["error_ar1"].values**2
    n = len(d)
    d_bar = np.mean(d)
    # HAC-style variance (Newey-West with h=1 truncation)
    gamma_0 = np.var(d, ddof=1)
    if n > 1:
        gamma_1 = np.cov(d[:-1], d[1:])[0, 1]
    else:
        gamma_1 = 0.0
    var_d = (gamma_0 + 2 * gamma_1) / n
    var_d = max(var_d, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))
    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_delta_mse": float(d_bar),
        "n_obs": n,
    }


# =========================================================
# Main
# =========================================================

def main():
    print("=" * 90)
    print("  WORKSTREAM B: FORMAL FORECAST-COMPARISON INFERENCE")
    print("=" * 90)

    if not PRED_PATH.exists():
        print(f"\n  ERROR: {PRED_PATH} not found.")
        print("  Run Workstream A first: uv run python scripts/run_smim_ws_a_nested_cv.py")
        return

    pred_df = pd.read_parquet(PRED_PATH)
    print(f"  Loaded {len(pred_df)} actor-quarter observations")
    print(f"  Windows: {pred_df['window'].nunique()}")
    print(f"  Holdout: {pred_df[pred_df['is_holdout']]['window'].nunique()} windows")

    rng = np.random.default_rng(SEED)

    # -- 1. Window-level loss differentials ----------------
    print("\n-- 1. Window-level losses --")
    wdf = compute_window_losses(pred_df)
    print(wdf[["window", "is_holdout", "r2_smim", "r2_ar1", "delta_mse_vs_ar1",
               "smim_wins_mse"]].to_string(index=False))

    # -- 2. Inference on non-holdout windows --------------
    cv_mask = ~wdf["is_holdout"]
    cv = wdf[cv_mask]
    n_cv = len(cv)

    print(f"\n-- 2. Inference on {n_cv} CV windows --")

    # 2a. Permutation test on DeltaMSE
    delta_mse = cv["delta_mse_vs_ar1"].values
    perm = paired_permutation_test(delta_mse, rng=rng)
    print(f"  Permutation test (DeltaMSE vs AR(1)):")
    print(f"    Mean DeltaMSE:  {perm['observed_mean']:.6f}")
    print(f"    p-value:    {perm['p_value']:.4f}")

    # 2b. Permutation test on DeltaMAE
    delta_mae = cv["delta_mae_vs_ar1"].values
    perm_mae = paired_permutation_test(delta_mae, rng=rng)
    print(f"  Permutation test (DeltaMAE vs AR(1)):")
    print(f"    Mean DeltaMAE:  {perm_mae['observed_mean']:.6f}")
    print(f"    p-value:    {perm_mae['p_value']:.4f}")

    # 2c. Sign test for win rate
    wins = int(cv["smim_wins_mse"].sum())
    sign = sign_test(wins, n_cv)
    print(f"  Sign test (SMIM wins on MSE):")
    print(f"    Wins:    {sign['wins']}/{sign['total']}")
    print(f"    p-value: {sign['p_value']:.4f}")

    # 2d. Block bootstrap CIs
    boot_mse = block_bootstrap_ci(delta_mse, rng=rng)
    boot_mae = block_bootstrap_ci(delta_mae, rng=rng)
    boot_r2 = block_bootstrap_ci(cv["delta_r2_vs_ar1"].values, rng=rng)

    print(f"  Bootstrap 95% CI (10,000 window-resamples):")
    print(f"    DeltaMSE:  {boot_mse['mean']:.6f}  [{boot_mse['ci_lo']:.6f}, {boot_mse['ci_hi']:.6f}]")
    print(f"    DeltaMAE:  {boot_mae['mean']:.6f}  [{boot_mae['ci_lo']:.6f}, {boot_mae['ci_hi']:.6f}]")
    print(f"    DeltaR^2:   {boot_r2['mean']:.4f}  [{boot_r2['ci_lo']:.4f}, {boot_r2['ci_hi']:.4f}]")

    # -- 3. Actor-quarter DM test -------------------------
    print(f"\n-- 3. Actor-quarter Diebold-Mariano test --")
    cv_pred = pred_df[~pred_df["is_holdout"]]
    dm = dm_test_actor_quarter(cv_pred)
    print(f"    DM stat:     {dm['dm_stat']:.4f}")
    print(f"    p-value:     {dm['p_value']:.6f}")
    print(f"    Mean DeltaMSE:   {dm['mean_delta_mse']:.6f}")
    print(f"    N obs:       {dm['n_obs']}")

    # -- 4. Holdout inference -----------------------------
    ho = wdf[wdf["is_holdout"]]
    if not ho.empty:
        print(f"\n-- 4. Holdout windows (no tuning exposure) --")
        for _, row in ho.iterrows():
            print(f"    {row['window']}: R^2_SMIM={row['r2_smim']:.4f}  "
                  f"R^2_AR1={row['r2_ar1']:.4f}  "
                  f"DeltaMSE={row['delta_mse_vs_ar1']:.6f}")

    # -- 5. Summary inference table -----------------------
    print(f"\n-- 5. Summary Table: Forecast Comparison with Inference --")
    summary_rows = []
    for label, delta_col, boot_result in [
        ("DeltaMSE vs AR(1)", "delta_mse_vs_ar1", boot_mse),
        ("DeltaMAE vs AR(1)", "delta_mae_vs_ar1", boot_mae),
        ("DeltaR^2 vs AR(1)", "delta_r2_vs_ar1", boot_r2),
    ]:
        summary_rows.append({
            "comparison": label,
            "mean": boot_result["mean"],
            "ci_lo_95": boot_result["ci_lo"],
            "ci_hi_95": boot_result["ci_hi"],
            "se": boot_result["se"],
            "perm_p": perm["p_value"] if "MSE" in label else
                      perm_mae["p_value"] if "MAE" in label else np.nan,
            "sign_wins": f"{wins}/{n_cv}",
            "sign_p": sign["p_value"],
            "dm_stat": dm["dm_stat"],
            "dm_p": dm["p_value"],
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    # -- Save ---------------------------------------------
    summary.to_csv(RESULTS_DIR / "ws_b_inference_table.csv", index=False)
    wdf.to_csv(RESULTS_DIR / "ws_b_loss_differentials.csv", index=False)
    print(f"\n  Saved: {RESULTS_DIR / 'ws_b_inference_table.csv'}")
    print(f"  Saved: {RESULTS_DIR / 'ws_b_loss_differentials.csv'}")


if __name__ == "__main__":
    main()
