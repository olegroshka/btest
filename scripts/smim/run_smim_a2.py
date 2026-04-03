#!/usr/bin/env python
"""
A2-BASELINES experiment runner.

Produces baseline performance numbers for 8 naive forecasting models on the
same experiment_a1 universe and FULL-ROLL windows as A1. These baselines are
the denominators for all future delta-R^2 claims.

Models:
  1. historical_mean           per-actor trailing mean
  2. random_walk               last observation carried forward
  3. sector_mean               cross-sectional sector average
  4. ar1_per_actor             per-actor AR(1)
  5. dynamic_factor_model_k5   DFM with K=5 factors (PCA + VAR(1))
  6. dynamic_factor_model_k10  DFM with K=10 factors
  7. var_bic                   VAR with BIC-selected lag (on top-5 PCA factors)
  8. symmetric_laplacian_spectral  spectral with symmetric Laplacian

Output:
  results/metrics/level1_A2-BASELINES.parquet   (8 models x 10 windows)
  results/configs/A2-BASELINES.yaml

Usage::
    uv run python scripts/smim/run_smim_a2.py
    uv run python scripts/smim/run_smim_a2.py --single-window 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# --- constants ---------------------------------------------------------------

EXP_ID = "A2-BASELINES"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"

TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


# --- data loading (reuse A1 pattern) ----------------------------------------


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load intensities and actor-sector mapping.

    Returns (intensities_wide, actor_sector_map).
    """
    with open(REGISTRY_PATH) as fh:
        reg_data = json.load(fh)

    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors_with_int = set(int_df["actor_id"].unique())
    actor_data = [a for a in reg_data["actors"] if a["actor_id"] in actors_with_int]
    actor_ids = [a["actor_id"] for a in actor_data]
    actor_sector = {a["actor_id"]: a.get("sector", "unknown") for a in actor_data}

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], actor_sector


def make_windows(test_years: list[int] | None = None) -> list[dict[str, Any]]:
    years = test_years or TEST_YEARS
    return [
        {
            "window_id": f"W{ty}",
            "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
            "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
            "test_start": pd.Timestamp(f"{ty}-01-01"),
            "test_end": pd.Timestamp(f"{ty}-12-31"),
        }
        for ty in years
    ]


# --- baseline models ---------------------------------------------------------


def model_historical_mean(obs_train: np.ndarray, obs_test: np.ndarray) -> np.ndarray:
    """Per-actor trailing mean from training period."""
    means = obs_train.mean(axis=0, keepdims=True)  # (1, N)
    return np.tile(means, (obs_test.shape[0], 1))  # (T_test, N)


def model_random_walk(obs_train: np.ndarray, obs_test: np.ndarray) -> np.ndarray:
    """Last observation carried forward."""
    last = obs_train[-1:, :]  # (1, N)
    return np.tile(last, (obs_test.shape[0], 1))


def model_sector_mean(
    obs_train: np.ndarray,
    obs_test: np.ndarray,
    actor_ids: list[str],
    actor_sector: dict[str, str],
) -> np.ndarray:
    """Cross-sectional sector average from training period."""
    sectors = [actor_sector.get(a, "unknown") for a in actor_ids]
    unique_sectors = list(set(sectors))
    pred = np.zeros_like(obs_test)
    for sec in unique_sectors:
        mask = np.array([s == sec for s in sectors])
        if mask.sum() == 0:
            continue
        sec_mean = obs_train[:, mask].mean(axis=1).mean()  # scalar
        pred[:, mask] = sec_mean
    return pred


def model_ar1(obs_train: np.ndarray, obs_test: np.ndarray) -> np.ndarray:
    """Per-actor AR(1): y_{t+1} = a + b * y_t."""
    T, N = obs_train.shape
    pred = np.zeros_like(obs_test)
    for i in range(N):
        y = obs_train[:, i]
        y_lag = y[:-1]
        y_curr = y[1:]
        if np.std(y_lag) < 1e-10:
            pred[:, i] = y.mean()
            continue
        # OLS: y_curr = a + b * y_lag
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta = np.linalg.lstsq(X, y_curr, rcond=None)[0]
        except np.linalg.LinAlgError:
            pred[:, i] = y.mean()
            continue
        # Multi-step forecast
        last_val = obs_train[-1, i]
        for t in range(obs_test.shape[0]):
            pred[t, i] = beta[0] + beta[1] * last_val
            last_val = pred[t, i]
    return pred


def model_dfm(obs_train: np.ndarray, obs_test: np.ndarray, K: int) -> np.ndarray:
    """Dynamic Factor Model: PCA(K) + VAR(1) on factors."""
    T, N = obs_train.shape
    # Demean
    mu = obs_train.mean(axis=0)
    X = obs_train - mu

    # PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    K_actual = min(K, len(S), T - 1)
    loadings = Vt[:K_actual].T  # (N, K)
    factors = X @ loadings  # (T, K)

    # VAR(1) on factors: F_{t+1} = A @ F_t + noise
    F_lag = factors[:-1]
    F_curr = factors[1:]
    try:
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0]  # (K, K)
    except np.linalg.LinAlgError:
        return np.tile(mu, (obs_test.shape[0], 1))

    # Forecast
    pred = np.zeros_like(obs_test)
    f_last = factors[-1]
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ loadings.T + mu
        f_last = f_next
    return pred


def model_var_bic(obs_train: np.ndarray, obs_test: np.ndarray) -> np.ndarray:
    """VAR on top-5 PCA factors with BIC lag selection (1-4)."""
    T, N = obs_train.shape
    mu = obs_train.mean(axis=0)
    X = obs_train - mu

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    K = min(5, len(S), T - 2)
    loadings = Vt[:K].T
    factors = X @ loadings  # (T, K)

    # BIC lag selection
    best_lag, best_bic = 1, np.inf
    for p in range(1, min(5, T // 3)):
        Y = factors[p:]
        Xr = np.hstack([factors[p - j - 1 : T - j - 1] for j in range(p)])
        try:
            B = np.linalg.lstsq(Xr, Y, rcond=None)[0]
            resid = Y - Xr @ B
            sse = np.sum(resid ** 2)
            n_params = p * K * K
            bic = len(Y) * np.log(sse / len(Y) + 1e-10) + n_params * np.log(len(Y))
            if bic < best_bic:
                best_bic = bic
                best_lag = p
        except np.linalg.LinAlgError:
            continue

    # Fit VAR(best_lag)
    p = best_lag
    Y = factors[p:]
    Xr = np.hstack([factors[p - j - 1 : T - j - 1] for j in range(p)])
    try:
        B = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.tile(mu, (obs_test.shape[0], 1))

    # Forecast
    pred = np.zeros_like(obs_test)
    history = list(factors[-(p):])
    for t in range(obs_test.shape[0]):
        xr = np.concatenate([history[-(j + 1)] for j in range(p)])
        f_next = xr @ B
        pred[t] = f_next @ loadings.T + mu
        history.append(f_next)
    return pred


def model_symmetric_laplacian(
    obs_train: np.ndarray, obs_test: np.ndarray
) -> np.ndarray:
    """Spectral with symmetric Laplacian: L_sym = D^{-1/2} W D^{-1/2}, K=3."""
    T, N = obs_train.shape
    mu = obs_train.mean(axis=0)
    X = obs_train - mu

    # Symmetric correlation as weight matrix
    W = np.corrcoef(X.T)
    W = np.nan_to_num(W, nan=0.0)
    np.fill_diagonal(W, 0.0)
    W = np.abs(W)  # symmetric, non-negative
    W[W < 0.1] = 0.0

    # Symmetric normalized Laplacian eigenvectors
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Take smallest non-trivial eigenvectors (= smoothest graph signals)
    K = min(3, N - 1)
    U = eigvecs[:, 1 : K + 1]  # skip first (constant) eigenvector

    # Project + Kalman-like filter (simple: project, AR(1) on factors)
    factors = X @ U  # (T, K)
    F_lag = factors[:-1]
    F_curr = factors[1:]
    try:
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.tile(mu, (obs_test.shape[0], 1))

    pred = np.zeros_like(obs_test)
    f_last = factors[-1]
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T + mu
        f_last = f_next
    return pred


# --- main --------------------------------------------------------------------


MODELS = {
    "historical_mean": lambda tr, te, **kw: model_historical_mean(tr, te),
    "random_walk": lambda tr, te, **kw: model_random_walk(tr, te),
    "sector_mean": lambda tr, te, **kw: model_sector_mean(
        tr, te, kw["actor_ids"], kw["actor_sector"]
    ),
    "ar1_per_actor": lambda tr, te, **kw: model_ar1(tr, te),
    "dfm_k5": lambda tr, te, **kw: model_dfm(tr, te, K=5),
    "dfm_k10": lambda tr, te, **kw: model_dfm(tr, te, K=10),
    "var_bic": lambda tr, te, **kw: model_var_bic(tr, te),
    "sym_laplacian_k3": lambda tr, te, **kw: model_symmetric_laplacian(tr, te),
}


def run_a2(
    verbose: bool = False,
    single_window: int | None = None,
) -> None:
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  8 baseline models  10 windows  same universe as A1")
    print("=" * 70)

    print("\n  Loading data ...", end="", flush=True)
    intensities_wide, actor_sector = load_data()
    print(f" done. {len(intensities_wide.columns)} actors")

    if single_window is not None:
        windows = [make_windows()[single_window]]
    else:
        windows = make_windows()

    l1_rows: list[dict[str, Any]] = []

    for wi, window in enumerate(windows):
        wid = window["window_id"]
        ts, te = window["train_start"], window["train_end"]
        test_s, test_e = window["test_start"], window["test_end"]

        int_train = intensities_wide[(intensities_wide.index >= ts) & (intensities_wide.index <= te)]
        int_test = intensities_wide[(intensities_wide.index >= test_s) & (intensities_wide.index <= test_e)]
        valid_cols = int_train.columns[int_train.notna().any()]
        int_train = int_train[valid_cols]
        int_test = int_test[valid_cols]
        actor_ids_final = list(valid_cols)

        col_means = int_train.mean()
        obs_train = int_train.fillna(col_means).values.astype(np.float64)
        obs_test = int_test.fillna(col_means).values.astype(np.float64)

        N = obs_train.shape[1]
        T_test = obs_test.shape[0]
        actual = obs_test.T.ravel()

        print(f"\n  [{wi + 1}/{len(windows)}] {wid}  N={N}  ", end="")

        for model_name, model_fn in MODELS.items():
            try:
                pred = model_fn(
                    obs_train, obs_test,
                    actor_ids=actor_ids_final,
                    actor_sector=actor_sector,
                )
                pred_flat = pred.T.ravel()
                r2 = float(oos_r_squared(pred_flat, actual))
                mae = float(np.mean(np.abs(actual - pred_flat)))
                coverage = float(np.mean(np.isfinite(pred_flat)))
            except Exception as exc:
                r2 = float("nan")
                mae = float("nan")
                coverage = 0.0
                if verbose:
                    print(f"\n    [WARN] {model_name}: {exc}")

            l1_rows.append({
                "experiment_id": EXP_ID,
                "window_id": wid,
                "train_start": ts,
                "train_end": te,
                "test_start": test_s,
                "test_end": test_e,
                "model": model_name,
                "oos_r2": r2,
                "mae": mae,
                "coverage": coverage,
                "n_actors": N,
                "n_test_periods": T_test,
            })

        # Print summary for this window
        window_models = [r for r in l1_rows if r["window_id"] == wid]
        best = max(window_models, key=lambda r: r["oos_r2"] if np.isfinite(r["oos_r2"]) else -999)
        rw_r2 = next((r["oos_r2"] for r in window_models if r["model"] == "random_walk"), float("nan"))
        print(f"RW={rw_r2:.4f}  best={best['model']}({best['oos_r2']:.4f})")

    # --- Summary table -------------------------------------------------------
    print("\n" + "-" * 70)
    print("  BASELINE SUMMARY (mean across 10 windows)")
    print(f"  {'Model':<28} {'Mean R2':>10} {'Std R2':>10} {'Mean MAE':>10} {'Rank':>6}")

    df = pd.DataFrame(l1_rows)
    summary = df.groupby("model").agg(
        mean_r2=("oos_r2", "mean"),
        std_r2=("oos_r2", "std"),
        mean_mae=("mae", "mean"),
    ).sort_values("mean_r2", ascending=False)

    for rank, (model, row) in enumerate(summary.iterrows(), 1):
        print(f"  {model:<28} {row['mean_r2']:>10.4f} {row['std_r2']:>10.4f} "
              f"{row['mean_mae']:>10.4f} {rank:>6}")

    rw_mean = summary.loc["random_walk", "mean_r2"] if "random_walk" in summary.index else 0.0
    print(f"\n  Random walk baseline R2 = {rw_mean:.4f}")
    print(f"  (This is the primary denominator for all future delta-R2 claims)")

    # --- Save ----------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    l1_path = METRICS_DIR / f"level1_{EXP_ID}.parquet"
    df.to_parquet(l1_path, index=False)
    print(f"\n  Wrote: {l1_path.relative_to(PROJECT_ROOT)}")

    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_models": len(MODELS),
        "n_windows": len(windows),
        "models": list(MODELS.keys()),
        "summary": {
            model: {"mean_r2": round(float(row["mean_r2"]), 6), "std_r2": round(float(row["std_r2"]), 6)}
            for model, row in summary.iterrows()
        },
    }
    yaml_path = CONFIGS_DIR / f"{EXP_ID}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-BASELINES experiment runner")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--single-window", type=int, default=None)
    args = parser.parse_args()
    run_a2(verbose=args.verbose, single_window=args.single_window)


if __name__ == "__main__":
    main()
