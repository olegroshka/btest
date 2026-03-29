#!/usr/bin/env python
"""
B2-SPECTRAL-METHODS experiment runner.

Compare 7 spectral decomposition methods on the same A1 universe/windows.
Tests H2a (directed > symmetric) and H2b (polar/dilation > Schur on
ill-conditioned operators).

Uses L1 depth (graph-factor OLS from B1 finding) for fair comparison:
the decomposition method IS the variable, not the downstream filter.

Output:
  results/metrics/level1_B2-SPECTRAL-METHODS.parquet  (7 methods x 10 windows)
  results/configs/B2-SPECTRAL-METHODS.yaml

Usage::
    uv run python scripts/run_smim_b2.py
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

from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.spectral.polar import PolarDecomposer
from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer
from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer, ExtendedDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# --- constants ---------------------------------------------------------------

EXP_ID = "B2-SPECTRAL-METHODS"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"

K_MIN = 3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"

# Method registry: name -> decomposer instance
METHODS = {
    "schur": SchurDecomposer(),
    "polar": PolarDecomposer(),
    "hermitian_dilation": HermitianDilationDecomposer(),
    "directed_variation": DirectedVariationDecomposer(),
    "dmd": ExactDMDDecomposer(),
    "extended_dmd": ExtendedDMDDecomposer(),
}


# --- data + operator (reuse B1 pattern) --------------------------------------


def load_data() -> tuple[pd.DataFrame, list[str]]:
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], actor_ids


def make_windows() -> list[dict]:
    return [
        {"window_id": f"W{ty}",
         "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
         "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
         "test_start": pd.Timestamp(f"{ty}-01-01"),
         "test_end": pd.Timestamp(f"{ty}-12-31")}
        for ty in TEST_YEARS
    ]


def build_operator(obs_train: np.ndarray, N: int) -> np.ndarray:
    """Intensity cross-correlation operator (simple, no optimisation)."""
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0
    return corr


def pca_decompose(operator: np.ndarray, k: int) -> ModalFrame:
    """Symmetric baseline: PCA of symmetrised operator."""
    sym = (operator + operator.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    # Top-k by magnitude
    idx = np.argsort(np.abs(eigvals))[::-1][:k]
    return ModalFrame(
        basis=eigvecs[:, idx],
        eigenvalues=eigvals[idx].astype(complex),
        method="pca",
    )


def evaluate_method(
    method_name: str,
    operator: np.ndarray,
    obs_train: np.ndarray,
    obs_test: np.ndarray,
    obs_test_raw: np.ndarray,
    obs_mean: np.ndarray,
    K: int,
) -> dict[str, Any]:
    """Run one method: decompose -> project -> AR(1) forecast -> R^2."""
    N = operator.shape[0]

    # Decompose
    if method_name == "pca":
        mf = pca_decompose(operator, k=min(K_MAX, N))
    elif method_name in ("dmd", "extended_dmd"):
        # DMD needs snapshot data, not operator
        decomposer = METHODS[method_name]
        mf = decomposer.decompose_snapshots(obs_train.T, k=min(K_MAX, N))
    else:
        decomposer = METHODS[method_name]
        mf = decomposer.decompose(operator, k=min(K_MAX, N))

    # Take top-K modes
    k_actual = min(K, mf.K)
    U = mf.basis[:, :k_actual].real  # ensure real for AR(1)

    # Condition number of operator
    try:
        cond = float(np.linalg.cond(operator))
    except Exception:
        cond = float("nan")

    # L1-style evaluation: graph factors + AR(1)
    factors_train = obs_train @ U  # (T_train, K)
    F_lag = factors_train[:-1]
    F_curr = factors_train[1:]
    try:
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(k_actual) * 0.9

    f_last = factors_train[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next

    pred_restored = (pred + obs_mean).T.ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_restored, actual))

    return {
        "oos_r2": r2,
        "k_actual": k_actual,
        "cond_number": cond,
    }


# --- main --------------------------------------------------------------------


def run_b2(verbose: bool = False, single_window: int | None = None) -> None:
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  7 spectral methods x 10 windows  (L1 depth: graph-factor OLS)")
    print("=" * 70)

    print("\n  Loading data ...", end="", flush=True)
    intensities_wide, actor_ids = load_data()
    print(f" done. {len(actor_ids)} actors")

    windows = [make_windows()[single_window]] if single_window is not None else make_windows()
    all_methods = list(METHODS.keys()) + ["pca"]

    l1_rows: list[dict] = []

    for wi, window in enumerate(windows):
        wid = window["window_id"]
        ts, te = window["train_start"], window["train_end"]
        test_s, test_e = window["test_start"], window["test_end"]

        int_train = intensities_wide[(intensities_wide.index >= ts) & (intensities_wide.index <= te)]
        int_test = intensities_wide[(intensities_wide.index >= test_s) & (intensities_wide.index <= test_e)]
        valid = int_train.columns[int_train.notna().any()]
        int_train, int_test = int_train[valid], int_test[valid]

        col_means = int_train.mean()
        obs_train_raw = int_train.fillna(col_means).values.astype(np.float64)
        obs_test_raw = int_test.fillna(col_means).values.astype(np.float64)
        obs_mean = obs_train_raw.mean(axis=0, keepdims=True)
        obs_train = obs_train_raw - obs_mean
        obs_test = obs_test_raw - obs_mean
        N = obs_train.shape[1]

        operator = build_operator(obs_train, N)

        print(f"\n  [{wi + 1}/{len(windows)}] {wid}  N={N}  cond={np.linalg.cond(operator):.0f}")

        for method in all_methods:
            try:
                res = evaluate_method(method, operator, obs_train, obs_test, obs_test_raw, obs_mean, K_MIN)
                r2 = res["oos_r2"]
                print(f"    {method:<24} R2={r2:.4f}  K={res['k_actual']}")
            except Exception as exc:
                r2 = float("nan")
                res = {"k_actual": 0, "cond_number": float("nan")}
                print(f"    {method:<24} FAILED: {exc}")

            l1_rows.append({
                "experiment_id": EXP_ID,
                "window_id": wid,
                "method": method,
                "oos_r2": r2,
                "k_actual": res["k_actual"],
                "cond_number": res.get("cond_number", float("nan")),
                "n_actors": N,
            })

    # --- Summary -------------------------------------------------------------
    df = pd.DataFrame(l1_rows)
    print("\n" + "-" * 70)
    print("  SPECTRAL METHOD RANKING (mean across windows)")
    print(f"  {'Method':<28} {'Mean R2':>10} {'Std R2':>10} {'Rank':>6}")

    summary = df.groupby("method").agg(
        mean_r2=("oos_r2", "mean"),
        std_r2=("oos_r2", "std"),
    ).sort_values("mean_r2", ascending=False)

    for rank, (method, row) in enumerate(summary.iterrows(), 1):
        marker = " <-- WINNER" if rank == 1 else ""
        pca_marker = " (symmetric baseline)" if method == "pca" else ""
        print(f"  {method:<28} {row['mean_r2']:>10.4f} {row['std_r2']:>10.4f} {rank:>6}{marker}{pca_marker}")

    # H2a test
    pca_r2 = summary.loc["pca", "mean_r2"] if "pca" in summary.index else float("nan")
    directed_methods = [m for m in summary.index if m != "pca"]
    best_directed = summary.loc[directed_methods].iloc[0] if directed_methods else None
    if best_directed is not None:
        delta_h2a = best_directed["mean_r2"] - pca_r2
        print(f"\n  H2a: best directed ({directed_methods[0]}) vs pca: "
              f"{best_directed['mean_r2']:.4f} vs {pca_r2:.4f}  delta={delta_h2a:+.4f}")
        print(f"  H2a verdict: {'SUPPORTED' if delta_h2a > 0.005 else 'NOT SUPPORTED'}")

    print("=" * 70)

    # --- Save ----------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    l1_path = METRICS_DIR / f"level1_{EXP_ID}.parquet"
    df.to_parquet(l1_path, index=False)
    print(f"\n  Wrote: {l1_path.relative_to(PROJECT_ROOT)}")

    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": all_methods,
        "n_windows": len(windows),
        "K": K_MIN,
        "winner": summary.index[0] if len(summary) > 0 else "unknown",
    }
    yaml_path = CONFIGS_DIR / f"{EXP_ID}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B2-SPECTRAL-METHODS runner")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--single-window", type=int, default=None)
    args = parser.parse_args()
    run_b2(verbose=args.verbose, single_window=args.single_window)


if __name__ == "__main__":
    main()
