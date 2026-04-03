#!/usr/bin/env python
"""
B8-NOISE + B9-EDGE-DEGRADE combined runner.

B8: Add Gaussian noise to intensity observations before building operator.
B9: Randomly flip edge directions in the operator.

Both test robustness at L1 depth (graph-factor OLS).

Output:
  results/metrics/level1_B8-NOISE.parquet
  results/metrics/level1_B9-EDGE-DEGRADE.parquet
  results/configs/B8-NOISE.yaml
  results/configs/B9-EDGE-DEGRADE.yaml

Usage::
    uv run python scripts/smim/run_smim_b8_b9.py
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

from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"

K_MIN = 3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
EDGE_CORRUPT_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_data():
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
    return wide[cols]


def evaluate_l1(obs_train, obs_test, obs_test_raw, obs_mean, K):
    N = obs_train.shape[1]
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0

    mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    k_act = min(K, mf.K)
    U = mf.basis[:, :k_act].real

    factors = obs_train @ U
    try:
        A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(k_act) * 0.9

    f_last = factors[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next

    actual = obs_test_raw.T.ravel()
    return float(oos_r_squared((pred + obs_mean).T.ravel(), actual))


def evaluate_with_corrupted_operator(obs_train, obs_test, obs_test_raw, obs_mean, K, corrupt_frac, rng):
    N = obs_train.shape[1]
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0

    # Corrupt edges: randomly flip direction
    if corrupt_frac > 0:
        mask = rng.random((N, N)) < corrupt_frac
        # For each selected edge, swap (i,j) and (j,i) values
        corr_corrupted = corr.copy()
        swap = corr[mask]
        corr_corrupted[mask] = corr.T[mask]  # swap direction
        corr = corr_corrupted

    mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    k_act = min(K, mf.K)
    U = mf.basis[:, :k_act].real

    factors = obs_train @ U
    try:
        A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(k_act) * 0.9

    f_last = factors[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next

    actual = obs_test_raw.T.ravel()
    return float(oos_r_squared((pred + obs_mean).T.ravel(), actual))


def make_windows():
    return [
        {"window_id": f"W{ty}",
         "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
         "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
         "test_start": pd.Timestamp(f"{ty}-01-01"),
         "test_end": pd.Timestamp(f"{ty}-12-31")}
        for ty in TEST_YEARS
    ]


def main():
    print("=" * 70)
    print(f"  B8-NOISE + B9-EDGE-DEGRADE  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    intensities = load_data()
    windows = make_windows()
    rng = np.random.default_rng(42)

    # ── B8: NOISE INJECTION ──────────────────────────────────────────────────
    print("\n  === B8: NOISE INJECTION ===")
    b8_rows = []

    for wi, w in enumerate(windows):
        ts, te = w["train_start"], w["train_end"]
        test_s, test_e = w["test_start"], w["test_end"]

        int_train = intensities[(intensities.index >= ts) & (intensities.index <= te)]
        int_test = intensities[(intensities.index >= test_s) & (intensities.index <= test_e)]
        valid = int_train.columns[int_train.notna().any()]
        int_train, int_test = int_train[valid], int_test[valid]

        col_means = int_train.mean()
        obs_train_raw = int_train.fillna(col_means).values.astype(np.float64)
        obs_test_raw = int_test.fillna(col_means).values.astype(np.float64)
        obs_mean = obs_train_raw.mean(axis=0, keepdims=True)
        obs_train_clean = obs_train_raw - obs_mean
        obs_test = obs_test_raw - obs_mean

        # Per-actor signal std
        actor_std = np.std(obs_train_clean, axis=0, keepdims=True)
        actor_std = np.maximum(actor_std, 1e-10)

        if wi == 0:
            print(f"  Window {w['window_id']}  noise levels: {NOISE_LEVELS}")

        for sigma in NOISE_LEVELS:
            noise = rng.standard_normal(obs_train_clean.shape) * actor_std * sigma
            obs_noisy = obs_train_clean + noise
            try:
                r2 = evaluate_l1(obs_noisy, obs_test, obs_test_raw, obs_mean, K_MIN)
            except Exception:
                r2 = float("nan")

            b8_rows.append({
                "experiment_id": "B8-NOISE",
                "window_id": w["window_id"],
                "noise_sigma": sigma,
                "oos_r2": r2,
            })

    df_b8 = pd.DataFrame(b8_rows)
    print("\n  B8 NOISE DEGRADATION (mean across windows)")
    print(f"  {'sigma':>8} {'Mean R2':>10} {'Retention':>10}")
    summary_b8 = df_b8.groupby("noise_sigma")["oos_r2"].mean()
    r2_clean = summary_b8.get(0.0, 0)
    for sigma, r2 in summary_b8.items():
        ret = r2 / r2_clean * 100 if r2_clean > 0 else 0
        print(f"  {sigma:>8.1f} {r2:>10.4f} {ret:>9.1f}%")

    # ── B9: EDGE DEGRADATION ─────────────────────────────────────────────────
    print("\n  === B9: EDGE DEGRADATION ===")
    b9_rows = []

    for wi, w in enumerate(windows):
        ts, te = w["train_start"], w["train_end"]
        test_s, test_e = w["test_start"], w["test_end"]

        int_train = intensities[(intensities.index >= ts) & (intensities.index <= te)]
        int_test = intensities[(intensities.index >= test_s) & (intensities.index <= test_e)]
        valid = int_train.columns[int_train.notna().any()]
        int_train, int_test = int_train[valid], int_test[valid]

        col_means = int_train.mean()
        obs_train_raw = int_train.fillna(col_means).values.astype(np.float64)
        obs_test_raw = int_test.fillna(col_means).values.astype(np.float64)
        obs_mean = obs_train_raw.mean(axis=0, keepdims=True)
        obs_train = obs_train_raw - obs_mean
        obs_test = obs_test_raw - obs_mean

        for frac in EDGE_CORRUPT_LEVELS:
            try:
                r2 = evaluate_with_corrupted_operator(
                    obs_train, obs_test, obs_test_raw, obs_mean, K_MIN, frac, rng)
            except Exception:
                r2 = float("nan")

            b9_rows.append({
                "experiment_id": "B9-EDGE-DEGRADE",
                "window_id": w["window_id"],
                "corrupt_fraction": frac,
                "oos_r2": r2,
            })

    df_b9 = pd.DataFrame(b9_rows)
    print("\n  B9 EDGE DEGRADATION (mean across windows)")
    print(f"  {'corrupt':>8} {'Mean R2':>10} {'Retention':>10}")
    summary_b9 = df_b9.groupby("corrupt_fraction")["oos_r2"].mean()
    r2_clean_b9 = summary_b9.get(0.0, 0)
    for frac, r2 in summary_b9.items():
        ret = r2 / r2_clean_b9 * 100 if r2_clean_b9 > 0 else 0
        print(f"  {frac:>8.1f} {r2:>10.4f} {ret:>9.1f}%")

    robust = summary_b9.get(0.3, 0) >= 0.7 * r2_clean_b9 if r2_clean_b9 > 0 else False
    print(f"\n  Robustness criterion (30% corruption >= 70% clean): {'PASS' if robust else 'FAIL'}")
    print("=" * 70)

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    df_b8.to_parquet(METRICS_DIR / "level1_B8-NOISE.parquet", index=False)
    df_b9.to_parquet(METRICS_DIR / "level1_B9-EDGE-DEGRADE.parquet", index=False)
    yaml.dump({"experiment_id": "B8-NOISE", "levels": NOISE_LEVELS},
              open(CONFIGS_DIR / "B8-NOISE.yaml", "w"))
    yaml.dump({"experiment_id": "B9-EDGE-DEGRADE", "levels": EDGE_CORRUPT_LEVELS},
              open(CONFIGS_DIR / "B9-EDGE-DEGRADE.yaml", "w"))
    print(f"  Wrote B8 + B9 results.")


if __name__ == "__main__":
    main()
