#!/usr/bin/env python
"""
B6-N-SWEEP experiment runner.

Find the performance cliff as N shrinks. Tests Q5: where does each
component stop adding value?

Uses RECENT period (single OOS window 2024-2025), US-LC universe at
N=20/50/100/200. L1 depth (graph-factor OLS from B1) + L2 (Kalman)
compared at each N.

Output:
  results/metrics/level1_B6-N-SWEEP.parquet
  results/configs/B6-N-SWEEP.yaml

Usage::
    uv run python scripts/smim/run_smim_b6.py
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

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EXP_ID = "B6-N-SWEEP"
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")

K_MIN = 3
K_MAX = 15
N_TARGETS = [20, 50, 100, 200]

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_universe(n_target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load US-LC intensities for N actors, return demeaned train/test."""
    reg_path = PROJECT_ROOT / "data" / "smim" / "registries" / "US-LC_registry.json"
    int_path = PROJECT_ROOT / "data" / "smim" / "intensities" / "US-LC_intensities.parquet"

    with open(reg_path) as fh:
        reg = json.load(fh)

    df = pd.read_parquet(int_path)
    # Filter to actors with data in training period
    recent = df[(df["period"] >= TRAIN_START) & (df["period"] <= TEST_END)
                & (df["normalisation_method"] == "capex_assets_xsrank")]
    avail = list(recent["actor_id"].unique())[:n_target]

    wide = recent[recent["actor_id"].isin(avail)].pivot_table(
        index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range(TRAIN_START, TEST_END, freq="QS")
    wide = wide.reindex(idx)

    # Drop all-NaN columns in training
    train_df = wide[wide.index <= TRAIN_END]
    valid = train_df.columns[train_df.notna().any()]
    train_df = train_df[valid]
    test_df = wide[valid][wide.index >= TEST_START]

    col_means = train_df.mean()
    obs_train_raw = train_df.fillna(col_means).values.astype(np.float64)
    obs_test_raw = test_df.fillna(col_means).values.astype(np.float64)
    obs_mean = obs_train_raw.mean(axis=0, keepdims=True)

    return (obs_train_raw - obs_mean, obs_test_raw - obs_mean,
            obs_test_raw, obs_mean, obs_train_raw.shape[1])


def run_at_n(n_target: int) -> dict[str, Any]:
    """Run L1 (graph-factor) and L2 (Kalman) at given N."""
    obs_train, obs_test, obs_test_raw, obs_mean, N = load_universe(n_target)
    T_train = obs_train.shape[0]
    actual = obs_test_raw.T.ravel()

    # Operator: intensity cross-correlation
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0

    K = min(K_MIN, N - 1)

    # --- L1: graph-factor OLS (Schur) ---
    mf_cand = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    U_schur = mf_cand.basis[:, :K].real

    factors = obs_train @ U_schur
    F_lag, F_curr = factors[:-1], factors[1:]
    try:
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.9

    f_last = factors[-1]
    pred_l1 = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred_l1[t] = f_next @ U_schur.T
        f_last = f_next
    r2_l1_schur = float(oos_r_squared((pred_l1 + obs_mean).T.ravel(), actual))

    # --- L1: graph-factor OLS (DMD) ---
    try:
        mf_dmd = ExactDMDDecomposer().decompose_snapshots(obs_train.T, k=min(K_MAX, N))
        U_dmd = mf_dmd.basis[:, :K].real
        factors_dmd = obs_train @ U_dmd
        F_lag_d, F_curr_d = factors_dmd[:-1], factors_dmd[1:]
        A_d = np.linalg.lstsq(F_lag_d, F_curr_d, rcond=None)[0]
        f_last_d = factors_dmd[-1]
        pred_dmd = np.zeros_like(obs_test)
        for t in range(obs_test.shape[0]):
            f_next = A_d.T @ f_last_d
            pred_dmd[t] = f_next @ U_dmd.T
            f_last_d = f_next
        r2_l1_dmd = float(oos_r_squared((pred_dmd + obs_mean).T.ravel(), actual))
    except Exception:
        r2_l1_dmd = float("nan")

    # --- L2: Kalman M=1 ---
    mf_k = ModalFrame(basis=U_schur, eigenvalues=mf_cand.eigenvalues[:K], method=mf_cand.method)
    try:
        kf = KalmanFilter()
        st = kf.em_estimate(obs_train, mf_k, max_iter=50)
        kf2 = KalmanFilter(F=st.params["F"], Q=st.params["Q"], R=st.params["R"])
        st_test = kf2.filter(obs_test, mf_k)
        pred_l2 = st_test.alpha_predicted @ U_schur.T
        r2_l2 = float(oos_r_squared((pred_l2 + obs_mean).T.ravel(), actual))
    except Exception:
        r2_l2 = float("nan")

    # --- Naive baselines ---
    r2_rw = float(oos_r_squared(
        np.tile(obs_train_raw := obs_train + obs_mean, (1, 1))[-1:].repeat(obs_test.shape[0], axis=0).T.ravel(),
        actual))
    r2_mean = float(oos_r_squared(
        np.tile(obs_mean, (obs_test.shape[0], 1)).T.ravel(), actual))

    return {
        "N_target": n_target, "N_actual": N, "T_train": T_train,
        "r2_l1_schur": r2_l1_schur, "r2_l1_dmd": r2_l1_dmd,
        "r2_l2_kalman": r2_l2,
        "r2_random_walk": r2_rw, "r2_hist_mean": r2_mean,
    }


def main() -> None:
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  N = {N_TARGETS}  RECENT period  K={K_MIN}")
    print("=" * 70)

    rows = []
    for n_target in N_TARGETS:
        print(f"\n  N={n_target} ...", end="", flush=True)
        try:
            res = run_at_n(n_target)
            print(f" N_actual={res['N_actual']}  L1_schur={res['r2_l1_schur']:.4f}  "
                  f"L1_dmd={res['r2_l1_dmd']:.4f}  L2_kalman={res['r2_l2_kalman']:.4f}  "
                  f"RW={res['r2_random_walk']:.4f}")
            rows.append(res)
        except Exception as exc:
            print(f" FAILED: {exc}")

    # Summary
    print("\n" + "-" * 70)
    print(f"  {'N':>6} {'L1_Schur':>10} {'L1_DMD':>10} {'L2_Kalman':>10} {'RW':>10} {'Mean':>10}")
    for r in rows:
        print(f"  {r['N_actual']:>6} {r['r2_l1_schur']:>10.4f} {r['r2_l1_dmd']:>10.4f} "
              f"{r['r2_l2_kalman']:>10.4f} {r['r2_random_walk']:>10.4f} {r['r2_hist_mean']:>10.4f}")
    print("=" * 70)

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(METRICS_DIR / f"level1_{EXP_ID}.parquet", index=False)
    print(f"\n  Wrote: results/metrics/level1_{EXP_ID}.parquet")

    config = {"experiment_id": EXP_ID, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "N_targets": N_TARGETS, "period": "RECENT"}
    with open(CONFIGS_DIR / f"{EXP_ID}.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: results/configs/{EXP_ID}.yaml")


if __name__ == "__main__":
    main()
