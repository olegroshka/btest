#!/usr/bin/env python
"""
B7-T-SWEEP experiment runner.

Find minimum training window length for each component.
Tests H3b: regime switching needs sufficient training length.

All iterations use test period 2024-2025. Training windows vary:
  T=5yr (2019-2023), T=8yr (2016-2023), T=10yr (2014-2023),
  T=15yr (2009-2023), T=20yr (2004-2023)

Output:
  results/metrics/level1_B7-T-SWEEP.parquet
  results/configs/B7-T-SWEEP.yaml

Usage::
    uv run python scripts/run_smim_b7.py
"""

from __future__ import annotations

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
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EXP_ID = "B7-T-SWEEP"
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")

K_MIN = 3
K_MAX = 15
TRAIN_LENGTHS = [5, 8, 10, 15, 20]  # years

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_intensities() -> pd.DataFrame:
    int_df = pd.read_parquet(
        PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
    )
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2000-01-01", "2025-12-31", freq="QS")
    return wide.reindex(idx)


def run_at_T(intensities: pd.DataFrame, t_years: int) -> dict[str, Any]:
    train_start = pd.Timestamp(f"{2024 - t_years}-01-01")
    train_end = pd.Timestamp("2023-12-31")

    train_df = intensities[(intensities.index >= train_start) & (intensities.index <= train_end)]
    test_df = intensities[(intensities.index >= TEST_START) & (intensities.index <= TEST_END)]

    valid = train_df.columns[train_df.notna().any()]
    train_df, test_df = train_df[valid], test_df[valid]
    col_means = train_df.mean()
    obs_train_raw = train_df.fillna(col_means).values.astype(np.float64)
    obs_test_raw = test_df.fillna(col_means).values.astype(np.float64)
    obs_mean = obs_train_raw.mean(axis=0, keepdims=True)
    obs_train = obs_train_raw - obs_mean
    obs_test = obs_test_raw - obs_mean
    N = obs_train.shape[1]
    T_train = obs_train.shape[0]
    actual = obs_test_raw.T.ravel()
    K = min(K_MIN, N - 1)

    # Operator
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0

    mf_cand = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    U = mf_cand.basis[:, :K].real
    mf_k = ModalFrame(basis=U, eigenvalues=mf_cand.eigenvalues[:K], method=mf_cand.method)

    # L1: graph-factor OLS
    factors = obs_train @ U
    try:
        A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.9
    f_last = factors[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next
    r2_l1 = float(oos_r_squared((pred + obs_mean).T.ravel(), actual))

    # L2: Kalman M=1
    try:
        kf = KalmanFilter()
        st = kf.em_estimate(obs_train, mf_k, max_iter=50)
        kf2 = KalmanFilter(F=st.params["F"], Q=st.params["Q"], R=st.params["R"])
        st_test = kf2.filter(obs_test, mf_k)
        pred_l2 = st_test.alpha_predicted @ U.T
        r2_l2 = float(oos_r_squared((pred_l2 + obs_mean).T.ravel(), actual))
    except Exception:
        r2_l2 = float("nan")

    # L3: Kim M=2
    try:
        kim = KimFilter(n_regimes=2)
        st_kim = kim.em_estimate(obs_train, mf_k, max_iter=50)
        kim2 = KimFilter(n_regimes=2)
        st_kim_test = kim2.filter(obs_test, mf_k,
            F_list=st_kim.params["F_list"], Q_list=st_kim.params["Q_list"],
            R=st_kim.params["R"], P_trans=st_kim.params["P_trans"])
        pred_l3 = st_kim_test.alpha_predicted @ U.T
        r2_l3 = float(oos_r_squared((pred_l3 + obs_mean).T.ravel(), actual))
    except Exception:
        r2_l3 = float("nan")

    # Random walk
    r2_rw = float(oos_r_squared(
        np.tile(obs_train_raw[-1:], (obs_test.shape[0], 1)).T.ravel(), actual))

    return {
        "T_years": t_years, "T_quarters": T_train, "N": N,
        "r2_l1_ols": r2_l1, "r2_l2_kalman": r2_l2, "r2_l3_kim": r2_l3,
        "r2_random_walk": r2_rw,
    }


def main() -> None:
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  T = {TRAIN_LENGTHS} years  test=2024-2025  K={K_MIN}")
    print("=" * 70)

    intensities = load_intensities()
    rows = []

    for t_yr in TRAIN_LENGTHS:
        print(f"\n  T={t_yr}yr ...", end="", flush=True)
        try:
            res = run_at_T(intensities, t_yr)
            print(f" T_q={res['T_quarters']} N={res['N']}  "
                  f"L1={res['r2_l1_ols']:.4f}  L2={res['r2_l2_kalman']:.4f}  "
                  f"L3={res['r2_l3_kim']:.4f}  RW={res['r2_random_walk']:.4f}")
            rows.append(res)
        except Exception as exc:
            print(f" FAILED: {exc}")

    print("\n" + "-" * 70)
    print(f"  {'T(yr)':>6} {'T(Q)':>6} {'N':>5} {'L1_OLS':>10} {'L2_Kalman':>10} "
          f"{'L3_Kim':>10} {'RW':>10}")
    for r in rows:
        print(f"  {r['T_years']:>6} {r['T_quarters']:>6} {r['N']:>5} "
              f"{r['r2_l1_ols']:>10.4f} {r['r2_l2_kalman']:>10.4f} "
              f"{r['r2_l3_kim']:>10.4f} {r['r2_random_walk']:>10.4f}")
    print("=" * 70)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / f"level1_{EXP_ID}.parquet", index=False)
    with open(CONFIGS_DIR / f"{EXP_ID}.yaml", "w") as fh:
        yaml.dump({"experiment_id": EXP_ID, "T_lengths": TRAIN_LENGTHS}, fh)
    print(f"\n  Wrote results.")


if __name__ == "__main__":
    main()
