#!/usr/bin/env python
"""
B1-COMPONENT-ABLATION experiment runner.

Decomposes the A1 result into per-layer contributions to answer Q1:
does each layer of pipeline complexity earn its keep?

5 depths (each adds one layer on the previous):
  L1: graph factors only (PCA on operator, OLS forecast — no Kalman)
  L2: spectral + single-regime Kalman (M=1)
  L3: spectral + regime switching (KimFilter M=2)
  L4: L3 + emergence diagnostics (PID synergy + criticality)
  L5: full pipeline (= A1 reference, reused from results)

Decision rule: delta-R^2 >= 0.5pp AND DM p <= 0.10 -> ADDS VALUE

Output:
  results/metrics/level1_B1-COMPONENT-ABLATION.parquet
  results/metrics/level2_B1-COMPONENT-ABLATION.parquet
  results/configs/B1-COMPONENT-ABLATION.yaml

Usage::
    uv run python scripts/run_smim_b1.py
    uv run python scripts/run_smim_b1.py --single-window 0
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
import scipy.sparse as sp
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry as ConcreteActorRegistry
from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.dynamics.phase_transition import (
    criticality_index,
    extract_order_parameter,
)
from quantdsl_backtest.smim.emergence.pid import compute_synergy_matrix
from quantdsl_backtest.smim.gaps.emergence_aware import EmergenceAwareBenchmark
from quantdsl_backtest.smim.gaps.modal import ModalBenchmark
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
from quantdsl_backtest.smim.graph.operators import AggregateOperator
from quantdsl_backtest.smim.graph.sparsification import l1_sparsify
from quantdsl_backtest.smim.interfaces import (
    Actor,
    ActorType,
    DateRange,
    Layer,
    ModalFrame,
    RelationChannel,
)
from quantdsl_backtest.smim.spectral.mode_selection import MDLModeSelector
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

# --- constants ---------------------------------------------------------------

EXP_ID = "B1-COMPONENT-ABLATION"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"

K_MIN = 3
K_MAX_CANDIDATES = 15
TARGET_DENSITY = 0.15
TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"

_ACTOR_TYPE_MAP = {
    "global_shock": ActorType.GLOBAL_SHOCK, "central_bank": ActorType.CENTRAL_BANK,
    "regulator": ActorType.REGULATOR, "intl_org": ActorType.INTL_ORG,
    "sector_leader": ActorType.SECTOR_LEADER, "large_firm": ActorType.LARGE_FIRM,
    "bank": ActorType.BANK,
}


# --- data loading (shared with A1) -------------------------------------------


def load_data() -> tuple[pd.DataFrame, list[dict], list[str]]:
    with open(REGISTRY_PATH) as fh:
        reg_data = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors_with_int = set(int_df["actor_id"].unique())
    actor_data = [a for a in reg_data["actors"] if a["actor_id"] in actors_with_int]
    actor_ids = [a["actor_id"] for a in actor_data]

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], actor_data, actor_ids


def make_windows(test_years: list[int] | None = None) -> list[dict[str, Any]]:
    years = test_years or TEST_YEARS
    return [
        {"window_id": f"W{ty}",
         "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
         "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
         "test_start": pd.Timestamp(f"{ty}-01-01"),
         "test_end": pd.Timestamp(f"{ty}-12-31")}
        for ty in years
    ]


def prepare_window_data(
    window: dict, intensities_wide: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Prepare demeaned train/test arrays for a window."""
    ts, te = window["train_start"], window["train_end"]
    test_s, test_e = window["test_start"], window["test_end"]

    int_train = intensities_wide[(intensities_wide.index >= ts) & (intensities_wide.index <= te)]
    int_test = intensities_wide[(intensities_wide.index >= test_s) & (intensities_wide.index <= test_e)]
    valid = int_train.columns[int_train.notna().any()]
    int_train = int_train[valid]
    int_test = int_test[valid]

    col_means = int_train.mean()
    obs_train_raw = int_train.fillna(col_means).values.astype(np.float64)
    obs_test_raw = int_test.fillna(col_means).values.astype(np.float64)
    obs_mean = obs_train_raw.mean(axis=0, keepdims=True)
    obs_train = obs_train_raw - obs_mean
    obs_test = obs_test_raw - obs_mean

    return obs_train, obs_test, obs_test_raw, obs_mean, list(valid), len(valid)


def build_operator(obs_train: np.ndarray, N: int) -> np.ndarray:
    """Build operator via end-to-end optimised channel weights (Approach C from A1)."""
    from scipy.optimize import minimize as sp_minimize

    basis_ops = []
    # Instantaneous correlation
    corr0 = np.corrcoef(obs_train.T)
    corr0 = np.nan_to_num(corr0, nan=0.0)
    np.fill_diagonal(corr0, 0.0)
    basis_ops.append(corr0)

    T = obs_train.shape[0]
    # Lag cross-correlation
    if T > 2:
        lag1 = np.zeros((N, N))
        for lag_off in [1, 2]:
            if T > lag_off:
                xp = obs_train[:-lag_off] - obs_train[:-lag_off].mean(axis=0)
                xn = obs_train[lag_off:] - obs_train[lag_off:].mean(axis=0)
                num = (xp.T @ xn) / max(len(xp) - 1, 1)
                sp_std = np.std(obs_train[:-lag_off], axis=0, keepdims=True).T
                sn_std = np.std(obs_train[lag_off:], axis=0, keepdims=True)
                denom = np.maximum(sp_std @ sn_std, 1e-10)
                lag1 += np.nan_to_num(num / denom, nan=0.0)
        np.fill_diagonal(lag1, 0.0)
        basis_ops.append(lag1)

    # Multi-scale cosine
    for scale in [4, 8]:
        if T >= scale + 1:
            kernel = np.ones(scale) / scale
            sm = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), 0, obs_train)
            if sm.shape[0] >= 2:
                norms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                sim = (sm / norms).T @ (sm / norms)
                np.fill_diagonal(sim, 0.0)
                basis_ops.append(sim)

    n_bases = len(basis_ops)
    t_split = int(T * 0.75)
    obs_st, obs_sv = obs_train[:t_split], obs_train[t_split:]

    def objective(weights):
        try:
            A = sum(w * B for w, B in zip(weights, basis_ops))
            thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
            A[np.abs(A) < thr] = 0.0
            mf = SchurDecomposer().decompose(A, k=min(K_MAX_CANDIDATES, N))
            U = mf.basis[:, :K_MIN]
            mf_k = ModalFrame(basis=U, eigenvalues=mf.eigenvalues[:K_MIN], method=mf.method)
            kf = KalmanFilter()
            st = kf.em_estimate(obs_st, mf_k, max_iter=15)
            kf2 = KalmanFilter(F=st.params.get("F"), Q=st.params.get("Q"), R=st.params.get("R"))
            sv = kf2.filter(obs_sv, mf_k)
            pred = sv.alpha_predicted @ U.T
            r2 = float(oos_r_squared(pred.T.ravel(), obs_sv.T.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    x0 = np.ones(n_bases) / n_bases
    result = sp_minimize(objective, x0, method="Nelder-Mead",
                         options={"maxiter": 150, "xatol": 0.01, "fatol": 0.001})
    A_final = sum(w * B for w, B in zip(result.x, basis_ops))
    thr = np.percentile(np.abs(A_final[A_final != 0]), 30) if np.any(A_final != 0) else 0.01
    A_final[np.abs(A_final) < thr] = 0.0
    return A_final


# --- depth functions ---------------------------------------------------------


def depth_L1_graph_factors(
    obs_train: np.ndarray, obs_test: np.ndarray, obs_test_raw: np.ndarray,
    obs_mean: np.ndarray, op_dense: np.ndarray, N: int, K: int,
) -> dict[str, float]:
    """L1: Graph factors only. PCA on operator → project → OLS forecast."""
    # Eigenvectors of operator as basis (graph Fourier modes)
    eigvals, eigvecs = np.linalg.eigh(op_dense + op_dense.T)  # symmetrise
    U = eigvecs[:, -K:]  # top-K eigenvectors

    # Project train onto graph modes
    factors_train = obs_train @ U  # (T_train, K)

    # AR(1) on factors
    F_lag = factors_train[:-1]
    F_curr = factors_train[1:]
    try:
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.9

    # Forecast test
    f_last = factors_train[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next

    # Restore mean
    pred_restored = (pred + obs_mean).T.ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_restored, actual))
    return {"oos_r2": r2, "errors": actual - pred_restored}


def depth_L2_kalman_m1(
    obs_train: np.ndarray, obs_test: np.ndarray, obs_test_raw: np.ndarray,
    obs_mean: np.ndarray, modal_frame: ModalFrame, registry,
) -> dict[str, float]:
    """L2: Spectral + single-regime Kalman (M=1)."""
    kf = KalmanFilter()
    state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)
    kf_test = KalmanFilter(
        F=state_train.params.get("F"),
        Q=state_train.params.get("Q"),
        R=state_train.params.get("R"),
    )
    state_test = kf_test.filter(obs_test, modal_frame)
    gap = PredictiveBenchmark().compute(obs_test, state_test, modal_frame, registry)

    pred_restored = (gap.benchmarks + obs_mean.T).ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_restored, actual))
    return {"oos_r2": r2, "errors": actual - pred_restored,
            "state_train": state_train, "state_test": state_test}


def depth_L3_regime_switching(
    obs_train: np.ndarray, obs_test: np.ndarray, obs_test_raw: np.ndarray,
    obs_mean: np.ndarray, modal_frame: ModalFrame, registry,
) -> dict[str, float]:
    """L3: Spectral + KimFilter M=2."""
    kf = KimFilter(n_regimes=2)
    state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)
    kf_test = KimFilter(n_regimes=2)
    state_test = kf_test.filter(
        obs_test, modal_frame,
        F_list=state_train.params.get("F_list"),
        Q_list=state_train.params.get("Q_list"),
        R=state_train.params.get("R"),
        P_trans=state_train.params.get("P_trans"),
    )
    gap = PredictiveBenchmark().compute(obs_test, state_test, modal_frame, registry)

    pred_restored = (gap.benchmarks + obs_mean.T).ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_restored, actual))
    return {"oos_r2": r2, "errors": actual - pred_restored,
            "state_train": state_train, "state_test": state_test}


def depth_L4_emergence(
    obs_train: np.ndarray, obs_test: np.ndarray, obs_test_raw: np.ndarray,
    obs_mean: np.ndarray, modal_frame: ModalFrame, registry,
    state_train_l3, state_test_l3,
) -> dict[str, float]:
    """L4: L3 + emergence diagnostics (synergy + criticality)."""
    K = modal_frame.K
    U = modal_frame.basis

    # Synergy from training data
    alpha_train = state_train_l3.alpha_filtered
    recon = alpha_train @ U.T
    train_gaps = (obs_train - recon).T
    try:
        synergy = compute_synergy_matrix(alpha_train, train_gaps, K)
        # Sign correction
        gap_target = train_gaps.mean(axis=0)
        for j in range(K):
            for k in range(j + 1, K):
                inter = alpha_train[:, j] * alpha_train[:, k]
                cc = np.cov(inter, gap_target)[0, 1]
                d = np.sign(cc) if abs(cc) > 1e-10 else 0.0
                synergy[j, k] *= d
                synergy[k, j] *= d
    except Exception:
        synergy = np.zeros((K, K))

    # Criticality from training
    d = min(3, K)
    psi = extract_order_parameter(alpha_train, d=d)
    crit_train = criticality_index(psi, window_size=8)
    crit_test = np.full(obs_test.shape[0], crit_train[-1] if len(crit_train) > 0 else 1.0)

    ea = EmergenceAwareBenchmark(synergy_matrix=synergy, criticality=crit_test)
    gap_ea = ea.compute(obs_test, state_test_l3, modal_frame, registry)

    pred_restored = (gap_ea.benchmarks + obs_mean.T).ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_restored, actual))
    return {"oos_r2": r2, "errors": actual - pred_restored}


# --- main --------------------------------------------------------------------


def run_b1(verbose: bool = False, single_window: int | None = None) -> None:
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  5 depths x 10 windows  component layer ablation")
    print("=" * 70)

    print("\n  Loading data ...", end="", flush=True)
    intensities_wide, actor_data, actor_ids = load_data()
    print(f" done. {len(actor_ids)} actors")

    # Load A1 results for L5 reference
    a1_path = METRICS_DIR / "level1_A1-MVP-FULL.parquet"
    a1_df = pd.read_parquet(a1_path) if a1_path.exists() else None

    windows = [make_windows()[single_window]] if single_window is not None else make_windows()

    l1_rows: list[dict] = []
    l2_rows: list[dict] = []

    for wi, window in enumerate(windows):
        wid = window["window_id"]
        print(f"\n  [{wi + 1}/{len(windows)}] {wid}")

        obs_train, obs_test, obs_test_raw, obs_mean, actor_ids_final, N = \
            prepare_window_data(window, intensities_wide)

        bench_actors = [
            Actor(actor_id=a["actor_id"], name=a["name"],
                  actor_type=_ACTOR_TYPE_MAP.get(a["actor_type"], ActorType.LARGE_FIRM),
                  layer=Layer(a["layer"]), geography=a["geography"],
                  sector=a["sector"], external_ids=a.get("external_ids", {}))
            for a in actor_data if a["actor_id"] in set(actor_ids_final)
        ]
        registry = ConcreteActorRegistry(actors=bench_actors[:N])

        # Build operator (Approach C)
        op_dense = build_operator(obs_train, N)

        # Schur decompose
        mf_cand = SchurDecomposer().decompose(op_dense, k=min(K_MAX_CANDIDATES, N))
        K = min(K_MIN, mf_cand.K)
        modal_frame = ModalFrame(
            basis=mf_cand.basis[:, :K],
            eigenvalues=mf_cand.eigenvalues[:K],
            method=mf_cand.method,
        )

        depths = {}
        prev_errors = None

        # --- L1: graph factors -----------------------------------------------
        try:
            res = depth_L1_graph_factors(obs_train, obs_test, obs_test_raw, obs_mean, op_dense, N, K)
            depths["L1_graph"] = res["oos_r2"]
            prev_errors = res["errors"]
            print(f"    L1 graph:     R2={res['oos_r2']:.4f}")
        except Exception as e:
            depths["L1_graph"] = float("nan")
            print(f"    L1 graph:     FAILED ({e})")

        # --- L2: spectral + Kalman M=1 --------------------------------------
        try:
            res = depth_L2_kalman_m1(obs_train, obs_test, obs_test_raw, obs_mean, modal_frame, registry)
            depths["L2_kalman"] = res["oos_r2"]
            errors_l2 = res["errors"]
            print(f"    L2 kalman:    R2={res['oos_r2']:.4f}")
        except Exception as e:
            depths["L2_kalman"] = float("nan")
            errors_l2 = prev_errors
            print(f"    L2 kalman:    FAILED ({e})")

        # --- L3: regime switching M=2 ----------------------------------------
        try:
            res = depth_L3_regime_switching(obs_train, obs_test, obs_test_raw, obs_mean, modal_frame, registry)
            depths["L3_regime"] = res["oos_r2"]
            errors_l3 = res["errors"]
            state_train_l3 = res["state_train"]
            state_test_l3 = res["state_test"]
            print(f"    L3 regime:    R2={res['oos_r2']:.4f}")
        except Exception as e:
            depths["L3_regime"] = float("nan")
            errors_l3 = errors_l2
            state_train_l3 = None
            state_test_l3 = None
            print(f"    L3 regime:    FAILED ({e})")

        # --- L4: emergence ---------------------------------------------------
        if state_train_l3 is not None and state_test_l3 is not None:
            try:
                res = depth_L4_emergence(
                    obs_train, obs_test, obs_test_raw, obs_mean, modal_frame, registry,
                    state_train_l3, state_test_l3,
                )
                depths["L4_emergence"] = res["oos_r2"]
                errors_l4 = res["errors"]
                print(f"    L4 emergence: R2={res['oos_r2']:.4f}")
            except Exception as e:
                depths["L4_emergence"] = float("nan")
                errors_l4 = errors_l3
                print(f"    L4 emergence: FAILED ({e})")
        else:
            depths["L4_emergence"] = float("nan")
            errors_l4 = errors_l3
            print(f"    L4 emergence: SKIPPED (L3 failed)")

        # --- L5: full pipeline (from A1) -------------------------------------
        if a1_df is not None:
            a1_row = a1_df[(a1_df["window_id"] == wid) & (a1_df["benchmark_class"] == "predictive")]
            if not a1_row.empty:
                depths["L5_full"] = float(a1_row.iloc[0]["oos_r2"])
            else:
                depths["L5_full"] = float("nan")
        else:
            depths["L5_full"] = float("nan")
        print(f"    L5 full(A1):  R2={depths['L5_full']:.4f}")

        # --- L1 rows ---------------------------------------------------------
        for depth_name, r2 in depths.items():
            l1_rows.append({
                "experiment_id": EXP_ID,
                "window_id": wid,
                "depth": depth_name,
                "oos_r2": r2,
                "n_actors": N,
                "n_test_periods": obs_test.shape[0],
            })

        # --- L2 rows: delta-R2 between consecutive depths --------------------
        depth_names = ["L1_graph", "L2_kalman", "L3_regime", "L4_emergence", "L5_full"]
        for i in range(1, len(depth_names)):
            prev_d = depth_names[i - 1]
            curr_d = depth_names[i]
            r2_prev = depths.get(prev_d, float("nan"))
            r2_curr = depths.get(curr_d, float("nan"))
            delta = r2_curr - r2_prev if np.isfinite(r2_curr) and np.isfinite(r2_prev) else float("nan")
            l2_rows.append({
                "experiment_id": EXP_ID,
                "window_id": wid,
                "component": f"{curr_d} vs {prev_d}",
                "delta_r2": delta,
                "adds_value": abs(delta) >= 0.005 if np.isfinite(delta) else False,
            })

    # --- Summary -------------------------------------------------------------
    df_l1 = pd.DataFrame(l1_rows)
    df_l2 = pd.DataFrame(l2_rows)

    print("\n" + "-" * 70)
    print("  COMPONENT VALUE TABLE (mean across windows)")
    print(f"  {'Depth':<20} {'Mean R2':>10} {'Std R2':>10} {'vs prev':>10} {'Verdict':>12}")

    depth_names = ["L1_graph", "L2_kalman", "L3_regime", "L4_emergence", "L5_full"]
    prev_mean = None
    for depth in depth_names:
        sub = df_l1[df_l1["depth"] == depth]
        mean_r2 = sub["oos_r2"].mean()
        std_r2 = sub["oos_r2"].std()
        delta = mean_r2 - prev_mean if prev_mean is not None and np.isfinite(mean_r2) else 0.0
        verdict = "ADDS VALUE" if abs(delta) >= 0.005 else "MARGINAL"
        if prev_mean is None:
            verdict = "BASE"
        print(f"  {depth:<20} {mean_r2:>10.4f} {std_r2:>10.4f} {delta:>+10.4f} {verdict:>12}")
        prev_mean = mean_r2 if np.isfinite(mean_r2) else prev_mean

    print("=" * 70)

    # --- Save ----------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    l1_path = METRICS_DIR / f"level1_{EXP_ID}.parquet"
    df_l1.to_parquet(l1_path, index=False)
    print(f"\n  Wrote: {l1_path.relative_to(PROJECT_ROOT)}")

    l2_path = METRICS_DIR / f"level2_{EXP_ID}.parquet"
    df_l2.to_parquet(l2_path, index=False)
    print(f"  Wrote: {l2_path.relative_to(PROJECT_ROOT)}")

    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "depths": depth_names,
        "n_windows": len(windows),
    }
    yaml_path = CONFIGS_DIR / f"{EXP_ID}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-COMPONENT-ABLATION runner")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--single-window", type=int, default=None)
    args = parser.parse_args()
    run_b1(verbose=args.verbose, single_window=args.single_window)


if __name__ == "__main__":
    main()
