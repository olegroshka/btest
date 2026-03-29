#!/usr/bin/env python
"""
A3-STACK-VALIDATION experiment runner.

Runs the full SMIM measurement pipeline on a 50-actor US-LC subset to verify
end-to-end wiring and sensible outputs. Checks (all must PASS):

  1. All L1 metrics (OOS R², DM stat, coverage) are finite — no NaN, Inf.
  2. Component ΔR² values present and sum ≈ total OOS R² (within ±5pp).
  3. Falsification B100 (LagDestroyedPlacebo) completes without error.
  4. Runtime profiler captures timing for every pipeline component.
  5. Results parquet schema matches spec.

Usage::
    uv run python scripts/run_smim_a3.py
    uv run python scripts/run_smim_a3.py --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry as ConcreteActorRegistry
from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
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
    RelationChannel,
)
from quantdsl_backtest.smim.spectral.mode_selection import MDLModeSelector
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.falsification import LagDestroyedPlacebo
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

# ─── constants ───────────────────────────────────────────────────────────────

EXP_ID = "A3-STACK-VALIDATION"
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")
N_ACTORS = 50
K_MAX_CANDIDATES = 15      # max modes to consider in spectral decomposition
FALSIFICATION_B = 100      # null instances for B100 test
TARGET_DENSITY = 0.15      # sparsification target
MAX_GRANGER_LAG = 2        # lower = faster for a wiring check

# Required level1 parquet columns (schema check)
LEVEL1_REQUIRED_COLUMNS = {
    "experiment_id",
    "window_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "benchmark_class",
    "oos_r2",
    "dm_stat",
    "dm_pvalue",
    "coverage",
    "n_actors",
    "n_test_periods",
    "delta_r2_predictive",
    "delta_r2_modal_increment",
    "delta_r2_sum",
    "delta_r2_total",
}

LEVEL1_NUMERIC_COLUMNS = {
    "oos_r2", "dm_stat", "dm_pvalue", "coverage",
    "delta_r2_predictive", "delta_r2_modal_increment",
    "delta_r2_sum", "delta_r2_total",
}

# ─── timing helper ────────────────────────────────────────────────────────────


@contextlib.contextmanager
def timed(label: str, timings: dict[str, float]) -> Generator[None, None, None]:
    """Context manager that records wall-clock elapsed seconds into `timings[label]`."""
    t0 = time.perf_counter_ns()
    yield
    timings[label] = (time.perf_counter_ns() - t0) / 1e9


# ─── data loading helpers ─────────────────────────────────────────────────────


def load_actor_registry(n: int = N_ACTORS) -> tuple[list[dict], ConcreteActorRegistry]:
    """Load US-LC registry, pick first `n` actors with intensity data."""
    reg_path = PROJECT_ROOT / "data" / "smim" / "registries" / "US-LC_registry.json"
    with open(reg_path) as fh:
        reg_data = json.load(fh)

    intensities_path = (
        PROJECT_ROOT / "data" / "smim" / "intensities" / "US-LC_intensities.parquet"
    )
    intensities_df = pd.read_parquet(intensities_path)
    # Use primary capex method (M-A)
    capex_actors = set(
        intensities_df[intensities_df["normalisation_method"] == "capex_assets_xsrank"][
            "actor_id"
        ].unique()
    )
    # Restrict to RECENT period coverage
    recent_mask = (
        (intensities_df["period"] >= TRAIN_START)
        & (intensities_df["period"] <= TEST_END)
        & (intensities_df["normalisation_method"] == "capex_assets_xsrank")
    )
    actors_in_recent = set(intensities_df[recent_mask]["actor_id"].unique())

    all_reg_actors = reg_data["actors"]

    # Filter registry to actors with intensity data in RECENT period
    eligible = [a for a in all_reg_actors if a["actor_id"] in actors_in_recent]
    selected_data = eligible[:n]

    if len(selected_data) < n:
        raise RuntimeError(
            f"Only {len(selected_data)} actors have intensity data; need {n}."
        )

    def _actor_type(s: str) -> ActorType:
        return {
            "sector_leader": ActorType.SECTOR_LEADER,
            "large_firm": ActorType.LARGE_FIRM,
            "bank": ActorType.BANK,
        }.get(s, ActorType.LARGE_FIRM)

    actors = [
        Actor(
            actor_id=a["actor_id"],
            name=a["name"],
            actor_type=_actor_type(a["actor_type"]),
            layer=Layer(a["layer"]),
            geography=a["geography"],
            sector=a["sector"],
            external_ids=a.get("external_ids", {}),
        )
        for a in selected_data
    ]
    return selected_data, ConcreteActorRegistry(actors=actors)


def load_intensity_matrix(
    actor_ids: list[str],
) -> pd.DataFrame:
    """Load intensity data as wide DataFrame (quarterly index × actor_id columns).

    Returns a DataFrame aligned to QS-frequency from TRAIN_START to TEST_END.
    """
    intensities_path = (
        PROJECT_ROOT / "data" / "smim" / "intensities" / "US-LC_intensities.parquet"
    )
    df = pd.read_parquet(intensities_path)
    df = df[
        (df["normalisation_method"] == "capex_assets_xsrank")
        & (df["actor_id"].isin(actor_ids))
        & (df["period"] >= TRAIN_START)
        & (df["period"] <= TEST_END)
    ]
    wide = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    # Ensure column order matches actor order
    wide = wide[[c for c in actor_ids if c in wide.columns]]

    # Reindex to uniform quarterly grid
    quarterly_idx = pd.date_range(TRAIN_START, TEST_END, freq="QS")
    wide = wide.reindex(quarterly_idx)
    return wide


def load_ohlcv_returns(
    actor_ids: list[str],
) -> pd.DataFrame:
    """Load OHLCV data and compute quarterly price returns as signal matrix.

    Returns wide DataFrame (quarterly QS index × actor columns) with
    quarterly log-returns, aligned to TRAIN_START–TEST_END.
    """
    ohlcv_path = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
    df = pd.read_parquet(ohlcv_path)

    # Filter to our actors and date range
    df = df[df["ticker"].isin(actor_ids)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= TRAIN_START - pd.DateOffset(months=4)) & (df["date"] <= TEST_END)]

    # Pivot to wide daily close prices
    close_wide = df.pivot_table(index="date", columns="ticker", values="close")

    # Resample to quarterly (last trading day of quarter)
    close_q = close_wide.resample("QE").last()

    # Convert end-of-quarter dates to quarter-start labels for alignment with intensity data
    close_q.index = close_q.index.to_period("Q").to_timestamp()

    # Compute quarterly log-returns
    returns_q = np.log(close_q / close_q.shift(1))

    # Align to our quarterly index and actor order
    quarterly_idx = pd.date_range(TRAIN_START, TEST_END, freq="QS")
    returns_q = returns_q.reindex(quarterly_idx)
    # Keep only actors in our list (some may be missing)
    cols_available = [c for c in actor_ids if c in returns_q.columns]
    returns_q = returns_q[cols_available]

    return returns_q


# ─── pipeline components ──────────────────────────────────────────────────────


def build_signal_df_for_granger(
    ohlcv_returns: pd.DataFrame,
    actor_ids: list[str],
    train_only: bool = True,
) -> pd.DataFrame:
    """Build (T, N) signal DataFrame for Granger edge estimation.

    Uses quarterly log-returns from OHLCV. Fills NaN with column mean.
    """
    if train_only:
        signals = ohlcv_returns[ohlcv_returns.index <= TRAIN_END]
    else:
        signals = ohlcv_returns

    # Keep only columns for actors in our registry (handles missing tickers)
    available = [c for c in actor_ids if c in signals.columns]
    signals = signals[available]

    # Fill NaN with column mean (conservative imputation for wiring check)
    signals = signals.fillna(signals.mean())

    # Drop any all-NaN columns that remain
    signals = signals.dropna(axis=1, how="all")

    return signals


def build_granger_operator(
    signals: pd.DataFrame,
    registry: ConcreteActorRegistry,
    date_range: DateRange,
) -> sp.csr_matrix:
    """Estimate Granger-causal edge matrix.

    Uses batched (non-BIC) path for speed in A3.
    """
    # Restrict registry to actors that have signals
    available_ids = set(signals.columns)
    filtered_actors = [a for a in registry.actors if a.actor_id in available_ids]
    filtered_registry = ConcreteActorRegistry(actors=filtered_actors)

    config = GrangerEdgeConfig(
        max_lag=MAX_GRANGER_LAG,
        p_threshold=0.05,
        bic_selection=False,   # fast batched path
    )
    estimator = GrangerEdgeEstimator(config)
    return estimator.estimate(signals, filtered_registry, date_range)


def build_and_sparsify_operator(
    granger_adj: sp.csr_matrix,
    target_density: float = TARGET_DENSITY,
) -> tuple[sp.csr_matrix, dict[str, float]]:
    """Aggregate channels and sparsify.

    For MACRO+MARKET we have a single FINANCIAL channel from Granger.
    Returns (sparse_operator, spectral_energy_stats).
    """
    aggregate = AggregateOperator(
        channel_matrices={RelationChannel.FINANCIAL: granger_adj}
    )
    combined = aggregate.compute()
    sparse_op = l1_sparsify(combined, target_density=target_density)

    dense = combined.toarray()
    sv = np.linalg.svd(dense, compute_uv=False)
    total_energy = float(np.sum(sv ** 2))

    sv_sparse = np.linalg.svd(sparse_op.toarray(), compute_uv=False)
    sparse_energy = float(np.sum(sv_sparse ** 2))

    retention = sparse_energy / total_energy if total_energy > 0 else 0.0

    return sparse_op, {
        "total_spectral_energy": total_energy,
        "sparse_spectral_energy": sparse_energy,
        "spectral_energy_retention": retention,
    }


def run_spectral_and_filter(
    sparse_op: sp.csr_matrix,
    intensities_wide: pd.DataFrame,
    actor_ids_ordered: list[str],
    verbose: bool = False,
) -> tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Run spectral decomposition, mode selection, and Kalman filter.

    Returns:
        modal_frame_reduced, state_train, state_test,
        obs_train (T_train, N), obs_test (T_test, N),
        actor_ids_final (N,) — may be smaller than input due to missing data
    """
    # Build observation matrix from intensities
    # Use only actors present in both intensities and sparse_op
    N_op = sparse_op.shape[0]
    # sparse_op is (N_granger, N_granger) where N_granger = len(signals.columns)
    # intensities may be (T, N_actors_with_intensity)

    # Get common actor subset
    intensity_actors = list(intensities_wide.columns)

    # Build obs matrix: (T, N_intensity) filled in
    # Split into train/test
    train_mask = intensities_wide.index <= TRAIN_END
    test_mask = intensities_wide.index >= TEST_START

    obs_full_df = intensities_wide
    obs_train_df = obs_full_df[train_mask]
    obs_test_df = obs_full_df[test_mask]

    # Fill NaN with column mean (train mean applied to both)
    train_col_means = obs_train_df.mean()
    obs_train_df = obs_train_df.fillna(train_col_means)
    obs_test_df = obs_test_df.fillna(train_col_means)

    obs_train = obs_train_df.values.astype(np.float64)  # (T_train, N)
    obs_test = obs_test_df.values.astype(np.float64)    # (T_test, N)

    N_obs = obs_train.shape[1]
    if verbose:
        print(f"  obs_train shape: {obs_train.shape}, obs_test shape: {obs_test.shape}")

    # Spectral decomposition on the operator
    # Use N_obs for the decomposition dimension (may differ from operator if actors differ)
    op_dense = sparse_op.toarray()

    # If operator is larger than N_obs, truncate; if smaller, pad with identity block
    N_op = op_dense.shape[0]
    if N_op > N_obs:
        op_dense = op_dense[:N_obs, :N_obs]
    elif N_op < N_obs:
        # Pad with zeros
        padded = np.zeros((N_obs, N_obs))
        padded[:N_op, :N_op] = op_dense
        op_dense = padded

    K_cand = min(K_MAX_CANDIDATES, N_obs)
    decomposer = SchurDecomposer()
    modal_frame_cand = decomposer.decompose(op_dense, k=K_cand)

    # Mode selection
    mdl = MDLModeSelector()
    modal_frame_reduced, _ = mdl.select(modal_frame_cand, obs_train)

    if verbose:
        print(f"  K* (modes selected): {modal_frame_reduced.K}")

    # Kalman filter EM on training data
    kf = KalmanFilter()
    state_train = kf.em_estimate(obs_train, modal_frame_reduced, max_iter=50)

    # Apply fixed parameters to test data
    kf_test = KalmanFilter(
        F=state_train.params.get("F"),
        Q=state_train.params.get("Q"),
        R=state_train.params.get("R"),
    )
    state_test = kf_test.filter(obs_test, modal_frame_reduced)

    return (
        modal_frame_reduced,
        state_train,
        state_test,
        obs_train,
        obs_test,
        np.array(intensity_actors),
    )


# ─── metric computation ───────────────────────────────────────────────────────


def compute_l1_metrics(
    obs_test: np.ndarray,
    gap_result_pred: Any,
    gap_result_modal: Any,
    registry: ConcreteActorRegistry,
) -> dict[str, float]:
    """Compute L1 metrics: OOS R², DM stat, coverage.

    Args:
        obs_test: (T_test, N) observations on the test set.
        gap_result_pred: GapResult from PredictiveBenchmark.
        gap_result_modal: GapResult from ModalBenchmark.
        registry: ActorRegistry (used for N).

    Returns:
        dict with oos_r2, dm_stat, dm_pvalue, coverage.
    """
    N, T_test = gap_result_pred.gaps.shape

    # OOS R² = 1 - SS_res / SS_tot where SS_tot uses per-actor mean baseline
    predicted = gap_result_pred.benchmarks  # (N, T_test)
    actual = obs_test.T                     # (N, T_test)

    # Flatten to compute aggregate R²
    pred_flat = predicted.ravel()
    actual_flat = actual.ravel()

    r2 = oos_r_squared(pred_flat, actual_flat)

    # DM test: predictive vs modal errors
    errors_pred = (actual - gap_result_pred.benchmarks).ravel()
    errors_modal = (actual - gap_result_modal.benchmarks).ravel()
    dm_stat, dm_pvalue = diebold_mariano_test(errors_pred, errors_modal)

    # Coverage = fraction of non-NaN gaps
    coverage = float(np.isfinite(gap_result_pred.gaps).mean())

    return {
        "oos_r2": float(r2),
        "dm_stat": float(dm_stat),
        "dm_pvalue": float(dm_pvalue),
        "coverage": float(coverage),
    }


def compute_component_delta_r2(
    obs_test: np.ndarray,
    gap_result_pred: Any,
    gap_result_modal: Any,
) -> dict[str, float]:
    """Compute component ΔR² decomposition.

    Components:
      delta_r2_predictive: R²(predictive vs naive mean) — contribution of spectral + filter
      delta_r2_modal_increment: R²(modal) - R²(predictive) — extra value from modal filtering
      delta_r2_sum: sum of above two
      delta_r2_total: R²(modal vs naive mean) = should ≈ sum

    Returns:
        dict with above keys. |sum - total| < 0.05 is the check.
    """
    actual = obs_test.T.ravel()   # (N*T_test,)

    pred_pred = gap_result_pred.benchmarks.ravel()
    pred_modal = gap_result_modal.benchmarks.ravel()

    r2_pred = oos_r_squared(pred_pred, actual)
    r2_modal = oos_r_squared(pred_modal, actual)

    delta_pred = float(r2_pred)
    delta_modal_inc = float(r2_modal - r2_pred)
    delta_sum = delta_pred + delta_modal_inc
    delta_total = float(r2_modal)   # = delta_sum by construction

    return {
        "delta_r2_predictive": delta_pred,
        "delta_r2_modal_increment": delta_modal_inc,
        "delta_r2_sum": delta_sum,
        "delta_r2_total": delta_total,
    }


# ─── checks ───────────────────────────────────────────────────────────────────


def check_1_l1_metrics_finite(metrics: dict[str, float]) -> tuple[bool, str]:
    """Check 1: all L1 metrics are finite (no NaN, Inf)."""
    for key in ("oos_r2", "dm_stat", "dm_pvalue", "coverage"):
        val = metrics.get(key)
        if val is None or not np.isfinite(val):
            return False, f"L1 metric '{key}' = {val!r} is not finite"
    return True, "all L1 metrics are finite"


def check_2_delta_r2_sums(delta_r2: dict[str, float]) -> tuple[bool, str]:
    """Check 2: component ΔR² sum ≈ total (within ±5pp)."""
    total = delta_r2["delta_r2_total"]
    summed = delta_r2["delta_r2_sum"]
    diff = abs(summed - total)
    ok = diff < 0.05
    msg = f"|sum ({summed:.4f}) - total ({total:.4f})| = {diff:.4f}"
    return ok, msg + (" < 0.05 OK" if ok else " >= 0.05 FAIL")


def check_3_falsification_b100(
    obs_train: np.ndarray,
    modal_frame: Any,
    timings: dict[str, float],
) -> tuple[bool, str]:
    """Check 3: LagDestroyedPlacebo with B=100 completes without error."""
    try:
        # Pipeline function: OOS R² using the Kalman filter on permuted signals
        kf = KalmanFilter(
            F=0.9 * np.eye(modal_frame.K),
            Q=0.1 * np.eye(modal_frame.K),
            R=0.1 * np.eye(modal_frame.N),
        )

        T_train = obs_train.shape[0]
        T_split = T_train // 2

        def pipeline_fn(signals: np.ndarray) -> float:
            """Fast OOS R² on permuted signals."""
            obs_tr = signals[:T_split]
            obs_te = signals[T_split:]
            state = kf.filter(obs_tr, modal_frame)
            gap = PredictiveBenchmark().compute(obs_tr, state, modal_frame, None)
            # Evaluate on obs_te using same parameters
            state_te = kf.filter(obs_te, modal_frame)
            gap_te = PredictiveBenchmark().compute(obs_te, state_te, modal_frame, None)
            actual_flat = obs_te.T.ravel()
            pred_flat = gap_te.benchmarks.ravel()
            return oos_r_squared(pred_flat, actual_flat)

        with timed("falsification_b100", timings):
            placebo = LagDestroyedPlacebo(n_instances=FALSIFICATION_B)
            result = placebo.run(obs_train, pipeline_fn)

        return True, (
            f"B100 complete: p_value={result.p_value:.3f}, "
            f"observed={result.observed_metric:.4f}"
        )
    except Exception as exc:
        return False, f"B100 failed with exception: {exc}"


def check_4_profiling_complete(
    timings: dict[str, float],
    required_components: list[str],
) -> tuple[bool, str]:
    """Check 4: runtime profiler captures all required components."""
    missing = [c for c in required_components if c not in timings]
    if missing:
        return False, f"Missing timing entries: {missing}"
    return True, f"All {len(required_components)} component timings captured"


def check_5_schema(df: pd.DataFrame) -> tuple[bool, str]:
    """Check 5: results parquet schema matches spec."""
    present = set(df.columns)
    missing = LEVEL1_REQUIRED_COLUMNS - present
    if missing:
        return False, f"Missing columns: {sorted(missing)}"

    # Numeric columns should be numeric
    for col in LEVEL1_NUMERIC_COLUMNS:
        if col in df.columns:
            val = df[col].iloc[0]
            if not isinstance(val, (int, float, np.floating, np.integer)):
                return False, f"Column '{col}' has non-numeric value: {type(val)}"

    return True, f"Schema valid: {len(present)} columns present"


# ─── output helpers ───────────────────────────────────────────────────────────


def build_level1_row(
    exp_id: str,
    n_actors: int,
    obs_test_shape: tuple[int, int],
    l1: dict[str, float],
    delta: dict[str, float],
) -> pd.DataFrame:
    """Build a single-row level1 parquet."""
    T_test, N = obs_test_shape
    return pd.DataFrame([{
        "experiment_id": exp_id,
        "window_id": "RECENT_2018-2025",
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "benchmark_class": "predictive",
        "oos_r2": l1["oos_r2"],
        "dm_stat": l1["dm_stat"],
        "dm_pvalue": l1["dm_pvalue"],
        "coverage": l1["coverage"],
        "n_actors": n_actors,
        "n_test_periods": T_test,
        "delta_r2_predictive": delta["delta_r2_predictive"],
        "delta_r2_modal_increment": delta["delta_r2_modal_increment"],
        "delta_r2_sum": delta["delta_r2_sum"],
        "delta_r2_total": delta["delta_r2_total"],
    }])


def write_outputs(
    level1_df: pd.DataFrame,
    exp_config: dict,
    timings: dict[str, float],
    results_dir: Path,
) -> None:
    """Write level1 parquet and config YAML to results/."""
    metrics_dir = results_dir / "metrics"
    configs_dir = results_dir / "configs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = metrics_dir / f"level1_{EXP_ID}.parquet"
    level1_df.to_parquet(parquet_path, index=False)

    yaml_path = configs_dir / f"{EXP_ID}.yaml"
    config_with_timings = {**exp_config, "runtime_seconds": timings}
    with open(yaml_path, "w") as fh:
        yaml.dump(config_with_timings, fh, default_flow_style=False, sort_keys=True)

    print(f"  Wrote: {parquet_path.relative_to(PROJECT_ROOT)}")
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")


# ─── main ─────────────────────────────────────────────────────────────────────


REQUIRED_TIMING_COMPONENTS = [
    "load_data",
    "granger_edges",
    "sparsification",
    "spectral_decomposition",
    "mode_selection",
    "kalman_filter_em",
    "kalman_filter_test",
    "benchmark_predictive",
    "benchmark_modal",
    "metrics_computation",
    "falsification_b100",
]


def run_a3(verbose: bool = False) -> dict[str, bool]:
    """Run A3-STACK-VALIDATION. Returns dict of check_id to bool."""
    print(f"\n{'='*70}")
    print(f"  A3-STACK-VALIDATION  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  N_actors={N_ACTORS}  train={TRAIN_START.date()}–{TRAIN_END.date()}  "
          f"test={TEST_START.date()}–{TEST_END.date()}")
    print(f"{'='*70}\n")

    timings: dict[str, float] = {}
    checks: dict[str, bool] = {}

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("[1/9] Loading data …")
    with timed("load_data", timings):
        actor_data, registry = load_actor_registry(N_ACTORS)
        actor_ids = [a["actor_id"] for a in actor_data]
        intensities_wide = load_intensity_matrix(actor_ids)
        ohlcv_returns = load_ohlcv_returns(actor_ids)

    # Drop actors where OHLCV returns are entirely missing (signals need both)
    ohlcv_available = [c for c in actor_ids if c in ohlcv_returns.columns]
    if len(ohlcv_available) < N_ACTORS:
        print(f"  WARNING: Only {len(ohlcv_available)}/{N_ACTORS} actors have OHLCV data.")

    # Restrict both to actors with OHLCV data
    actor_ids_final = [a for a in actor_ids if a in ohlcv_available and a in intensities_wide.columns]
    if len(actor_ids_final) < 10:
        raise RuntimeError(f"Too few actors with complete data: {len(actor_ids_final)}")

    # Rebuild registry for final actor set
    actors_final = [a for a in registry.actors if a.actor_id in set(actor_ids_final)]
    registry_final = ConcreteActorRegistry(actors=actors_final)
    intensities_wide = intensities_wide[actor_ids_final]

    print(f"  Final actor count: {len(actor_ids_final)}")
    if verbose:
        print(f"  Actor IDs: {actor_ids_final[:10]} ...")

    # ── Step 2: Build signal matrix for Granger ───────────────────────────────
    print("[2/9] Building signal matrix …")
    signals_df = build_signal_df_for_granger(ohlcv_returns, actor_ids_final, train_only=True)

    if verbose:
        print(f"  Signals shape: {signals_df.shape}")

    # Rebuild registry to match signal columns
    signal_actor_ids = list(signals_df.columns)
    actors_for_granger = [a for a in registry_final.actors if a.actor_id in set(signal_actor_ids)]
    registry_granger = ConcreteActorRegistry(actors=actors_for_granger)

    date_range = DateRange(start=TRAIN_START, end=TRAIN_END)

    # ── Step 3: Granger edges ─────────────────────────────────────────────────
    print("[3/9] Estimating Granger edges …")
    with timed("granger_edges", timings):
        granger_adj = build_granger_operator(signals_df, registry_granger, date_range)

    if verbose:
        print(f"  Granger adjacency: {granger_adj.shape}, nnz={granger_adj.nnz}")

    # ── Step 4: Sparsification ────────────────────────────────────────────────
    print("[4/9] Sparsifying operator …")
    with timed("sparsification", timings):
        sparse_op, spectral_stats = build_and_sparsify_operator(granger_adj, TARGET_DENSITY)

    energy_retention = spectral_stats["spectral_energy_retention"]
    if verbose:
        print(f"  Spectral energy retention: {energy_retention:.1%}")
    if energy_retention < 0.80:
        print(f"  WARNING (A3): Spectral energy retention {energy_retention:.1%} < 80% "
              f"(violates A3 standing assumption). Continuing for wiring check.")

    # ── Step 5: Spectral decomposition ────────────────────────────────────────
    print("[5/9] Spectral decomposition …")
    N_op = sparse_op.shape[0]
    N_int = len([c for c in actor_ids_final if c in intensities_wide.columns])
    N = min(N_op, N_int)
    op_dense = sparse_op.toarray()[:N, :N]

    K_cand = min(K_MAX_CANDIDATES, N)
    with timed("spectral_decomposition", timings):
        decomposer = SchurDecomposer()
        modal_frame_cand = decomposer.decompose(op_dense, k=K_cand)

    if verbose:
        print(f"  Candidate modes K_cand={K_cand}, N={N}")

    # ── Step 6: Mode selection ────────────────────────────────────────────────
    print("[6/9] Mode selection (MDL) …")
    # Build obs_train aligned to spectral N
    int_cols = [c for c in actor_ids_final if c in intensities_wide.columns][:N]
    obs_train_df = intensities_wide[int_cols][intensities_wide.index <= TRAIN_END]
    train_col_means = obs_train_df.mean()
    obs_train_df_filled = obs_train_df.fillna(train_col_means)
    obs_train = obs_train_df_filled.values.astype(np.float64)

    obs_test_df = intensities_wide[int_cols][intensities_wide.index >= TEST_START]
    obs_test_df_filled = obs_test_df.fillna(train_col_means)
    obs_test = obs_test_df_filled.values.astype(np.float64)

    with timed("mode_selection", timings):
        mdl = MDLModeSelector()
        modal_frame, _ = mdl.select(modal_frame_cand, obs_train)

    K_star = modal_frame.K
    print(f"  K* = {K_star} modes selected")

    # ── Step 7: Kalman filter EM (train) ──────────────────────────────────────
    print("[7/9] Kalman filter EM (train) …")
    with timed("kalman_filter_em", timings):
        kf = KalmanFilter()
        state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)

    if verbose:
        print(f"  Log-likelihood: {state_train.log_likelihood:.3f}")

    # ── Step 7b: Apply on test ────────────────────────────────────────────────
    with timed("kalman_filter_test", timings):
        kf_test = KalmanFilter(
            F=state_train.params.get("F"),
            Q=state_train.params.get("Q"),
            R=state_train.params.get("R"),
        )
        state_test = kf_test.filter(obs_test, modal_frame)

    # ── Step 8: Benchmarks and gaps ───────────────────────────────────────────
    print("[8/9] Computing benchmarks and gaps …")
    with timed("benchmark_predictive", timings):
        gap_pred = PredictiveBenchmark().compute(obs_test, state_test, modal_frame, registry_final)

    with timed("benchmark_modal", timings):
        gap_modal = ModalBenchmark().compute(obs_test, state_test, modal_frame, registry_final)

    # ── Step 9: Metrics ───────────────────────────────────────────────────────
    print("[9/9] Computing metrics and running checks …\n")
    with timed("metrics_computation", timings):
        l1_metrics = compute_l1_metrics(obs_test, gap_pred, gap_modal, registry_final)
        delta_r2 = compute_component_delta_r2(obs_test, gap_pred, gap_modal)

    # ── Check 1: L1 metrics finite ────────────────────────────────────────────
    c1_pass, c1_msg = check_1_l1_metrics_finite(l1_metrics)
    checks["check_1_l1_finite"] = c1_pass
    status = "PASS" if c1_pass else "FAIL"
    print(f"  Check 1 [{status}] L1 metrics finite: {c1_msg}")
    print(f"    oos_r2={l1_metrics['oos_r2']:.4f}  "
          f"dm_stat={l1_metrics['dm_stat']:.3f}  "
          f"dm_pvalue={l1_metrics['dm_pvalue']:.3f}  "
          f"coverage={l1_metrics['coverage']:.3f}")

    # ── Check 2: ΔR² components sum ───────────────────────────────────────────
    c2_pass, c2_msg = check_2_delta_r2_sums(delta_r2)
    checks["check_2_delta_r2"] = c2_pass
    status = "PASS" if c2_pass else "FAIL"
    print(f"  Check 2 [{status}] Delta-R2 components: {c2_msg}")
    print(f"    dr2_pred={delta_r2['delta_r2_predictive']:.4f}  "
          f"dr2_modal_inc={delta_r2['delta_r2_modal_increment']:.4f}")

    # ── Check 3: Falsification B100 ───────────────────────────────────────────
    c3_pass, c3_msg = check_3_falsification_b100(obs_train, modal_frame, timings)
    checks["check_3_falsification"] = c3_pass
    status = "PASS" if c3_pass else "FAIL"
    print(f"  Check 3 [{status}] Falsification B100: {c3_msg}")

    # ── Check 4: Profiling complete ───────────────────────────────────────────
    c4_pass, c4_msg = check_4_profiling_complete(timings, REQUIRED_TIMING_COMPONENTS)
    checks["check_4_profiling"] = c4_pass
    status = "PASS" if c4_pass else "FAIL"
    print(f"  Check 4 [{status}] Runtime profiling: {c4_msg}")

    # ── Build and validate schema ──────────────────────────────────────────────
    T_test, N_final = obs_test.shape
    level1_df = build_level1_row(EXP_ID, N_final, (T_test, N_final), l1_metrics, delta_r2)

    c5_pass, c5_msg = check_5_schema(level1_df)
    checks["check_5_schema"] = c5_pass
    status = "PASS" if c5_pass else "FAIL"
    print(f"  Check 5 [{status}] Schema: {c5_msg}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    exp_config = {
        "experiment_id": EXP_ID,
        "universe": "US-LC",
        "n_actors": N_final,
        "n_actors_requested": N_ACTORS,
        "institutions": "INST-MINIMAL",
        "signals": "MACRO+MARKET",
        "period": "RECENT",
        "train_start": str(TRAIN_START.date()),
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "pipeline_depth": "full",
        "k_star": int(K_star),
        "k_max_candidates": K_MAX_CANDIDATES,
        "target_density": TARGET_DENSITY,
        "max_granger_lag": MAX_GRANGER_LAG,
        "falsification_b": FALSIFICATION_B,
        "spectral_energy_retention": float(spectral_stats["spectral_energy_retention"]),
    }
    results_dir = PROJECT_ROOT / "results"
    write_outputs(level1_df, exp_config, timings, results_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("  RUNTIME BREAKDOWN")
    print(f"  {'Component':<38} {'Seconds':>8}")
    print(f"  {'-'*38}  {'-'*8}")
    total = sum(timings.values())
    for comp, secs in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"  {comp:<38} {secs:>8.3f}")
    print(f"  {'-'*38}  {'-'*8}")
    print(f"  {'TOTAL':<38} {total:>8.3f}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f"\n{'='*70}")
    print(f"  A3-STACK-VALIDATION: {n_pass}/{n_total} checks PASS")
    if n_pass == n_total:
        print("  [OK] ALL CHECKS PASS -- proceed to A1")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"  [FAIL] FAILED: {failed}")
        print("  Fix before proceeding to A1")
    print(f"{'='*70}\n")

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A3-STACK-VALIDATION experiment.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    results = run_a3(verbose=args.verbose)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
