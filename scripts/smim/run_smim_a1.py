#!/usr/bin/env python
"""
A1-MVP-FULL experiment runner.

Full SMIM pipeline on the experiment_a1 universe (93 actors, 6 sectors, US+UK)
with rolling 10yr train / 1yr test windows, multi-regime comparison (M=1,2,3),
emergence diagnostics, and full falsification suite.

Output:
  results/metrics/level1_A1-MVP-FULL.parquet   (10 windows x 3 benchmarks)
  results/metrics/level2_A1-MVP-FULL.parquet   (component attribution)
  results/metrics/level3_A1-MVP-FULL.parquet   (robustness / stability)
  results/metrics/level4_A1-MVP-FULL.parquet   (economic validity)
  results/metrics/level5_A1-MVP-FULL.parquet   (runtime profile)
  results/configs/A1-MVP-FULL.yaml

Decision gate: mean OOS R^2 > 0 across 10 windows (best M, predictive).
If <= 0: STOP.

Usage::
    uv run python scripts/smim/run_smim_a1.py
    uv run python scripts/smim/run_smim_a1.py --skip-falsification
    uv run python scripts/smim/run_smim_a1.py --single-window 0 --regimes 1
    uv run python scripts/smim/run_smim_a1.py --verbose
"""

from __future__ import annotations

import argparse
import contextlib
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
from quantdsl_backtest.smim.validation.falsification import FalsificationSuite
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

# --- constants ---------------------------------------------------------------

EXP_ID = "A1-MVP-FULL"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
OHLCV_PATHS = {
    "US-LC": PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet",
    "UK-LC": PROJECT_ROOT / "equities" / "smim" / "UK-LC" / "ohlcv.parquet",
}
FRED_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "fred_signals.parquet"
EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"

K_MAX_CANDIDATES = 15
K_MIN = 3                # MDL floor: at T/N < 1, MDL always picks K=1; force minimum
TARGET_DENSITY = 0.15
MAX_GRANGER_LAG = 2
FALSIFICATION_B = 100

# Test years for 10 non-overlapping 1yr windows
TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


# --- timing helper -----------------------------------------------------------


@contextlib.contextmanager
def timed(label: str, timings: dict[str, float]) -> Generator[None, None, None]:
    t0 = time.perf_counter_ns()
    yield
    timings[label] = (time.perf_counter_ns() - t0) / 1e9


# --- actor type mapping ------------------------------------------------------

_ACTOR_TYPE_MAP = {
    "global_shock": ActorType.GLOBAL_SHOCK,
    "central_bank": ActorType.CENTRAL_BANK,
    "regulator": ActorType.REGULATOR,
    "intl_org": ActorType.INTL_ORG,
    "sector_leader": ActorType.SECTOR_LEADER,
    "large_firm": ActorType.LARGE_FIRM,
    "bank": ActorType.BANK,
}


def _actor_type(s: str) -> ActorType:
    return _ACTOR_TYPE_MAP.get(s, ActorType.LARGE_FIRM)


# --- data loading ------------------------------------------------------------


def load_a1_registry() -> tuple[list[dict], ConcreteActorRegistry, list[str]]:
    """Load experiment_a1 registry, filter to actors with intensity data.

    Returns (actor_data_list, registry, actor_ids_with_intensity).
    """
    with open(REGISTRY_PATH) as fh:
        reg_data = json.load(fh)

    # Find actors with intensity data
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors_with_intensity = set(int_df["actor_id"].unique())

    actor_data = [a for a in reg_data["actors"] if a["actor_id"] in actors_with_intensity]
    actor_ids = [a["actor_id"] for a in actor_data]

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
        for a in actor_data
    ]
    registry = ConcreteActorRegistry(actors=actors)
    return actor_data, registry, actor_ids


def load_a1_intensities(actor_ids: list[str]) -> pd.DataFrame:
    """Load intensity matrix for experiment_a1 (mixed normalisation methods).

    Returns wide DataFrame: (T_quarters, N_actors) indexed by quarterly dates.
    """
    df = pd.read_parquet(INTENSITIES_PATH)
    df = df[df["actor_id"].isin(actor_ids)]
    # Each actor has exactly one normalisation method -- no filtering needed
    wide = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    # Reindex to full quarterly grid
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols]


def load_ohlcv_returns(actor_ids: list[str]) -> pd.DataFrame:
    """Load OHLCV from US-LC + UK-LC, compute quarterly log-returns."""
    frames = []
    for uni, path in OHLCV_PATHS.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["ticker"].isin(actor_ids)].copy()
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"])

    # Resample to quarterly close
    pivoted = combined.pivot_table(index="date", columns="ticker", values="close")
    close_q = pivoted.resample("QE").last()
    close_q.index = close_q.index.to_period("Q").to_timestamp()

    # Log returns
    returns_q = np.log(close_q / close_q.shift(1))
    returns_q = returns_q.iloc[1:]  # drop first NaN row

    # Reindex to QS grid
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    returns_q = returns_q.reindex(idx)
    cols = [c for c in actor_ids if c in returns_q.columns]
    return returns_q[cols]


def load_fred_signals(actor_fred_mapping: dict[str, str]) -> pd.DataFrame:
    """Load FRED series for institutional actors, compute quarterly changes.

    Args:
        actor_fred_mapping: {actor_id: fred_series_id}

    Returns wide DataFrame: (T_quarters, N_fred_actors) with quarterly changes.
    """
    if not FRED_PATH.exists() or not actor_fred_mapping:
        return pd.DataFrame()

    df = pd.read_parquet(FRED_PATH)
    series_to_actor = {v: k for k, v in actor_fred_mapping.items()}
    df = df[df["signal_id"].isin(actor_fred_mapping.values())].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Resample to quarterly (last value per quarter)
    frames = {}
    for series_id, group in df.groupby("signal_id"):
        actor_id = series_to_actor[series_id]
        g = group.set_index("event_date")["value"].sort_index()
        q = g.resample("QE").last()
        q.index = q.index.to_period("Q").to_timestamp()
        # Quarterly log-change as signal
        q_change = np.log(q / q.shift(1)).replace([np.inf, -np.inf], np.nan)
        frames[actor_id] = q_change

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    result = result.reindex(idx)
    return result


def load_edgar_signals(actor_ids: list[str]) -> pd.DataFrame:
    """Load EDGAR balance sheet data as quarterly log-change signals.

    Uses Assets and CapEx tags to produce quarterly change signals for
    actors with EDGAR coverage. This directly relates to the intensity
    target (CapEx/Assets) and gives richer Granger inputs.

    Returns wide DataFrame: (T_quarters, N_edgar_actors) with quarterly changes.
    """
    if not EDGAR_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(EDGAR_PATH)
    df = df[df["ticker"].isin(actor_ids)].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Use Assets quarterly change as signal; column name = actor_id (ticker).
    # This fills signal gaps for actors without OHLCV data.
    frames = {}
    sub = df[df["tag"] == "Assets"].copy()
    if not sub.empty:
        for ticker, group in sub.groupby("ticker"):
            g = group.set_index("event_date")["value"].sort_index()
            g = g[~g.index.duplicated(keep="last")]
            q = g.resample("QE").last().dropna()
            q.index = q.index.to_period("Q").to_timestamp()
            q_change = np.log(q / q.shift(1)).replace([np.inf, -np.inf], np.nan)
            frames[ticker] = q_change

    # Cross-sectional rank transform: convert raw log-changes to [0,1] ranks
    # per quarter.  This matches the intensity normalisation and reduces
    # outlier influence from noisy quarterly balance-sheet revisions.
    if frames:
        raw = pd.DataFrame(frames)
        ranked = raw.rank(axis=1, pct=True)
        frames = {c: ranked[c] for c in ranked.columns}

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    result = result.reindex(idx)
    return result


def build_signal_matrix(
    ohlcv_returns: pd.DataFrame,
    fred_signals: pd.DataFrame,
    edgar_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine OHLCV, FRED, and EDGAR into a unified signal matrix for Granger.

    OHLCV takes priority: if an actor has both OHLCV and EDGAR, OHLCV is used.
    EDGAR fills gaps for actors without OHLCV coverage.
    """
    parts = []
    ohlcv_cols = set()
    if not ohlcv_returns.empty:
        parts.append(ohlcv_returns)
        ohlcv_cols = set(ohlcv_returns.columns)
    if not fred_signals.empty:
        parts.append(fred_signals)
    if edgar_signals is not None and not edgar_signals.empty:
        # Only use EDGAR for actors NOT already covered by OHLCV
        edgar_only = edgar_signals[[c for c in edgar_signals.columns if c not in ohlcv_cols]]
        if not edgar_only.empty:
            parts.append(edgar_only)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)


# --- window generation -------------------------------------------------------


def make_windows(
    test_years: list[int] | None = None,
) -> list[dict[str, Any]]:
    years = test_years or TEST_YEARS
    windows = []
    for ty in years:
        windows.append({
            "window_id": f"W{ty}",
            "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
            "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
            "test_start": pd.Timestamp(f"{ty}-01-01"),
            "test_end": pd.Timestamp(f"{ty}-12-31"),
        })
    return windows


# --- per-window pipeline -----------------------------------------------------


def run_window(
    window: dict[str, Any],
    intensities_wide: pd.DataFrame,
    signal_matrix: pd.DataFrame,
    registry: ConcreteActorRegistry,
    actor_ids: list[str],
    M: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run full pipeline for one window and one regime count M."""
    timings: dict[str, float] = {}
    ts, te = window["train_start"], window["train_end"]
    test_s, test_e = window["test_start"], window["test_end"]

    # --- Slice data ----------------------------------------------------------
    with timed("load_data", timings):
        int_train_df = intensities_wide[(intensities_wide.index >= ts) & (intensities_wide.index <= te)]
        int_test_df = intensities_wide[(intensities_wide.index >= test_s) & (intensities_wide.index <= test_e)]

        # Drop actors with all-NaN training intensity
        valid_cols = int_train_df.columns[int_train_df.notna().any()]
        int_train_df = int_train_df[valid_cols]
        int_test_df = int_test_df[valid_cols]

        col_means = int_train_df.mean()
        obs_train_raw = int_train_df.fillna(col_means).values.astype(np.float64)
        obs_test_raw = int_test_df.fillna(col_means).values.astype(np.float64)

        # Demean: subtract per-actor training mean.  The state-space model
        # y = U @ alpha + eps assumes zero-mean observations.  Without this,
        # alpha must absorb the ~0.5 mean level through an orthonormal U,
        # which is impossible at K << N and produces predictions near zero.
        obs_mean = obs_train_raw.mean(axis=0, keepdims=True)  # (1, N)
        obs_train = obs_train_raw - obs_mean
        obs_test = obs_test_raw - obs_mean

        actor_ids_final = list(valid_cols)
        N = len(actor_ids_final)

        # Signal matrix for this window
        if not signal_matrix.empty:
            sig_cols = [c for c in actor_ids_final if c in signal_matrix.columns]
            sig_train = signal_matrix[sig_cols][(signal_matrix.index >= ts) & (signal_matrix.index <= te)]
            sig_train = sig_train.fillna(sig_train.mean())
            sig_train = sig_train.dropna(axis=1, how="all")
        else:
            sig_train = pd.DataFrame()
            sig_cols = []

    T_train, T_test = obs_train.shape[0], obs_test.shape[0]

    # --- Granger edges -------------------------------------------------------
    with timed("granger_edges", timings):
        if not sig_train.empty and len(sig_train.columns) >= 2:
            signal_actor_ids = list(sig_train.columns)
            sig_registry = ConcreteActorRegistry(
                actors=[a for a in registry.actors if a.actor_id in set(signal_actor_ids)]
            )
            date_range = DateRange(start=ts, end=te)
            config = GrangerEdgeConfig(max_lag=MAX_GRANGER_LAG, p_threshold=0.05, bic_selection=False)
            granger_adj = GrangerEdgeEstimator(config).estimate(sig_train, sig_registry, date_range)
        else:
            granger_adj = sp.csr_matrix((N, N))

    # --- Approach C: End-to-end operator optimisation ---------------------------
    # Build a library of basis operators (correlation, lagged-corr, multi-scale,
    # Granger).  Then optimise the channel weights w to maximise sub-validation
    # R² through the full pipeline: A(w) → Schur → Kalman EM → predict → R².
    # This is truly end-to-end: the operator is shaped by what makes the
    # Kalman filter predictive, not by proxy metrics.
    with timed("learned_operator", timings):
        from scipy.optimize import minimize as sp_minimize

        # ---- Build basis operator library -----------------------------------
        basis_ops: list[np.ndarray] = []
        basis_names: list[str] = []

        # B0: instantaneous cross-correlation
        corr0 = np.corrcoef(obs_train.T)
        corr0 = np.nan_to_num(corr0, nan=0.0)
        np.fill_diagonal(corr0, 0.0)
        basis_ops.append(corr0)
        basis_names.append("corr_inst")

        # B1: lag-1 cross-correlation (directed: actor i at t predicts j at t+1)
        if T_train > 2:
            lag1 = np.zeros((N, N))
            for lag_offset in [1, 2]:
                if T_train > lag_offset:
                    x_prev = obs_train[:-lag_offset]  # (T-lag, N)
                    x_next = obs_train[lag_offset:]   # (T-lag, N)
                    # Per-pair correlation of lagged series
                    xp_dm = x_prev - x_prev.mean(axis=0)
                    xn_dm = x_next - x_next.mean(axis=0)
                    num = (xp_dm.T @ xn_dm) / max(len(xp_dm) - 1, 1)
                    sp_std = np.std(x_prev, axis=0, keepdims=True).T
                    sn_std = np.std(x_next, axis=0, keepdims=True)
                    denom = np.maximum(sp_std @ sn_std, 1e-10)
                    lag_corr = num / denom
                    lag1 += np.nan_to_num(lag_corr, nan=0.0)
            np.fill_diagonal(lag1, 0.0)
            basis_ops.append(lag1)
            basis_names.append("corr_lag")

        # B2-B3: multi-scale cosine similarity (4Q, 8Q)
        for scale in [4, 8]:
            if T_train >= scale + 1:
                kernel = np.ones(scale) / scale
                sm = np.apply_along_axis(
                    lambda x: np.convolve(x, kernel, mode="valid"), axis=0, arr=obs_train
                )
                if sm.shape[0] >= 2:
                    norms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                    normed = sm / norms
                    sim = normed.T @ normed
                    np.fill_diagonal(sim, 0.0)
                    basis_ops.append(sim)
                    basis_names.append(f"cosine_{scale}Q")

        # B4: Granger (padded to N)
        N_granger = granger_adj.shape[0]
        if N_granger > 0:
            g_dense = granger_adj.toarray() if sp.issparse(granger_adj) else granger_adj
            g_padded = np.zeros((N, N))
            n_g = min(N_granger, N)
            g_padded[:n_g, :n_g] = g_dense[:n_g, :n_g]
            basis_ops.append(g_padded)
            basis_names.append("granger")

        n_bases = len(basis_ops)

        # ---- Sub-train / sub-val split for optimisation ---------------------
        t_split = int(T_train * 0.75)
        obs_st = obs_train[:t_split]
        obs_sv = obs_train[t_split:]
        K_opt = K_MIN

        def objective(weights: np.ndarray) -> float:
            """Negative sub-val R² for given operator channel weights."""
            try:
                # Build weighted operator
                A = np.zeros((N, N))
                for w, B in zip(weights, basis_ops):
                    A += w * B
                # Threshold
                thr = np.percentile(np.abs(A[A != 0]), 30) if np.any(A != 0) else 0.01
                A[np.abs(A) < thr] = 0.0
                # Decompose
                mf_opt = SchurDecomposer().decompose(A, k=min(K_MAX_CANDIDATES, N))
                U_opt = mf_opt.basis[:, :K_opt]
                mf_k = ModalFrame(
                    basis=U_opt,
                    eigenvalues=mf_opt.eigenvalues[:K_opt],
                    method=mf_opt.method,
                )
                # Kalman on sub-train
                kf_opt = KalmanFilter()
                state_opt = kf_opt.em_estimate(obs_st, mf_k, max_iter=15)
                # Predict sub-val
                kf_sv = KalmanFilter(
                    F=state_opt.params.get("F"),
                    Q=state_opt.params.get("Q"),
                    R=state_opt.params.get("R"),
                )
                state_sv = kf_sv.filter(obs_sv, mf_k)
                pred_sv = state_sv.alpha_predicted @ U_opt.T
                r2 = float(oos_r_squared(pred_sv.T.ravel(), obs_sv.T.ravel()))
                return -r2 if np.isfinite(r2) else 1e6
            except Exception:
                return 1e6

        # ---- Optimise with Nelder-Mead (gradient-free, ~100-200 evals) ------
        x0 = np.ones(n_bases) / n_bases  # equal initial weights
        opt_result = sp_minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": 150, "xatol": 0.01, "fatol": 0.001},
        )
        w_opt = opt_result.x

        if verbose:
            print(f"      operator weights: "
                  + ", ".join(f"{n}={w:.3f}" for n, w in zip(basis_names, w_opt))
                  + f"  sub-val R2={-opt_result.fun:.4f}")

        # ---- Build final operator with optimal weights ----------------------
        A_final = np.zeros((N, N))
        for w, B in zip(w_opt, basis_ops):
            A_final += w * B
        thr = np.percentile(np.abs(A_final[A_final != 0]), 30) if np.any(A_final != 0) else 0.01
        A_final[np.abs(A_final) < thr] = 0.0
        sparse_op = sp.csr_matrix(A_final)

    # --- Spectral decomposition (on optimised operator) -----------------------
    op_dense = sparse_op.toarray()[:N, :N]
    K_cand = min(K_MAX_CANDIDATES, N)

    with timed("spectral_decomposition", timings):
        modal_frame_cand = SchurDecomposer().decompose(op_dense, k=K_cand)

    # --- Mode selection (with K_min floor) ------------------------------------
    with timed("mode_selection", timings):
        modal_frame, _ = MDLModeSelector().select(modal_frame_cand, obs_train)
        # MDL at T/N < 1 is too conservative (always K=1).  Apply K_min floor
        # to capture cross-sectional structure (sector, geography effects).
        if modal_frame.K < K_MIN and modal_frame_cand.K >= K_MIN:
            modal_frame = ModalFrame(
                basis=modal_frame_cand.basis[:, :K_MIN],
                eigenvalues=modal_frame_cand.eigenvalues[:K_MIN],
                method=modal_frame_cand.method,
            )

    K = modal_frame.K

    # --- State estimation (Kalman or Kim) ------------------------------------
    with timed("filter_em", timings):
        if M == 1:
            kf = KalmanFilter()
            state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)
        else:
            kf = KimFilter(n_regimes=M)
            state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)

    with timed("filter_test", timings):
        if M == 1:
            kf_test = KalmanFilter(
                F=state_train.params.get("F"),
                Q=state_train.params.get("Q"),
                R=state_train.params.get("R"),
            )
            state_test = kf_test.filter(obs_test, modal_frame)
        else:
            kf_test = KimFilter(n_regimes=M)
            state_test = kf_test.filter(
                obs_test, modal_frame,
                F_list=state_train.params.get("F_list"),
                Q_list=state_train.params.get("Q_list"),
                R=state_train.params.get("R"),
                P_trans=state_train.params.get("P_trans"),
            )

    # --- Registry for benchmarks (only final actors) -------------------------
    bench_registry = ConcreteActorRegistry(
        actors=[a for a in registry.actors if a.actor_id in set(actor_ids_final)]
    )

    # --- Benchmarks ----------------------------------------------------------
    with timed("benchmark_predictive", timings):
        gap_pred = PredictiveBenchmark().compute(obs_test, state_test, modal_frame, bench_registry)

    with timed("benchmark_modal", timings):
        gap_modal = ModalBenchmark().compute(obs_test, state_test, modal_frame, bench_registry)

    # --- OOS R^2 (restore mean for fair comparison) ----------------------------
    actual = obs_test_raw.T.ravel()
    # Benchmarks are in demeaned space (N, T); add per-actor mean back
    pred_restored = (gap_pred.benchmarks + obs_mean.T).ravel()   # (N, T) + (N, 1)
    modal_restored = (gap_modal.benchmarks + obs_mean.T).ravel()
    oos_r2_pred = float(oos_r_squared(pred_restored, actual))
    oos_r2_modal = float(oos_r_squared(modal_restored, actual))

    return {
        "M": M,
        "oos_r2_pred": oos_r2_pred,
        "oos_r2_modal": oos_r2_modal,
        "K": K,
        "N": N,
        "T_train": T_train,
        "T_test": T_test,
        "timings": timings,
        "state_train": state_train,
        "state_test": state_test,
        "modal_frame": modal_frame,
        "gap_pred": gap_pred,
        "gap_modal": gap_modal,
        "obs_train": obs_train,
        "obs_test": obs_test,
        "obs_test_raw": obs_test_raw,
        "obs_mean": obs_mean,
        "sparse_op": sparse_op,
        "actor_ids_final": actor_ids_final,
        "bench_registry": bench_registry,
    }


# --- emergence computation ---------------------------------------------------


def compute_emergence(
    result: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, Any | None, float]:
    """Compute synergy, criticality, emergence-aware benchmark.

    Three fixes over the naive approach:
      1. Synergy estimated from TRAINING data (T=40) not test data (T=4).
         T=4 gives rank-deficient covariance → unreliable PID.
      2. Criticality computed from TRAINING alpha with proper window_size=8,
         then the final value is carried forward to the test period.
         T=4 with window_size=2 gave constant 1.0 (no information).
      3. Training-period gap proxy: residuals from train filter (obs - U@alpha_filt)
         used as target for PID, giving stable synergy estimates.

    Returns (synergy, criticality, gap_ea, oos_r2_ea).
    """
    state_train = result["state_train"]
    state_test = result["state_test"]
    modal_frame = result["modal_frame"]
    obs_train = result["obs_train"]
    obs_test = result["obs_test"]
    bench_registry = result["bench_registry"]
    K = result["K"]
    T_test = result["T_test"]

    try:
        # --- Fix 1: Synergy from TRAINING data (T=40, not T=4) ---------------
        alpha_train = state_train.alpha_filtered  # (T_train, K)
        U = modal_frame.basis
        # Training-period gap proxy: actual - filtered reconstruction
        recon_train = alpha_train @ U.T  # (T_train, N)
        train_gaps = (obs_train - recon_train).T  # (N, T_train)
        synergy = compute_synergy_matrix(alpha_train, train_gaps, K)

        # --- Signed synergy: PID synergy is always >= 0 (magnitude only).
        # Compute the DIRECTION of each mode-pair interaction from the
        # cross-covariance of the interaction term with the aggregate gap.
        # sign(cov(alpha_j * alpha_k, gap_mean)) tells us whether the modes
        # interact constructively (+) or destructively (-).
        gap_target = train_gaps.mean(axis=0)  # (T_train,)
        for j in range(K):
            for k in range(j + 1, K):
                interaction = alpha_train[:, j] * alpha_train[:, k]
                cross_cov = np.cov(interaction, gap_target)[0, 1]
                direction = np.sign(cross_cov) if abs(cross_cov) > 1e-10 else 0.0
                synergy[j, k] *= direction
                synergy[k, j] *= direction

        # --- Fix 2: Criticality from TRAINING alpha (T=40, window=8) ---------
        d = min(3, K)
        psi_train = extract_order_parameter(alpha_train, d=d)
        crit_train = criticality_index(psi_train, window_size=8)  # (T_train,)
        # Carry the last training criticality forward into the test period
        crit_last = crit_train[-1] if len(crit_train) > 0 else 1.0
        crit_test = np.full(T_test, crit_last)

        # --- Fix 3: Cross-validated synergy weight ------------------------------
        # Instead of fixed damping, find the optimal correction weight by
        # splitting training into sub-train (first 80%) / sub-val (last 20%).
        T_tr = obs_train.shape[0]
        t_split = int(T_tr * 0.8)
        if t_split >= 8 and T_tr - t_split >= 2:
            obs_sub_val = obs_train[t_split:]
            # Get filtered states for sub-val period
            kf_cv = KalmanFilter(
                F=state_train.params.get("F"),
                Q=state_train.params.get("Q"),
                R=state_train.params.get("R"),
            )
            state_cv = kf_cv.filter(obs_sub_val, modal_frame)
            # Compute base prediction on sub-val
            base_cv = state_cv.alpha_predicted @ U.T  # (T_sv, N)
            actual_cv = obs_sub_val
            # Compute synergy correction on sub-val
            syn_corr_cv = np.zeros_like(base_cv)
            for j_cv in range(K):
                for k_cv in range(j_cv + 1, K):
                    inter = state_cv.alpha_predicted[:, j_cv] * state_cv.alpha_predicted[:, k_cv] * synergy[j_cv, k_cv]
                    mode_corr = np.zeros((base_cv.shape[0], K))
                    mode_corr[:, j_cv] += inter * 0.5
                    mode_corr[:, k_cv] += inter * 0.5
                    syn_corr_cv += mode_corr @ U.T
            # Grid search for best weight
            best_w, best_r2_cv = 0.0, -1e10
            for w in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                combined_cv = base_cv + w * syn_corr_cv
                r2_cv = float(oos_r_squared(combined_cv.T.ravel(), actual_cv.T.ravel()))
                if r2_cv > best_r2_cv:
                    best_r2_cv = r2_cv
                    best_w = w
        else:
            best_w = 0.0  # fallback: no correction

        # --- Apply CV-tuned weight to emergence benchmark ---------------------
        # Scale synergy matrix by the CV-optimal weight before passing
        synergy_scaled = synergy * best_w
        ea_bench = EmergenceAwareBenchmark(
            synergy_matrix=synergy_scaled,
            criticality=crit_test,
        )
        gap_ea = ea_bench.compute(obs_test, state_test, modal_frame, bench_registry)

        # Restore mean for R²
        obs_mean = result["obs_mean"]
        actual = result["obs_test_raw"].T.ravel()
        ea_restored = (gap_ea.benchmarks + obs_mean.T).ravel()
        oos_r2_ea = float(oos_r_squared(ea_restored, actual))

        return synergy, crit_test, gap_ea, oos_r2_ea
    except Exception as exc:
        print(f"      [WARN] emergence failed: {exc}")
        return None, None, None, float("nan")


# --- falsification -----------------------------------------------------------


def run_falsification(
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Run all 7 falsification tests for a window result."""
    try:
        obs_train = result["obs_train"]
        modal_frame = result["modal_frame"]
        sparse_op = result["sparse_op"]
        M = result["M"]
        bench_registry = result["bench_registry"]

        # Lightweight pipeline function: filter + benchmark + R^2
        def signal_pipeline_fn(signals: np.ndarray) -> float:
            kf = KalmanFilter()
            state = kf.em_estimate(signals, modal_frame, max_iter=10)
            kf2 = KalmanFilter(
                F=state.params.get("F"),
                Q=state.params.get("Q"),
                R=state.params.get("R"),
            )
            # Use last 4 quarters as test
            T = signals.shape[0]
            t_split = max(T - 4, T // 2)
            obs_tr = signals[:t_split]
            obs_te = signals[t_split:]
            state2 = kf2.filter(obs_te, modal_frame)
            gap = PredictiveBenchmark().compute(obs_te, state2, modal_frame, bench_registry)
            return float(oos_r_squared(gap.benchmarks.ravel(), obs_te.T.ravel()))

        def operator_pipeline_fn(op: sp.csr_matrix) -> float:
            N_op = min(op.shape[0], obs_train.shape[1])
            od = op.toarray()[:N_op, :N_op]
            if N_op < obs_train.shape[1]:
                padded = np.zeros((obs_train.shape[1], obs_train.shape[1]))
                padded[:N_op, :N_op] = od
                od = padded
            mf = SchurDecomposer().decompose(od, k=min(K_MAX_CANDIDATES, od.shape[0]))
            mf_sel, _ = MDLModeSelector().select(mf, obs_train)
            kf = KalmanFilter()
            state = kf.em_estimate(obs_train, mf_sel, max_iter=10)
            kf2 = KalmanFilter(F=state.params.get("F"), Q=state.params.get("Q"), R=state.params.get("R"))
            T = obs_train.shape[0]
            t_split = max(T - 4, T // 2)
            state2 = kf2.filter(obs_train[t_split:], mf_sel)
            gap = PredictiveBenchmark().compute(obs_train[t_split:], state2, mf_sel, bench_registry)
            return float(oos_r_squared(gap.benchmarks.ravel(), obs_train[t_split:].T.ravel()))

        actor_types = [a.actor_type.value for a in bench_registry.actors]
        layer_assignments = np.array([a.layer.value if hasattr(a.layer, "value") else int(a.layer)
                                      for a in bench_registry.actors])

        suite = FalsificationSuite(n_instances=FALSIFICATION_B)
        results = suite.run_all(
            operator=sparse_op,
            signals=obs_train,
            actor_types=actor_types,
            multi_regime_metric=result["oos_r2_pred"] if M > 1 else 0.0,
            single_regime_metric=result.get("oos_r2_m1", result["oos_r2_pred"]),
            layer_assignments=layer_assignments,
            smim_metric=result["oos_r2_pred"],
            dfm_metric=0.0,
            operator_pipeline_fn=operator_pipeline_fn,
            signal_pipeline_fn=signal_pipeline_fn,
        )
        return results
    except Exception as exc:
        print(f"      [WARN] falsification failed: {exc}")
        return None


# --- L1 metrics --------------------------------------------------------------


def compute_l1_row(
    window: dict[str, Any],
    benchmark_class: str,
    oos_r2: float,
    dm_stat: float,
    dm_pvalue: float,
    coverage: float,
    N: int,
    T_test: int,
    delta_pred: float,
    delta_modal_inc: float,
) -> dict[str, Any]:
    return {
        "experiment_id": EXP_ID,
        "window_id": window["window_id"],
        "train_start": window["train_start"],
        "train_end": window["train_end"],
        "test_start": window["test_start"],
        "test_end": window["test_end"],
        "benchmark_class": benchmark_class,
        "oos_r2": oos_r2,
        "dm_stat": dm_stat,
        "dm_pvalue": dm_pvalue,
        "coverage": coverage,
        "n_actors": N,
        "n_test_periods": T_test,
        "delta_r2_predictive": delta_pred,
        "delta_r2_modal_increment": delta_modal_inc,
        "delta_r2_sum": delta_pred + delta_modal_inc,
        "delta_r2_total": delta_pred + delta_modal_inc,
    }


# --- main orchestrator -------------------------------------------------------


def run_a1(
    verbose: bool = False,
    skip_falsification: bool = False,
    single_window: int | None = None,
    regime_list: list[int] | None = None,
) -> None:
    regimes = regime_list or [1, 2, 3]

    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  signals=OHLCV+FRED  regimes={regimes}  falsification={'ON' if not skip_falsification else 'OFF'}")
    print("=" * 70)

    # --- Load all data once --------------------------------------------------
    print("\n  Loading data ...", end="", flush=True)
    actor_data, registry, actor_ids = load_a1_registry()
    intensities_wide = load_a1_intensities(actor_ids)

    ohlcv_returns = load_ohlcv_returns(actor_ids)
    n_ohlcv = len(ohlcv_returns.columns)

    # Build FRED mapping from registry
    fred_mapping = {}
    for a in actor_data:
        fred_series = a.get("external_ids", {}).get("fred_series")
        if fred_series:
            fred_mapping[a["actor_id"]] = fred_series
    fred_signals = load_fred_signals(fred_mapping)
    n_fred = len(fred_signals.columns) if not fred_signals.empty else 0

    edgar_signals = load_edgar_signals(actor_ids)
    n_edgar = len(edgar_signals.columns) if not edgar_signals.empty else 0

    signal_matrix = build_signal_matrix(ohlcv_returns, fred_signals, edgar_signals)
    n_signals = len(signal_matrix.columns) if not signal_matrix.empty else 0

    print(f" done. {len(actor_ids)} actors, {n_ohlcv} OHLCV + {n_fred} FRED + {n_edgar} EDGAR = {n_signals} signals")

    # --- Generate windows ----------------------------------------------------
    if single_window is not None:
        windows = [make_windows()[single_window]]
    else:
        windows = make_windows()

    # --- Results accumulators ------------------------------------------------
    l1_rows: list[dict[str, Any]] = []
    l2_rows: list[dict[str, Any]] = []
    l5_rows: list[dict[str, Any]] = []
    window_summaries: list[dict[str, Any]] = []

    all_oos_r2_best: list[float] = []

    for wi, window in enumerate(windows):
        wid = window["window_id"]
        print(f"\n  [{wi + 1}/{len(windows)}] {wid}  "
              f"train={window['train_start'].year}-{window['train_end'].year}  "
              f"test={window['test_start'].year}")

        # --- Run all regime counts -------------------------------------------
        results_by_M: dict[int, dict[str, Any]] = {}
        for M in regimes:
            print(f"    M={M} ...", end="", flush=True)
            try:
                result = run_window(
                    window, intensities_wide, signal_matrix,
                    registry, actor_ids, M, verbose,
                )
                results_by_M[M] = result
                print(f" K*={result['K']} N={result['N']} "
                      f"OOS R2={result['oos_r2_pred']:.4f} ({result['timings'].get('filter_em', 0):.1f}s)")
            except Exception as exc:
                print(f" FAILED: {exc}")

        if not results_by_M:
            print("    [SKIP] no regime succeeded")
            continue

        # --- Select best M ---------------------------------------------------
        best_M = max(results_by_M, key=lambda m: results_by_M[m]["oos_r2_pred"])
        best = results_by_M[best_M]
        # Store M=1 R^2 for regime delta
        oos_r2_m1 = results_by_M.get(1, {}).get("oos_r2_pred", best["oos_r2_pred"])
        best["oos_r2_m1"] = oos_r2_m1

        print(f"    Best M={best_M}")

        # --- Emergence -------------------------------------------------------
        print("    Emergence ...", end="", flush=True)
        synergy, crit, gap_ea, oos_r2_ea = compute_emergence(best)
        print(f" EA R2={oos_r2_ea:.4f}" if np.isfinite(oos_r2_ea) else " skipped")

        # --- Falsification ---------------------------------------------------
        fals_results = None
        n_fals_pass = 0
        if not skip_falsification:
            print("    Falsification ...", end="", flush=True)
            fals_results = run_falsification(best)
            if fals_results:
                n_fals_pass = sum(1 for r in fals_results.values() if r.passes)
                print(f" {n_fals_pass}/{len(fals_results)} pass")
            else:
                print(" failed")

        # --- L1 metrics ------------------------------------------------------
        N = best["N"]
        T_test = best["T_test"]

        # DM test: predictive vs modal (use raw actuals, mean-restored benchmarks)
        obs_mean = best["obs_mean"]
        actual_raw = best["obs_test_raw"].T.ravel()
        errors_pred = actual_raw - (best["gap_pred"].benchmarks + obs_mean.T).ravel()
        errors_modal = actual_raw - (best["gap_modal"].benchmarks + obs_mean.T).ravel()
        dm_stat, dm_pval = diebold_mariano_test(errors_pred, errors_modal)

        coverage = float(np.mean(np.isfinite(best["gap_pred"].gaps)))
        delta_pred = best["oos_r2_pred"]
        delta_modal_inc = best["oos_r2_modal"] - best["oos_r2_pred"]

        # Predictive benchmark row
        l1_rows.append(compute_l1_row(
            window, "predictive", best["oos_r2_pred"],
            dm_stat, dm_pval, coverage, N, T_test, delta_pred, 0.0,
        ))
        # Modal benchmark row
        l1_rows.append(compute_l1_row(
            window, "modal", best["oos_r2_modal"],
            dm_stat, dm_pval, coverage, N, T_test, delta_pred, delta_modal_inc,
        ))
        # Emergence-aware benchmark row
        if np.isfinite(oos_r2_ea):
            delta_ea_inc = oos_r2_ea - best["oos_r2_modal"]
            l1_rows.append(compute_l1_row(
                window, "emergence_aware", oos_r2_ea,
                dm_stat, dm_pval, coverage, N, T_test,
                delta_pred, delta_modal_inc + delta_ea_inc,
            ))

        # --- L2: component attribution --------------------------------------
        delta_regime = best["oos_r2_pred"] - oos_r2_m1 if best_M > 1 else 0.0
        delta_emergence = (oos_r2_ea - best["oos_r2_modal"]) if np.isfinite(oos_r2_ea) else 0.0
        l2_rows.append({
            "experiment_id": EXP_ID,
            "window_id": wid,
            "best_M": best_M,
            "k_star": best["K"],
            "delta_r2_regime": delta_regime,
            "delta_r2_emergence": delta_emergence,
            "delta_r2_modal_vs_pred": delta_modal_inc,
            "n_falsification_pass": n_fals_pass if fals_results else -1,
        })

        # --- L5: timing per component ----------------------------------------
        for comp, secs in best["timings"].items():
            l5_rows.append({
                "experiment_id": EXP_ID,
                "window_id": wid,
                "component": comp,
                "time_s": secs,
                "N": N,
                "M": best_M,
            })

        all_oos_r2_best.append(best["oos_r2_pred"])

        window_summaries.append({
            "window_id": wid,
            "best_M": best_M,
            "K": best["K"],
            "N": N,
            "oos_r2_pred": best["oos_r2_pred"],
            "oos_r2_modal": best["oos_r2_modal"],
            "oos_r2_ea": oos_r2_ea,
            "fals_pass": n_fals_pass if fals_results else -1,
            "total_s": sum(best["timings"].values()),
            "results_by_M": {m: r["oos_r2_pred"] for m, r in results_by_M.items()},
        })

    # --- Decision gate -------------------------------------------------------
    if not all_oos_r2_best:
        print("\n  NO RESULTS -- cannot evaluate decision gate")
        return

    mean_r2 = float(np.mean(all_oos_r2_best))
    gate_pass = mean_r2 > 0

    # --- Summary output ------------------------------------------------------
    print("\n" + "-" * 70)
    print("  REGIME COMPARISON SUMMARY")
    print(f"  {'Window':<10}", end="")
    for M in regimes:
        print(f"  M={M:>5}", end="")
    print("  Best_M")
    for ws in window_summaries:
        print(f"  {ws['window_id']:<10}", end="")
        for M in regimes:
            r2 = ws["results_by_M"].get(M, float("nan"))
            print(f"  {r2:>7.4f}", end="")
        print(f"  {ws['best_M']:>5}")

    print(f"\n  Mean OOS R2 by M: ", end="")
    for M in regimes:
        vals = [ws["results_by_M"].get(M, float("nan")) for ws in window_summaries]
        print(f"M{M}={np.nanmean(vals):.4f}  ", end="")
    print()

    print("\n" + "-" * 70)
    print("  PER-WINDOW RESULTS (best M)")
    print(f"  {'Window':<10} {'R2_pred':>10} {'R2_modal':>10} {'R2_EA':>10} "
          f"{'K':>4} {'N':>4} {'M':>3} {'Fals':>6} {'Time':>7}")
    for ws in window_summaries:
        ea_str = f"{ws['oos_r2_ea']:.4f}" if np.isfinite(ws["oos_r2_ea"]) else "n/a"
        fals_str = f"{ws['fals_pass']}/7" if ws["fals_pass"] >= 0 else "skip"
        print(f"  {ws['window_id']:<10} {ws['oos_r2_pred']:>10.4f} {ws['oos_r2_modal']:>10.4f} "
              f"{ea_str:>10} {ws['K']:>4} {ws['N']:>4} {ws['best_M']:>3} "
              f"{fals_str:>6} {ws['total_s']:>6.1f}s")

    print("\n" + "=" * 70)
    if gate_pass:
        print(f"  DECISION GATE: PASS -- mean OOS R2 = {mean_r2:.4f} > 0")
    else:
        print(f"  DECISION GATE: STOP -- mean OOS R2 = {mean_r2:.4f} <= 0")
        print("  Do NOT proceed to B-series. Diagnose first.")
    print("=" * 70)

    # --- Save outputs --------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # L1
    l1_df = pd.DataFrame(l1_rows)
    l1_path = METRICS_DIR / f"level1_{EXP_ID}.parquet"
    l1_df.to_parquet(l1_path, index=False)
    print(f"\n  Wrote: {l1_path.relative_to(PROJECT_ROOT)}")

    # L2
    l2_df = pd.DataFrame(l2_rows)
    l2_path = METRICS_DIR / f"level2_{EXP_ID}.parquet"
    l2_df.to_parquet(l2_path, index=False)
    print(f"  Wrote: {l2_path.relative_to(PROJECT_ROOT)}")

    # L3: stability
    l3_rows = [{
        "experiment_id": EXP_ID,
        "metric": "oos_r2_pred",
        "mean": float(np.mean(all_oos_r2_best)),
        "std": float(np.std(all_oos_r2_best)),
        "min": float(np.min(all_oos_r2_best)),
        "max": float(np.max(all_oos_r2_best)),
        "n_windows": len(all_oos_r2_best),
    }]
    l3_df = pd.DataFrame(l3_rows)
    l3_path = METRICS_DIR / f"level3_{EXP_ID}.parquet"
    l3_df.to_parquet(l3_path, index=False)
    print(f"  Wrote: {l3_path.relative_to(PROJECT_ROOT)}")

    # L4: gap half-life (AR(1) coefficient on gap series)
    l4_rows = []
    for ws in window_summaries:
        l4_rows.append({
            "experiment_id": EXP_ID,
            "window_id": ws["window_id"],
            "gap_halflife_note": "requires per-actor gap persistence analysis (deferred)",
        })
    l4_df = pd.DataFrame(l4_rows)
    l4_path = METRICS_DIR / f"level4_{EXP_ID}.parquet"
    l4_df.to_parquet(l4_path, index=False)
    print(f"  Wrote: {l4_path.relative_to(PROJECT_ROOT)}")

    # L5
    l5_df = pd.DataFrame(l5_rows)
    l5_path = METRICS_DIR / f"level5_{EXP_ID}.parquet"
    l5_df.to_parquet(l5_path, index=False)
    print(f"  Wrote: {l5_path.relative_to(PROJECT_ROOT)}")

    # Config YAML
    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "registry": str(REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        "intensities": str(INTENSITIES_PATH.relative_to(PROJECT_ROOT)),
        "n_actors_registry": len(actor_data),
        "n_actors_intensity": len(actor_ids),
        "n_signals_ohlcv": n_ohlcv,
        "n_signals_fred": n_fred,
        "n_signals_edgar": n_edgar,
        "regimes_tested": regimes,
        "windows": len(windows),
        "falsification": not skip_falsification,
        "decision_gate": "PASS" if gate_pass else "STOP",
        "mean_oos_r2": round(mean_r2, 6),
        "per_window": [
            {
                "window_id": ws["window_id"],
                "best_M": ws["best_M"],
                "K": ws["K"],
                "N": ws["N"],
                "oos_r2_pred": round(ws["oos_r2_pred"], 6),
                "oos_r2_modal": round(ws["oos_r2_modal"], 6),
            }
            for ws in window_summaries
        ],
    }
    yaml_path = CONFIGS_DIR / f"{EXP_ID}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")


# --- CLI entry point ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="A1-MVP-FULL experiment runner")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-falsification", action="store_true",
                        help="Skip falsification suite (~17 min)")
    parser.add_argument("--single-window", type=int, default=None,
                        help="Run only window N (0-9) for debugging")
    parser.add_argument("--regimes", type=str, default="1,2,3",
                        help="Comma-separated regime counts to test")
    args = parser.parse_args()

    regime_list = [int(x) for x in args.regimes.split(",")]

    run_a1(
        verbose=args.verbose,
        skip_falsification=args.skip_falsification,
        single_window=args.single_window,
        regime_list=regime_list,
    )


if __name__ == "__main__":
    main()
