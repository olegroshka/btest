"""
Benchmark: full SMIM pipeline (GPU-accelerated path).

Exercises all GPU-accelerated components in sequence:
  1. Polar decomposition → modal frame
  2. Batch Granger edge estimation
  3. Kalman filter (numpy, but modal frame is device-derived)
  4. Batch PID bootstrap

Parametrised over N (actors) and device (CPU / CUDA).
T is fixed at 80 to match profiling_results.json baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.compute.batch_granger import batch_granger_test
from quantdsl_backtest.smim.compute.batch_pid import batch_pid_synergy
from quantdsl_backtest.smim.compute.linalg import polar_decompose
from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame

_K = 15   # modal dimension — matches profiling baseline
_T = 80   # time steps — one experiment year (quarterly)


def _build_pipeline(N: int, seed: int = 42) -> dict:
    """Pre-generate all synthetic inputs so they are excluded from timing."""
    rng = np.random.default_rng(seed)

    # Observations: (T, N)
    obs = rng.standard_normal((_T, N)).astype(np.float64)

    # Raw modal candidate basis: (N, K)
    A_raw = rng.standard_normal((N, _K)).astype(np.float64)

    # Kalman parameters
    F = 0.8 * np.eye(_K)
    Q = 0.05 * np.eye(_K)
    R = 0.1 * np.eye(N)

    # Modal states for PID (pre-run filter so PID timing is isolated)
    orth, _ = polar_decompose(A_raw)
    modal_frame = ModalFrame(
        basis=orth, eigenvalues=np.linspace(0.9, 0.5, _K),
        method=DecompositionMethod.SCHUR,
    )
    kf = KalmanFilter(F=F, Q=Q, R=R)
    state = kf.filter(obs, modal_frame)
    modal_states = state.alpha_filtered  # (T, K)
    target = obs[:, 0]  # proxy for a gap signal

    return dict(
        obs=obs, A_raw=A_raw,
        F=F, Q=Q, R=R,
        modal_states=modal_states, target=target,
    )


def _run_pipeline(inputs: dict) -> None:
    """Execute all GPU-accelerated pipeline stages.

    This is the callable passed to ``benchmark()``.  Each call mirrors
    one full pipeline iteration for a single experiment run.
    """
    obs = inputs["obs"]
    A_raw = inputs["A_raw"]
    N = obs.shape[1]
    F, Q, R = inputs["F"], inputs["Q"], inputs["R"]

    # Stage 1: spectral — polar decomposition (GPU SVD)
    orth, _ = polar_decompose(A_raw)
    modal_frame = ModalFrame(
        basis=orth, eigenvalues=np.linspace(0.9, 0.5, _K),
        method=DecompositionMethod.SCHUR,
    )

    # Stage 2: Granger edge estimation (GPU batch lstsq)
    _ = batch_granger_test(obs.T, max_lag=4, p_threshold=0.05)

    # Stage 3: Kalman filter (numpy — inherits modal frame from GPU stage)
    kf = KalmanFilter(F=F, Q=Q, R=R)
    state = kf.filter(obs, modal_frame)

    # Stage 4: PID bootstrap (GPU batch covariance + log-det)
    _ = batch_pid_synergy(
        state.alpha_filtered, inputs["target"], n_bootstrap=100
    )


@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_full_pipeline(
    benchmark: pytest.fixture, device: object, N: int
) -> None:
    """Benchmark GPU-accelerated pipeline stages end-to-end (N actors, T=80)."""
    inputs = _build_pipeline(N)

    # Warm-up run excluded from timing
    _run_pipeline(inputs)

    benchmark(_run_pipeline, inputs)
