"""Tests for smim/gaps/predictive.py — PredictiveBenchmark."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    DecompositionMethod,
    FilteredState,
    GapResult,
    ModalFrame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_modal_frame(N: int = 5, K: int = 2) -> ModalFrame:
    rng = np.random.default_rng(0)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_filtered_state(T: int, K: int, alpha_predicted: np.ndarray | None = None) -> FilteredState:
    rng = np.random.default_rng(1)
    if alpha_predicted is None:
        alpha_predicted = rng.standard_normal((T, K))
    alpha_filtered = rng.standard_normal((T, K))
    return FilteredState(
        alpha_filtered=alpha_filtered,
        alpha_predicted=alpha_predicted,
        regime_probs=np.ones((T, 1)),
        log_likelihood=-100.0,
        regimes_selected=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictiveBenchmark:
    def test_benchmark_class_is_predictive(self) -> None:
        bench = PredictiveBenchmark()
        assert bench.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_gap_shape(self) -> None:
        T, N, K = 30, 5, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(2).standard_normal((T, N))  # (T, N)
        fs = _make_filtered_state(T, K)
        bench = PredictiveBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert isinstance(result, GapResult)
        assert result.gaps.shape == (N, T)
        assert result.benchmarks.shape == (N, T)

    def test_perfect_prediction_gives_zero_gap(self) -> None:
        """When alpha_predicted recovers observations exactly, gap should be ~0."""
        T, N, K = 20, 3, 2
        mf = _make_modal_frame(N=N, K=K)
        U = mf.basis  # (N, K)

        # Construct observations that lie exactly on the modal subspace
        rng = np.random.default_rng(3)
        alpha_true = rng.standard_normal((T, K))
        obs_TN = alpha_true @ U.T  # (T, N)

        fs = _make_filtered_state(T, K, alpha_predicted=alpha_true)
        bench = PredictiveBenchmark()
        result = bench.compute(obs_TN, fs, mf, actors=None)
        np.testing.assert_allclose(result.gaps, 0.0, atol=1e-12)

    def test_benchmark_class_in_result(self) -> None:
        T, N, K = 20, 4, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(4).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = PredictiveBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert result.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_accepts_NT_observations(self) -> None:
        """Should also accept (N, T) shaped observations."""
        T, N, K = 25, 5, 2
        mf = _make_modal_frame(N=N, K=K)
        obs_NT = np.random.default_rng(5).standard_normal((N, T))  # (N, T)
        fs = _make_filtered_state(T, K)
        bench = PredictiveBenchmark()
        result = bench.compute(obs_NT, fs, mf, actors=None)
        assert result.gaps.shape == (N, T)
