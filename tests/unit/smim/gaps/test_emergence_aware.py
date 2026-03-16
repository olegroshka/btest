"""Tests for smim/gaps/emergence_aware.py — EmergenceAwareBenchmark."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps.emergence_aware import EmergenceAwareBenchmark
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

def _make_modal_frame(N: int = 5, K: int = 3) -> ModalFrame:
    rng = np.random.default_rng(0)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_filtered_state(T: int, K: int, seed: int = 1) -> FilteredState:
    rng = np.random.default_rng(seed)
    return FilteredState(
        alpha_filtered=rng.standard_normal((T, K)),
        alpha_predicted=rng.standard_normal((T, K)),
        regime_probs=np.ones((T, 1)),
        log_likelihood=-100.0,
        regimes_selected=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmergenceAwareBenchmark:
    def test_benchmark_class(self) -> None:
        bench = EmergenceAwareBenchmark()
        assert bench.benchmark_class == BenchmarkClass.EMERGENCE_AWARE

    def test_with_zero_synergy_equals_predictive(self) -> None:
        """Zero synergy matrix → same gaps as predictive benchmark."""
        T, N, K = 30, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(2).standard_normal((T, N))
        fs = _make_filtered_state(T, K)

        zero_synergy = np.zeros((K, K))
        em_bench = EmergenceAwareBenchmark(synergy_matrix=zero_synergy)
        pred_bench = PredictiveBenchmark()

        em_result = em_bench.compute(obs, fs, mf, actors=None)
        pred_result = pred_bench.compute(obs, fs, mf, actors=None)

        np.testing.assert_allclose(em_result.gaps, pred_result.gaps, atol=1e-10)

    def test_with_nonzero_synergy_differs(self) -> None:
        """Non-zero synergy → gaps differ from zero-synergy case."""
        T, N, K = 30, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(3).standard_normal((T, N))
        fs = _make_filtered_state(T, K)

        zero_synergy = np.zeros((K, K))
        high_synergy = np.ones((K, K)) * 0.5
        np.fill_diagonal(high_synergy, 0.0)

        em_zero = EmergenceAwareBenchmark(synergy_matrix=zero_synergy)
        em_high = EmergenceAwareBenchmark(synergy_matrix=high_synergy)

        result_zero = em_zero.compute(obs, fs, mf, actors=None)
        result_high = em_high.compute(obs, fs, mf, actors=None)

        # Gaps should differ
        assert not np.allclose(result_zero.gaps, result_high.gaps)

    def test_gap_shape(self) -> None:
        T, N, K = 25, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(4).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = EmergenceAwareBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert result.gaps.shape == (N, T)
        assert result.benchmarks.shape == (N, T)

    def test_result_is_gap_result(self) -> None:
        T, N, K = 20, 4, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(5).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = EmergenceAwareBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert isinstance(result, GapResult)

    def test_with_criticality_scaling(self) -> None:
        """Criticality scaling changes benchmarks."""
        T, N, K = 25, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(6).standard_normal((T, N))
        fs = _make_filtered_state(T, K)

        crit_ones = np.ones(T)  # scaling = 1 → no change
        crit_twos = np.full(T, 2.0)  # scaling = 2 → doubles benchmark

        em_ones = EmergenceAwareBenchmark(criticality=crit_ones)
        em_twos = EmergenceAwareBenchmark(criticality=crit_twos)

        r_ones = em_ones.compute(obs, fs, mf, actors=None)
        r_twos = em_twos.compute(obs, fs, mf, actors=None)

        # Benchmarks should differ
        assert not np.allclose(r_ones.benchmarks, r_twos.benchmarks)
