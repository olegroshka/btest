"""Tests for smim/gaps/modal.py — ModalBenchmark."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps.modal import ModalBenchmark
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
    rng = np.random.default_rng(7)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_filtered_state(T: int, K: int) -> FilteredState:
    rng = np.random.default_rng(8)
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

class TestModalBenchmark:
    def test_benchmark_class_is_modal(self) -> None:
        bench = ModalBenchmark()
        assert bench.benchmark_class == BenchmarkClass.MODAL

    def test_modal_attribution_not_none(self) -> None:
        T, N, K = 30, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(9).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = ModalBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert result.modal_attribution is not None

    def test_modal_attribution_shape(self) -> None:
        T, N, K = 25, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(10).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = ModalBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert result.modal_attribution.shape == (N, T, K)

    def test_attribution_sums_to_prediction_gap(self) -> None:
        """sum over k of attribution[i, t, :] should equal predicted - filtered gap projected."""
        T, N, K = 20, 4, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(11).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = ModalBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)

        # attribution sums over k give the projection of (alpha_p - alpha_s) onto U
        attribution_sum = result.modal_attribution.sum(axis=2)  # (N, T)
        U = mf.basis
        expected = ((fs.alpha_predicted - fs.alpha_filtered) @ U.T).T  # (N, T)
        np.testing.assert_allclose(attribution_sum, expected, atol=1e-12)

    def test_gap_shape(self) -> None:
        T, N, K = 30, 5, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(12).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = ModalBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert result.gaps.shape == (N, T)
        assert result.benchmarks.shape == (N, T)

    def test_result_is_gap_result(self) -> None:
        T, N, K = 20, 4, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(13).standard_normal((T, N))
        fs = _make_filtered_state(T, K)
        bench = ModalBenchmark()
        result = bench.compute(obs, fs, mf, actors=None)
        assert isinstance(result, GapResult)
