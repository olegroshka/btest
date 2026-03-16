"""Tests for structural channel decomposition and benchmark (M6.1)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps.structural import (
    ChannelDecomposer,
    ChannelStabilityResult,
    StructuralBenchmarkComputer,
)
from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    DecompositionMethod,
    FilteredState,
    GapResult,
    ModalFrame,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_symmetric(N: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((N, N))
    return (A + A.T) / 2


def _make_modal_frame(N: int, K: int, rng: np.random.Generator) -> ModalFrame:
    basis, _ = np.linalg.qr(rng.standard_normal((N, K)))
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_filtered_state(T: int, K: int, rng: np.random.Generator) -> FilteredState:
    alpha = rng.standard_normal((T, K))
    return FilteredState(
        alpha_filtered=alpha,
        alpha_predicted=alpha,
        regime_probs=np.ones((T, 1)),
        log_likelihood=-100.0,
        regimes_selected=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# M6.1-T1: ChannelDecomposer
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelDecomposer:

    def test_stable_channel_classified_correctly(self):
        """Channel with consistent R² across windows → is_stable=True."""
        rng = np.random.default_rng(42)
        N, T = 8, 50

        # Build a low-rank matrix that consistently captures variance
        U_true = np.linalg.qr(rng.standard_normal((N, 3)))[0]
        # Generate observations dominated by this subspace
        alpha = rng.standard_normal((T, 3)) * 2
        obs = alpha @ U_true.T + rng.standard_normal((T, N)) * 0.05
        # The stable channel matrix projects well onto U_true
        stable_mat = U_true @ U_true.T + np.eye(N) * 0.01

        decomposer = ChannelDecomposer(stability_threshold=1.0, n_windows=5)
        results = decomposer.decompose({"stable": stable_mat}, obs, n_modes=3)

        assert len(results) == 1
        assert results[0].channel == "stable"
        assert results[0].is_stable is True

    def test_unstable_channel_classified_correctly(self):
        """Channel with high variance R² → is_stable=False."""
        rng = np.random.default_rng(123)
        N, T = 8, 50

        # Observations that vary wildly across windows
        obs = rng.standard_normal((T, N))
        # Make some windows very different
        obs[:10] *= 5.0
        obs[10:20] *= 0.01

        # A random matrix that won't consistently capture variance
        unstable_mat = rng.standard_normal((N, N))
        unstable_mat = (unstable_mat + unstable_mat.T) / 2

        # Use high threshold so even mild variance → unstable
        decomposer = ChannelDecomposer(stability_threshold=100.0, n_windows=5)
        results = decomposer.decompose({"unstable": unstable_mat}, obs, n_modes=2)

        assert len(results) == 1
        assert results[0].is_stable is False

    def test_decompose_returns_one_result_per_channel(self):
        """decompose returns exactly one result per channel."""
        rng = np.random.default_rng(7)
        N, T = 6, 30
        obs = rng.standard_normal((T, N))
        matrices = {
            "C1": _make_symmetric(N, rng),
            "C2": _make_symmetric(N, rng),
            "C3": _make_symmetric(N, rng),
        }
        decomposer = ChannelDecomposer()
        results = decomposer.decompose(matrices, obs)
        assert len(results) == 3
        names = {r.channel for r in results}
        assert names == {"C1", "C2", "C3"}

    def test_stability_ratio_formula(self):
        """stability_ratio = mean_contribution / (std_contribution + 1e-9)."""
        rng = np.random.default_rng(99)
        N, T = 6, 30
        obs = rng.standard_normal((T, N))
        mat = _make_symmetric(N, rng)
        decomposer = ChannelDecomposer(n_windows=4)
        results = decomposer.decompose({"ch": mat}, obs, n_modes=2)
        r = results[0]
        expected = r.mean_contribution / (r.std_contribution + 1e-9)
        assert abs(r.stability_ratio - expected) < 1e-10

    def test_stable_and_distortionary_helpers(self):
        """stable_channels and distortionary_channels filter correctly."""
        results = [
            ChannelStabilityResult("A", 0.8, 0.1, 8.0, True),
            ChannelStabilityResult("B", 0.2, 0.5, 0.4, False),
            ChannelStabilityResult("C", 0.9, 0.05, 18.0, True),
        ]
        decomposer = ChannelDecomposer()
        assert decomposer.stable_channels(results) == ["A", "C"]
        assert decomposer.distortionary_channels(results) == ["B"]


# ─────────────────────────────────────────────────────────────────────────────
# M6.1-T2: StructuralBenchmarkComputer
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralBenchmarkComputer:

    def test_structural_benchmark_produces_gap_result(self):
        """compute() returns a valid GapResult."""
        rng = np.random.default_rng(42)
        N, T, K = 6, 20, 3
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)
        fs = _make_filtered_state(T, K, rng)

        computer = StructuralBenchmarkComputer(n_modes=3)
        result = computer.compute(obs, fs, mf, actors=None)

        assert isinstance(result, GapResult)
        assert result.gaps.shape == (N, T)
        assert result.benchmarks.shape == (N, T)

    def test_structural_differs_from_predictive_with_distortionary_channels(self):
        """Structural with stable_operator gives different results than pure modal."""
        rng = np.random.default_rng(55)
        N, T, K = 8, 30, 3

        # Stable subspace
        U_stable, _ = np.linalg.qr(rng.standard_normal((N, 3)))
        stable_op = U_stable @ U_stable.T

        # Observations with significant noise (distortionary component)
        alpha = rng.standard_normal((T, K))
        obs = alpha @ _make_modal_frame(N, K, rng).basis.T + rng.standard_normal((T, N)) * 2.0

        mf = _make_modal_frame(N, K, rng)
        fs = _make_filtered_state(T, K, rng)

        computer_structural = StructuralBenchmarkComputer(n_modes=3)
        computer_modal = StructuralBenchmarkComputer(n_modes=3)

        result_structural = computer_structural.compute(obs, fs, mf, None, stable_operator=stable_op)
        result_modal = computer_modal.compute(obs, fs, mf, None, stable_operator=None)

        # They should differ (different bases)
        assert not np.allclose(result_structural.benchmarks, result_modal.benchmarks)

    def test_benchmark_class_is_structural(self):
        """benchmark_class attribute is BenchmarkClass.STRUCTURAL."""
        computer = StructuralBenchmarkComputer()
        assert computer.benchmark_class == BenchmarkClass.STRUCTURAL

    def test_structural_benchmark_with_stable_operator(self):
        """With stable_operator, eigenvectors are used for projection."""
        rng = np.random.default_rng(77)
        N, T, K = 6, 20, 3
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)
        fs = _make_filtered_state(T, K, rng)

        stable_op = _make_symmetric(N, rng)
        computer = StructuralBenchmarkComputer(n_modes=2)
        result = computer.compute(obs, fs, mf, None, stable_operator=stable_op)

        assert result.gaps.shape == (N, T)
        assert result.benchmark_class == BenchmarkClass.STRUCTURAL

    def test_gaps_equal_observations_minus_benchmarks(self):
        """gaps = observations - benchmarks (shape-consistent)."""
        rng = np.random.default_rng(11)
        N, T, K = 5, 15, 2
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)
        fs = _make_filtered_state(T, K, rng)

        computer = StructuralBenchmarkComputer(n_modes=2)
        result = computer.compute(obs, fs, mf, None)

        expected_gaps = obs.T - result.benchmarks
        np.testing.assert_allclose(result.gaps, expected_gaps, atol=1e-12)
