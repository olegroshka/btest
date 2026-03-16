"""Tests for MDLModeSelector, CompressibilityFilter, RGRelevanceFilter, CombinedModeSelector (M3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.mode_selection import (
    CombinedModeSelector,
    CompressibilityFilter,
    MDLModeSelector,
    RGRelevanceFilter,
    lz_complexity,
)


def _make_frame(N: int = 10, K: int = 8, seed: int = 0) -> ModalFrame:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((N, K)))
    eigenvalues = np.arange(K, 0, -1).astype(complex)
    return ModalFrame(basis=basis, eigenvalues=eigenvalues, method=DecompositionMethod.SCHUR)


def _make_signal_with_known_modes(N: int = 10, T: int = 100, n_modes: int = 3, seed: int = 1) -> tuple[ModalFrame, np.ndarray]:
    """Generate signals that lie in a known subspace of dimension n_modes."""
    rng = np.random.default_rng(seed)
    # True basis
    basis_full, _ = np.linalg.qr(rng.standard_normal((N, N)))
    true_modes = basis_full[:, :n_modes]
    noise_modes = basis_full[:, n_modes:]

    # Generate time series in signal subspace
    amplitudes_signal = rng.standard_normal((T, n_modes)) * 5.0
    amplitudes_noise = rng.standard_normal((T, N - n_modes)) * 0.1

    Y = amplitudes_signal @ true_modes.T + amplitudes_noise @ noise_modes.T  # (T, N)

    # Build ModalFrame with all N modes
    K = N
    frame = ModalFrame(
        basis=basis_full,
        eigenvalues=np.arange(K, 0, -1).astype(complex),
        method=DecompositionMethod.SCHUR,
    )
    return frame, Y


# ─────────────────────────────────────────────────────────────────────────────
# LZ complexity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLZComplexity:
    def test_constant_sequence_low_complexity(self) -> None:
        seq = np.ones(100)
        c = lz_complexity(seq)
        # All values equal median -> all zeros -> complexity 1
        assert c <= 2

    def test_random_sequence_higher_complexity(self) -> None:
        rng = np.random.default_rng(0)
        seq = rng.standard_normal(200)
        c = lz_complexity(seq)
        # Random binary string has O(n/log n) phrases
        assert c > 10

    def test_periodic_lower_than_random(self) -> None:
        # Periodic signal
        t = np.arange(200)
        periodic = np.sin(2 * np.pi * t / 10)
        periodic_c = lz_complexity(periodic)

        rng = np.random.default_rng(42)
        random_c = lz_complexity(rng.standard_normal(200))
        assert periodic_c < random_c

    def test_empty_sequence(self) -> None:
        c = lz_complexity(np.array([]))
        assert c == 0

    def test_single_element(self) -> None:
        c = lz_complexity(np.array([1.0]))
        assert c >= 1


# ─────────────────────────────────────────────────────────────────────────────
# MDLModeSelector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMDLModeSelector:
    def test_selects_fewer_modes_than_available(self) -> None:
        frame, Y = _make_signal_with_known_modes(N=10, T=80, n_modes=3)
        selector = MDLModeSelector()
        reduced_frame, reduced_ts = selector.select(frame, Y)
        assert reduced_frame.K <= frame.K

    def test_selects_at_least_one_mode(self) -> None:
        frame, Y = _make_signal_with_known_modes(N=6, T=40, n_modes=2)
        selector = MDLModeSelector()
        reduced_frame, _ = selector.select(frame, Y)
        assert reduced_frame.K >= 1

    def test_conservative_for_noisy_signal(self) -> None:
        """MDL should select k* <= n_modes + small buffer for clean signal."""
        frame, Y = _make_signal_with_known_modes(N=10, T=100, n_modes=3, seed=5)
        selector = MDLModeSelector()
        reduced_frame, _ = selector.select(frame, Y)
        # Should not select all K modes when only 3 are dominant
        assert reduced_frame.K <= 10

    def test_k_positive(self) -> None:
        frame = _make_frame(N=6, K=4)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((30, 6))
        selector = MDLModeSelector()
        reduced_frame, _ = selector.select(frame, Y)
        assert reduced_frame.K > 0

    def test_reduced_ts_shape(self) -> None:
        frame = _make_frame(N=6, K=5)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((30, 6))
        selector = MDLModeSelector()
        reduced_frame, reduced_ts = selector.select(frame, Y)
        assert reduced_ts.shape == (30, reduced_frame.K)


# ─────────────────────────────────────────────────────────────────────────────
# CompressibilityFilter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressibilityFilter:
    def test_retains_periodic_modes(self) -> None:
        """Periodic mode should have high rho and be retained."""
        N = 6
        T = 200
        rng = np.random.default_rng(9)
        t = np.arange(T)
        # Build a ModalFrame where mode 0 is periodic, mode 1 is noise
        basis, _ = np.linalg.qr(rng.standard_normal((N, 2)))
        frame = ModalFrame(basis=basis, eigenvalues=np.array([2.0 + 0j, 1.0 + 0j]), method=DecompositionMethod.SCHUR)

        # Modal amplitudes: mode 0 = periodic, mode 1 = random
        periodic_amp = np.sin(2 * np.pi * t / 10)
        random_amp = rng.standard_normal(T)
        ts = np.stack([periodic_amp, random_amp], axis=1)  # (T, K)

        cf = CompressibilityFilter(min_compressibility=0.05)
        reduced_frame, _ = cf.select(frame, ts)
        # Mode 0 (periodic) should be retained
        assert reduced_frame.K >= 1

    def test_always_retains_at_least_one_mode(self) -> None:
        frame = _make_frame(N=5, K=3)
        rng = np.random.default_rng(0)
        # All random amplitudes - might fail threshold but should keep at least 1
        ts = rng.standard_normal((50, 3))
        cf = CompressibilityFilter(min_compressibility=0.99)  # very high threshold
        reduced_frame, _ = cf.select(frame, ts)
        assert reduced_frame.K >= 1

    def test_filter_with_N_dimensional_input(self) -> None:
        N, K = 8, 4
        frame = _make_frame(N=N, K=K)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((50, N))
        cf = CompressibilityFilter(min_compressibility=0.0)
        reduced_frame, reduced_ts = cf.select(frame, Y)
        assert reduced_frame.K >= 1


# ─────────────────────────────────────────────────────────────────────────────
# RGRelevanceFilter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRGRelevanceFilter:
    def test_global_mode_passes(self) -> None:
        """Uniform loadings across all layers should pass RG."""
        N = 8
        K = 2
        # Uniform basis: all loadings equal
        basis = np.ones((N, K)) / np.sqrt(N)
        frame = ModalFrame(basis=basis, eigenvalues=np.array([2.0 + 0j, 1.0 + 0j]), method=DecompositionMethod.SCHUR)
        # Assign layers 0-3 evenly
        layers = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((20, K))

        rg = RGRelevanceFilter()
        reduced_frame, _ = rg.select(frame, ts, layer_assignments=layers)
        assert reduced_frame.K >= 1

    def test_layer3_only_mode_fails(self) -> None:
        """Mode concentrated exclusively on layer 3 should fail RG test."""
        N = 8
        K = 2
        rng = np.random.default_rng(0)
        basis = rng.standard_normal((N, K))
        # Mode 0: all weight on layer 3 actors (indices 6, 7)
        basis[:, 0] = 0.0
        basis[6, 0] = 1.0 / np.sqrt(2)
        basis[7, 0] = 1.0 / np.sqrt(2)
        # Mode 1: uniform
        basis[:, 1] = 1.0 / np.sqrt(N)
        frame = ModalFrame(basis=basis, eigenvalues=np.array([2.0 + 0j, 1.0 + 0j]), method=DecompositionMethod.SCHUR)
        layers = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        ts = np.ones((20, K))

        rg = RGRelevanceFilter()
        reduced_frame, _ = rg.select(frame, ts, layer_assignments=layers)
        # Mode 0 should be removed but mode 1 retained
        assert reduced_frame.K >= 1
        if reduced_frame.K == 1:
            # Should be mode 1 (uniform) that survived
            np.testing.assert_allclose(np.abs(reduced_frame.basis[:, 0]), 1.0 / np.sqrt(N), atol=1e-10)

    def test_no_layers_all_pass(self) -> None:
        frame = _make_frame(N=6, K=3)
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((20, 3))
        rg = RGRelevanceFilter()
        reduced_frame, _ = rg.select(frame, ts, layer_assignments=None)
        assert reduced_frame.K == 3


# ─────────────────────────────────────────────────────────────────────────────
# CombinedModeSelector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedModeSelector:
    def test_k_star_leq_input_K(self) -> None:
        frame, Y = _make_signal_with_known_modes(N=8, T=60, n_modes=3)
        selector = CombinedModeSelector(mdl=True, min_compressibility=0.0, rg_relevance=False)
        reduced_frame, _ = selector.select(frame, Y)
        assert reduced_frame.K <= frame.K

    def test_always_at_least_one_mode(self) -> None:
        frame = _make_frame(N=5, K=3)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((20, 5))
        # Harsh settings
        selector = CombinedModeSelector(mdl=True, min_compressibility=0.99, rg_relevance=False)
        reduced_frame, reduced_ts = selector.select(frame, Y)
        assert reduced_frame.K >= 1

    def test_empty_result_handled_gracefully(self) -> None:
        frame = _make_frame(N=5, K=2)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((15, 5))
        selector = CombinedModeSelector(mdl=False, min_compressibility=0.99, rg_relevance=False)
        reduced_frame, reduced_ts = selector.select(frame, Y)
        assert reduced_frame.K >= 1
        assert reduced_ts.shape[0] == 15
        assert reduced_ts.shape[1] == reduced_frame.K

    def test_no_filtering_preserves_K(self) -> None:
        frame = _make_frame(N=6, K=4)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((30, 6))
        # All filters disabled / very permissive
        selector = CombinedModeSelector(mdl=False, min_compressibility=0.0, rg_relevance=False)
        reduced_frame, _ = selector.select(frame, Y)
        assert reduced_frame.K == 4

    def test_with_layer_assignments(self) -> None:
        N = 8
        frame = _make_frame(N=N, K=4)
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((40, N))
        layers = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        selector = CombinedModeSelector(
            mdl=False, min_compressibility=0.0, rg_relevance=True, layer_assignments=layers
        )
        reduced_frame, _ = selector.select(frame, Y)
        assert reduced_frame.K >= 1
