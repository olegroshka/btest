"""Tests for smim/emergence/tda.py — sliding-window TDA."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.emergence.tda import (
    persistence_landscape_norms,
    sliding_window_persistence,
    topological_complexity,
    wasserstein_distances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alpha(T: int = 60, K: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, K))


# ---------------------------------------------------------------------------
# M4.7-T1 tests
# ---------------------------------------------------------------------------

class TestSlidingWindowPersistence:
    def test_sliding_window_returns_list(self) -> None:
        alpha = _make_alpha()
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        assert isinstance(diagrams, list)
        assert len(diagrams) > 0

    def test_n_windows_matches_config(self) -> None:
        """Verify correct window count for given T, window_size, overlap."""
        T, W, O = 60, 20, 10
        alpha = _make_alpha(T=T)
        step = W - O
        expected = len(range(0, T - W + 1, step))
        diagrams = sliding_window_persistence(alpha, window_size=W, overlap=O)
        assert len(diagrams) == expected

    def test_each_diagram_has_two_parts(self) -> None:
        alpha = _make_alpha()
        diagrams = sliding_window_persistence(alpha, window_size=15, overlap=5)
        for dgm in diagrams:
            assert len(dgm) >= 2  # at least H0 and H1

    def test_diagram_arrays_have_two_columns(self) -> None:
        alpha = _make_alpha()
        diagrams = sliding_window_persistence(alpha, window_size=15, overlap=5)
        for dgm in diagrams:
            for dim_dgm in dgm:
                if len(dim_dgm) > 0:
                    assert dim_dgm.shape[1] == 2


# ---------------------------------------------------------------------------
# M4.7-T2 tests
# ---------------------------------------------------------------------------

class TestTopologicalComplexity:
    def test_topological_complexity_non_negative(self) -> None:
        alpha = _make_alpha()
        diagrams = sliding_window_persistence(alpha, window_size=15, overlap=5)
        C = topological_complexity(diagrams)
        assert np.all(C >= 0.0)

    def test_topological_complexity_shape(self) -> None:
        alpha = _make_alpha(T=60)
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        C = topological_complexity(diagrams)
        assert C.shape == (len(diagrams),)

    def test_wasserstein_distances_length(self) -> None:
        """Length of Wasserstein distances = n_windows - 1."""
        alpha = _make_alpha(T=60)
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        dists = wasserstein_distances(diagrams)
        assert len(dists) == len(diagrams) - 1

    def test_wasserstein_distances_non_negative(self) -> None:
        alpha = _make_alpha(T=60)
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        dists = wasserstein_distances(diagrams)
        assert np.all(dists >= 0.0)

    def test_persistence_landscape_norms_shape(self) -> None:
        alpha = _make_alpha(T=60)
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        norms = persistence_landscape_norms(diagrams)
        assert norms.shape == (len(diagrams), 2)

    def test_persistence_landscape_norms_non_negative(self) -> None:
        alpha = _make_alpha(T=60)
        diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
        norms = persistence_landscape_norms(diagrams)
        assert np.all(norms >= 0.0)
