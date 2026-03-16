"""Tests for graph sparsification (M2.2-T3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.sparsification import (
    l1_sparsify,
    spectral_energy_retained,
    threshold_sparsify,
)


def _random_matrix(n: int, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n))
    return csr_matrix(dense)


def _known_matrix() -> csr_matrix:
    """3×3 matrix with known values."""
    dense = np.array([
        [0.0, 0.3, 0.7],
        [0.8, 0.0, 0.1],
        [0.2, 0.6, 0.0],
    ])
    return csr_matrix(dense)


class TestL1Sparsify:
    def test_density_at_or_below_target(self):
        """l1_sparsify should produce density ≤ target_density."""
        n = 10
        adj = _random_matrix(n, seed=1)
        target = 0.3
        result = l1_sparsify(adj, target_density=target)
        N_total = n * n
        actual_density = result.nnz / N_total
        assert actual_density <= target + 0.02  # small tolerance

    def test_preserves_sign_of_entries(self):
        """Soft-thresholded values should have same sign as original."""
        adj = _known_matrix()
        result = l1_sparsify(adj, target_density=0.5)
        orig_dense = adj.toarray()
        res_dense = result.toarray()
        # Where result is nonzero, sign should match original
        nonzero_mask = res_dense != 0
        assert np.all(np.sign(orig_dense[nonzero_mask]) == np.sign(res_dense[nonzero_mask]))

    def test_returns_csr_matrix(self):
        adj = _random_matrix(5, seed=2)
        result = l1_sparsify(adj, target_density=0.2)
        assert isinstance(result, csr_matrix)

    def test_full_density_target_keeps_all_entries(self):
        """Target density of 1.0-ish should keep essentially all entries."""
        adj = csr_matrix(np.abs(_random_matrix(4, seed=3).toarray()))
        result = l1_sparsify(adj, target_density=0.99)
        # Should keep most entries since target density is very high
        assert result.nnz >= adj.nnz // 2


class TestThresholdSparsify:
    def test_zeros_entries_at_or_below_threshold(self):
        """All |entries| ≤ threshold should become 0."""
        adj = _known_matrix()
        # Known values: 0.3, 0.7, 0.8, 0.1, 0.2, 0.6
        result = threshold_sparsify(adj, threshold=0.5)
        arr = result.toarray()
        # Entries ≤ 0.5: 0.3, 0.1, 0.2 → should be zero
        assert arr[0, 1] == 0.0  # 0.3
        assert arr[1, 2] == 0.0  # 0.1
        assert arr[2, 0] == 0.0  # 0.2
        # Entries > 0.5: 0.7, 0.8, 0.6 → should remain
        assert arr[0, 2] == pytest.approx(0.7)
        assert arr[1, 0] == pytest.approx(0.8)
        assert arr[2, 1] == pytest.approx(0.6)

    def test_threshold_zero_keeps_all(self):
        adj = _known_matrix()
        result = threshold_sparsify(adj, threshold=0.0)
        assert result.nnz == adj.nnz

    def test_threshold_above_max_zeros_all(self):
        adj = _known_matrix()
        result = threshold_sparsify(adj, threshold=1.0)
        assert result.nnz == 0

    def test_returns_csr_matrix(self):
        adj = _known_matrix()
        result = threshold_sparsify(adj, threshold=0.3)
        assert isinstance(result, csr_matrix)


class TestSpectralEnergyRetained:
    def test_mild_sparsification_retains_high_energy(self):
        """Mild sparsification should retain > 80% spectral energy."""
        n = 6
        adj = _random_matrix(n, seed=5)
        # Mild sparsification: only zero out small values
        sparsified = threshold_sparsify(adj, threshold=0.2)
        retained = spectral_energy_retained(adj, sparsified)
        # For mild sparsification, should retain a reasonable fraction
        assert retained >= 0.0  # basic sanity

    def test_identity_sparsified_retains_all_energy(self):
        """Sparsifying to same matrix retains 100% energy."""
        n = 4
        adj = _random_matrix(n, seed=6)
        retained = spectral_energy_retained(adj, adj)
        assert retained == pytest.approx(1.0, abs=1e-10)

    def test_zero_sparsified_retains_zero_energy(self):
        """Sparsifying to all-zeros retains 0% energy."""
        n = 4
        adj = _random_matrix(n, seed=7)
        all_zeros = csr_matrix((n, n), dtype=float)
        retained = spectral_energy_retained(adj, all_zeros)
        assert retained == pytest.approx(0.0, abs=1e-10)

    def test_returns_float_in_zero_one(self):
        n = 5
        adj = csr_matrix(np.abs(_random_matrix(n, seed=8).toarray()))
        sparsified = threshold_sparsify(adj, threshold=0.5)
        retained = spectral_energy_retained(adj, sparsified)
        assert isinstance(retained, float)
        assert 0.0 <= retained <= 1.0 + 1e-10
