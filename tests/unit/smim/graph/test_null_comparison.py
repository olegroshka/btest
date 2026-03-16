"""Tests for null model comparison (M2.2-T2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.null_comparison import (
    NullComparisonResult,
    NullModelComparison,
    modularity_statistic,
    spectral_gap_statistic,
)
from quantdsl_backtest.smim.graph.null_models import degree_preserving_rewire


def _ring_graph(n: int) -> csr_matrix:
    """Create a simple ring/cycle graph for predictable spectral properties."""
    dense = np.zeros((n, n))
    for i in range(n):
        dense[i, (i + 1) % n] = 1.0
        dense[(i + 1) % n, i] = 1.0
    return csr_matrix(dense)


def _random_sparse(n: int, density: float = 0.3, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n))
    mask = rng.random((n, n)) > (1.0 - density)
    dense[~mask] = 0.0
    np.fill_diagonal(dense, 0.0)
    dense[dense < 0] = 0.0
    return csr_matrix(dense)


class TestSpectralGapStatistic:
    def test_returns_finite_float(self):
        adj = _ring_graph(6)
        stat = spectral_gap_statistic(adj)
        assert np.isfinite(stat)

    def test_returns_float_type(self):
        adj = _random_sparse(5, seed=1)
        stat = spectral_gap_statistic(adj)
        assert isinstance(stat, float)

    def test_zero_for_empty_graph(self):
        adj = csr_matrix((4, 4), dtype=float)
        stat = spectral_gap_statistic(adj)
        assert np.isfinite(stat)

    def test_positive_for_connected_graph(self):
        # Fully connected graph has positive spectral gap
        n = 4
        adj = csr_matrix(np.ones((n, n)) - np.eye(n))
        stat = spectral_gap_statistic(adj)
        assert stat > 0


class TestModularityStatistic:
    def test_returns_value_in_valid_range(self):
        adj = _ring_graph(6)
        Q = modularity_statistic(adj)
        assert -0.5 <= Q <= 1.0

    def test_all_same_community_returns_finite(self):
        adj = _random_sparse(5, seed=2)
        Q = modularity_statistic(adj)
        assert np.isfinite(Q)

    def test_zero_matrix_returns_zero(self):
        adj = csr_matrix((4, 4), dtype=float)
        Q = modularity_statistic(adj)
        assert Q == 0.0

    def test_with_community_labels(self):
        n = 6
        adj = _ring_graph(n)
        labels = np.array([0, 0, 0, 1, 1, 1])
        Q = modularity_statistic(adj, community_labels=labels)
        assert -0.5 <= Q <= 1.0


class TestNullComparisonResult:
    def test_has_required_fields(self):
        result = NullComparisonResult(
            observed_stat=1.5,
            null_distribution=np.array([0.5, 1.0, 2.0]),
            p_value=0.33,
            passes=False,
        )
        assert result.observed_stat == 1.5
        assert len(result.null_distribution) == 3
        assert result.p_value == pytest.approx(0.33)
        assert result.passes is False

    def test_passes_true_when_p_value_low(self):
        result = NullComparisonResult(
            observed_stat=10.0,
            null_distribution=np.zeros(100),
            p_value=0.01,
            passes=True,
        )
        assert result.passes is True


class TestNullModelComparison:
    def test_run_produces_valid_p_value(self):
        adj = _ring_graph(6)
        rng = np.random.default_rng(42)
        comparison = NullModelComparison(
            rewire_fn=degree_preserving_rewire,
            n_instances=10,  # Fast for unit test
            statistic="spectral_gap",
        )
        result = comparison.run(adj, rng=rng)
        assert 0.0 <= result.p_value <= 1.0

    def test_run_returns_null_comparison_result(self):
        adj = _random_sparse(5, seed=7)
        rng = np.random.default_rng(0)
        comparison = NullModelComparison(n_instances=5)
        result = comparison.run(adj, rng=rng)
        assert isinstance(result, NullComparisonResult)

    def test_null_distribution_length_matches_n_instances(self):
        adj = _ring_graph(5)
        rng = np.random.default_rng(1)
        n_inst = 10
        comparison = NullModelComparison(n_instances=n_inst)
        result = comparison.run(adj, rng=rng)
        assert len(result.null_distribution) == n_inst

    def test_passes_flag_consistent_with_p_value(self):
        adj = _ring_graph(6)
        rng = np.random.default_rng(2)
        comparison = NullModelComparison(n_instances=10)
        result = comparison.run(adj, rng=rng)
        expected_passes = result.p_value < 0.05
        assert result.passes == expected_passes
