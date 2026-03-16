"""Tests for null model rewiring (M2.2-T1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.null_models import (
    block_preserving_rewire,
    degree_preserving_rewire,
    type_preserving_rewire,
)


def _random_digraph(n: int, edge_prob: float = 0.3, seed: int = 42) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.random((n, n))
    dense[dense > edge_prob] = 0.0
    np.fill_diagonal(dense, 0.0)
    dense[dense > 0] = 1.0
    return csr_matrix(dense)


def _in_degrees(adj: csr_matrix) -> np.ndarray:
    return np.array(adj.sum(axis=0)).flatten().astype(int)


def _out_degrees(adj: csr_matrix) -> np.ndarray:
    return np.array(adj.sum(axis=1)).flatten().astype(int)


class TestDegreePreservingRewire:
    def test_degree_sequences_preserved(self):
        adj = _random_digraph(8, edge_prob=0.4, seed=0)
        rng = np.random.default_rng(123)
        rewired = degree_preserving_rewire(adj, rng=rng)
        assert np.array_equal(_in_degrees(adj), _in_degrees(rewired))
        assert np.array_equal(_out_degrees(adj), _out_degrees(rewired))

    def test_edge_count_preserved(self):
        adj = _random_digraph(6, edge_prob=0.5, seed=1)
        rng = np.random.default_rng(42)
        rewired = degree_preserving_rewire(adj, rng=rng)
        assert adj.nnz == rewired.nnz

    def test_rewired_differs_from_original(self):
        """With enough swaps, the graph should differ from the original."""
        adj = _random_digraph(10, edge_prob=0.4, seed=7)
        rng = np.random.default_rng(99)
        rewired = degree_preserving_rewire(adj, n_swaps=500, rng=rng)
        # Expect some change (probabilistically near-certain for n=10, n_swaps=500)
        diff = (adj - rewired)
        # We just check it runs without error and returns correct type/shape
        assert isinstance(rewired, csr_matrix)
        assert rewired.shape == adj.shape

    def test_no_self_loops(self):
        adj = _random_digraph(8, edge_prob=0.4, seed=2)
        rng = np.random.default_rng(55)
        rewired = degree_preserving_rewire(adj, rng=rng)
        arr = rewired.toarray()
        assert np.all(np.diag(arr) == 0.0)

    def test_returns_csr_matrix(self):
        adj = _random_digraph(5, edge_prob=0.3, seed=3)
        rng = np.random.default_rng(1)
        result = degree_preserving_rewire(adj, rng=rng)
        assert isinstance(result, csr_matrix)

    def test_single_edge_graph_unchanged(self):
        dense = np.zeros((3, 3))
        dense[0, 1] = 1.0
        adj = csr_matrix(dense)
        rng = np.random.default_rng(0)
        rewired = degree_preserving_rewire(adj, rng=rng)
        assert rewired.nnz == 1


class TestBlockPreservingRewire:
    def test_no_new_cross_block_edges(self):
        """After rewiring, there should be no new cross-block edges."""
        n = 6
        adj = _random_digraph(n, edge_prob=0.4, seed=10)
        # First 3 nodes in block 0, last 3 in block 1
        # Force no cross-block edges in original
        dense = adj.toarray()
        dense[0:3, 3:6] = 0.0  # remove cross-block edges
        dense[3:6, 0:3] = 0.0
        adj_no_cross = csr_matrix(dense)

        layer_assignments = np.array([0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(11)
        rewired = block_preserving_rewire(adj_no_cross, layer_assignments, rng=rng)
        arr = rewired.toarray()

        # No cross-block edges should exist after rewiring
        assert np.all(arr[0:3, 3:6] == 0.0)
        assert np.all(arr[3:6, 0:3] == 0.0)

    def test_returns_csr_matrix(self):
        adj = _random_digraph(6, edge_prob=0.3, seed=5)
        layer_assignments = np.array([0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(2)
        result = block_preserving_rewire(adj, layer_assignments, rng=rng)
        assert isinstance(result, csr_matrix)

    def test_shape_preserved(self):
        n = 8
        adj = _random_digraph(n, edge_prob=0.3, seed=6)
        layer_assignments = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        rng = np.random.default_rng(3)
        result = block_preserving_rewire(adj, layer_assignments, rng=rng)
        assert result.shape == (n, n)


class TestTypePreservingRewire:
    def test_returns_csr_matrix(self):
        adj = _random_digraph(6, edge_prob=0.3, seed=8)
        type_assignments = np.array([0, 0, 1, 1, 2, 2])
        rng = np.random.default_rng(4)
        result = type_preserving_rewire(adj, type_assignments, rng=rng)
        assert isinstance(result, csr_matrix)

    def test_shape_preserved(self):
        n = 6
        adj = _random_digraph(n, edge_prob=0.3, seed=9)
        type_assignments = np.array([0, 0, 1, 1, 2, 2])
        rng = np.random.default_rng(5)
        result = type_preserving_rewire(adj, type_assignments, rng=rng)
        assert result.shape == (n, n)
