"""Tests for graph storage (save/load round-trip) (M2.1-T8)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.storage import (
    load_edges,
    load_operator,
    save_edges,
    save_operator,
)
from quantdsl_backtest.smim.interfaces import RelationChannel


def _random_sparse(n: int, density: float = 0.3, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, n))
    mask = rng.random((n, n)) > (1.0 - density)
    dense[~mask] = 0.0
    return csr_matrix(dense)


class TestSaveLoadEdgesRoundTrip:
    def test_round_trip_single_channel(self, tmp_path):
        n = 4
        original = _random_sparse(n, density=0.4, seed=1)
        edges = {RelationChannel.FINANCIAL: original}
        save_edges(edges, tmp_path, "2020Q1")
        loaded = load_edges(tmp_path, "2020Q1")
        assert RelationChannel.FINANCIAL in loaded
        diff = (original - loaded[RelationChannel.FINANCIAL])
        assert diff.nnz == 0

    def test_round_trip_multiple_channels(self, tmp_path):
        n = 5
        edges = {
            RelationChannel.FINANCIAL: _random_sparse(n, seed=0),
            RelationChannel.NARRATIVE: _random_sparse(n, seed=1),
            RelationChannel.SUPPLY_CHAIN: _random_sparse(n, seed=2),
        }
        save_edges(edges, tmp_path, "period1")
        loaded = load_edges(tmp_path, "period1")
        for channel, original in edges.items():
            assert channel in loaded
            diff = (original - loaded[channel])
            assert diff.nnz == 0

    def test_creates_subdirectory(self, tmp_path):
        n = 3
        edges = {RelationChannel.FISCAL: _random_sparse(n, seed=5)}
        save_edges(edges, tmp_path, "2023Q2")
        out_dir = tmp_path / "2023Q2"
        assert out_dir.exists()
        assert (out_dir / "edges_C3.npz").exists()

    def test_load_returns_empty_dict_if_no_files(self, tmp_path):
        # Create directory with no files
        (tmp_path / "empty_period").mkdir()
        loaded = load_edges(tmp_path, "empty_period")
        assert isinstance(loaded, dict)
        assert len(loaded) == 0


class TestSaveLoadOperatorRoundTrip:
    def test_round_trip_operator(self, tmp_path):
        n = 6
        original = _random_sparse(n, density=0.5, seed=10)
        save_operator(original, tmp_path, "2021Q1")
        loaded = load_operator(tmp_path, "2021Q1")
        diff = (original - loaded)
        assert diff.nnz == 0

    def test_operator_file_created(self, tmp_path):
        n = 3
        op = csr_matrix(np.eye(n))
        save_operator(op, tmp_path, "test_period")
        file_path = tmp_path / "test_period" / "operator.npz"
        assert file_path.exists()

    def test_round_trip_exact_equality(self, tmp_path):
        """Verify exact matrix equality after save-reload."""
        n = 4
        original = _random_sparse(n, density=0.6, seed=99)
        save_operator(original, tmp_path, "exact")
        loaded = load_operator(tmp_path, "exact")
        orig_dense = original.toarray()
        loaded_dense = loaded.toarray()
        assert np.allclose(orig_dense, loaded_dense)
