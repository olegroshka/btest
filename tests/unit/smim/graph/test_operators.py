"""Tests for AggregateOperator (M2.1-T5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.operators import AggregateOperator
from quantdsl_backtest.smim.interfaces import RelationChannel


def _eye_sparse(n: int, scale: float = 1.0) -> csr_matrix:
    return csr_matrix(np.eye(n) * scale)


def _make_two_channel_matrices(n: int = 4) -> dict[RelationChannel, csr_matrix]:
    rng = np.random.default_rng(0)
    A = csr_matrix(np.abs(rng.standard_normal((n, n))))
    B = csr_matrix(np.abs(rng.standard_normal((n, n))))
    return {RelationChannel.FINANCIAL: A, RelationChannel.NARRATIVE: B}


class TestAggregateOperatorCompute:
    def test_uniform_weights_give_equal_weight_average(self):
        n = 3
        A = _eye_sparse(n, scale=2.0)
        B = _eye_sparse(n, scale=4.0)
        matrices = {RelationChannel.FINANCIAL: A, RelationChannel.NARRATIVE: B}
        op = AggregateOperator(channel_matrices=matrices).with_uniform_weights()
        result = op.compute()
        expected_diag = (2.0 + 4.0) * (1 / 2)  # = 3.0
        arr = result.toarray()
        assert arr[0, 0] == pytest.approx(expected_diag)

    def test_explicit_weights_applied_correctly(self):
        n = 2
        A = csr_matrix(np.ones((n, n)) * 1.0)
        B = csr_matrix(np.ones((n, n)) * 1.0)
        matrices = {RelationChannel.FINANCIAL: A, RelationChannel.NARRATIVE: B}
        weights = {RelationChannel.FINANCIAL: 0.3, RelationChannel.NARRATIVE: 0.7}
        op = AggregateOperator(channel_matrices=matrices, weights=weights)
        result = op.compute()
        expected = 0.3 * 1.0 + 0.7 * 1.0
        assert result.toarray()[0, 0] == pytest.approx(expected)

    def test_shape_preserved(self):
        n = 5
        matrices = _make_two_channel_matrices(n)
        op = AggregateOperator(channel_matrices=matrices)
        result = op.compute()
        assert result.shape == (n, n)

    def test_empty_matrices_raises_value_error(self):
        op = AggregateOperator(channel_matrices={})
        with pytest.raises(ValueError):
            op.compute()


class TestWithUniformWeights:
    def test_sets_uniform_weights(self):
        n = 3
        matrices = _make_two_channel_matrices(n)
        op = AggregateOperator(channel_matrices=matrices)
        op.with_uniform_weights()
        R = len(matrices)
        for ch in matrices:
            assert op.weights[ch] == pytest.approx(1.0 / R)

    def test_returns_self_for_chaining(self):
        n = 3
        matrices = _make_two_channel_matrices(n)
        op = AggregateOperator(channel_matrices=matrices)
        returned = op.with_uniform_weights()
        assert returned is op


class TestSpectralEnergy:
    def test_spectral_energy_positive_for_nonzero_matrix(self):
        n = 4
        matrices = _make_two_channel_matrices(n)
        op = AggregateOperator(channel_matrices=matrices).with_uniform_weights()
        energy = op.spectral_energy()
        assert energy > 0.0

    def test_spectral_energy_zero_for_zero_matrix(self):
        n = 3
        Z = csr_matrix((n, n), dtype=float)
        matrices = {RelationChannel.FINANCIAL: Z}
        op = AggregateOperator(channel_matrices=matrices)
        energy = op.spectral_energy()
        assert energy == pytest.approx(0.0)


class TestDensity:
    def test_density_in_zero_one_range(self):
        n = 4
        matrices = _make_two_channel_matrices(n)
        op = AggregateOperator(channel_matrices=matrices).with_uniform_weights()
        d = op.density()
        assert 0.0 <= d <= 1.0

    def test_density_zero_for_zero_matrix(self):
        n = 3
        Z = csr_matrix((n, n), dtype=float)
        matrices = {RelationChannel.FINANCIAL: Z}
        op = AggregateOperator(channel_matrices=matrices)
        assert op.density() == pytest.approx(0.0)

    def test_density_one_for_full_matrix(self):
        n = 2
        F = csr_matrix(np.ones((n, n)))
        matrices = {RelationChannel.FINANCIAL: F}
        op = AggregateOperator(channel_matrices=matrices)
        assert op.density() == pytest.approx(1.0)
