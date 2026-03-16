"""Tests for AblationRunner (M2.5-T1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import EdgeConfig
from quantdsl_backtest.smim.graph.ablation import AblationResult, AblationRunner
from quantdsl_backtest.smim.interfaces import RelationChannel


def _random_sparse(n: int, seed: int = 0, scale: float = 1.0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = np.abs(rng.standard_normal((n, n))) * scale
    np.fill_diagonal(dense, 0.0)
    return csr_matrix(dense)


def _two_channel_matrices(n: int = 5) -> dict[RelationChannel, csr_matrix]:
    return {
        RelationChannel.FINANCIAL: _random_sparse(n, seed=0),
        RelationChannel.NARRATIVE: _random_sparse(n, seed=1),
    }


class TestRunChannelAblation:
    def test_returns_dict_with_one_key_per_channel(self):
        n = 4
        matrices = _two_channel_matrices(n)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run_channel_ablation(matrices)
        # Should have one entry per channel in the input
        assert len(result) == len(matrices)
        # Keys should be channel names
        assert "financial" in result
        assert "narrative" in result

    def test_values_are_floats(self):
        matrices = _two_channel_matrices()
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run_channel_ablation(matrices)
        for v in result.values():
            assert isinstance(v, float)

    def test_single_channel_ablation_returns_full_metric(self):
        """Removing the only channel gives a specific delta."""
        n = 3
        matrices = {RelationChannel.FINANCIAL: _random_sparse(n, seed=0)}
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run_channel_ablation(matrices)
        # With only one channel, removing it gives delta = full_metric - 0 = full_metric
        assert "financial" in result
        assert result["financial"] > 0  # Frobenius norm is always positive


class TestRunDensitySweep:
    def test_returns_dict_with_correct_keys(self):
        n = 4
        operator = _random_sparse(n, seed=2)
        config = EdgeConfig()
        runner = AblationRunner(config)
        densities = [0.1, 0.3, 0.5]
        result = runner.run_density_sweep(operator, densities=densities)
        assert set(result.keys()) == set(densities)

    def test_values_are_in_zero_one(self):
        n = 5
        operator = _random_sparse(n, seed=3)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run_density_sweep(operator)
        for v in result.values():
            assert 0.0 <= v <= 1.0 + 1e-10

    def test_default_densities_populated(self):
        n = 4
        operator = _random_sparse(n, seed=4)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run_density_sweep(operator)
        assert len(result) > 0


class TestAblationRun:
    def test_run_returns_ablation_result(self):
        n = 4
        matrices = _two_channel_matrices(n)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run(matrices)
        assert isinstance(result, AblationResult)

    def test_result_has_summary_dataframe(self):
        n = 4
        matrices = _two_channel_matrices(n)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run(matrices)
        assert isinstance(result.summary, pd.DataFrame)
        assert len(result.summary) > 0

    def test_result_channel_ablation_not_empty(self):
        n = 4
        matrices = _two_channel_matrices(n)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run(matrices)
        assert len(result.channel_ablation) > 0

    def test_result_density_sweep_not_empty(self):
        n = 4
        matrices = _two_channel_matrices(n)
        config = EdgeConfig()
        runner = AblationRunner(config)
        result = runner.run(matrices)
        assert len(result.density_sweep) > 0
