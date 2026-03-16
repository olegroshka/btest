"""Tests for AbstractEdgeEstimator and EdgeEstimatorFactory (M2.1-T1, M2.1-T6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import EdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.graph.edges.base import AbstractEdgeEstimator, EdgeEstimatorFactory
from quantdsl_backtest.smim.interfaces import Actor, ActorType, DateRange, Layer, RelationChannel


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_actors(n: int = 3) -> ActorRegistry:
    actors = [
        Actor(
            actor_id=f"actor{i}",
            name=f"Actor {i}",
            actor_type=ActorType.LARGE_FIRM,
            layer=Layer.TRANSMISSION,
            geography="US",
            sector="energy",
        )
        for i in range(n)
    ]
    return ActorRegistry(actors=actors)


def _make_signals(actors: ActorRegistry, n_rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="QS")
    data = np.random.default_rng(42).standard_normal((n_rows, actors.N))
    return pd.DataFrame(
        data,
        index=dates,
        columns=[a.actor_id for a in actors.actors],
    )


def _make_date_range(signals: pd.DataFrame) -> DateRange:
    return DateRange(start=signals.index[0], end=signals.index[-1])


# ─── Concrete stub for testing AbstractEdgeEstimator ─────────────────────────

class StubEstimator(AbstractEdgeEstimator):
    channel = RelationChannel.FINANCIAL

    def __init__(self, config=None):
        self._config = config

    def estimate(self, signals, actors, date_range):
        N = actors.N
        return csr_matrix((N, N), dtype=float)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAbstractEdgeEstimatorValidation:
    def test_validate_inputs_raises_for_extra_column(self):
        actors = _make_actors(2)
        signals = pd.DataFrame(
            {"actor0": [1.0], "actor1": [2.0], "actor99": [3.0]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        est = StubEstimator()
        with pytest.raises(ValueError, match="actor99"):
            est._validate_inputs(signals, actors, dr)

    def test_validate_inputs_raises_for_missing_column(self):
        actors = _make_actors(3)
        # Only 2 columns, one actor missing
        signals = pd.DataFrame(
            {"actor0": [1.0], "actor1": [2.0]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        est = StubEstimator()
        with pytest.raises(ValueError):
            est._validate_inputs(signals, actors, dr)

    def test_validate_inputs_raises_for_high_nan_fraction(self):
        actors = _make_actors(2)
        data = {"actor0": [np.nan] * 9 + [1.0], "actor1": [1.0] * 10}
        signals = pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=10))
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        est = StubEstimator()
        with pytest.raises(ValueError, match="NaN"):
            est._validate_inputs(signals, actors, dr)

    def test_validate_inputs_passes_for_valid_data(self):
        actors = _make_actors(3)
        signals = _make_signals(actors)
        dr = _make_date_range(signals)
        est = StubEstimator()
        # Should not raise
        est._validate_inputs(signals, actors, dr)


class TestToSparse:
    def test_zeros_below_threshold(self):
        est = StubEstimator()
        dense = np.array([[0.0, 0.5, 0.1], [0.3, 0.0, 0.8], [0.05, 0.2, 0.0]])
        result = est._to_sparse(dense, threshold=0.2)
        assert isinstance(result, csr_matrix)
        arr = result.toarray()
        # Entries <= 0.2 should be zero
        assert arr[0, 1] == 0.5   # 0.5 > 0.2 → kept
        assert arr[0, 2] == 0.0   # 0.1 <= 0.2 → zeroed
        assert arr[1, 0] == 0.3   # 0.3 > 0.2 → kept
        assert arr[2, 0] == 0.0   # 0.05 <= 0.2 → zeroed

    def test_produces_csr_matrix(self):
        est = StubEstimator()
        dense = np.eye(4)
        result = est._to_sparse(dense, threshold=0.0)
        assert isinstance(result, csr_matrix)

    def test_correct_sparsity(self):
        est = StubEstimator()
        dense = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = est._to_sparse(dense, threshold=0.0)
        assert result.nnz == 2


class TestApplyPitFilter:
    def test_keeps_rows_within_date_range(self):
        est = StubEstimator()
        dates = pd.date_range("2020-01-01", periods=8, freq="QS")
        signals = pd.DataFrame({"a": range(8)}, index=dates)
        dr = DateRange(start=dates[2], end=dates[5])
        filtered = est._apply_pit_filter(signals, dr)
        assert filtered.index[0] == dates[2]
        assert filtered.index[-1] == dates[5]
        assert len(filtered) == 4

    def test_excludes_rows_outside_range(self):
        est = StubEstimator()
        dates = pd.date_range("2020-01-01", periods=8, freq="QS")
        signals = pd.DataFrame({"a": range(8)}, index=dates)
        dr = DateRange(start=dates[3], end=dates[3])
        filtered = est._apply_pit_filter(signals, dr)
        assert len(filtered) == 1


class TestEdgeEstimatorFactory:
    def test_raises_key_error_for_unknown_channel(self):
        factory = EdgeEstimatorFactory()
        config = EdgeConfig()
        with pytest.raises(KeyError, match="unknown_channel"):
            factory.create("unknown_channel", config)

    def test_register_and_create(self):
        factory = EdgeEstimatorFactory()
        factory.register("financial", StubEstimator)
        config = EdgeConfig()
        estimator = factory.create("financial", config)
        assert isinstance(estimator, StubEstimator)

    def test_estimate_all_with_mocked_estimators(self):
        """estimate_all returns dict with correct RelationChannel keys."""
        factory = EdgeEstimatorFactory()

        # Create mock estimator classes that return known matrices
        N = 3

        class MockFinancialEstimator(StubEstimator):
            channel = RelationChannel.FINANCIAL

        class MockNarrativeEstimator(StubEstimator):
            channel = RelationChannel.NARRATIVE

        factory.register("financial", MockFinancialEstimator)
        factory.register("narrative", MockNarrativeEstimator)

        actors = _make_actors(N)
        signals = _make_signals(actors)
        dr = _make_date_range(signals)
        config = EdgeConfig(channels=["financial", "narrative"])

        results = factory.estimate_all(signals, actors, dr, config)

        assert RelationChannel.FINANCIAL in results
        assert RelationChannel.NARRATIVE in results
        assert len(results) == 2
        for mat in results.values():
            assert isinstance(mat, csr_matrix)
            assert mat.shape == (N, N)
