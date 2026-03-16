"""Tests for GrangerEdgeEstimator (M2.1-T2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
from quantdsl_backtest.smim.interfaces import Actor, ActorType, DateRange, Layer, RelationChannel


def _make_registry(*actor_ids: str) -> ActorRegistry:
    actors = [
        Actor(
            actor_id=aid,
            name=aid,
            actor_type=ActorType.LARGE_FIRM,
            layer=Layer.TRANSMISSION,
            geography="US",
            sector="energy",
        )
        for aid in actor_ids
    ]
    return ActorRegistry(actors=actors)


def _make_var1_data(n: int = 60, seed: int = 42) -> tuple[pd.DataFrame, ActorRegistry]:
    """Generate synthetic VAR(1): A causes B (0.8 coefficient), B does not cause A.

    A_t = noise
    B_t = 0.8 * A_{t-1} + noise
    """
    rng = np.random.default_rng(seed)
    a_data = rng.standard_normal(n)
    noise_b = rng.standard_normal(n) * 0.3
    b_data = np.zeros(n)
    for t in range(1, n):
        b_data[t] = 0.8 * a_data[t - 1] + noise_b[t]

    dates = pd.date_range("2005-01-01", periods=n, freq="ME")
    registry = _make_registry("actor_A", "actor_B")
    signals = pd.DataFrame(
        {"actor_A": a_data, "actor_B": b_data},
        index=dates,
    )
    return signals, registry


class TestGrangerEdgeEstimator:
    def test_output_is_csr_matrix(self):
        signals, registry = _make_var1_data()
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.05, bic_selection=False)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        assert isinstance(result, csr_matrix)

    def test_output_shape_is_n_by_n(self):
        signals, registry = _make_var1_data()
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.05, bic_selection=False)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        N = registry.N
        assert result.shape == (N, N)

    def test_channel_is_financial(self):
        est = GrangerEdgeEstimator(GrangerEdgeConfig())
        assert est.channel == RelationChannel.FINANCIAL

    def test_recovers_a_causes_b_edge(self):
        """A→B edge should be detected (A[b_idx, a_idx] > 0)."""
        signals, registry = _make_var1_data(n=60, seed=42)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        # Use relaxed threshold to make test robust
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.10, bic_selection=False)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)

        a_idx = registry.index_of("actor_A")
        b_idx = registry.index_of("actor_B")

        arr = result.toarray()
        # A causes B: arr[b_idx, a_idx] should be > 0
        assert arr[b_idx, a_idx] > 0, (
            f"Expected A→B edge A[{b_idx}, {a_idx}] > 0, got {arr[b_idx, a_idx]}"
        )

    def test_diagonal_is_zero(self):
        """Self-causation entries should always be zero."""
        signals, registry = _make_var1_data()
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=1, p_threshold=0.05, bic_selection=False)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        arr = result.toarray()
        assert np.all(np.diag(arr) == 0.0)
