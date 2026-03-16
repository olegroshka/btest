"""Tests for NarrativeEdgeEstimator (M2.1-T3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import NarrativeEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.graph.edges.narrative import NarrativeEdgeEstimator
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


def _make_text_signals(registry: ActorRegistry) -> tuple[pd.DataFrame, DateRange]:
    """Create text signals where actor1 and actor2 share topics, actor3 is different."""
    dates = pd.date_range("2020-01-01", periods=3, freq="QS")

    data = {
        "actor1": [
            "energy investment capex oil gas infrastructure",
            "capital expenditure energy sector expansion",
            "investment energy infrastructure projects",
        ],
        "actor2": [
            "energy investment capex renewable power",
            "capital investment energy oil drilling",
            "energy infrastructure capex spending",
        ],
        "actor3": [
            "climate change sea level rising ocean",
            "global warming sea temperature arctic ice",
            "climate sea level coast flood risk",
        ],
    }
    signals = pd.DataFrame(data, index=dates)
    dr = DateRange(start=dates[0], end=dates[-1])
    return signals, dr


class TestNarrativeEdgeEstimator:
    def test_output_is_csr_matrix(self):
        registry = _make_registry("actor1", "actor2", "actor3")
        signals, dr = _make_text_signals(registry)
        config = NarrativeEdgeConfig(similarity_threshold=0.3, method="tfidf")
        est = NarrativeEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        assert isinstance(result, csr_matrix)

    def test_output_shape_is_n_by_n(self):
        registry = _make_registry("actor1", "actor2", "actor3")
        signals, dr = _make_text_signals(registry)
        config = NarrativeEdgeConfig(similarity_threshold=0.3, method="tfidf")
        est = NarrativeEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        N = registry.N
        assert result.shape == (N, N)

    def test_channel_is_narrative(self):
        est = NarrativeEdgeEstimator(NarrativeEdgeConfig())
        assert est.channel == RelationChannel.NARRATIVE

    def test_similar_actors_share_edge(self):
        """actor1 and actor2 share energy/investment topics → edge should exist."""
        registry = _make_registry("actor1", "actor2", "actor3")
        signals, dr = _make_text_signals(registry)
        # Use low threshold to ensure similar actors connect
        config = NarrativeEdgeConfig(similarity_threshold=0.1, method="tfidf")
        est = NarrativeEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)

        a1_idx = registry.index_of("actor1")
        a2_idx = registry.index_of("actor2")
        arr = result.toarray()

        # At least one direction should have an edge
        assert arr[a1_idx, a2_idx] > 0 or arr[a2_idx, a1_idx] > 0, (
            "Expected similarity edge between actor1 and actor2 (both have energy/investment text)"
        )

    def test_diagonal_is_zero(self):
        """Self-similarity entries should be zeroed out."""
        registry = _make_registry("actor1", "actor2", "actor3")
        signals, dr = _make_text_signals(registry)
        config = NarrativeEdgeConfig(similarity_threshold=0.0, method="tfidf")
        est = NarrativeEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        arr = result.toarray()
        assert np.all(np.diag(arr) == 0.0)

    def test_threshold_applied(self):
        """With threshold=0.99, very few edges should exist."""
        registry = _make_registry("actor1", "actor2", "actor3")
        signals, dr = _make_text_signals(registry)
        config = NarrativeEdgeConfig(similarity_threshold=0.99, method="tfidf")
        est = NarrativeEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        # Very high threshold → no edges
        assert result.nnz == 0
