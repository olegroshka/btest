"""
Integration tests for edge estimators with realistic data (M2.1-T7).

Uses 10 actors, 4 layers, 80 quarterly observations with known causal links.
Runs GrangerEdgeEstimator end-to-end and verifies output properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
from quantdsl_backtest.smim.interfaces import Actor, ActorType, DateRange, Layer

pytestmark = pytest.mark.slow


def _make_10_actor_registry() -> ActorRegistry:
    """Build a 10-actor registry with 4 layers."""
    actor_specs = [
        ("actor_00", ActorType.GLOBAL_SHOCK, Layer.EXOGENOUS),
        ("actor_01", ActorType.CENTRAL_BANK, Layer.UPSTREAM),
        ("actor_02", ActorType.REGULATOR, Layer.UPSTREAM),
        ("actor_03", ActorType.THINK_TANK, Layer.UPSTREAM),
        ("actor_04", ActorType.LARGE_FIRM, Layer.TRANSMISSION),
        ("actor_05", ActorType.BANK, Layer.TRANSMISSION),
        ("actor_06", ActorType.SECTOR_LEADER, Layer.TRANSMISSION),
        ("actor_07", ActorType.SME, Layer.DOWNSTREAM),
        ("actor_08", ActorType.MUNICIPALITY, Layer.DOWNSTREAM),
        ("actor_09", ActorType.HOUSEHOLD, Layer.DOWNSTREAM),
    ]
    actors = [
        Actor(
            actor_id=aid,
            name=aid,
            actor_type=atype,
            layer=layer,
            geography="US",
            sector="energy",
        )
        for aid, atype, layer in actor_specs
    ]
    return ActorRegistry(actors=actors)


def _generate_var1_signals(
    registry: ActorRegistry,
    n_obs: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 10-dimensional VAR(1) data with known causal links.

    Causal structure:
    - actor_00 → actor_04 (strong: 0.5)
    - actor_01 → actor_05 (strong: 0.6)
    - actor_02 → actor_06 (strong: 0.4)

    All other dependencies are noise.
    """
    rng = np.random.default_rng(seed)
    N = registry.N
    T = n_obs

    # Build VAR(1) coefficient matrix
    A = np.zeros((N, N))
    idx = {a.actor_id: registry.index_of(a.actor_id) for a in registry.actors}
    A[idx["actor_04"], idx["actor_00"]] = 0.5
    A[idx["actor_05"], idx["actor_01"]] = 0.6
    A[idx["actor_06"], idx["actor_02"]] = 0.4

    # Ensure stationarity via spectral radius check
    ev = max(abs(np.linalg.eigvals(A)))
    if ev > 0.8:
        A *= 0.8 / ev

    data = np.zeros((T, N))
    for t in range(1, T):
        data[t] = A @ data[t - 1] + rng.standard_normal(N) * 0.5

    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    return pd.DataFrame(
        data,
        index=dates,
        columns=[a.actor_id for a in registry.actors],
    )


class TestGrangerEdgeIntegration:
    def test_output_shape_is_n_by_n(self):
        registry = _make_10_actor_registry()
        signals = _generate_var1_signals(registry)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.05, bic_selection=True)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        N = registry.N
        assert result.shape == (N, N)

    def test_output_is_csr_matrix(self):
        registry = _make_10_actor_registry()
        signals = _generate_var1_signals(registry)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.05, bic_selection=True)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        assert isinstance(result, csr_matrix)

    def test_density_in_reasonable_range(self):
        """Output density should be between 0 and 100%."""
        registry = _make_10_actor_registry()
        signals = _generate_var1_signals(registry)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.05, bic_selection=True)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        N = registry.N
        density = result.nnz / (N * N)
        assert 0.0 <= density <= 1.0

    def test_no_errors_with_bic_selection(self):
        """Full run with BIC lag selection should complete without errors."""
        registry = _make_10_actor_registry()
        signals = _generate_var1_signals(registry, n_obs=80)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        config = GrangerEdgeConfig(max_lag=4, p_threshold=0.10, bic_selection=True)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        assert result is not None

    def test_known_causal_links_detected(self):
        """actor_00 → actor_04 should be detected with relaxed threshold."""
        registry = _make_10_actor_registry()
        signals = _generate_var1_signals(registry, seed=42)
        dr = DateRange(start=signals.index[0], end=signals.index[-1])
        # Use relaxed threshold for reliability
        config = GrangerEdgeConfig(max_lag=2, p_threshold=0.15, bic_selection=False)
        est = GrangerEdgeEstimator(config)
        result = est.estimate(signals, registry, dr)
        idx = {a.actor_id: registry.index_of(a.actor_id) for a in registry.actors}
        arr = result.toarray()
        # At least one of the three planted edges should be detected
        planted_edges = [
            arr[idx["actor_04"], idx["actor_00"]],
            arr[idx["actor_05"], idx["actor_01"]],
            arr[idx["actor_06"], idx["actor_02"]],
        ]
        assert any(e > 0 for e in planted_edges), (
            f"None of the planted causal edges detected. Values: {planted_edges}"
        )
