"""Tests for smim/interfaces.py — enums, dataclasses, and Protocol structure."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from quantdsl_backtest.smim.interfaces import (
    Actor,
    ActorRegistry,
    ActorType,
    BenchmarkClass,
    DateRange,
    DecompositionMethod,
    DiagnosticResult,
    FalsificationResult,
    FilteredState,
    GapResult,
    Layer,
    ModalFrame,
    RelationChannel,
    DataAdapter,
    EdgeEstimator,
    EmergenceDiagnostic,
    FalsificationTest,
    InvestmentIntensityMapper,
    ModeSelector,
    SpectralDecomposer,
    StateFilter,
    BenchmarkComputer,
)


class TestEnums:
    def test_actor_type_values(self) -> None:
        assert ActorType.CENTRAL_BANK == "central_bank"
        assert ActorType.LARGE_FIRM == "large_firm"

    def test_layer_values(self) -> None:
        assert Layer.EXOGENOUS == 0
        assert Layer.DOWNSTREAM == 3

    def test_relation_channel_codes(self) -> None:
        assert RelationChannel.REGULATORY == "C1"
        assert RelationChannel.NARRATIVE == "C4"

    def test_benchmark_class_coverage(self) -> None:
        classes = {bc.value for bc in BenchmarkClass}
        assert "predictive" in classes
        assert "structural" in classes

    def test_decomposition_methods(self) -> None:
        assert DecompositionMethod.SCHUR == "schur"
        assert DecompositionMethod.DMD == "dmd"


class TestDataClasses:
    def _make_actor(self, actor_id: str = "a1") -> Actor:
        return Actor(
            actor_id=actor_id,
            name="Test Actor",
            actor_type=ActorType.LARGE_FIRM,
            layer=Layer.TRANSMISSION,
            geography="US",
            sector="energy",
        )

    def test_actor_frozen(self) -> None:
        actor = self._make_actor()
        with pytest.raises(Exception):
            actor.actor_id = "changed"  # type: ignore[misc]

    def test_actor_registry_n(self) -> None:
        actors = [self._make_actor(f"a{i}") for i in range(5)]
        registry = ActorRegistry(actors=actors)
        assert registry.N == 5

    def test_modal_frame_properties(self) -> None:
        basis = np.random.randn(10, 3)
        eigenvalues = np.array([1.0 + 0j, 0.5 + 0.1j, 0.2 + 0j])
        frame = ModalFrame(
            basis=basis,
            eigenvalues=eigenvalues,
            method=DecompositionMethod.SCHUR,
        )
        assert frame.N == 10
        assert frame.K == 3

    def test_gap_result_requires_benchmark_class(self) -> None:
        gaps = np.zeros((5, 10))
        benchmarks = np.ones((5, 10))
        result = GapResult(
            gaps=gaps,
            benchmark_class=BenchmarkClass.PREDICTIVE,
            benchmarks=benchmarks,
        )
        assert result.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_filtered_state_shapes(self) -> None:
        T, K, M = 20, 4, 2
        state = FilteredState(
            alpha_filtered=np.zeros((T, K)),
            alpha_predicted=np.zeros((T, K)),
            regime_probs=np.ones((T, M)) / M,
            log_likelihood=-100.0,
            regimes_selected=M,
        )
        assert state.alpha_filtered.shape == (T, K)
        assert state.regime_probs.shape == (T, M)

    def test_falsification_result_passes_flag(self) -> None:
        result = FalsificationResult(
            test_name="shuffled_edges",
            observed_statistic=0.8,
            null_distribution=np.random.uniform(0, 0.5, 100),
            p_value=0.02,
            passes=True,
        )
        assert result.passes is True


class TestProtocolCheckability:
    """Smoke-test that runtime_checkable protocols can be checked with isinstance."""

    def test_data_adapter_is_protocol(self) -> None:
        assert hasattr(DataAdapter, "__protocol_attrs__") or hasattr(
            DataAdapter, "_is_protocol"
        )

    def test_edge_estimator_is_protocol(self) -> None:
        assert hasattr(EdgeEstimator, "__protocol_attrs__") or hasattr(
            EdgeEstimator, "_is_protocol"
        )
