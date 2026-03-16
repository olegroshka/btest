"""Tests for smim/gaps/ — BenchmarkFactory (real) + benchmark implementations."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps import BenchmarkFactory, get_benchmark
from quantdsl_backtest.smim.gaps.modal import ModalBenchmark
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.gaps.structural import StructuralBenchmarkComputer
from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    DecompositionMethod,
    FilteredState,
    GapResult,
    ModalFrame,
)


# ═══════════════════════════════════════════════════════════
# BenchmarkFactory (real tests — these must always pass)
# ═══════════════════════════════════════════════════════════

class TestBenchmarkFactory:
    def setup_method(self) -> None:
        self.factory = BenchmarkFactory()

    def test_all_benchmark_classes_registered(self) -> None:
        for cls in BenchmarkClass:
            computer = self.factory.get(cls)
            assert computer is not None

    def test_get_returns_correct_benchmark_class_property(self) -> None:
        for cls in BenchmarkClass:
            computer = self.factory.get(cls)
            assert computer.benchmark_class == cls

    def test_placeholders_raise_not_implemented(self) -> None:
        for cls in BenchmarkClass:
            computer = self.factory.get(cls)
            with pytest.raises(NotImplementedError, match=cls.value):
                computer.compute(None, None, None, None)  # type: ignore[arg-type]

    def test_register_replaces_implementation(self) -> None:
        class _Stub:
            @property
            def benchmark_class(self) -> BenchmarkClass:
                return BenchmarkClass.PREDICTIVE
            def compute(self, *args: object, **kwargs: object) -> None:
                return None

        stub = _Stub()
        self.factory.register(BenchmarkClass.PREDICTIVE, stub)  # type: ignore[arg-type]
        assert self.factory.get(BenchmarkClass.PREDICTIVE) is stub

    def test_registered_classes_covers_all(self) -> None:
        assert set(self.factory.registered_classes()) == set(BenchmarkClass)

    def test_module_level_get_benchmark(self) -> None:
        for cls in BenchmarkClass:
            computer = get_benchmark(cls)
            assert computer.benchmark_class == cls


# ═══════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════

def _make_fixtures(N: int = 4, T: int = 20, K: int = 3):
    """Build minimal ModalFrame and FilteredState for testing."""
    rng = np.random.default_rng(42)

    # Modal basis (N, K) — orthonormal columns
    basis_raw = rng.standard_normal((N, K))
    basis, _ = np.linalg.qr(basis_raw)
    basis = basis[:, :K]

    modal_frame = ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )

    alpha_filtered = rng.standard_normal((T, K))
    alpha_predicted = rng.standard_normal((T, K))
    regime_probs = np.ones((T, 1))

    filtered_state = FilteredState(
        alpha_filtered=alpha_filtered,
        alpha_predicted=alpha_predicted,
        regime_probs=regime_probs,
        log_likelihood=-100.0,
        regimes_selected=1,
    )

    observations = rng.standard_normal((N, T))
    return observations, filtered_state, modal_frame


# ═══════════════════════════════════════════════════════════
# PredictiveBenchmark
# ═══════════════════════════════════════════════════════════

class TestPredictiveBenchmark:
    def setup_method(self) -> None:
        self.benchmark = PredictiveBenchmark()
        self.observations, self.filtered_state, self.modal_frame = _make_fixtures()

    def test_compute_returns_gap_result(self) -> None:
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert isinstance(result, GapResult)

    def test_benchmark_class_is_predictive(self) -> None:
        assert self.benchmark.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_gap_result_has_benchmark_label(self) -> None:
        """Mandatory rule: every GapResult must carry a BenchmarkClass."""
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_gap_shape(self) -> None:
        N, T = self.observations.shape
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.gaps.shape == (N, T)
        assert result.benchmarks.shape == (N, T)


# ═══════════════════════════════════════════════════════════
# StructuralBenchmark
# ═══════════════════════════════════════════════════════════

class TestStructuralBenchmark:
    def setup_method(self) -> None:
        self.benchmark = StructuralBenchmarkComputer()
        self.observations, self.filtered_state, self.modal_frame = _make_fixtures()

    def test_compute_returns_gap_result(self) -> None:
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert isinstance(result, GapResult)

    def test_benchmark_class_is_structural(self) -> None:
        assert self.benchmark.benchmark_class == BenchmarkClass.STRUCTURAL

    def test_gap_result_carries_structural_label(self) -> None:
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.benchmark_class == BenchmarkClass.STRUCTURAL

    def test_gap_shape(self) -> None:
        N, T = self.observations.shape
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.gaps.shape == (N, T)


# ═══════════════════════════════════════════════════════════
# ModalBenchmark
# ═══════════════════════════════════════════════════════════

class TestModalBenchmark:
    def setup_method(self) -> None:
        self.benchmark = ModalBenchmark()
        self.observations, self.filtered_state, self.modal_frame = _make_fixtures()

    def test_compute_returns_gap_result(self) -> None:
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert isinstance(result, GapResult)

    def test_benchmark_class_is_modal(self) -> None:
        assert self.benchmark.benchmark_class == BenchmarkClass.MODAL

    def test_modal_attribution_shape(self) -> None:
        N, T = self.observations.shape
        K = self.modal_frame.K
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.modal_attribution is not None
        assert result.modal_attribution.shape == (N, T, K)

    def test_gap_result_carries_modal_label(self) -> None:
        result = self.benchmark.compute(
            self.observations, self.filtered_state, self.modal_frame, actors=None
        )
        assert result.benchmark_class == BenchmarkClass.MODAL
