"""Stub tests for smim/gaps/ benchmark computers (M5.x / M6.x tasks)."""

import pytest


@pytest.mark.xfail(reason="PredictiveBenchmark not yet implemented", strict=False)
class TestPredictiveBenchmark:
    def test_compute_returns_gap_result(self) -> None:
        pytest.fail("not implemented")

    def test_benchmark_class_is_predictive(self) -> None:
        pytest.fail("not implemented")

    def test_gap_result_has_benchmark_label(self) -> None:
        """Mandatory rule: every GapResult must carry a BenchmarkClass."""
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="StructuralBenchmark not yet implemented", strict=False)
class TestStructuralBenchmark:
    def test_compute_returns_gap_result(self) -> None:
        pytest.fail("not implemented")

    def test_benchmark_class_is_structural(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="ModalBenchmark not yet implemented", strict=False)
class TestModalBenchmark:
    def test_compute_returns_gap_result(self) -> None:
        pytest.fail("not implemented")

    def test_modal_attribution_shape(self) -> None:
        pytest.fail("not implemented")
