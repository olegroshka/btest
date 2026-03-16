"""Tests for smim/gaps/ — BenchmarkFactory (real) + implementation stubs (xfail)."""

import pytest

from quantdsl_backtest.smim.gaps import BenchmarkFactory, get_benchmark
from quantdsl_backtest.smim.interfaces import BenchmarkClass


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
