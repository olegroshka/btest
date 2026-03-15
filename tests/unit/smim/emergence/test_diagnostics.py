"""Stub tests for smim/emergence/ diagnostics (M5.x tasks)."""

import pytest


@pytest.mark.xfail(reason="PIDSynergy not yet implemented", strict=False)
class TestPIDSynergy:
    def test_compute_returns_diagnostic_result(self) -> None:
        pytest.fail("not implemented")

    def test_name_is_pid_synergy(self) -> None:
        pytest.fail("not implemented")

    def test_confidence_intervals_shape(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="TransferEntropyProfile not yet implemented", strict=False)
class TestTransferEntropyProfile:
    def test_compute_returns_diagnostic_result(self) -> None:
        pytest.fail("not implemented")

    def test_conditional_te_controlled_for_intermediaries(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="TopologicalComplexity not yet implemented", strict=False)
class TestTopologicalComplexity:
    def test_compute_returns_diagnostic_result(self) -> None:
        pytest.fail("not implemented")

    def test_min_window_count_respected(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="CriticalityIndex not yet implemented", strict=False)
class TestCriticalityIndex:
    def test_compute_returns_diagnostic_result(self) -> None:
        pytest.fail("not implemented")
