"""Stub tests for smim/signals/ bridge to btest DSL (M6.x tasks)."""

import pytest


@pytest.mark.xfail(reason="GapSignalBridge not yet implemented", strict=False)
class TestGapSignalBridge:
    def test_converts_gap_result_to_btest_signal(self) -> None:
        pytest.fail("not implemented")

    def test_output_conforms_to_dsl_signal_convention(self) -> None:
        pytest.fail("not implemented")

    def test_benchmark_label_preserved_in_metadata(self) -> None:
        pytest.fail("not implemented")
