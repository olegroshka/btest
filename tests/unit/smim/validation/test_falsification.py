"""Stub tests for smim/validation/ falsification tests (M6.x tasks)."""

import pytest


@pytest.mark.xfail(reason="ShuffledEdgesTest not yet implemented", strict=False)
class TestShuffledEdgesTest:
    def test_run_returns_falsification_result(self) -> None:
        pytest.fail("not implemented")

    def test_null_distribution_has_100_samples(self) -> None:
        pytest.fail("not implemented")

    def test_test_name_is_shuffled_edges(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="LagDestroyedTest not yet implemented", strict=False)
class TestLagDestroyedTest:
    def test_run_returns_falsification_result(self) -> None:
        pytest.fail("not implemented")

    def test_p_value_in_unit_interval(self) -> None:
        pytest.fail("not implemented")
