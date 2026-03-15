"""Stub tests for smim/dynamics/ state-space filters (M4.x tasks)."""

import pytest


@pytest.mark.xfail(reason="KalmanFilter not yet implemented", strict=False)
class TestKalmanFilter:
    def test_filter_returns_filtered_state(self) -> None:
        pytest.fail("not implemented")

    def test_single_regime(self) -> None:
        pytest.fail("not implemented")

    def test_alpha_filtered_shape(self) -> None:
        pytest.fail("not implemented")

    def test_em_estimation_converges(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="KimFilter not yet implemented", strict=False)
class TestKimFilter:
    def test_filter_returns_filtered_state(self) -> None:
        pytest.fail("not implemented")

    def test_regime_probs_sum_to_one(self) -> None:
        pytest.fail("not implemented")

    def test_a5_regime_persistence_minimum_8_quarters(self) -> None:
        """A5: average regime duration > 8 quarters."""
        pytest.fail("not implemented")
