"""Stub tests for smim/dynamics/phase_transition.py (M4.x tasks)."""

import pytest


@pytest.mark.xfail(reason="PhaseTransitionDetector not yet implemented", strict=False)
class TestPhaseTransitionDetector:
    def test_order_parameter_shape(self) -> None:
        pytest.fail("not implemented")

    def test_criticality_index_scalar(self) -> None:
        pytest.fail("not implemented")

    def test_gl_landscape_computation(self) -> None:
        pytest.fail("not implemented")
