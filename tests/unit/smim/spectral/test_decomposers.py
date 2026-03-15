"""Stub tests for smim/spectral/ decomposers (M3.x tasks)."""

import pytest


@pytest.mark.xfail(reason="SchurDecomposer not yet implemented", strict=False)
class TestSchurDecomposer:
    def test_decompose_returns_modal_frame(self) -> None:
        pytest.fail("not implemented")

    def test_method_is_schur(self) -> None:
        pytest.fail("not implemented")

    def test_eigenvalue_count_equals_k(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="PolarDecomposer not yet implemented", strict=False)
class TestPolarDecomposer:
    def test_decompose_returns_modal_frame(self) -> None:
        pytest.fail("not implemented")

    def test_method_is_polar(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="HermitianDilationDecomposer not yet implemented", strict=False)
class TestHermitianDilationDecomposer:
    def test_decompose_returns_modal_frame(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="DMDDecomposer not yet implemented", strict=False)
class TestDMDDecomposer:
    def test_decompose_returns_modal_frame(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="ModeSelector not yet implemented", strict=False)
class TestModeSelector:
    def test_select_reduces_k(self) -> None:
        pytest.fail("not implemented")

    def test_mdl_criterion_applied(self) -> None:
        pytest.fail("not implemented")

    def test_compressibility_threshold_respected(self) -> None:
        pytest.fail("not implemented")
