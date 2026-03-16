"""Tests for PolarDecomposer (M3.1-T3)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.polar import PolarDecomposer


class TestPolarDecomposer:
    def setup_method(self) -> None:
        self.decomposer = PolarDecomposer()
        rng = np.random.default_rng(99)
        self.A = rng.standard_normal((5, 5))

    def test_method_is_polar(self) -> None:
        assert self.decomposer.method == DecompositionMethod.POLAR

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        frame = self.decomposer.decompose(self.A, k=3)
        assert isinstance(frame, ModalFrame)

    def test_basis_shape(self) -> None:
        frame = self.decomposer.decompose(self.A, k=3)
        assert frame.basis.shape == (5, 3)

    def test_U_orthogonal(self) -> None:
        frame = self.decomposer.decompose(self.A, k=5)
        U = frame.metadata["U"]
        np.testing.assert_allclose(U.T @ U, np.eye(5), atol=1e-10)

    def test_P_symmetric(self) -> None:
        frame = self.decomposer.decompose(self.A, k=5)
        P = frame.metadata["P"]
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_P_positive_semidefinite(self) -> None:
        frame = self.decomposer.decompose(self.A, k=5)
        P = frame.metadata["P"]
        p_vals = np.linalg.eigvalsh(P)
        assert np.all(p_vals >= -1e-10), f"Non-PSD eigenvalues: {p_vals}"

    def test_reconstruction_U_P_equals_A(self) -> None:
        frame = self.decomposer.decompose(self.A, k=5)
        U = frame.metadata["U"]
        P = frame.metadata["P"]
        np.testing.assert_allclose(U @ P, self.A, atol=1e-10)

    def test_eigenvalues_descending_magnitude(self) -> None:
        frame = self.decomposer.decompose(self.A, k=5)
        mags = np.abs(frame.eigenvalues.real)
        assert list(mags) == sorted(mags, reverse=True)

    def test_k_clamps_to_N(self) -> None:
        frame = self.decomposer.decompose(self.A, k=100)
        assert frame.K == 5

    def test_eigenvalues_non_negative(self) -> None:
        """Eigenvalues of P are non-negative (singular values)."""
        frame = self.decomposer.decompose(self.A, k=5)
        assert np.all(frame.eigenvalues.real >= -1e-10)
