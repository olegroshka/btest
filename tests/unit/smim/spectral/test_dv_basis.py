"""Tests for DirectedVariationDecomposer (M3.1-T5)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer


class TestDirectedVariationDecomposer:
    def setup_method(self) -> None:
        self.decomposer = DirectedVariationDecomposer()
        rng = np.random.default_rng(13)
        # Simple adjacency-like matrix
        A = rng.uniform(0, 1, (6, 6))
        np.fill_diagonal(A, 0)
        self.A = A

    def test_method_is_directed_variation(self) -> None:
        assert self.decomposer.method == DecompositionMethod.DIRECTED_VARIATION

    def test_basis_shape(self) -> None:
        frame = self.decomposer.decompose(self.A, k=4)
        assert frame.basis.shape == (6, 4)

    def test_output_basis_orthonormal(self) -> None:
        """Laplacian eigenvectors should be orthonormal."""
        frame = self.decomposer.decompose(self.A, k=6)
        U = frame.basis
        np.testing.assert_allclose(U.T @ U, np.eye(6), atol=1e-10)

    def test_dv_values_non_decreasing(self) -> None:
        """Rayleigh quotients u^T L u should be non-decreasing across modes."""
        frame = self.decomposer.decompose(self.A, k=4)
        L = frame.metadata["laplacian"]
        U = frame.basis
        rayleigh = np.array([float(U[:, k].T @ L @ U[:, k]) for k in range(frame.K)])
        # Each should be >= the previous
        for i in range(1, len(rayleigh)):
            assert rayleigh[i] >= rayleigh[i - 1] - 1e-10, \
                f"DV values not non-decreasing: {rayleigh}"

    def test_eigenvalues_non_negative(self) -> None:
        """Laplacian eigenvalues are non-negative."""
        frame = self.decomposer.decompose(self.A, k=4)
        assert np.all(frame.eigenvalues.real >= -1e-10)

    def test_k_clamps_to_N(self) -> None:
        frame = self.decomposer.decompose(self.A, k=100)
        assert frame.K == 6

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        frame = self.decomposer.decompose(self.A, k=3)
        assert isinstance(frame, ModalFrame)
