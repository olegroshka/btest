"""Tests for SchurDecomposer (M3.1-T2)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

# 4x4 test matrix with known eigenvalue structure
A4 = np.array([
    [2, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 3, 1],
    [0, 0, 0, 2],
], dtype=float)


class TestSchurDecomposer:
    def setup_method(self) -> None:
        self.decomposer = SchurDecomposer()

    def test_method_is_schur(self) -> None:
        assert self.decomposer.method == DecompositionMethod.SCHUR

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        frame = self.decomposer.decompose(A4, k=4)
        assert isinstance(frame, ModalFrame)

    def test_eigenvalue_count_equals_k(self) -> None:
        frame = self.decomposer.decompose(A4, k=3)
        assert frame.K == 3

    def test_eigenvalue_count_clamped_to_N(self) -> None:
        frame = self.decomposer.decompose(A4, k=100)
        assert frame.K == 4

    def test_basis_shape(self) -> None:
        frame = self.decomposer.decompose(A4, k=3)
        assert frame.basis.shape == (4, 3)

    def test_T_upper_triangular(self) -> None:
        frame = self.decomposer.decompose(A4, k=4)
        T = frame.metadata["T"]
        N = T.shape[0]
        for i in range(1, N):
            for j in range(i):
                assert abs(T[i, j]) < 1e-10, f"T[{i},{j}] = {T[i,j]}"

    def test_Q_unitary(self) -> None:
        frame = self.decomposer.decompose(A4, k=4)
        Q = frame.metadata["Q"]
        identity = Q.conj().T @ Q
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-10)

    def test_eigenvalues_match_diag_T(self) -> None:
        frame = self.decomposer.decompose(A4, k=4)
        T = frame.metadata["T"]
        full_eigs = frame.metadata["full_eigenvalues"]
        np.testing.assert_allclose(np.sort(np.abs(full_eigs)), np.sort(np.abs(np.diag(T))), atol=1e-10)

    def test_reconstruction(self) -> None:
        """Q @ T @ Q^H should reconstruct A."""
        frame = self.decomposer.decompose(A4, k=4)
        T = frame.metadata["T"]
        Q = frame.metadata["Q"]
        A_reconstructed = (Q @ T @ Q.conj().T).real
        np.testing.assert_allclose(A_reconstructed, A4, atol=1e-10)

    def test_eigenvalues_complex_type(self) -> None:
        frame = self.decomposer.decompose(A4, k=4)
        assert np.iscomplexobj(frame.eigenvalues)

    def test_on_random_symmetric_matrix(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        A = A + A.T  # symmetric
        frame = self.decomposer.decompose(A, k=4)
        T = frame.metadata["T"]
        Q = frame.metadata["Q"]
        A_rec = (Q @ T @ Q.conj().T).real
        np.testing.assert_allclose(A_rec, A, atol=1e-9)
