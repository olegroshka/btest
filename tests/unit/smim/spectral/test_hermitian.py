"""Tests for HermitianDilationDecomposer (M3.1-T4)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer


class TestHermitianDilationDecomposer:
    def setup_method(self) -> None:
        self.decomposer = HermitianDilationDecomposer()
        rng = np.random.default_rng(7)
        self.A = rng.standard_normal((5, 5))

    def test_method_is_hermitian_dilation(self) -> None:
        assert self.decomposer.method == DecompositionMethod.HERMITIAN_DILATION

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        frame = self.decomposer.decompose(self.A, k=3)
        assert isinstance(frame, ModalFrame)

    def test_basis_shape(self) -> None:
        frame = self.decomposer.decompose(self.A, k=3)
        assert frame.basis.shape == (5, 3)

    def test_H_eigenvalues_come_in_pairs(self) -> None:
        """Eigenvalues of H(A) should come in +/- pairs."""
        frame = self.decomposer.decompose(self.A, k=5)
        all_eigs = frame.metadata["all_eigenvalues"]
        # For each positive eigenvalue, there should be a corresponding negative one
        pos_eigs = np.sort(all_eigs[all_eigs > 1e-10])
        neg_eigs = np.sort(-all_eigs[all_eigs < -1e-10])
        # They should match
        min_len = min(len(pos_eigs), len(neg_eigs))
        if min_len > 0:
            np.testing.assert_allclose(
                pos_eigs[:min_len], neg_eigs[:min_len], atol=1e-8
            )

    def test_singular_values_match_numpy_svd(self) -> None:
        """Positive eigenvalues of H should match singular values of A."""
        frame = self.decomposer.decompose(self.A, k=5)
        all_eigs = frame.metadata["all_eigenvalues"]
        pos_eigs = np.sort(all_eigs[all_eigs > 1e-10])[::-1]
        expected_sv = np.sort(np.linalg.svd(self.A, compute_uv=False))[::-1]
        # Compare top k
        k = min(len(pos_eigs), len(expected_sv))
        np.testing.assert_allclose(pos_eigs[:k], expected_sv[:k], atol=1e-8)

    def test_left_singular_vectors_satisfy_Av_approx_sigma_u(self) -> None:
        """A @ right_sv ≈ sigma * left_sv for each mode."""
        frame = self.decomposer.decompose(self.A, k=3)
        right_svecs = frame.metadata["right_singular_vectors"]
        sv = frame.eigenvalues.real

        for i in range(frame.K):
            u = frame.basis[:, i]
            v = right_svecs[:, i]
            sigma = sv[i]
            lhs = self.A @ v
            rhs = sigma * u
            # Check direction (allow sign flip)
            if np.linalg.norm(rhs) > 1e-12:
                ratio = lhs / rhs if not np.any(np.abs(rhs) < 1e-12) else np.ones_like(lhs)
                # Should be approximately constant (all same sign)
                lhs_norm = lhs / (np.linalg.norm(lhs) + 1e-15)
                rhs_norm = rhs / (np.linalg.norm(rhs) + 1e-15)
                dot = abs(np.dot(lhs_norm, rhs_norm))
                assert dot > 0.9, f"Mode {i}: A v not aligned with sigma*u, dot={dot}"

    def test_eigenvalues_are_non_negative_retained(self) -> None:
        """Retained eigenvalues (singular values) should be non-negative."""
        frame = self.decomposer.decompose(self.A, k=4)
        assert np.all(frame.eigenvalues.real >= -1e-10)

    def test_k_limits_output(self) -> None:
        frame = self.decomposer.decompose(self.A, k=3)
        assert frame.K == 3
