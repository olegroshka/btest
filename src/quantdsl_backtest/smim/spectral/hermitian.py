"""Hermitian dilation decomposer (M3.1-T4)."""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer
from quantdsl_backtest.smim.compute.linalg import hermitian_dilation_decompose


class HermitianDilationDecomposer(AbstractSpectralDecomposer):
    """Spectral decomposer via Hermitian dilation H(A) = [[0, A], [A^T, 0]]."""

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.HERMITIAN_DILATION

    def decompose(self, operator, k: int) -> ModalFrame:
        A = self._validate_operator(operator)
        N = A.shape[0]
        k_actual = min(k, N)
        sigma, U_L, U_R = hermitian_dilation_decompose(A, k_actual)
        # Reconstruct H and all_eigenvalues for metadata (tests access these)
        H = np.block([[np.zeros((N, N)), A], [A.T, np.zeros((N, N))]])
        all_eigenvalues = np.linalg.eigvalsh(H)  # ascending
        return ModalFrame(
            basis=U_L,
            eigenvalues=sigma.astype(complex),
            method=self.method,
            metadata={
                "right_singular_vectors": U_R,
                "H": H,
                "all_eigenvalues": all_eigenvalues,
            },
        )

    def _decompose_scipy(self, operator, k: int) -> ModalFrame:
        """Fallback: original numpy eigh implementation."""
        A = self._validate_operator(operator)
        N = A.shape[0]
        H = np.block([[np.zeros((N, N)), A], [A.T, np.zeros((N, N))]])
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        pos_idx = np.where(eigenvalues > 0)[0]
        if len(pos_idx) == 0:
            order = np.argsort(-np.abs(eigenvalues))
            pos_idx = order
        else:
            pos_idx = pos_idx[np.argsort(-eigenvalues[pos_idx])]
        k_actual = min(k, len(pos_idx))
        idx = pos_idx[:k_actual]
        sqrt2 = np.sqrt(2.0)
        left_svecs = sqrt2 * eigenvectors[:N, idx]
        right_svecs = sqrt2 * eigenvectors[N:, idx]
        sv = eigenvalues[idx]
        return ModalFrame(
            basis=left_svecs,
            eigenvalues=sv.astype(complex),
            method=self.method,
            metadata={
                "right_singular_vectors": right_svecs,
                "H": H,
                "all_eigenvalues": eigenvalues,
            },
        )
