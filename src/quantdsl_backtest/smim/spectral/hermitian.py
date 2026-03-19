"""Hermitian dilation decomposer (M3.1-T4)."""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer


class HermitianDilationDecomposer(AbstractSpectralDecomposer):
    """Spectral decomposer via Hermitian dilation H(A) = [[0, A], [A^T, 0]]."""

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.HERMITIAN_DILATION

    def decompose(self, operator, k: int) -> ModalFrame:
        A = self._validate_operator(operator)
        N = A.shape[0]
        # Construct 2N x 2N Hermitian dilation
        H = np.block([[np.zeros((N, N)), A], [A.T, np.zeros((N, N))]])
        # Eigendecompose symmetric matrix -> real eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        # Positive eigenvalues correspond to singular values
        pos_idx = np.where(eigenvalues > 0)[0]
        if len(pos_idx) == 0:
            # Fallback: take top-k by magnitude
            order = np.argsort(-np.abs(eigenvalues))
            pos_idx = order
        else:
            # Sort positive eigenvalues descending
            pos_idx = pos_idx[np.argsort(-eigenvalues[pos_idx])]
        k_actual = min(k, len(pos_idx))
        idx = pos_idx[:k_actual]
        # Recover singular vectors via u_L = √2 · v[:N],  u_R = √2 · v[N:]
        # (standard result: H eigenvectors for +σ are [u_L; u_R]/√2)
        sqrt2 = np.sqrt(2.0)
        left_svecs = sqrt2 * eigenvectors[:N, idx]
        # Right singular vectors: bottom-N components
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
