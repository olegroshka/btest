"""Polar decomposition-based spectral decomposer (M3.1-T3)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import polar

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer


class PolarDecomposer(AbstractSpectralDecomposer):
    """Spectral decomposer using polar factorisation A = U @ P."""

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.POLAR

    def decompose(self, operator, k: int) -> ModalFrame:
        A = self._validate_operator(operator)
        # A = U @ P, U orthogonal/unitary, P symmetric PSD
        U, P = polar(A)
        # Eigendecompose P (symmetric -> real eigenvalues, ascending order)
        p_vals, p_vecs = np.linalg.eigh(P)
        # Sort descending by eigenvalue
        order = np.argsort(-p_vals)
        k_actual = min(k, len(p_vals))
        idx = order[:k_actual]
        basis = p_vecs[:, idx]
        eigenvalues = p_vals[idx].astype(complex)
        # Also eigendecompose U for metadata
        u_vals, u_vecs = np.linalg.eig(U)
        return ModalFrame(
            basis=basis,
            eigenvalues=eigenvalues,
            method=self.method,
            metadata={"U": U, "P": P, "u_eigenvalues": u_vals, "u_eigenvectors": u_vecs},
        )
