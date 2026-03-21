"""Schur decomposition-based spectral decomposer (M3.1-T2)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur as _schur_scipy

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer
from quantdsl_backtest.smim.compute.linalg import schur_decompose


class SchurDecomposer(AbstractSpectralDecomposer):
    """Spectral decomposer using Schur factorisation A = Q T Q^H."""

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.SCHUR

    def decompose(self, operator, k: int) -> ModalFrame:
        A = self._validate_operator(operator)
        # A = Q @ T @ Q^H  (complex Schur) — stays on CPU (torch has no schur)
        Q, T = schur_decompose(A)
        # Eigenvalues from diagonal of T
        eigenvalues = np.diag(T)
        # Order by magnitude descending
        order = np.argsort(-np.abs(eigenvalues))
        k_actual = min(k, len(eigenvalues))
        idx = order[:k_actual]
        # Use real part of Q columns for ModalFrame basis
        basis = Q[:, idx].real
        return ModalFrame(
            basis=basis,
            eigenvalues=eigenvalues[idx],
            method=self.method,
            metadata={"T": T, "Q": Q, "full_eigenvalues": eigenvalues},
        )

    def _decompose_scipy(self, operator, k: int) -> ModalFrame:
        """Fallback: original scipy implementation."""
        A = self._validate_operator(operator)
        T, Q = _schur_scipy(A, output="complex")
        eigenvalues = np.diag(T)
        order = np.argsort(-np.abs(eigenvalues))
        k_actual = min(k, len(eigenvalues))
        idx = order[:k_actual]
        basis = Q[:, idx].real
        return ModalFrame(
            basis=basis,
            eigenvalues=eigenvalues[idx],
            method=self.method,
            metadata={"T": T, "Q": Q, "full_eigenvalues": eigenvalues},
        )
