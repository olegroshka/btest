"""Directed-variation basis decomposer via graph Laplacian (M3.1-T5)."""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer


class DirectedVariationDecomposer(AbstractSpectralDecomposer):
    """Directed-variation basis via symmetrised graph Laplacian.

    Practical approximation: symmetrise the adjacency, form L = D - A_sym,
    and return eigenvectors ordered by ascending eigenvalue (smoothest modes first).
    These minimise directed variation energy on the graph.
    """

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.DIRECTED_VARIATION

    def decompose(self, operator, k: int) -> ModalFrame:
        A = self._validate_operator(operator)
        N = A.shape[0]
        # Symmetrise for DV-ordered basis
        A_sym = (A + A.T) / 2.0
        # Degree matrix
        d = A_sym.sum(axis=1)
        D = np.diag(d)
        L = D - A_sym  # graph Laplacian
        # Eigendecompose Laplacian (ascending = smoothest modes first)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Return k smallest (smoothest = lowest DV)
        k_actual = min(k, N)
        basis = eigenvectors[:, :k_actual]
        evals = eigenvalues[:k_actual].astype(complex)
        return ModalFrame(
            basis=basis,
            eigenvalues=evals,
            method=self.method,
            metadata={"laplacian": L, "all_eigenvalues": eigenvalues},
        )
