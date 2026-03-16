"""Abstract base classes and factory for spectral decomposers (M3.1-T1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sparse

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


class AbstractSpectralDecomposer(ABC):
    """Abstract base for spectral decomposers. Implements SpectralDecomposer protocol."""

    @property
    @abstractmethod
    def method(self) -> DecompositionMethod: ...

    @abstractmethod
    def decompose(self, operator: sparse.csr_matrix, k: int) -> ModalFrame: ...

    def _validate_operator(self, op: sparse.csr_matrix) -> np.ndarray:
        """Convert to dense, verify square. Raises ValueError if not square."""
        if sparse.issparse(op):
            A = op.toarray().astype(float)
        else:
            A = np.asarray(op, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Operator must be square, got shape {A.shape}")
        return A

    def _truncate_modes(self, frame: ModalFrame, k: int) -> ModalFrame:
        """Retain only top-k modes by eigenvalue magnitude."""
        K = frame.basis.shape[1]
        if k >= K:
            return frame
        order = np.argsort(-np.abs(frame.eigenvalues))[:k]
        return ModalFrame(
            basis=frame.basis[:, order],
            eigenvalues=frame.eigenvalues[order],
            method=frame.method,
            metadata=frame.metadata,
        )


class SpectralDecomposerFactory:
    """Maps DecompositionMethod -> decomposer class."""

    def __init__(self) -> None:
        self._registry: dict[DecompositionMethod, type[AbstractSpectralDecomposer]] = {}

    def register(self, method: DecompositionMethod, cls: type) -> None:
        self._registry[method] = cls

    def create(self, method: DecompositionMethod | str) -> AbstractSpectralDecomposer:
        if isinstance(method, str):
            try:
                method = DecompositionMethod(method)
            except ValueError as exc:
                raise KeyError(
                    f"Unknown decomposition method: {method!r}. "
                    f"Available: {list(self._registry)}"
                ) from exc
        if method not in self._registry:
            raise KeyError(
                f"Unknown decomposition method: {method!r}. "
                f"Available: {list(self._registry)}"
            )
        return self._registry[method]()

    def available_methods(self) -> list[DecompositionMethod]:
        return list(self._registry)
