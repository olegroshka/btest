"""Spectral method comparison harness (M3.1-T6)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer


@dataclass
class MethodComparisonRow:
    method: DecompositionMethod
    condition_number: float
    reconstruction_error: float  # Frobenius ||A - U diag(lambda) U^T||_F / ||A||_F
    compute_time_ms: float
    k_retained: int


class SpectralComparisonHarness:
    """Run multiple decomposition methods on the same operator and compare metrics."""

    def __init__(
        self,
        methods: list[DecompositionMethod | str],
        k: int = 10,
    ) -> None:
        from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer
        from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer
        from quantdsl_backtest.smim.spectral.polar import PolarDecomposer
        from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

        self.k = k
        self._decomposers: dict[DecompositionMethod, AbstractSpectralDecomposer] = {}

        _cls_map: dict[DecompositionMethod, type[AbstractSpectralDecomposer]] = {
            DecompositionMethod.SCHUR: SchurDecomposer,
            DecompositionMethod.POLAR: PolarDecomposer,
            DecompositionMethod.HERMITIAN_DILATION: HermitianDilationDecomposer,
            DecompositionMethod.DIRECTED_VARIATION: DirectedVariationDecomposer,
        }

        for m in methods:
            dm = DecompositionMethod(m) if isinstance(m, str) else m
            if dm in _cls_map:
                self._decomposers[dm] = _cls_map[dm]()

    def run(self, operator: sparse.csr_matrix) -> pd.DataFrame:
        """Run each method, compute metrics, return sorted DataFrame."""
        if sparse.issparse(operator):
            A_dense = operator.toarray().astype(float)
        else:
            A_dense = np.asarray(operator, dtype=float)

        rows = []
        for dm, decomposer in self._decomposers.items():
            t0 = time.perf_counter()
            try:
                frame = decomposer.decompose(operator, self.k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                cond = float(np.linalg.cond(A_dense))
                rec_err = self._reconstruction_error(A_dense, frame)
                rows.append(
                    MethodComparisonRow(
                        method=dm,
                        condition_number=cond,
                        reconstruction_error=rec_err,
                        compute_time_ms=elapsed_ms,
                        k_retained=frame.K,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                rows.append(
                    MethodComparisonRow(
                        method=dm,
                        condition_number=float("nan"),
                        reconstruction_error=float("nan"),
                        compute_time_ms=elapsed_ms,
                        k_retained=0,
                    )
                )

        df = pd.DataFrame(
            [
                {
                    "method": r.method.value,
                    "condition_number": r.condition_number,
                    "reconstruction_error": r.reconstruction_error,
                    "compute_time_ms": r.compute_time_ms,
                    "k_retained": r.k_retained,
                }
                for r in rows
            ]
        )
        if not df.empty:
            df = df.sort_values("reconstruction_error").reset_index(drop=True)
        return df

    def _reconstruction_error(self, A: np.ndarray, frame: ModalFrame) -> float:
        """||A - U diag(lambda.real) U^T||_F / ||A||_F."""
        U = frame.basis
        lam = frame.eigenvalues.real
        A_approx = U @ np.diag(lam) @ U.T
        norm_A = np.linalg.norm(A, "fro")
        if norm_A == 0:
            return 0.0
        return float(np.linalg.norm(A - A_approx, "fro") / norm_A)
