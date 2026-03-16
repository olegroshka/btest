"""
AggregateOperator: weighted combination of multi-channel adjacency matrices.

M2.1-T5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.interfaces import RelationChannel


@dataclass
class AggregateOperator:
    """Weighted sum of per-channel adjacency matrices.

    Computes A_t = Σ_r ω_r A^{(r)}.

    Attributes:
        channel_matrices: Mapping from RelationChannel to its csr_matrix.
        weights: Per-channel weights ω_r. If empty, uniform weights are used.
    """

    channel_matrices: dict[RelationChannel, csr_matrix]
    weights: dict[RelationChannel, float] = field(default_factory=dict)

    def compute(self) -> csr_matrix:
        """Return the weighted sum Σ_r ω_r A^{(r)}.

        Uses equal weights (1/R) if self.weights is empty.
        """
        if not self.channel_matrices:
            raise ValueError("channel_matrices is empty; nothing to aggregate.")

        channels = list(self.channel_matrices.keys())
        R = len(channels)

        if self.weights:
            w = self.weights
        else:
            equal = 1.0 / R
            w = {ch: equal for ch in channels}

        result: csr_matrix | None = None
        for ch in channels:
            mat = self.channel_matrices[ch]
            weighted = mat.multiply(w.get(ch, 0.0))
            if result is None:
                result = weighted
            else:
                result = result + weighted

        return csr_matrix(result)

    def with_uniform_weights(self) -> AggregateOperator:
        """Set all weights to 1/R and return self for chaining."""
        R = len(self.channel_matrices)
        if R == 0:
            self.weights = {}
        else:
            equal = 1.0 / R
            self.weights = {ch: equal for ch in self.channel_matrices}
        return self

    def spectral_energy(self) -> float:
        """Sum of squared singular values of compute() (= ‖A‖_F²)."""
        mat = self.compute()
        dense = mat.toarray()
        sv = np.linalg.svd(dense, compute_uv=False)
        return float(np.sum(sv ** 2))

    def density(self) -> float:
        """Fraction of non-zero entries in compute()."""
        mat = self.compute()
        N_total = mat.shape[0] * mat.shape[1]
        if N_total == 0:
            return 0.0
        return mat.nnz / N_total
