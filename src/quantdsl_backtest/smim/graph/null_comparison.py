"""
Null model comparison: observed statistic vs. null distribution.

M2.2-T2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from quantdsl_backtest.smim.graph.null_models import degree_preserving_rewire


@dataclass
class NullComparisonResult:
    """Result from comparing an observed graph statistic to a null distribution.

    Attributes:
        observed_stat: Statistic computed on the observed graph.
        null_distribution: Array of shape (B,) from null instances.
        p_value: Fraction of null statistics ≥ observed_stat.
        passes: True if p_value < 0.05.
    """

    observed_stat: float
    null_distribution: np.ndarray  # (B,)
    p_value: float
    passes: bool  # True if p_value < 0.05


def spectral_gap_statistic(adj: csr_matrix) -> float:
    """Compute λ₂ of the symmetrised Laplacian.

    Laplacian L = D - (A + Aᵀ)/2 where D is degree matrix.

    Returns:
        Second smallest eigenvalue of L (spectral gap).
    """
    sym = (adj + adj.T) / 2.0
    sym_dense = sym.toarray()
    N = sym_dense.shape[0]
    if N == 0:
        return 0.0

    degree = np.array(sym_dense.sum(axis=1)).flatten()
    D = np.diag(degree)
    L = D - sym_dense

    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues_sorted = np.sort(eigenvalues)

    if len(eigenvalues_sorted) < 2:
        return float(eigenvalues_sorted[0]) if len(eigenvalues_sorted) > 0 else 0.0

    return float(eigenvalues_sorted[1])


def modularity_statistic(
    adj: csr_matrix,
    community_labels: np.ndarray | None = None,
) -> float:
    """Compute Newman modularity Q.

    If community_labels is None, all nodes are in a single community.
    Q = (1/2m) Σ_{ij} [A_{ij} - k_i k_j / 2m] δ(c_i, c_j)

    Args:
        adj: Adjacency matrix.
        community_labels: Integer array of length N. Defaults to all-same community.

    Returns:
        Modularity Q ∈ [-0.5, 1].
    """
    sym = (adj + adj.T) / 2.0
    dense = sym.toarray()
    N = dense.shape[0]

    if community_labels is None:
        community_labels = np.zeros(N, dtype=int)

    m = dense.sum() / 2.0
    if m == 0:
        return 0.0

    k = dense.sum(axis=1)
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if community_labels[i] == community_labels[j]:
                Q += dense[i, j] - k[i] * k[j] / (2 * m)

    return float(Q / (2 * m))


class NullModelComparison:
    """Runs a statistical test of observed graph statistic vs. null distribution.

    Args:
        rewire_fn: Callable that takes (adj, **kwargs) and returns a rewired
                   csr_matrix.
        n_instances: Number of null instances B. Default 100.
        statistic: Which built-in statistic to use ("spectral_gap" or "modularity").
    """

    _STATISTICS: dict[str, Callable[[csr_matrix], float]] = {
        "spectral_gap": spectral_gap_statistic,
        "modularity": modularity_statistic,
    }

    def __init__(
        self,
        rewire_fn: Callable = degree_preserving_rewire,
        n_instances: int = 100,
        statistic: str = "spectral_gap",
    ) -> None:
        self.rewire_fn = rewire_fn
        self.n_instances = n_instances
        self.statistic = statistic

    def run(
        self,
        observed: csr_matrix,
        layer_assignments: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> NullComparisonResult:
        """Generate null distribution and compute p-value.

        Args:
            observed: The observed adjacency matrix.
            layer_assignments: Optional block labels passed to block_preserving
                               rewire functions.
            rng: NumPy Generator for reproducibility.

        Returns:
            NullComparisonResult with p_value and passes flag.
        """
        if rng is None:
            rng = np.random.default_rng()

        stat_fn = self._STATISTICS.get(self.statistic, spectral_gap_statistic)
        observed_stat = stat_fn(observed)

        null_stats = np.empty(self.n_instances, dtype=float)
        for b in range(self.n_instances):
            if layer_assignments is not None:
                null_mat = self.rewire_fn(observed, layer_assignments=layer_assignments, rng=rng)
            else:
                null_mat = self.rewire_fn(observed, rng=rng)
            null_stats[b] = stat_fn(null_mat)

        p_value = float(np.mean(null_stats >= observed_stat))

        return NullComparisonResult(
            observed_stat=observed_stat,
            null_distribution=null_stats,
            p_value=p_value,
            passes=p_value < 0.05,
        )
