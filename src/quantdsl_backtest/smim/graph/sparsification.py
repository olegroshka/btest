"""
Graph sparsification utilities.

M2.2-T3
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


def l1_sparsify(adj: csr_matrix, target_density: float) -> csr_matrix:
    """Soft-threshold to achieve approximately target_density.

    Uses binary search to find λ such that density(max(|A| - λ, 0)) ≈ target.

    Args:
        adj: Input adjacency matrix.
        target_density: Desired density ∈ (0, 1).

    Returns:
        Sparsified csr_matrix with density ≤ target_density.
    """
    dense = adj.toarray().astype(float)
    abs_vals = np.abs(dense)
    N_total = dense.size

    if N_total == 0:
        return csr_matrix(dense)

    # Binary search on threshold λ
    lo, hi = 0.0, float(abs_vals.max()) + 1e-12

    for _ in range(50):
        mid = (lo + hi) / 2.0
        shrunk = np.maximum(abs_vals - mid, 0.0)
        density = np.count_nonzero(shrunk) / N_total
        if density > target_density:
            lo = mid
        else:
            hi = mid

    lam = (lo + hi) / 2.0
    shrunk = np.maximum(abs_vals - lam, 0.0)
    # Restore sign
    result = np.sign(dense) * shrunk
    return csr_matrix(result)


def threshold_sparsify(adj: csr_matrix, threshold: float) -> csr_matrix:
    """Hard threshold: zero entries with |weight| ≤ threshold.

    Args:
        adj: Input adjacency matrix.
        threshold: Entries with |weight| ≤ this value are set to zero.

    Returns:
        Sparsified csr_matrix.
    """
    dense = adj.toarray().astype(float)
    dense[np.abs(dense) <= threshold] = 0.0
    return csr_matrix(dense)


def spectral_energy_retained(
    original: csr_matrix,
    sparsified: csr_matrix,
) -> float:
    """Fraction of spectral energy (sum of squared singular values) retained.

    = Σ sv_sparsified² / Σ sv_original²

    Uses np.linalg.svd on dense matrices (appropriate for small N).

    Args:
        original: Original adjacency matrix.
        sparsified: Sparsified version.

    Returns:
        Float in [0, 1] (or 0 if original has zero energy).
    """
    orig_dense = original.toarray().astype(float)
    spar_dense = sparsified.toarray().astype(float)

    sv_orig = np.linalg.svd(orig_dense, compute_uv=False)
    sv_spar = np.linalg.svd(spar_dense, compute_uv=False)

    energy_orig = float(np.sum(sv_orig ** 2))
    energy_spar = float(np.sum(sv_spar ** 2))

    if energy_orig == 0.0:
        return 0.0
    return energy_spar / energy_orig
