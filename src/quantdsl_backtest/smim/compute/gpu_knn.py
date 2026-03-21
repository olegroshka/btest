"""Brute-force k-nearest-neighbours via PyTorch.

For KSG transfer entropy, we need KNN on (T, d) point clouds where
T ≤ 10,000 and d ≤ 6. Brute-force pairwise distances on GPU is faster
than CPU KD-trees for this regime.

Strategy: compute full pairwise distance matrix on GPU via the expansion trick:
  ||a-b||² = ||a||² + ||b||² - 2 a·b
Then torch.topk for k smallest per row.

Memory: if T² × 8 bytes > 2 GB, compute distances in row chunks to stay
within device memory limits.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

from .torch_ops import ensure_tensor, get_device, to_numpy

# Maximum bytes of distance matrix to materialise at once (2 GiB).
_MAX_BYTES = 2 * 1024 ** 3


def knn_query(
    data: NDArray,
    k: int,
    p: float = float("inf"),
    device: torch.device | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute k nearest neighbours for every point in *data* (self-query).

    Matches the output of ``scipy.spatial.KDTree(data).query(data, k=k+1, p=p)``
    — the first column (self, distance 0) is included in the returned arrays.

    Args:
        data: (T, d) point cloud.
        k:    Number of neighbours to return (including the query point itself,
              so the effective number of *other* neighbours is k-1).
        p:    Minkowski exponent. ``float("inf")`` gives the L∞ (Chebyshev) norm
              used by the KSG estimator.  Currently only ``float("inf")`` and
              ``2.0`` (Euclidean) are supported.
        device: Target device. Defaults to auto-detected SMIM device.

    Returns:
        distances: (T, k) float64 array — distances to the k nearest neighbours,
                   sorted ascending. First column is 0 (self-distance).
        indices:   (T, k) int64 array — indices into *data*.
    """
    if device is None:
        device = get_device()

    data_t: torch.Tensor = ensure_tensor(data, device=device, dtype=torch.float64)
    T, d = data_t.shape

    # How many rows fit in one chunk?
    bytes_per_row = T * 8  # float64
    chunk_rows = max(1, _MAX_BYTES // bytes_per_row)
    chunk_rows = min(chunk_rows, T)

    all_dists = torch.empty(T, k, dtype=torch.float64, device=device)
    all_idx = torch.empty(T, k, dtype=torch.int64, device=device)

    # Precompute squared norms (used in expansion trick for Euclidean only).
    if p == 2.0:
        sq_norms = (data_t * data_t).sum(dim=1)  # (T,)

    n_chunks = math.ceil(T / chunk_rows)
    for c in range(n_chunks):
        row_start = c * chunk_rows
        row_end = min(row_start + chunk_rows, T)
        q = data_t[row_start:row_end]  # (B, d)
        B = q.shape[0]

        if p == float("inf"):
            # L∞: max over dims of |a - b|
            # q: (B, d) → (B, 1, d); data_t: (T, d) → (1, T, d)
            dist_mat = (q.unsqueeze(1) - data_t.unsqueeze(0)).abs().amax(dim=2)
            # shape: (B, T)
        elif p == 2.0:
            # Euclidean via expansion: ||a-b||² = ||a||² + ||b||² - 2 a·b
            sq_q = sq_norms[row_start:row_end]  # (B,)
            cross = q @ data_t.t()              # (B, T)
            dist_sq = sq_q.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * cross
            # Clamp negatives from floating-point error before sqrt.
            dist_mat = dist_sq.clamp(min=0.0).sqrt()
        else:
            raise ValueError(f"Unsupported p={p}. Use p=2.0 or p=inf.")

        # k smallest per row (including self at distance 0).
        top_dists, top_idx = dist_mat.topk(k, dim=1, largest=False, sorted=True)
        all_dists[row_start:row_end] = top_dists
        all_idx[row_start:row_end] = top_idx

    return to_numpy(all_dists), to_numpy(all_idx)


def knn_count_within_radius(
    data: NDArray,
    queries: NDArray,
    r_max: NDArray,
    p: float = float("inf"),
    device: torch.device | None = None,
) -> NDArray:
    """Count neighbours in *data* within per-query radius *r_max* (L∞ norm).

    Equivalent to::

        tree = cKDTree(data)
        counts = [len(tree.query_ball_point(queries[i], r_max[i], p=inf)) - 1
                  for i in range(len(queries))]

    Args:
        data:    (T, d) reference points.
        queries: (T, d) query points (may equal data).
        r_max:   (T,) per-query radii.
        p:       Minkowski exponent (only ``float("inf")`` supported).
        device:  Target device.

    Returns:
        (T,) integer counts (excluding the query point itself), clipped to ≥ 0.
    """
    if p != float("inf"):
        raise ValueError(f"knn_count_within_radius only supports p=inf, got p={p}")

    if device is None:
        device = get_device()

    data_t = ensure_tensor(data, device=device, dtype=torch.float64)
    queries_t = ensure_tensor(queries, device=device, dtype=torch.float64)
    r_t = ensure_tensor(r_max, device=device, dtype=torch.float64)

    T_q = queries_t.shape[0]
    T_d = data_t.shape[0]

    bytes_per_row = T_d * 8
    chunk_rows = max(1, _MAX_BYTES // bytes_per_row)
    chunk_rows = min(chunk_rows, T_q)

    counts = torch.empty(T_q, dtype=torch.int64, device=device)

    n_chunks = math.ceil(T_q / chunk_rows)
    for c in range(n_chunks):
        row_start = c * chunk_rows
        row_end = min(row_start + chunk_rows, T_q)
        q = queries_t[row_start:row_end]   # (B, d)
        r = r_t[row_start:row_end]         # (B,)
        B = q.shape[0]

        # L∞ distance: (B, T_d)
        dist_mat = (q.unsqueeze(1) - data_t.unsqueeze(0)).abs().amax(dim=2)

        # Count points strictly within radius (exclude self by subtracting 1).
        within = (dist_mat <= r.unsqueeze(1)).sum(dim=1) - 1  # (B,)
        counts[row_start:row_end] = within.clamp(min=0)

    return to_numpy(counts).astype(np.int64)
