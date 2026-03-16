"""
Null model rewiring algorithms for significance testing.

M2.2-T1

Three rewiring strategies:
- degree_preserving_rewire: preserves in-degree and out-degree per node
- block_preserving_rewire: rewires within blocks only
- type_preserving_rewire: rewires within actor-type groups only
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def degree_preserving_rewire(
    adj: csr_matrix,
    n_swaps: int | None = None,
    rng: np.random.Generator | None = None,
) -> csr_matrix:
    """Degree-preserving edge rewiring (Maslov-Sneppen algorithm).

    Repeatedly picks 2 random edges (a→b, c→d) and attempts a swap to
    (a→d, c→b) if:
    - a ≠ c and b ≠ d
    - No self-loops created
    - No existing edge in swap positions

    Preserves in-degree and out-degree per node.

    Args:
        adj: Input adjacency matrix (N × N csr_matrix).
        n_swaps: Number of swap attempts. Defaults to 10 * nnz.
        rng: NumPy random Generator. Uses default if None.

    Returns:
        Rewired csr_matrix with same shape and degree sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Work on COO representation for fast edge manipulation
    adj_coo = adj.tocoo()
    rows = list(adj_coo.row.astype(int))
    cols = list(adj_coo.col.astype(int))
    data = list(adj_coo.data.copy())
    N = adj.shape[0]

    if len(rows) < 2:
        return adj.copy()

    if n_swaps is None:
        n_swaps = 10 * len(rows)

    # Build an edge set for O(1) membership queries
    edge_set = set(zip(rows, cols))

    for _ in range(n_swaps):
        if len(rows) < 2:
            break

        # Pick two distinct edge indices
        idx1, idx2 = rng.choice(len(rows), size=2, replace=False)
        a, b = rows[idx1], cols[idx1]
        c, d = rows[idx2], cols[idx2]

        # Validity checks
        if a == c or b == d:
            continue
        if a == d or c == b:
            continue  # would create self-loops
        if (a, d) in edge_set or (c, b) in edge_set:
            continue  # edges already exist

        # Remove old edges
        edge_set.discard((a, b))
        edge_set.discard((c, d))

        # Insert new edges
        rows[idx1], cols[idx1] = a, d
        rows[idx2], cols[idx2] = c, b
        edge_set.add((a, d))
        edge_set.add((c, b))

    rows_arr = np.array(rows, dtype=np.int32)
    cols_arr = np.array(cols, dtype=np.int32)
    data_arr = np.array(data, dtype=float)

    return csr_matrix((data_arr, (rows_arr, cols_arr)), shape=adj.shape)


def block_preserving_rewire(
    adj: csr_matrix,
    layer_assignments: np.ndarray,
    rng: np.random.Generator | None = None,
) -> csr_matrix:
    """Rewire edges within each block (layer) separately.

    Edges between different blocks are preserved as-is; only intra-block
    edges are rewired using degree_preserving_rewire.

    Args:
        adj: Input adjacency matrix (N × N csr_matrix).
        layer_assignments: Integer array of length N mapping nodes to blocks.
        rng: NumPy random Generator. Uses default if None.

    Returns:
        Rewired csr_matrix, no new cross-block edges.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = adj.shape[0]
    dense = adj.toarray().copy()
    unique_layers = np.unique(layer_assignments)

    for layer in unique_layers:
        node_indices = np.where(layer_assignments == layer)[0]
        if len(node_indices) < 2:
            continue
        # Extract intra-block sub-matrix
        sub = dense[np.ix_(node_indices, node_indices)]
        sub_sparse = csr_matrix(sub)
        rewired_sub = degree_preserving_rewire(sub_sparse, rng=rng)
        dense[np.ix_(node_indices, node_indices)] = rewired_sub.toarray()

    return csr_matrix(dense)


def type_preserving_rewire(
    adj: csr_matrix,
    type_assignments: np.ndarray,
    rng: np.random.Generator | None = None,
) -> csr_matrix:
    """Rewire edges within each actor-type group separately.

    Same algorithm as block_preserving_rewire but keyed on actor types.

    Args:
        adj: Input adjacency matrix (N × N csr_matrix).
        type_assignments: Array of length N (any hashable values) mapping nodes
                          to actor types.
        rng: NumPy random Generator. Uses default if None.

    Returns:
        Rewired csr_matrix, no new cross-type edges.
    """
    return block_preserving_rewire(adj, type_assignments, rng=rng)
