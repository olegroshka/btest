"""Transfer entropy estimation using the KSG (Kraskov-Stögbauer-Grassberger) estimator.

M4.6-T3: Transfer entropy TE_{X→Y} = I(Y_{t+1}; X_t | Y_t).
M4.6-T4: Conditional transfer entropy TE_{X→Y|Z} for mediation analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


# ---------------------------------------------------------------------------
# M4.6-T3: KSG Transfer Entropy
# ---------------------------------------------------------------------------

def _knn_count(data: np.ndarray, queries: np.ndarray, r_max: np.ndarray) -> np.ndarray:
    """Count neighbours within radius r_max[i] for each query point i.

    Uses the L∞ (Chebyshev) norm consistent with KSG estimator.

    Args:
        data:    (n, d) reference points.
        queries: (n, d) query points.
        r_max:   (n,) per-query radii.

    Returns:
        (n,) integer counts (excluding the point itself).
    """
    tree = cKDTree(data)
    counts = np.array(
        [
            len(tree.query_ball_point(queries[i], r_max[i], p=np.inf)) - 1
            for i in range(len(queries))
        ]
    )
    return np.maximum(counts, 0)


def ksg_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k_neighbours: int = 5,
) -> float:
    """Estimate transfer entropy TE_{X→Y} = I(Y_{t+1}; X_t | Y_t) via KSG.

    Uses the KSG conditional mutual information estimator (Kraskov et al. 2004).

    Args:
        source:       (T,) source time series X.
        target:       (T,) target time series Y.
        lag:          Time lag (default 1).
        k_neighbours: Number of nearest neighbours.

    Returns:
        Non-negative transfer entropy estimate in nats.
    """
    T = len(source)
    n = T - lag

    Y_fut = target[lag:]         # (n,)
    X_past = source[:-lag]       # (n,)
    Y_past = target[:-lag]       # (n,)

    # Joint space: (Y_fut, X_past, Y_past)
    joint = np.column_stack([Y_fut, X_past, Y_past])  # (n, 3)
    xz = np.column_stack([X_past, Y_past])             # (n, 2)
    yz = np.column_stack([Y_fut, Y_past])               # (n, 2)
    z = Y_past.reshape(-1, 1)                           # (n, 1)

    # Radius from k-NN in joint space
    tree_joint = cKDTree(joint)
    dists, _ = tree_joint.query(joint, k=k_neighbours + 1, p=np.inf)
    r = dists[:, -1]

    n_xz = _knn_count(xz, xz, r)
    n_yz = _knn_count(yz, yz, r)
    n_z = _knn_count(z, z, r)

    te = (
        digamma(k_neighbours)
        + np.mean(digamma(n_z + 1))
        - np.mean(digamma(n_xz + 1))
        - np.mean(digamma(n_yz + 1))
    )
    return max(float(te), 0.0)


def layer_transfer_entropy(
    alpha_filtered: np.ndarray,
    modal_frame,
    actors,
    lag: int = 1,
) -> np.ndarray:
    """Estimate a 4×4 transfer entropy matrix between layers.

    Args:
        alpha_filtered: (T, K) filtered modal states.
        modal_frame:    ModalFrame (unused here, for interface compatibility).
        actors:         ActorRegistry providing layer membership.
        lag:            Time lag for TE estimation.

    Returns:
        (4, 4) matrix TE[i, j] = TE from layer i to layer j.
    """
    from quantdsl_backtest.smim.interfaces import Layer

    T, K = alpha_filtered.shape
    n_layers = 4
    TE = np.zeros((n_layers, n_layers))

    # Build one signal per layer by summing the first available modal state
    layer_signals = []
    for lay in range(n_layers):
        layer_actors = actors.actors_in_layer(Layer(lay))
        if not layer_actors:
            layer_signals.append(np.zeros(T))
            continue
        # Use the index of the first actor in this layer as mode index
        idx = actors.index_of(layer_actors[0].actor_id)
        mode_idx = idx % K  # wrap to available modes
        layer_signals.append(alpha_filtered[:, mode_idx])

    for i in range(n_layers):
        for j in range(n_layers):
            if i != j:
                TE[i, j] = ksg_transfer_entropy(
                    layer_signals[i], layer_signals[j], lag=lag
                )

    return TE


# ---------------------------------------------------------------------------
# M4.6-T4: Conditional Transfer Entropy
# ---------------------------------------------------------------------------

def conditional_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray,
    lag: int = 1,
    k: int = 5,
) -> float:
    """Estimate conditional TE_{X→Y|Z} = I(Y_{t+1}; X_t | Y_t, Z_t) via KSG.

    Args:
        source:       (T,) source time series X.
        target:       (T,) target time series Y.
        conditioning: (T,) conditioning variable Z.
        lag:          Time lag (default 1).
        k:            Number of nearest neighbours.

    Returns:
        Non-negative conditional transfer entropy estimate in nats.
    """
    T = len(source)

    Y_fut = target[lag:]
    X_past = source[:-lag]
    Y_past = target[:-lag]
    Z_past = conditioning[:-lag]

    joint = np.column_stack([Y_fut, X_past, Y_past, Z_past])  # (n, 4)
    xz = np.column_stack([X_past, Y_past, Z_past])             # (n, 3)
    yz = np.column_stack([Y_fut, Y_past, Z_past])               # (n, 3)
    z = np.column_stack([Y_past, Z_past])                       # (n, 2)

    tree_joint = cKDTree(joint)
    dists, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    r = dists[:, -1]

    n_xz = _knn_count(xz, xz, r)
    n_yz = _knn_count(yz, yz, r)
    n_z = _knn_count(z, z, r)

    cte = (
        digamma(k)
        + np.mean(digamma(n_z + 1))
        - np.mean(digamma(n_xz + 1))
        - np.mean(digamma(n_yz + 1))
    )
    return max(float(cte), 0.0)
