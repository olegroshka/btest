"""Topological Data Analysis for SMIM emergence detection.

M4.7-T1: Sliding-window persistent homology.
M4.7-T2: Topological complexity and Wasserstein distance summaries.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# M4.7-T1: Sliding-window persistent homology
# ---------------------------------------------------------------------------

def sliding_window_persistence(
    alpha_filtered: np.ndarray,
    window_size: int = 20,
    overlap: int = 10,
) -> list:
    """Compute sliding-window persistence diagrams.

    For each window of length ``window_size`` (stepped by ``window_size - overlap``),
    compute the persistence diagram of the point cloud formed by the modal state vectors.

    If ``ripser`` is installed it is used; otherwise a simple fallback based on
    pairwise distance histograms is employed.

    Args:
        alpha_filtered: (T, K) filtered modal states.
        window_size:    Length of each sliding window.
        overlap:        Number of overlapping time steps between successive windows.

    Returns:
        List of persistence diagrams.  Each diagram is a list of two arrays
        [H0_dgm, H1_dgm] where each array has shape (n_bars, 2) with columns
        [birth, death].
    """
    T, K = alpha_filtered.shape
    step = max(window_size - overlap, 1)
    diagrams = []

    try:
        from ripser import ripser as _ripser  # type: ignore[import]
        use_ripser = True
    except ImportError:
        use_ripser = False

    t = 0
    while t + window_size <= T:
        window = alpha_filtered[t: t + window_size]  # (window_size, K)

        if use_ripser:
            result = _ripser(window, maxdim=1)
            diagrams.append(result["dgms"])
        else:
            # Fallback: pairwise distance proxy
            from scipy.spatial.distance import pdist

            d = pdist(window)
            # H0 diagram: birth=0, death=sorted distances (up to 10 bars)
            h0 = np.column_stack([np.zeros(min(len(d), 10)), np.sort(d)[: min(len(d), 10)]])
            diagrams.append([h0, np.zeros((0, 2))])

        t += step

    return diagrams


# ---------------------------------------------------------------------------
# M4.7-T2: Topological summary statistics
# ---------------------------------------------------------------------------

def persistence_landscape_norms(diagrams: list, p: int = 2) -> np.ndarray:
    """Compute L^p norms of persistence for each window and dimension.

    Args:
        diagrams: List of persistence diagrams (output of sliding_window_persistence).
        p:        L^p norm exponent.

    Returns:
        (n_windows, 2) array of norms for H0 (col 0) and H1 (col 1).
    """
    norms = []
    for dgm in diagrams:
        row = []
        for dim_dgm in dgm[:2]:  # H0, H1
            if len(dim_dgm) == 0:
                row.append(0.0)
            else:
                persist = dim_dgm[:, 1] - dim_dgm[:, 0]
                persist = persist[np.isfinite(persist)]
                if len(persist) > 0:
                    row.append(float(np.sum(persist ** p) ** (1.0 / p)))
                else:
                    row.append(0.0)
        norms.append(row)
    return np.array(norms)


def topological_complexity(diagrams: list) -> np.ndarray:
    """Compute topological complexity T_t = Σ_k (k+1) · total_persistence_k.

    Args:
        diagrams: List of persistence diagrams.

    Returns:
        (n_windows,) array of complexity values.
    """
    C = []
    for dgm in diagrams:
        val = 0.0
        for k, dim_dgm in enumerate(dgm):
            if len(dim_dgm) == 0:
                continue
            persist = dim_dgm[:, 1] - dim_dgm[:, 0]
            persist = persist[np.isfinite(persist)]
            val += (k + 1) * float(np.sum(persist))
        C.append(val)
    return np.array(C)


def wasserstein_distances(diagrams: list, dim: int = 0) -> np.ndarray:
    """Compute L2 Wasserstein distances between successive diagrams.

    If ``persim`` is installed it is used; otherwise a simple L2 norm of sorted
    persistence lengths is used as a proxy.

    Args:
        diagrams: List of persistence diagrams.
        dim:      Homological dimension (0=H0, 1=H1).

    Returns:
        (n_windows - 1,) array of distances.
    """
    try:
        from persim import wasserstein  # type: ignore[import]
        use_persim = True
    except ImportError:
        use_persim = False

    dists = []
    for i in range(len(diagrams) - 1):
        dgm_a = diagrams[i][dim] if dim < len(diagrams[i]) else np.zeros((0, 2))
        dgm_b = diagrams[i + 1][dim] if dim < len(diagrams[i + 1]) else np.zeros((0, 2))

        if use_persim:
            try:
                d = wasserstein(dgm_a, dgm_b)
                dists.append(float(d))
            except Exception:
                dists.append(0.0)
        else:
            # Fallback: L2 norm of persistence length difference
            def _persist(dgm: np.ndarray) -> np.ndarray:
                if len(dgm) == 0:
                    return np.zeros(5)
                p = np.sort(dgm[:, 1] - dgm[:, 0])[:5]
                return p

            pa = _persist(dgm_a)
            pb = _persist(dgm_b)
            n = min(len(pa), len(pb))
            dists.append(float(np.linalg.norm(pa[:n] - pb[:n])))

    return np.array(dists)
