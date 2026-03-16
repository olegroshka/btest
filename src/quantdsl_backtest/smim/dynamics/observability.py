"""Observability matrix computation and diagnostics for SMIM state-space models.

M4.3-T1: Compute observability matrix O_z = [U; UF; UF²; ...; UF^{K-1}]
         and derive condition-number-based diagnostics.
"""

from __future__ import annotations

import numpy as np


def compute_observability_matrix(
    F_z: np.ndarray, U_K: np.ndarray
) -> np.ndarray:
    """Compute the observability matrix for a state-space system.

    O_z = [U; U F; U F²; ...; U F^{K-1}]   shape (N*K, K)

    A system is observable iff O_z has full column rank.

    Args:
        F_z: (K, K) state transition matrix for regime z.
        U_K: (N, K) observation matrix (modal basis).

    Returns:
        Stacked observability matrix of shape (N*K, K).
    """
    K = F_z.shape[0]
    rows: list[np.ndarray] = [U_K]
    F_pow = np.eye(K)
    for _ in range(K - 1):
        F_pow = F_pow @ F_z
        rows.append(U_K @ F_pow)
    return np.vstack(rows)  # (N*K, K)


def observability_diagnostics(
    F_z: np.ndarray,
    U_K: np.ndarray,
    condition_max: float = 1e6,
) -> dict:
    """Compute observability diagnostics from state-space parameters.

    Args:
        F_z:           (K, K) state transition matrix.
        U_K:           (N, K) observation matrix (modal basis).
        condition_max: Threshold above which the system is considered near-unobservable.

    Returns:
        Dictionary with keys:
            - ``observability_matrix``: (N*K, K) stacked observability matrix.
            - ``singular_values``:      (K,) singular values in descending order.
            - ``condition_number``:     Ratio of largest to smallest singular value.
            - ``passes``:               True if condition_number < condition_max.
            - ``suggested_k_star``:     Number of singular values above 1e-6 × max.
    """
    O = compute_observability_matrix(F_z, U_K)
    sv = np.linalg.svd(O, compute_uv=False)  # (K,) descending order
    cond = float(sv[0]) / max(float(sv[-1]), 1e-12)

    return {
        "observability_matrix": O,
        "singular_values": sv,
        "condition_number": cond,
        "passes": cond < condition_max,
        "suggested_k_star": int(np.sum(sv > sv[0] * 1e-6)),
    }
