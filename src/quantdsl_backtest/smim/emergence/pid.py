"""Partial Information Decomposition (PID) via Gaussian MMI.

M4.6-T1: PID synergy computation with Gaussian MMI redundancy measure.
M4.6-T2: Bootstrap confidence intervals for PID synergy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from quantdsl_backtest.smim.compute.batch_pid import batch_pid_synergy


# ---------------------------------------------------------------------------
# M4.6-T1: Gaussian PID
# ---------------------------------------------------------------------------

@dataclass
class PIDResult:
    """Partial Information Decomposition result.

    Attributes:
        redundancy: Shared information min(I_j, I_k) [bits, nats, or natural units].
        unique_j:   Information unique to source j.
        unique_k:   Information unique to source k.
        synergy:    Synergistic information only available from j and k jointly.
    """

    redundancy: float
    unique_j: float
    unique_k: float
    synergy: float


def _mutual_info_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute mutual information I(X; Y) assuming joint Gaussianity.

    Uses the formula I = H(X) + H(Y) - H(X, Y) where
    H(Z) = ½ (n + n log 2π + log det Σ_Z).

    Args:
        X: (T,) or (T, p) samples from X.
        Y: (T,) or (T, q) samples from Y.

    Returns:
        Mutual information in natural units (nats).
    """
    X = np.atleast_2d(X).T if X.ndim == 1 else np.asarray(X)
    Y = np.atleast_2d(Y).T if Y.ndim == 1 else np.asarray(Y)
    XY = np.hstack([X, Y])

    def _entropy_gaussian(Z: np.ndarray) -> float:
        n = Z.shape[1]
        C = np.cov(Z.T) + np.eye(n) * 1e-8
        if n == 1:
            C = np.atleast_2d(C)
        sign, logdet = np.linalg.slogdet(C)
        return 0.5 * (n * (1.0 + np.log(2.0 * np.pi)) + logdet)

    return _entropy_gaussian(X) + _entropy_gaussian(Y) - _entropy_gaussian(XY)


def gaussian_mmi_pid(
    alpha_j: np.ndarray,
    alpha_k: np.ndarray,
    target: np.ndarray,
) -> PIDResult:
    """Compute PID decomposition using the MMI (min-mutual-information) redundancy.

    PID decomposes I(J,K;T) into four non-negative atoms:
        R  = min(I(J;T), I(K;T))    — redundancy
        U_j = I(J;T) - R            — unique from j
        U_k = I(K;T) - R            — unique from k
        S  = I(J,K;T) - R - U_j - U_k  — synergy

    Args:
        alpha_j: (T,) time series for source j.
        alpha_k: (T,) time series for source k.
        target:  (T,) target time series.

    Returns:
        PIDResult with non-negative decomposition atoms.
    """
    I_j = _mutual_info_gaussian(alpha_j, target)
    I_k = _mutual_info_gaussian(alpha_k, target)
    jk = np.column_stack([alpha_j, alpha_k])
    I_jk = _mutual_info_gaussian(jk, target)

    # MMI redundancy: min of individual mutual informations
    R = max(min(I_j, I_k), 0.0)
    U_j = max(I_j - R, 0.0)
    U_k = max(I_k - R, 0.0)
    S = max(I_jk - R - U_j - U_k, 0.0)

    return PIDResult(redundancy=R, unique_j=U_j, unique_k=U_k, synergy=S)


def compute_synergy_matrix(
    alpha_filtered: np.ndarray,
    gaps: np.ndarray,
    K: int,
) -> np.ndarray:
    """Build a symmetric pairwise synergy matrix across modal states.

    Uses batched PyTorch implementation for all K*(K-1)/2 pairs simultaneously.

    Args:
        alpha_filtered: (T, K) filtered modal states.
        gaps:           (N, T) gap matrix, collapsed to scalar target via mean.
        K:              Number of modes (clipped to available columns).

    Returns:
        (K, K) symmetric matrix S[j, k] = synergy(mode_j, mode_k → aggregate_gap).
    """
    K = min(K, alpha_filtered.shape[1])
    target = gaps.mean(axis=0)  # (T,) aggregate gap signal
    synergy, _, _ = batch_pid_synergy(alpha_filtered[:, :K], target, n_bootstrap=1)
    return synergy


def _compute_synergy_matrix_sequential(
    alpha_filtered: np.ndarray,
    gaps: np.ndarray,
    K: int,
) -> np.ndarray:
    """Fallback: original sequential numpy implementation.

    Args:
        alpha_filtered: (T, K) filtered modal states.
        gaps:           (N, T) gap matrix, collapsed to scalar target via mean.
        K:              Number of modes (clipped to available columns).

    Returns:
        (K, K) symmetric matrix S[j, k] = synergy(mode_j, mode_k → aggregate_gap).
    """
    K = min(K, alpha_filtered.shape[1])
    target = gaps.mean(axis=0)
    S = np.zeros((K, K))

    for j in range(K):
        for k in range(j + 1, K):
            pid = gaussian_mmi_pid(alpha_filtered[:, j], alpha_filtered[:, k], target)
            S[j, k] = pid.synergy
            S[k, j] = pid.synergy

    return S


# ---------------------------------------------------------------------------
# M4.6-T2: Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_synergy(
    alpha_j: np.ndarray,
    alpha_k: np.ndarray,
    target: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    block_size: int = 5,
) -> tuple[float, float, float]:
    """Estimate bootstrap confidence interval for PID synergy.

    Uses batched PyTorch implementation for all bootstrap samples at once.

    Args:
        alpha_j:    (T,) source time series j.
        alpha_k:    (T,) source time series k.
        target:     (T,) target time series.
        n_bootstrap: Number of bootstrap samples.
        ci_level:   Coverage level (e.g., 0.95).
        block_size: Block length for block bootstrap.

    Returns:
        (observed_synergy, ci_lower, ci_upper)
    """
    modal_states = np.column_stack([alpha_j, alpha_k])  # (T, 2)
    synergy, ci_lower, ci_upper = batch_pid_synergy(
        modal_states, target,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        block_size=block_size,
    )
    # synergy[0, 1] = synergy between modes 0 and 1
    observed = float(synergy[0, 1])
    lower = float(ci_lower[0, 1])
    upper = float(ci_upper[0, 1])
    return observed, lower, upper


def _bootstrap_synergy_sequential(
    alpha_j: np.ndarray,
    alpha_k: np.ndarray,
    target: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    block_size: int = 5,
) -> tuple[float, float, float]:
    """Fallback: original sequential block-bootstrap implementation.

    Args:
        alpha_j:    (T,) source time series j.
        alpha_k:    (T,) source time series k.
        target:     (T,) target time series.
        n_bootstrap: Number of bootstrap samples.
        ci_level:   Coverage level (e.g., 0.95).
        block_size: Block length for block bootstrap.

    Returns:
        (observed_synergy, ci_lower, ci_upper)
    """
    T = len(alpha_j)
    synergies: list[float] = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        n_blocks = max(T // block_size, 1)
        starts = rng.integers(0, max(T - block_size + 1, 1), size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_size) for s in starts]
        )[:T]
        pid = gaussian_mmi_pid(
            alpha_j[indices], alpha_k[indices], target[indices]
        )
        synergies.append(pid.synergy)

    alpha_ci = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(synergies, alpha_ci))
    upper = float(np.quantile(synergies, 1.0 - alpha_ci))
    observed = gaussian_mmi_pid(alpha_j, alpha_k, target).synergy
    return observed, lower, upper
