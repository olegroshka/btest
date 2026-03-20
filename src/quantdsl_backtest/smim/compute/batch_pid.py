"""Batch PID bootstrap via PyTorch.

For Gaussian MMI PID, synergy depends only on covariance matrices.
The bootstrap generates B resampled covariance matrices — each is a
matrix multiply, trivially parallelisable.

Strategy:
1. Generate B block-bootstrap index arrays on device
2. Build all K*(K-1)/2 pairs in a single (n_pairs, T, 3) tensor
3. Gather all B bootstrap samples: (n_pairs*B, T, 3)
4. Compute all covariances at once: (n_pairs*B, 3, 3) via batched matrix multiply
5. Extract MI terms via torch.linalg.slogdet — all pairs × all samples in one call
6. Return synergy matrix (K, K) and optional CIs as numpy

Reference implementation ``_pid_bootstrap_sequential`` uses the identical
block-bootstrap formula via NumPy and is kept for parity testing.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from .torch_ops import ensure_tensor, get_device, to_numpy

_REG = 1e-8  # covariance regularisation constant


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _block_bootstrap_indices(
    T: int,
    n_bootstrap: int,
    block_size: int,
    rng: np.random.Generator,
) -> NDArray:
    """Return (n_bootstrap, T) integer index array via block bootstrap.

    Args:
        T: Series length.
        n_bootstrap: Number of bootstrap samples.
        block_size: Block length (should be ~sqrt(T) for time series).
        rng: NumPy random generator.

    Returns:
        (n_bootstrap, T) int64 index array clipped to [0, T).
    """
    n_blocks = max(T // block_size, 1)
    max_start = max(T - block_size + 1, 1)
    starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))  # (B, n_blocks)
    offsets = np.arange(block_size)  # (block_size,)
    # (B, n_blocks, block_size) → (B, n_blocks * block_size) → truncate to T
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_bootstrap, -1)
    return indices[:, :T]


def _batched_cov(X: torch.Tensor) -> torch.Tensor:
    """Compute batched unbiased covariance matrices with regularisation.

    Args:
        X: (B, T, C) data tensor (uncentred).

    Returns:
        (B, C, C) regularised covariance matrices.
    """
    B, T, C = X.shape
    X_c = X - X.mean(dim=1, keepdim=True)              # (B, T, C)
    cov = torch.bmm(X_c.transpose(1, 2), X_c) / (T - 1)  # (B, C, C)
    eye = torch.eye(C, dtype=X.dtype, device=X.device).unsqueeze(0)
    return cov + _REG * eye                              # (B, C, C)


def _synergy_from_3x3_cov(cov: torch.Tensor) -> torch.Tensor:
    """Compute MMI PID synergy from batched 3×3 covariance matrices.

    Column layout: 0 = alpha_j, 1 = alpha_k, 2 = target.

    I(X; Y) = 0.5 * (logdet Σ_X + logdet Σ_Y − logdet Σ_{XY})

    MMI PID:
        R   = max(min(I_j, I_k), 0)
        U_j = max(I_j − R, 0)
        U_k = max(I_k − R, 0)
        S   = max(I_jk − R − U_j − U_k, 0)

    Args:
        cov: (B, 3, 3) regularised covariance matrices.

    Returns:
        (B,) synergy values (≥ 0).
    """
    # Scalar log-variances
    _, ld_j = torch.linalg.slogdet(cov[:, 0:1, 0:1])  # (B,) logdet of var(alpha_j)
    _, ld_k = torch.linalg.slogdet(cov[:, 1:2, 1:2])  # (B,) logdet of var(alpha_k)
    _, ld_t = torch.linalg.slogdet(cov[:, 2:3, 2:3])  # (B,) logdet of var(target)

    # 2×2 sub-matrices for pair MI terms
    cov_jt = cov[:, [0, 2], :][:, :, [0, 2]]  # (B, 2, 2)
    cov_kt = cov[:, [1, 2], :][:, :, [1, 2]]  # (B, 2, 2)
    cov_jk = cov[:, :2, :2]                   # (B, 2, 2) — for I_jk joint source

    _, ld_jt = torch.linalg.slogdet(cov_jt)  # (B,)
    _, ld_kt = torch.linalg.slogdet(cov_kt)  # (B,)
    _, ld_jk = torch.linalg.slogdet(cov_jk)  # (B,)
    _, ld_3  = torch.linalg.slogdet(cov)     # (B,) full 3×3

    # Mutual information terms
    I_j  = 0.5 * (ld_j + ld_t - ld_jt)
    I_k  = 0.5 * (ld_k + ld_t - ld_kt)
    I_jk = 0.5 * (ld_jk + ld_t - ld_3)

    # MMI PID decomposition
    R   = torch.minimum(I_j, I_k).clamp(min=0.0)
    U_j = (I_j  - R).clamp(min=0.0)
    U_k = (I_k  - R).clamp(min=0.0)
    S   = (I_jk - R - U_j - U_k).clamp(min=0.0)
    return S


# ---------------------------------------------------------------------------
# Sequential fallback (CPU / NumPy)
# ---------------------------------------------------------------------------


def _pid_bootstrap_sequential(
    alpha_j: NDArray,
    alpha_k: NDArray,
    target: NDArray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    block_size: int = 5,
) -> tuple[float, float, float]:
    """Block-bootstrap synergy CI — sequential NumPy fallback.

    Identical logic to the original ``bootstrap_synergy`` in emergence/pid.py.
    Kept so the batched implementation can be validated against it.

    Args:
        alpha_j: (T,) source time series j.
        alpha_k: (T,) source time series k.
        target:  (T,) target time series.
        n_bootstrap: Number of bootstrap samples.
        ci_level: Coverage level (e.g. 0.95).
        block_size: Block length for block bootstrap.

    Returns:
        (observed_synergy, ci_lower, ci_upper)
    """
    from quantdsl_backtest.smim.emergence.pid import gaussian_mmi_pid

    T = len(alpha_j)
    synergies: list[float] = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        n_blocks = max(T // block_size, 1)
        starts = rng.integers(0, max(T - block_size + 1, 1), size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_size) for s in starts]
        )[:T]
        pid = gaussian_mmi_pid(alpha_j[indices], alpha_k[indices], target[indices])
        synergies.append(pid.synergy)

    alpha_ci = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(synergies, alpha_ci))
    upper = float(np.quantile(synergies, 1.0 - alpha_ci))
    observed = gaussian_mmi_pid(alpha_j, alpha_k, target).synergy
    return observed, lower, upper


# ---------------------------------------------------------------------------
# Batched PyTorch implementation
# ---------------------------------------------------------------------------


def batch_pid_synergy(
    modal_states: NDArray,
    target: NDArray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    block_size: int = 5,
    device: torch.device | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Batch PID bootstrap across all mode pairs using PyTorch.

    Computes K*(K-1)/2 pairs × B bootstrap samples in a single batched
    covariance computation on the target device (CPU or CUDA).

    Args:
        modal_states: (T, K) filtered modal state time series.
        target:       (T,) target time series (e.g. aggregate gap signal).
        n_bootstrap:  Number of block-bootstrap samples.
        ci_level:     Confidence interval coverage (e.g. 0.95).
        block_size:   Block length for block bootstrap (~sqrt(T) recommended).
        device:       PyTorch device.  None → auto-detect via SMIM_DEVICE env var.

    Returns:
        synergy:  (K, K) symmetric synergy matrix — point estimates from the
                  full (non-bootstrapped) data.  Matches ``compute_synergy_matrix``
                  / ``gaussian_mmi_pid`` within atol=1e-4.
        ci_lower: (K, K) lower CI bound from bootstrap distribution.
        ci_upper: (K, K) upper CI bound from bootstrap distribution.
    """
    if device is None:
        device = get_device()

    T, K = modal_states.shape

    # All unique pairs (j < k)
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
    n_pairs = len(pairs)

    if n_pairs == 0:
        empty = np.zeros((K, K))
        return empty, empty, empty

    # ── Move data to device ──────────────────────────────────────────────
    m_t = ensure_tensor(modal_states, device=device)   # (T, K)
    t_t = ensure_tensor(target, device=device)          # (T,)

    # Build all-pairs data tensor: (n_pairs, T, 3)  [alpha_j, alpha_k, target]
    j_idx = [p[0] for p in pairs]
    k_idx = [p[1] for p in pairs]
    alpha_j_all = m_t[:, j_idx].T                           # (n_pairs, T)
    alpha_k_all = m_t[:, k_idx].T                           # (n_pairs, T)
    target_rep  = t_t.unsqueeze(0).expand(n_pairs, -1)      # (n_pairs, T)
    data_pt = torch.stack([alpha_j_all, alpha_k_all, target_rep], dim=-1)  # (n_pairs, T, 3)

    # ── Point estimates (non-bootstrapped full data) ─────────────────────
    cov_obs = _batched_cov(data_pt)                          # (n_pairs, 3, 3)
    S_obs   = to_numpy(_synergy_from_3x3_cov(cov_obs))      # (n_pairs,)

    # ── Bootstrap sampling ───────────────────────────────────────────────
    rng    = np.random.default_rng(42)
    idx_np = _block_bootstrap_indices(T, n_bootstrap, block_size, rng)  # (B, T)
    idx_t  = torch.as_tensor(idx_np, dtype=torch.long, device=device)   # (B, T)

    # Gather bootstrap samples for all pairs: (n_pairs, B, T, 3)
    X_pbt = data_pt[:, idx_t, :]  # advanced indexing on T dim → (n_pairs, B, T, 3)

    # Flatten pairs × bootstraps for single batched covariance call
    X_flat   = X_pbt.reshape(n_pairs * n_bootstrap, T, 3)   # (n_pairs*B, T, 3)
    cov_flat = _batched_cov(X_flat)                           # (n_pairs*B, 3, 3)

    # Synergy for all pairs × bootstraps
    S_flat = _synergy_from_3x3_cov(cov_flat)                 # (n_pairs*B,)
    S_pb   = to_numpy(S_flat.reshape(n_pairs, n_bootstrap))  # (n_pairs, B)

    # CI from bootstrap distribution
    alpha_ci = (1.0 - ci_level) / 2.0
    ci_lo = np.quantile(S_pb, alpha_ci,       axis=1)  # (n_pairs,)
    ci_hi = np.quantile(S_pb, 1.0 - alpha_ci, axis=1)  # (n_pairs,)

    # ── Assemble symmetric output matrices ───────────────────────────────
    synergy  = np.zeros((K, K))
    ci_lower = np.zeros((K, K))
    ci_upper = np.zeros((K, K))

    for idx, (j, k) in enumerate(pairs):
        synergy[j, k]  = synergy[k, j]  = S_obs[idx]
        ci_lower[j, k] = ci_lower[k, j] = ci_lo[idx]
        ci_upper[j, k] = ci_upper[k, j] = ci_hi[idx]

    return synergy, ci_lower, ci_upper
