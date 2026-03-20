"""Batch Granger causality via PyTorch batched least-squares.

For N actors and max_lag L, the Granger test for i→j is::

    Restricted:    y_j[t] = Σ_{l=1}^L a_l y_j[t-l] + c + ε           (L+1 params)
    Unrestricted:  y_j[t] = Σ_{l=1}^L a_l y_j[t-l]
                           + Σ_{l=1}^L b_l y_i[t-l] + c + ε           (2L+1 params)
    F = ((RSS_r − RSS_u) / L) / (RSS_u / (T_eff − 2L − 1))

Strategy: construct ALL N*(N-1) regression matrices in a single batch tensor
and solve via torch.linalg.lstsq. This replaces the Python loop over N²
pairs with a single GPU kernel launch per chunk.

Memory management: pairs are processed in chunks of `chunk_size` to bound
peak GPU memory. At N=500, L=4, T=2000 each chunk of 10,000 pairs uses
~700MB of VRAM; 25 chunks cover all 249,500 pairs.

Reference implementation `_granger_sequential` uses the identical OLS
formula via numpy and is kept for parity testing.
"""

from __future__ import annotations

import numpy as np
import torch
import scipy.stats
from numpy.typing import NDArray

from .torch_ops import ensure_tensor, get_device, to_numpy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_lagged_features(signals: NDArray, lag: int) -> NDArray:
    """Construct lag matrix from a (T, N) signal array.

    Returns:
        lags: float64 array of shape (T_eff, N, lag) where
              lags[t, n, l] = signals[t + lag - 1 - l, n]   (i.e. lag l+1 of actor n)
              and T_eff = T - lag.
    """
    T, N = signals.shape
    T_eff = T - lag
    lags = np.empty((T_eff, N, lag), dtype=np.float64)
    for l in range(lag):
        # l=0 → lag-1 value; l=lag-1 → lag-lag value
        lags[:, :, l] = signals[lag - l - 1: T - l - 1, :]
    return lags


def _solve_rss(
    X: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched OLS solve returning (solution, residual sum of squares).

    Args:
        X: (batch, T_eff, n_params) design matrix
        y: (batch, T_eff, 1) response

    Returns:
        sol: (batch, n_params, 1)
        rss: (batch,)  — ||y − X @ sol||²
    """
    device = X.device
    driver = "gels" if device.type == "cuda" else None
    sol = torch.linalg.lstsq(X, y, driver=driver).solution
    residuals = y - X @ sol
    rss = (residuals ** 2).sum(dim=1).squeeze(-1)
    return sol, rss


# ---------------------------------------------------------------------------
# Sequential reference implementation (numpy, CPU only)
# ---------------------------------------------------------------------------

def _granger_sequential(
    signals: NDArray,
    lag: int,
    p_threshold: float,
) -> NDArray:
    """Sequential OLS Granger test — reference implementation for parity tests.

    Uses the identical F-statistic formula as ``batch_granger_test``.
    Does NOT perform BIC lag selection; uses the supplied fixed ``lag``.

    Args:
        signals:     (T, N) float64 time-series array, no NaNs.
        lag:         Fixed lag order.
        p_threshold: Significance level for edge inclusion.

    Returns:
        adj: (N, N) where adj[j, i] = F-stat if i Granger-causes j, else 0.
    """
    T, N = signals.shape
    T_eff = T - lag
    df_num = lag
    df_denom = T_eff - 2 * lag - 1
    if df_denom <= 0:
        raise ValueError(
            f"Not enough observations: T_eff={T_eff}, lag={lag}, need T_eff > 2*lag+1"
        )

    lags_np = _build_lagged_features(signals, lag)   # (T_eff, N, lag)
    y_all = signals[lag:, :]                          # (T_eff, N)
    ones = np.ones((T_eff, 1))
    adj = np.zeros((N, N))

    for j in range(N):  # effect (caused series)
        y = y_all[:, j]
        X_r = np.hstack([lags_np[:, j, :], ones])    # (T_eff, lag+1)
        beta_r, _, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
        RSS_r = float(np.sum((y - X_r @ beta_r) ** 2))

        for i in range(N):  # cause
            if i == j:
                continue
            X_u = np.hstack([lags_np[:, j, :], lags_np[:, i, :], ones])  # (T_eff, 2L+1)
            beta_u, _, _, _ = np.linalg.lstsq(X_u, y, rcond=None)
            RSS_u = float(np.sum((y - X_u @ beta_u) ** 2))

            if RSS_u <= 0 or RSS_r <= RSS_u:
                continue
            F = ((RSS_r - RSS_u) / df_num) / (RSS_u / df_denom)
            p = float(scipy.stats.f.sf(F, df_num, df_denom))
            if p < p_threshold:
                adj[j, i] = F

    return adj


# ---------------------------------------------------------------------------
# Batch GPU implementation
# ---------------------------------------------------------------------------

def batch_granger_test(
    signals: NDArray,
    max_lag: int = 4,
    p_threshold: float = 0.05,
    chunk_size: int = 10000,
) -> NDArray:
    """Batch Granger causality test via PyTorch batched lstsq.

    Constructs all N*(N-1) bivariate OLS regressions simultaneously as a
    3-D batch tensor and solves them in chunks via ``torch.linalg.lstsq``.
    Runs on CUDA if available (set via SMIM_DEVICE env var or auto-detect).

    Uses a fixed lag equal to ``max_lag`` for all pairs (no BIC selection).
    For BIC-based lag selection use ``GrangerEdgeEstimator`` with
    ``bic_selection=True``.

    Args:
        signals:     (T, N) float64 time-series array, no NaNs.
        max_lag:     Lag order applied uniformly to all pairs.
        p_threshold: Significance level for edge inclusion.
        chunk_size:  Max number of pairs per GPU batch. Tune for VRAM.
                     Default 10,000 uses ~700 MB per chunk at N=500, L=4.

    Returns:
        adj: (N, N) float64 where adj[j, i] = F-stat if i Granger-causes j
             (p < p_threshold), otherwise 0.
    """
    signals = np.asarray(signals, dtype=np.float64)
    T, N = signals.shape
    L = max_lag
    T_eff = T - L
    df_num = L
    df_denom = T_eff - 2 * L - 1

    if df_denom <= 0:
        raise ValueError(
            f"Not enough observations: T={T}, lag={L}, "
            f"need T > 3*lag+1 (T_eff={T_eff} > 2*lag+1={2*L+1})"
        )

    device = get_device()

    # Precompute lag features and upload to device once
    lags_np = _build_lagged_features(signals, L)          # (T_eff, N, L)
    lags_t = ensure_tensor(lags_np, device=device)         # (T_eff, N, L)
    y_all_t = ensure_tensor(signals[L:, :], device=device) # (T_eff, N)
    ones_t = torch.ones(T_eff, 1, device=device, dtype=torch.float64)

    # Enumerate all (effect=j, cause=i) pairs with j ≠ i
    js = [j for j in range(N) for i in range(N) if j != i]
    is_ = [i for j in range(N) for i in range(N) if j != i]
    num_pairs = len(js)

    js_t = torch.tensor(js, device=device, dtype=torch.long)
    is_t = torch.tensor(is_, device=device, dtype=torch.long)

    fstats_flat = np.zeros(num_pairs)
    pvals_flat = np.ones(num_pairs)

    for start in range(0, num_pairs, chunk_size):
        end = min(start + chunk_size, num_pairs)
        j_chunk = js_t[start:end]   # (B,)
        i_chunk = is_t[start:end]   # (B,)
        B = end - start

        # Gather lag tensors for this chunk — (B, T_eff, L)
        eff_lags = lags_t[:, j_chunk, :].permute(1, 0, 2)
        cau_lags = lags_t[:, i_chunk, :].permute(1, 0, 2)

        # Ones column broadcast to (B, T_eff, 1)
        ones_b = ones_t.unsqueeze(0).expand(B, -1, -1)

        # Response: (B, T_eff, 1)
        y_b = y_all_t[:, j_chunk].T.unsqueeze(-1)

        # Design matrices
        X_r = torch.cat([eff_lags, ones_b], dim=-1)             # (B, T_eff, L+1)
        X_u = torch.cat([eff_lags, cau_lags, ones_b], dim=-1)   # (B, T_eff, 2L+1)

        _, RSS_r_t = _solve_rss(X_r, y_b)
        _, RSS_u_t = _solve_rss(X_u, y_b)

        RSS_r_np = to_numpy(RSS_r_t)
        RSS_u_np = to_numpy(RSS_u_t)

        # F-stat is valid only when RSS_u < RSS_r (adding predictors helps)
        valid = (RSS_u_np > 0) & (RSS_r_np > RSS_u_np)
        F_np = np.where(
            valid,
            ((RSS_r_np - RSS_u_np) / df_num) / (RSS_u_np / df_denom),
            0.0,
        )
        p_np = np.where(
            valid,
            scipy.stats.f.sf(F_np, df_num, df_denom),
            1.0,
        )

        fstats_flat[start:end] = F_np
        pvals_flat[start:end] = p_np

    # Assemble adjacency matrix: adj[effect, cause] = F-stat
    adj = np.zeros((N, N))
    for k, (j, i) in enumerate(zip(js, is_)):
        if pvals_flat[k] < p_threshold:
            adj[j, i] = fstats_flat[k]

    return adj
