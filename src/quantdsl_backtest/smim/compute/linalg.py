"""Linear algebra operations — single implementation for CPU and CUDA.

Every function: numpy in → numpy out. PyTorch used internally.
Schur decomposition falls back to scipy (not available in torch).
"""
import numpy as np
import torch
import scipy.linalg
from numpy.typing import NDArray
from .torch_ops import ensure_tensor, to_numpy, get_device


def svd(A: NDArray, k: int | None = None) -> tuple[NDArray, NDArray, NDArray]:
    """SVD with optional truncation. Works on CPU or CUDA."""
    A_t = ensure_tensor(A)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
    if k is not None:
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    return to_numpy(U), to_numpy(S), to_numpy(Vh)


def eigh(H: NDArray, k: int | None = None) -> tuple[NDArray, NDArray]:
    """Symmetric eigendecomposition. Works on CPU or CUDA."""
    H_t = ensure_tensor(H)
    vals, vecs = torch.linalg.eigh(H_t)
    if k is not None:
        # Return top-k by magnitude
        idx = torch.argsort(vals.abs(), descending=True)[:k]
        vals, vecs = vals[idx], vecs[:, idx]
    return to_numpy(vals), to_numpy(vecs)


def polar_decompose(A: NDArray) -> tuple[NDArray, NDArray]:
    """Polar decomposition A = UP via SVD. Works on CPU or CUDA."""
    A_t = ensure_tensor(A)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
    orth = U @ Vh
    psd = (Vh.conj().mT * S.unsqueeze(-2)) @ Vh
    return to_numpy(orth), to_numpy(psd)


def schur_decompose(A: NDArray, k: int | None = None) -> tuple[NDArray, NDArray]:
    """Schur decomposition A = QTQ^H. Always runs on CPU (scipy).

    torch.linalg.schur does not exist. This is the one operation that
    cannot be GPU-accelerated. It takes <0.1% of pipeline time, so
    the impact is negligible.
    """
    T, Q = scipy.linalg.schur(np.asarray(A), output='complex')
    if k is not None:
        eig_magnitudes = np.abs(np.diag(T))
        idx = np.argsort(eig_magnitudes)[::-1][:k]
        Q, T = Q[:, idx], T[np.ix_(idx, idx)]
    return Q, T


def hermitian_dilation_decompose(
    A: NDArray, k: int
) -> tuple[NDArray, NDArray, NDArray]:
    """Hermitian dilation: eigendecompose H = [[0,A],[A^T,0]].

    Returns (eigenvalues, U_L, U_R). Works on CPU or CUDA.
    Supports non-square A of shape (M, N): H is (M+N) × (M+N).
    """
    M, N = A.shape
    A_t = ensure_tensor(A)
    zeros_MM = torch.zeros((M, M), device=A_t.device, dtype=A_t.dtype)
    zeros_NN = torch.zeros((N, N), device=A_t.device, dtype=A_t.dtype)
    H = torch.cat([
        torch.cat([zeros_MM, A_t], dim=1),
        torch.cat([A_t.T, zeros_NN], dim=1),
    ], dim=0)
    vals, vecs = torch.linalg.eigh(H)
    # Select top-k positive eigenvalues
    pos_mask = vals > 1e-10
    pos_vals = vals[pos_mask]
    pos_vecs = vecs[:, pos_mask]
    # Sort descending
    idx = torch.argsort(pos_vals, descending=True)[:k]
    sigma = pos_vals[idx]
    # Extract U_L (first M rows) and U_R (last N rows)
    U_L = pos_vecs[:M, idx] * (2 ** 0.5)
    U_R = pos_vecs[M:, idx] * (2 ** 0.5)
    return to_numpy(sigma), to_numpy(U_L), to_numpy(U_R)


def batch_covariance(X: NDArray) -> NDArray:
    """Covariance matrix. Works on CPU or CUDA."""
    X_t = ensure_tensor(X)
    mean = X_t.mean(dim=0, keepdim=True)
    centered = X_t - mean
    cov = (centered.T @ centered) / (X_t.shape[0] - 1)
    return to_numpy(cov)


def batch_solve(A: NDArray, B: NDArray) -> NDArray:
    """Batched linear solve A @ X = B. Shape (batch, N, N) and (batch, N, M).

    Works on CPU or CUDA — this is the key primitive for batch Granger.
    """
    A_t = ensure_tensor(A)
    B_t = ensure_tensor(B)
    X_t = torch.linalg.solve(A_t, B_t)
    return to_numpy(X_t)
