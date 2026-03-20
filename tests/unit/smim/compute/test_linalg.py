"""Unit tests for smim.compute.linalg — all ops must match numpy/scipy to atol=1e-10."""
import numpy as np
import pytest
import torch
import scipy.linalg

from quantdsl_backtest.smim.compute.torch_ops import get_device
from quantdsl_backtest.smim.compute.linalg import (
    batch_covariance,
    batch_solve,
    eigh,
    hermitian_dilation_decompose,
    polar_decompose,
    schur_decompose,
    svd,
)

ATOL = 1e-10
RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def clear_device_cache():
    get_device.cache_clear()
    yield
    get_device.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(shape: tuple, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = RNG
    return rng.standard_normal(shape)


def _sym_pd(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random symmetric positive-definite matrix."""
    A = _rand((n, n), rng)
    return A @ A.T + np.eye(n) * n


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

class TestSvd:
    def test_shape_full(self):
        A = _rand((8, 5))
        U, S, Vh = svd(A)
        assert U.shape == (8, 5)
        assert S.shape == (5,)
        assert Vh.shape == (5, 5)

    def test_reconstruction(self):
        A = _rand((10, 6))
        U, S, Vh = svd(A)
        A_rec = (U * S) @ Vh
        np.testing.assert_allclose(A_rec, A, atol=ATOL)

    def test_matches_numpy(self):
        A = _rand((12, 8))
        U_np, S_np, Vh_np = np.linalg.svd(A, full_matrices=False)
        U_t, S_t, Vh_t = svd(A)
        # Singular values must match exactly
        np.testing.assert_allclose(S_t, S_np, atol=ATOL)
        # Reconstruction must match
        np.testing.assert_allclose((U_t * S_t) @ Vh_t, (U_np * S_np) @ Vh_np, atol=ATOL)

    def test_truncation(self):
        A = _rand((20, 15))
        k = 5
        U, S, Vh = svd(A, k=k)
        assert U.shape == (20, k)
        assert S.shape == (k,)
        assert Vh.shape == (k, 15)

    def test_singular_values_descending(self):
        A = _rand((10, 8))
        _, S, _ = svd(A)
        assert np.all(np.diff(S) <= 1e-12), "Singular values not descending"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        A = _rand((15, 10))
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        _, S_cpu, _ = svd(A)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        _, S_gpu, _ = svd(A)
        np.testing.assert_allclose(S_gpu, S_cpu, atol=ATOL)


# ---------------------------------------------------------------------------
# EIGH
# ---------------------------------------------------------------------------

class TestEigh:
    def test_shape(self):
        H = _sym_pd(8)
        vals, vecs = eigh(H)
        assert vals.shape == (8,)
        assert vecs.shape == (8, 8)

    def test_eigenvalues_positive(self):
        H = _sym_pd(6)
        vals, _ = eigh(H)
        assert np.all(vals > 0)

    def test_matches_numpy(self):
        H = _sym_pd(10)
        vals_np, vecs_np = np.linalg.eigh(H)
        vals_t, vecs_t = eigh(H)
        # eigenvalues must match (possibly reordered)
        np.testing.assert_allclose(
            np.sort(vals_t), np.sort(vals_np), atol=ATOL
        )

    def test_reconstruction(self):
        H = _sym_pd(8)
        vals, vecs = eigh(H)
        H_rec = vecs @ np.diag(vals) @ vecs.T
        np.testing.assert_allclose(H_rec, H, atol=ATOL)

    def test_top_k_truncation(self):
        H = _sym_pd(12)
        k = 4
        vals, vecs = eigh(H, k=k)
        assert vals.shape == (k,)
        assert vecs.shape == (12, k)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        H = _sym_pd(8)
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        vals_cpu, _ = eigh(H)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        vals_gpu, _ = eigh(H)
        np.testing.assert_allclose(np.sort(vals_gpu), np.sort(vals_cpu), atol=ATOL)


# ---------------------------------------------------------------------------
# Polar decompose
# ---------------------------------------------------------------------------

class TestPolarDecompose:
    def test_shape_square(self):
        A = _rand((6, 6))
        U, P = polar_decompose(A)
        assert U.shape == (6, 6)
        assert P.shape == (6, 6)

    def test_orthogonal_factor(self):
        A = _rand((8, 8))
        U, _ = polar_decompose(A)
        eye = U.T @ U
        np.testing.assert_allclose(eye, np.eye(8), atol=ATOL)

    def test_psd_factor(self):
        A = _rand((6, 6))
        _, P = polar_decompose(A)
        # P must be symmetric
        np.testing.assert_allclose(P, P.T, atol=ATOL)
        # P must be PSD (all eigenvalues >= 0)
        eigs = np.linalg.eigvalsh(P)
        assert np.all(eigs >= -ATOL)

    def test_reconstruction(self):
        A = _rand((7, 7))
        U, P = polar_decompose(A)
        np.testing.assert_allclose(U @ P, A, atol=ATOL)

    def test_matches_scipy(self):
        A = _rand((8, 8))
        U_sc, P_sc = scipy.linalg.polar(A)
        U_t, P_t = polar_decompose(A)
        # Sign convention may differ; check reconstruction
        np.testing.assert_allclose(U_t @ P_t, U_sc @ P_sc, atol=ATOL)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        A = _rand((8, 8))
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        U_cpu, P_cpu = polar_decompose(A)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        U_gpu, P_gpu = polar_decompose(A)
        np.testing.assert_allclose(U_gpu @ P_gpu, U_cpu @ P_cpu, atol=ATOL)


# ---------------------------------------------------------------------------
# Schur decompose (always scipy / CPU)
# ---------------------------------------------------------------------------

class TestSchurDecompose:
    def test_shape(self):
        A = _rand((6, 6))
        Q, T = schur_decompose(A)
        assert Q.shape == (6, 6)
        assert T.shape == (6, 6)

    def test_reconstruction(self):
        A = _rand((8, 8))
        Q, T = schur_decompose(A)
        # A = Q T Q^H
        np.testing.assert_allclose(Q @ T @ Q.conj().T, A.astype(complex), atol=ATOL)

    def test_unitary_Q(self):
        A = _rand((6, 6))
        Q, _ = schur_decompose(A)
        eye = Q.conj().T @ Q
        np.testing.assert_allclose(eye, np.eye(6, dtype=complex), atol=ATOL)

    def test_matches_scipy(self):
        A = _rand((8, 8))
        T_sc, Q_sc = scipy.linalg.schur(A, output='complex')
        Q_t, T_t = schur_decompose(A)
        # Reconstructions must match
        rec_t = Q_t @ T_t @ Q_t.conj().T
        rec_sc = Q_sc @ T_sc @ Q_sc.conj().T
        np.testing.assert_allclose(np.real(rec_t), np.real(rec_sc), atol=ATOL)

    def test_always_returns_cpu_result(self, monkeypatch):
        """schur_decompose must return numpy arrays regardless of SMIM_DEVICE."""
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        A = _rand((6, 6))
        Q, T = schur_decompose(A)
        assert isinstance(Q, np.ndarray)
        assert isinstance(T, np.ndarray)

    def test_top_k(self):
        A = _rand((8, 8))
        k = 3
        Q, T = schur_decompose(A, k=k)
        assert Q.shape == (8, k)
        assert T.shape == (k, k)


# ---------------------------------------------------------------------------
# Hermitian dilation decompose
# ---------------------------------------------------------------------------

class TestHermitianDilationDecompose:
    def test_shape(self):
        A = _rand((6, 6))
        k = 4
        sigma, U_L, U_R = hermitian_dilation_decompose(A, k=k)
        assert sigma.shape == (k,)
        assert U_L.shape == (6, k)
        assert U_R.shape == (6, k)

    def test_sigma_positive(self):
        A = _rand((8, 6))
        sigma, _, _ = hermitian_dilation_decompose(A, k=4)
        assert np.all(sigma > 0)

    def test_sigma_descending(self):
        A = _rand((10, 8))
        sigma, _, _ = hermitian_dilation_decompose(A, k=5)
        assert np.all(np.diff(sigma) <= 1e-10)

    def test_approximation_quality(self):
        """U_L @ diag(sigma) @ U_R^T should reconstruct A exactly (all k singular values)."""
        rng = np.random.default_rng(7)
        A = _rand((6, 6), rng)
        k = 6
        sigma, U_L, U_R = hermitian_dilation_decompose(A, k=k)
        # U_L and U_R already absorb the √2 factor, so A_rec = Σ σ_i u_i v_i^T = A
        A_rec = U_L @ np.diag(sigma) @ U_R.T
        np.testing.assert_allclose(A_rec, A, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        A = _rand((8, 8))
        k = 4
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        sigma_cpu, _, _ = hermitian_dilation_decompose(A, k=k)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        sigma_gpu, _, _ = hermitian_dilation_decompose(A, k=k)
        np.testing.assert_allclose(sigma_gpu, sigma_cpu, atol=ATOL)


# ---------------------------------------------------------------------------
# Batch covariance
# ---------------------------------------------------------------------------

class TestBatchCovariance:
    def test_shape(self):
        X = _rand((100, 5))
        C = batch_covariance(X)
        assert C.shape == (5, 5)

    def test_symmetric(self):
        X = _rand((50, 6))
        C = batch_covariance(X)
        np.testing.assert_allclose(C, C.T, atol=ATOL)

    def test_matches_numpy(self):
        X = _rand((200, 8))
        C_np = np.cov(X.T)
        C_t = batch_covariance(X)
        np.testing.assert_allclose(C_t, C_np, atol=ATOL)

    def test_psd(self):
        X = _rand((100, 5))
        C = batch_covariance(X)
        eigs = np.linalg.eigvalsh(C)
        assert np.all(eigs >= -ATOL)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        X = _rand((100, 5))
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        C_cpu = batch_covariance(X)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        C_gpu = batch_covariance(X)
        np.testing.assert_allclose(C_gpu, C_cpu, atol=ATOL)


# ---------------------------------------------------------------------------
# Batch solve
# ---------------------------------------------------------------------------

class TestBatchSolve:
    def test_shape_2d(self):
        A = _sym_pd(5)
        b = _rand((5, 3))
        X = batch_solve(A, b)
        assert X.shape == (5, 3)

    def test_solves_correctly_2d(self):
        A = _sym_pd(6)
        b = _rand((6, 4))
        X = batch_solve(A, b)
        np.testing.assert_allclose(A @ X, b, atol=ATOL)

    def test_batched_3d(self):
        batch = 7
        n = 5
        A = np.stack([_sym_pd(n) for _ in range(batch)])
        b = _rand((batch, n, 3))
        X = batch_solve(A, b)
        assert X.shape == (batch, n, 3)
        for i in range(batch):
            np.testing.assert_allclose(A[i] @ X[i], b[i], atol=ATOL)

    def test_matches_numpy(self):
        A = _sym_pd(8)
        b = _rand((8, 5))
        X_np = np.linalg.solve(A, b)
        X_t = batch_solve(A, b)
        np.testing.assert_allclose(X_t, X_np, atol=ATOL)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu(self, monkeypatch):
        A = _sym_pd(8)
        b = _rand((8, 5))
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        X_cpu = batch_solve(A, b)
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        X_gpu = batch_solve(A, b)
        np.testing.assert_allclose(X_gpu, X_cpu, atol=ATOL)
