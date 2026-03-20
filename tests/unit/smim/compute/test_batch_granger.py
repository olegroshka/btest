"""Unit tests for smim.compute.batch_granger.

Covers:
- _granger_sequential: known edge detection, F-stat formula
- batch_granger_test: matches sequential on same data
- Chunking consistency (different chunk sizes → same result)
- No false positives on i.i.d. data
- CUDA parity with CPU
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from quantdsl_backtest.smim.compute.torch_ops import get_device
from quantdsl_backtest.smim.compute.batch_granger import (
    _build_lagged_features,
    _granger_sequential,
    batch_granger_test,
)

# ---------------------------------------------------------------------------
# Shared VAR(1) fixture  — same DGP as acceptance test A-GR-1
# ---------------------------------------------------------------------------

def _make_var1_data(seed: int = 42, T: int = 500) -> np.ndarray:
    """4-actor VAR(1) with two known causal edges: 0→1 and 3→2."""
    A = np.zeros((4, 4))
    A[1, 0] = 0.6   # 0 → 1
    A[2, 3] = 0.5   # 3 → 2
    A[0, 0] = 0.8
    A[1, 1] = 0.7
    A[2, 2] = 0.75
    A[3, 3] = 0.85
    rng = np.random.default_rng(seed)
    data = np.zeros((T, 4))
    data[0] = rng.standard_normal(4) * 0.1
    for t in range(1, T):
        data[t] = A @ data[t - 1] + 0.5 * rng.standard_normal(4)
    return data


@pytest.fixture(autouse=True)
def clear_device_cache():
    get_device.cache_clear()
    yield
    get_device.cache_clear()


# ---------------------------------------------------------------------------
# _build_lagged_features
# ---------------------------------------------------------------------------

class TestBuildLaggedFeatures:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        lags = _build_lagged_features(X, lag=3)
        assert lags.shape == (97, 5, 3)

    def test_lag1_values(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        lags = _build_lagged_features(X, lag=1)  # T_eff=9
        # lags[t, n, 0] = X[t, n]  (lag-1 of X[t+1])
        np.testing.assert_array_equal(lags[:, :, 0], X[:9, :])

    def test_lag2_values(self):
        X = np.arange(20, dtype=float).reshape(10, 2)
        lags = _build_lagged_features(X, lag=2)  # T_eff=8
        np.testing.assert_array_equal(lags[:, :, 0], X[1:9, :])  # lag-1
        np.testing.assert_array_equal(lags[:, :, 1], X[0:8, :])  # lag-2


# ---------------------------------------------------------------------------
# _granger_sequential
# ---------------------------------------------------------------------------

class TestGrangerSequential:
    def test_detects_known_edges(self):
        data = _make_var1_data(T=2000)
        adj = _granger_sequential(data, lag=1, p_threshold=0.01)
        assert adj[1, 0] > 0, "True edge 0→1 not detected"
        assert adj[2, 3] > 0, "True edge 3→2 not detected"

    def test_rejects_reverse_edges(self):
        data = _make_var1_data(T=2000)
        adj = _granger_sequential(data, lag=1, p_threshold=0.01)
        assert adj[0, 1] == 0.0, "Spurious reverse edge 1→0"
        assert adj[3, 2] == 0.0, "Spurious reverse edge 2→3"

    def test_no_false_positives_iid(self, monkeypatch):
        """i.i.d. Gaussian — no edges at p<0.001 with T=2000 (12 pairs × 0.001 = 1.2% chance)."""
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        rng = np.random.default_rng(7)
        data = rng.standard_normal((2000, 4))
        adj = _granger_sequential(data, lag=1, p_threshold=0.001)
        assert np.all(adj == 0), f"False positives: {np.argwhere(adj != 0).tolist()}"

    def test_self_edges_always_zero(self):
        data = _make_var1_data(T=300)
        adj = _granger_sequential(data, lag=1, p_threshold=0.05)
        assert np.all(np.diag(adj) == 0)

    def test_insufficient_obs_raises(self):
        data = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(ValueError, match="Not enough"):
            _granger_sequential(data, lag=4, p_threshold=0.05)


# ---------------------------------------------------------------------------
# batch_granger_test — correctness
# ---------------------------------------------------------------------------

class TestBatchGrangerCorrectness:
    def test_detects_known_edges(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = _make_var1_data(T=2000)
        adj = batch_granger_test(data, max_lag=1, p_threshold=0.01, chunk_size=100)
        assert adj[1, 0] > 0, "True edge 0→1 not detected by batch"
        assert adj[2, 3] > 0, "True edge 3→2 not detected by batch"

    def test_rejects_reverse_edges(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = _make_var1_data(T=2000)
        adj = batch_granger_test(data, max_lag=1, p_threshold=0.01, chunk_size=100)
        assert adj[0, 1] == 0.0, "Spurious edge 1→0"
        assert adj[3, 2] == 0.0, "Spurious edge 2→3"

    def test_no_false_positives_iid(self, monkeypatch):
        """i.i.d. Gaussian — no edges at p<0.001 with T=2000."""
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        rng = np.random.default_rng(7)
        data = rng.standard_normal((2000, 4))
        adj = batch_granger_test(data, max_lag=1, p_threshold=0.001)
        assert np.all(adj == 0), f"False positives: {np.argwhere(adj != 0).tolist()}"

    def test_self_edges_always_zero(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = _make_var1_data(T=300)
        adj = batch_granger_test(data, max_lag=1, p_threshold=0.05)
        assert np.all(np.diag(adj) == 0)

    def test_insufficient_obs_raises(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(ValueError):
            batch_granger_test(data, max_lag=4, p_threshold=0.05)


# ---------------------------------------------------------------------------
# Parity: batch_granger_test must match _granger_sequential
# ---------------------------------------------------------------------------

class TestBatchVsSequentialParity:
    """Batch (PyTorch) and sequential (numpy) must produce identical results."""

    @pytest.mark.parametrize("lag", [1, 2])
    def test_fstats_agree(self, monkeypatch, lag):
        """F-statistics agree to within rtol=1e-4 on CPU."""
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = _make_var1_data(T=1000)
        threshold = 0.05

        adj_seq = _granger_sequential(data, lag=lag, p_threshold=threshold)
        adj_bat = batch_granger_test(
            data, max_lag=lag, p_threshold=threshold, chunk_size=6
        )

        # Edge structure must be identical
        assert np.array_equal(
            adj_seq != 0, adj_bat != 0
        ), f"Edge mismatch at lag={lag}: seq={np.argwhere(adj_seq!=0).tolist()} bat={np.argwhere(adj_bat!=0).tolist()}"

        # F-stats on detected edges must agree
        mask = adj_seq != 0
        if mask.any():
            np.testing.assert_allclose(
                adj_bat[mask], adj_seq[mask], rtol=1e-4,
                err_msg=f"F-stat disagreement at lag={lag}",
            )

    def test_full_matrix_close(self, monkeypatch):
        """All N*(N-1) F-stats agree even for pairs below the threshold."""
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        rng = np.random.default_rng(99)
        data = rng.standard_normal((300, 3))
        lag = 1

        # Compute raw F-stats via both paths at p=1.0 (include everything)
        adj_seq = _granger_sequential(data, lag=lag, p_threshold=1.0)
        adj_bat = batch_granger_test(data, max_lag=lag, p_threshold=1.0, chunk_size=4)

        # All non-zero entries should agree
        mask = (adj_seq > 0) | (adj_bat > 0)
        np.testing.assert_allclose(
            adj_bat[mask], adj_seq[mask], rtol=1e-4,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_matches_cpu_fstats(self, monkeypatch):
        """CUDA batch F-stats agree with CPU sequential to rtol=1e-4."""
        data = _make_var1_data(T=1000)
        lag = 1

        adj_seq = _granger_sequential(data, lag=lag, p_threshold=1.0)

        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        adj_bat_cuda = batch_granger_test(
            data, max_lag=lag, p_threshold=1.0, chunk_size=6
        )

        mask = adj_seq > 0
        if mask.any():
            np.testing.assert_allclose(
                adj_bat_cuda[mask], adj_seq[mask], rtol=1e-4,
                err_msg="CUDA vs CPU F-stat disagreement",
            )


# ---------------------------------------------------------------------------
# Chunking consistency
# ---------------------------------------------------------------------------

class TestChunkingConsistency:
    """Different chunk sizes must produce identical results."""

    @pytest.mark.parametrize("chunk_size", [1, 3, 6, 1000])
    def test_chunk_sizes_agree(self, monkeypatch, chunk_size):
        """Different chunk sizes produce bitwise-identical or negligibly different results.

        torch.linalg.lstsq (QR path) can have floating-point differences ~1e-13
        across chunk boundaries due to different order of floating-point operations.
        Threshold p-values are computed from F-stats, so edge structure is stable.
        """
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        data = _make_var1_data(T=500)
        adj_ref = batch_granger_test(data, max_lag=1, p_threshold=0.05, chunk_size=12)
        adj_test = batch_granger_test(
            data, max_lag=1, p_threshold=0.05, chunk_size=chunk_size
        )
        # Edge structure must be identical
        assert np.array_equal(adj_ref != 0, adj_test != 0), "Edge structures differ"
        # F-values on detected edges agree to machine precision
        mask = adj_ref != 0
        if mask.any():
            np.testing.assert_allclose(adj_test[mask], adj_ref[mask], rtol=1e-10)


# ---------------------------------------------------------------------------
# CUDA correctness
# ---------------------------------------------------------------------------

class TestBatchGrangerCuda:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_detects_known_edges_cuda(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        data = _make_var1_data(T=2000)
        adj = batch_granger_test(data, max_lag=1, p_threshold=0.01, chunk_size=6)
        assert adj[1, 0] > 0, "True edge 0→1 not detected on CUDA"
        assert adj[2, 3] > 0, "True edge 3→2 not detected on CUDA"
        assert adj[0, 1] == 0.0
        assert adj[3, 2] == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_cpu_edge_structure_identical(self, monkeypatch):
        """CUDA and CPU must detect exactly the same edges."""
        data = _make_var1_data(T=2000)

        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        get_device.cache_clear()
        adj_cpu = batch_granger_test(data, max_lag=1, p_threshold=0.01, chunk_size=6)

        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        get_device.cache_clear()
        adj_gpu = batch_granger_test(data, max_lag=1, p_threshold=0.01, chunk_size=6)

        assert np.array_equal(
            adj_cpu != 0, adj_gpu != 0
        ), "CPU and CUDA edge structures differ"
