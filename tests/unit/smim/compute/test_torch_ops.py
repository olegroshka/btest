"""Unit tests for smim.compute.torch_ops — device management and tensor utilities."""
import os

import numpy as np
import pytest
import torch

from quantdsl_backtest.smim.compute.torch_ops import ensure_tensor, get_device, to_numpy

ATOL = 1e-10


@pytest.fixture(autouse=True)
def clear_device_cache():
    """Reset the lru_cache between tests so SMIM_DEVICE env changes take effect."""
    get_device.cache_clear()
    yield
    get_device.cache_clear()


class TestGetDevice:
    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_cpu_forced_via_env(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cpu")
        d = get_device()
        assert d.type == "cpu"

    def test_auto_without_cuda_returns_cpu(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "auto")
        d = get_device()
        # On this machine CUDA is not available, so auto → cpu
        if not torch.cuda.is_available():
            assert d.type == "cpu"

    def test_cache_returns_same_object(self):
        d1 = get_device()
        d2 = get_device()
        assert d1 is d2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_forced_via_env(self, monkeypatch):
        monkeypatch.setenv("SMIM_DEVICE", "cuda")
        d = get_device()
        assert d.type == "cuda"


class TestEnsureTensor:
    def test_numpy_to_tensor(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = ensure_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float64
        np.testing.assert_allclose(t.cpu().numpy(), arr, atol=ATOL)

    def test_tensor_passthrough(self):
        t_in = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        t_out = ensure_tensor(t_in)
        assert t_out.dtype == torch.float64
        assert t_out.shape == t_in.shape

    def test_explicit_device_cpu(self):
        arr = np.ones((3, 3))
        t = ensure_tensor(arr, device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_dtype_preserved_as_float64(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        t = ensure_tensor(arr)
        assert t.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_device(self):
        arr = np.ones((4, 4))
        t = ensure_tensor(arr, device=torch.device("cuda"))
        assert t.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_result_matches_cpu(self):
        arr = np.random.default_rng(0).standard_normal((5, 5))
        t_cpu = ensure_tensor(arr, device=torch.device("cpu"))
        t_gpu = ensure_tensor(arr, device=torch.device("cuda"))
        np.testing.assert_allclose(t_gpu.cpu().numpy(), t_cpu.numpy(), atol=ATOL)


class TestToNumpy:
    def test_basic_conversion(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [[1.0, 2.0], [3.0, 4.0]], atol=ATOL)

    def test_shape_preserved(self):
        t = torch.ones((3, 4, 5), dtype=torch.float64)
        arr = to_numpy(t)
        assert arr.shape == (3, 4, 5)

    def test_dtype_is_float64(self):
        t = torch.ones(4, dtype=torch.float64)
        arr = to_numpy(t)
        assert arr.dtype == np.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cuda_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0], atol=ATOL)
