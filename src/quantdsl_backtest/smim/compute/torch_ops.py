"""PyTorch-based compute operations with automatic CPU/CUDA dispatch.

Usage::

    from quantdsl_backtest.smim.compute import get_device, ensure_tensor, to_numpy

    device = get_device()  # auto-detect CUDA, or force via SMIM_DEVICE env var
    A_t = ensure_tensor(A_numpy, device=device)
    result_numpy = to_numpy(some_tensor)

All functions accept numpy arrays and return numpy arrays at the boundary.
PyTorch tensors are used internally for computation only.
"""
import os
import logging
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device(force: str | None = None) -> torch.device:
    """Get the compute device.

    Priority: force arg > SMIM_DEVICE env var > auto-detect.

    Args:
        force: "cpu", "cuda", "cuda:0", "cuda:1", or None (auto-detect).

    Returns:
        torch.device
    """
    if force is not None:
        device_str = force
    else:
        device_str = os.environ.get("SMIM_DEVICE", "auto")

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            logger.info(f"SMIM compute: using CUDA device {name}")
        else:
            device = torch.device("cpu")
            logger.info("SMIM compute: using CPU (no CUDA available)")
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("SMIM compute: using CPU (forced)")
    else:
        device = torch.device(device_str)
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device.index or 0)
            logger.info(f"SMIM compute: using CUDA device {name}")
    return device


def ensure_tensor(
    arr: NDArray | torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert numpy array to torch tensor on the target device.

    Always uses float64 for numerical precision matching scipy.
    """
    if device is None:
        device = get_device()
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def to_numpy(t: torch.Tensor) -> NDArray:
    """Convert torch tensor back to numpy (on CPU)."""
    return t.detach().cpu().numpy()
