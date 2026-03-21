"""
Benchmark conftest — shared fixtures for SMIM performance benchmarks.

The ``device`` fixture parametrises all benchmarks over CPU and CUDA.
Tests are skipped automatically on machines without a CUDA GPU.
"""
from __future__ import annotations

import os

import pytest
import torch

from quantdsl_backtest.smim.compute.torch_ops import get_device


@pytest.fixture(params=["cpu", "cuda"], ids=["CPU", "CUDA"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """CPU/CUDA parametrised device fixture.

    Tests using this fixture are automatically skipped on CUDA when no
    GPU is available.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _set_smim_device(device: torch.device) -> None:
    """Push SMIM_DEVICE into the environment for the duration of each benchmark."""
    os.environ["SMIM_DEVICE"] = device.type
    get_device.cache_clear()
    yield  # type: ignore[misc]
    os.environ.pop("SMIM_DEVICE", None)
    get_device.cache_clear()
