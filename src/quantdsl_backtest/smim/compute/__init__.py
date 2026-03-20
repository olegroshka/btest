"""SMIM compute layer — PyTorch-based CPU/CUDA dispatch.

Usage::

    from quantdsl_backtest.smim.compute import get_device, ensure_tensor, to_numpy
    from quantdsl_backtest.smim.compute import linalg
"""
from .torch_ops import get_device, ensure_tensor, to_numpy

__all__ = ["get_device", "ensure_tensor", "to_numpy", "linalg"]
