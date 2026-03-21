"""
Benchmark: PyTorch linear-algebra primitives (SVD, polar, hermitian dilation).

Parametrised over device (CPU / CUDA) and matrix size N.
These are the spectral operations that dominate at large N.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.compute.linalg import (
    hermitian_dilation_decompose,
    polar_decompose,
    svd,
)

_K_TRUNCATE = 20  # representative number of retained modes


@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_svd(benchmark: pytest.fixture, device: object, N: int) -> None:
    """Benchmark truncated SVD on an N×N matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N)).astype(np.float64)

    benchmark(svd, A, k=_K_TRUNCATE)


@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_polar(benchmark: pytest.fixture, device: object, N: int) -> None:
    """Benchmark polar decomposition of an N×K_TRUNCATE matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, _K_TRUNCATE)).astype(np.float64)

    benchmark(polar_decompose, A)


@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_hermitian_dilation(
    benchmark: pytest.fixture, device: object, N: int
) -> None:
    """Benchmark Hermitian dilation decomposition of an N×N matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N)).astype(np.float64)

    benchmark(hermitian_dilation_decompose, A, k=_K_TRUNCATE)
