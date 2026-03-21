"""
Benchmark: batch PID bootstrap and GPU KNN for transfer entropy.

PID is parametrised over K (number of modes); KNN over T (time-series length).
"""
from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.compute.batch_pid import batch_pid_synergy
from quantdsl_backtest.smim.compute.gpu_knn import knn_query

_N_BOOTSTRAP = 200   # matches SMIM default
_KSG_K = 4          # number of neighbours for KSG


@pytest.mark.parametrize("K", [5, 10, 15, 20])
def test_bench_pid_bootstrap(
    benchmark: pytest.fixture, device: object, K: int
) -> None:
    """Benchmark batch PID bootstrap for K modal states, T=2000 samples."""
    rng = np.random.default_rng(42)
    modal = rng.standard_normal((2000, K)).astype(np.float64)
    target = rng.standard_normal(2000).astype(np.float64)

    benchmark(batch_pid_synergy, modal, target, n_bootstrap=_N_BOOTSTRAP)


@pytest.mark.parametrize("T", [500, 1000, 2000, 5000])
def test_bench_knn(benchmark: pytest.fixture, device: object, T: int) -> None:
    """Benchmark GPU KNN on a (T, 4) point cloud — representative of KSG TE inputs."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((T, 4)).astype(np.float64)

    benchmark(knn_query, data, k=_KSG_K + 1)
