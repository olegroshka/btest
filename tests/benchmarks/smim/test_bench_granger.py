"""
Benchmark: batch Granger causality via PyTorch.

Parametrised over device (CPU / CUDA) and problem size N (number of actors).
T is fixed at 80 (one experiment year of quarterly data) to match the
profiling baseline in profiling_results.json.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.compute.batch_granger import batch_granger_test


@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_granger(benchmark: pytest.fixture, device: object, N: int) -> None:
    """Benchmark batch Granger edge estimation (N actors, T=80, lag=4).

    On CUDA the entire N² pair batch is solved in a single kernel launch.
    On CPU it uses torch on CPU — still faster than sequential statsmodels.
    """
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((N, 80)).astype(np.float64)

    # Warm up (not timed): ensures JIT / CUDA initialisation is excluded
    _ = batch_granger_test(signals[:10, :], max_lag=1, p_threshold=0.05)

    benchmark(batch_granger_test, signals, max_lag=4, p_threshold=0.05)
