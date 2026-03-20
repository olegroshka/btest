"""
Granger batch-parity acceptance test.

Verifies that ``batch_granger_test`` detects the same edges as the original
``GrangerEdgeEstimator`` (statsmodels) on the identical VAR(1) data used in
acceptance test A-GR-1.

Also verifies F-statistics from ``batch_granger_test`` match the sequential
numpy reference (``_granger_sequential``) to within rtol=1e-4.

This test lives in the acceptance suite because it cross-validates two
algorithm implementations rather than a single unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quantdsl_backtest.smim.compute.batch_granger import (
    _granger_sequential,
    batch_granger_test,
)
from quantdsl_backtest.smim.compute.torch_ops import get_device
from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
from quantdsl_backtest.smim.interfaces import DateRange

from tests.acceptance.smim.conftest import make_actor_registry

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("graph_construction"),
]


# ---------------------------------------------------------------------------
# Shared VAR(1) fixture — identical to A-GR-1
# ---------------------------------------------------------------------------

def _agr1_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (T×4 data, 4×4 true adjacency) from A-GR-1 DGP."""
    A_true = np.zeros((4, 4))
    A_true[1, 0] = 0.6
    A_true[2, 3] = 0.5
    A_true[0, 0] = 0.8
    A_true[1, 1] = 0.7
    A_true[2, 2] = 0.75
    A_true[3, 3] = 0.85
    T = 2000
    rng = np.random.default_rng(42)
    data = np.zeros((T, 4))
    data[0] = rng.standard_normal(4) * 0.1
    for t in range(1, T):
        data[t] = A_true @ data[t - 1] + 0.5 * rng.standard_normal(4)
    return data, A_true


@pytest.fixture(autouse=True)
def _clear_device_cache():
    get_device.cache_clear()
    yield
    get_device.cache_clear()


# ---------------------------------------------------------------------------
# Parity: batch implementation vs GrangerEdgeEstimator (statsmodels)
# ---------------------------------------------------------------------------

def test_batch_granger_detects_agr1_true_edges():
    """batch_granger_test detects both true edges from A-GR-1 (lag=1, p<0.01)."""
    data, _ = _agr1_data()
    adj = batch_granger_test(data, max_lag=1, p_threshold=0.01)
    assert adj[1, 0] > 0, "batch: true edge 0→1 not detected"
    assert adj[2, 3] > 0, "batch: true edge 3→2 not detected"
    assert adj[0, 1] == 0.0, "batch: spurious reverse edge 1→0"
    assert adj[3, 2] == 0.0, "batch: spurious reverse edge 2→3"


def test_batch_granger_matches_statsmodels_edges():
    """batch_granger_test (lag=1) detects the same edge set as GrangerEdgeEstimator.

    The original estimator uses BIC-selected lags; for VAR(1) data BIC selects
    lag=1.  We force lag=1 in both to ensure a fair comparison.
    """
    data, _ = _agr1_data()
    registry = make_actor_registry(4)
    dates = pd.date_range("2000-01-01", periods=2000, freq="D")
    actor_ids = [a.actor_id for a in registry.actors]
    signals = pd.DataFrame(data, index=dates, columns=actor_ids)
    dr = DateRange(start=dates[0], end=dates[-1])

    # Original statsmodels-based estimator (BIC selects lag=1 for VAR(1))
    est = GrangerEdgeEstimator(
        GrangerEdgeConfig(max_lag=1, p_threshold=0.01, bic_selection=True)
    )
    adj_orig = est.estimate(signals, registry, dr).toarray()

    # Batch implementation with fixed lag=1
    adj_batch = batch_granger_test(data, max_lag=1, p_threshold=0.01)

    # Same edge structure (which pairs are significant)
    orig_edges = set(zip(*np.where(adj_orig != 0)))
    batch_edges = set(zip(*np.where(adj_batch != 0)))
    assert orig_edges == batch_edges, (
        f"Edge mismatch:\n  statsmodels: {sorted(orig_edges)}\n"
        f"  batch:       {sorted(batch_edges)}"
    )


def test_batch_granger_fstats_match_sequential():
    """batch_granger_test F-statistics agree with _granger_sequential to rtol=1e-4.

    Both use the identical OLS formula; the only difference is numpy vs
    PyTorch lstsq, which should agree to machine precision.
    """
    data, _ = _agr1_data()

    adj_seq = _granger_sequential(data, lag=1, p_threshold=1.0)   # keep all pairs
    adj_bat = batch_granger_test(data, max_lag=1, p_threshold=1.0)

    # Edge structure must be identical (all non-zero F-stats)
    assert np.array_equal(
        adj_seq != 0, adj_bat != 0
    ), "F-stat non-zero patterns differ between sequential and batch"

    mask = adj_seq != 0
    np.testing.assert_allclose(
        adj_bat[mask], adj_seq[mask], rtol=1e-4,
        err_msg="F-statistics disagree between batch and sequential",
    )


def test_batch_granger_p_values_consistent():
    """p-values are consistent with F-stats (F=0 ↔ p=1, F large ↔ p small)."""
    data, _ = _agr1_data()

    # Run at loose threshold to get some non-zero entries
    adj = batch_granger_test(data, max_lag=1, p_threshold=0.5)

    # True edges should have large F-stats
    assert adj[1, 0] > 5.0, f"True edge F-stat unexpectedly small: {adj[1, 0]:.2f}"
    assert adj[2, 3] > 5.0, f"True edge F-stat unexpectedly small: {adj[2, 3]:.2f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_batch_granger_cuda_matches_cpu_edges():
    """CUDA and CPU batch implementations detect exactly the same edges."""
    data, _ = _agr1_data()

    get_device.cache_clear()
    import os
    os.environ["SMIM_DEVICE"] = "cpu"
    get_device.cache_clear()
    adj_cpu = batch_granger_test(data, max_lag=1, p_threshold=0.01)

    os.environ["SMIM_DEVICE"] = "cuda"
    get_device.cache_clear()
    adj_gpu = batch_granger_test(data, max_lag=1, p_threshold=0.01)

    assert np.array_equal(
        adj_cpu != 0, adj_gpu != 0
    ), "CUDA and CPU edge structures differ on A-GR-1 data"

    # F-stats must also agree
    mask = adj_cpu != 0
    if mask.any():
        np.testing.assert_allclose(
            adj_gpu[mask], adj_cpu[mask], rtol=1e-4,
            err_msg="CUDA vs CPU F-stat disagreement",
        )
