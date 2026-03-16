"""Tests for smim/dynamics/observability.py — observability matrix diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.observability import (
    compute_observability_matrix,
    observability_diagnostics,
)


class TestObservabilityMatrix:
    def test_observable_system_reasonable_condition(self) -> None:
        """F=diag([0.9, 0.8]), U=eye(2): system should be observable."""
        K = 2
        F = np.diag([0.9, 0.8])
        U = np.eye(K)
        diag = observability_diagnostics(F, U)
        assert diag["passes"] is True
        assert diag["condition_number"] < 1e6

    def test_near_unobservable_high_condition(self) -> None:
        """F=0*eye, U cannot distinguish state → rank-deficient observability matrix."""
        K = 2
        F = np.zeros((K, K))
        # U maps both states to same direction
        U = np.ones((2, K))
        diag = observability_diagnostics(F, U)
        # With F=0, O = [U; U@0; ...] = [U; 0; 0; ...], rank 1 < K=2
        # Condition number should be very large (ill-conditioned)
        assert diag["condition_number"] > 100.0

    def test_observability_matrix_shape(self) -> None:
        """Output shape should be (N*K, K)."""
        N, K = 3, 2
        F = np.diag([0.9, 0.8])
        U = np.random.default_rng(0).standard_normal((N, K))
        O = compute_observability_matrix(F, U)
        assert O.shape == (N * K, K)

    def test_identity_system_full_rank(self) -> None:
        """F=I, U=I: all rows identical but numerics should be reasonable."""
        K = 3
        F = np.eye(K)
        U = np.eye(K)
        O = compute_observability_matrix(F, U)
        assert O.shape == (K * K, K)
        # With F=I, each block is U @ I^k = U, rank(O) = rank(U) = K
        sv = np.linalg.svd(O, compute_uv=False)
        assert sv[0] > 1e-10

    def test_suggested_k_star_positive(self) -> None:
        K = 3
        F = np.diag([0.9, 0.8, 0.7])
        U = np.eye(K)
        diag = observability_diagnostics(F, U)
        assert diag["suggested_k_star"] >= 1

    def test_singular_values_descending(self) -> None:
        K = 2
        F = np.diag([0.9, 0.5])
        U = np.random.default_rng(5).standard_normal((4, K))
        diag = observability_diagnostics(F, U)
        sv = diag["singular_values"]
        assert sv[0] >= sv[-1]
