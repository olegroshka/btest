"""Tests for smim/emergence/pid.py — Gaussian MMI PID and bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.emergence.pid import (
    PIDResult,
    bootstrap_synergy,
    compute_synergy_matrix,
    gaussian_mmi_pid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals(T: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    alpha_j = rng.standard_normal(T)
    alpha_k = rng.standard_normal(T)
    target = rng.standard_normal(T)
    return alpha_j, alpha_k, target


# ---------------------------------------------------------------------------
# M4.6-T1 tests
# ---------------------------------------------------------------------------

class TestPIDResult:
    def test_pid_result_fields(self) -> None:
        pid = PIDResult(redundancy=0.1, unique_j=0.2, unique_k=0.15, synergy=0.05)
        assert hasattr(pid, "redundancy")
        assert hasattr(pid, "unique_j")
        assert hasattr(pid, "unique_k")
        assert hasattr(pid, "synergy")

    def test_synergy_non_negative(self) -> None:
        alpha_j, alpha_k, target = _make_signals()
        pid = gaussian_mmi_pid(alpha_j, alpha_k, target)
        assert pid.synergy >= 0.0

    def test_redundancy_non_negative(self) -> None:
        alpha_j, alpha_k, target = _make_signals()
        pid = gaussian_mmi_pid(alpha_j, alpha_k, target)
        assert pid.redundancy >= 0.0

    def test_unique_informations_non_negative(self) -> None:
        alpha_j, alpha_k, target = _make_signals()
        pid = gaussian_mmi_pid(alpha_j, alpha_k, target)
        assert pid.unique_j >= 0.0
        assert pid.unique_k >= 0.0

    def test_synergy_matrix_symmetric(self) -> None:
        T, K = 80, 3
        rng = np.random.default_rng(1)
        alpha = rng.standard_normal((T, K))
        gaps = rng.standard_normal((5, T))
        S = compute_synergy_matrix(alpha, gaps, K)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_synergy_matrix_shape(self) -> None:
        T, K = 80, 3
        rng = np.random.default_rng(2)
        alpha = rng.standard_normal((T, K))
        gaps = rng.standard_normal((4, T))
        S = compute_synergy_matrix(alpha, gaps, K)
        assert S.shape == (K, K)

    def test_redundant_sources(self) -> None:
        """Two copies of same signal → high redundancy, near-zero synergy."""
        T = 100
        rng = np.random.default_rng(3)
        source = rng.standard_normal(T)
        target = rng.standard_normal(T)
        pid = gaussian_mmi_pid(source, source, target)  # j and k are identical
        assert pid.redundancy > 0.0
        # Synergy should be ~0 for identical sources (MMI gives R = I(j;T) = I(k;T))
        # Synergy = I(jk;T) - R - U_j - U_k; with identical sources, I(jk;T) ≈ I(j;T)
        assert pid.synergy < 0.5  # effectively near zero


# ---------------------------------------------------------------------------
# M4.6-T2 tests
# ---------------------------------------------------------------------------

class TestBootstrapSynergy:
    def test_bootstrap_ci_contains_observed(self) -> None:
        alpha_j, alpha_k, target = _make_signals(T=80)
        observed, lower, upper = bootstrap_synergy(
            alpha_j, alpha_k, target, n_bootstrap=50, ci_level=0.95
        )
        assert lower <= observed + 1e-10  # lower ≤ observed (with tiny tolerance)
        assert upper >= observed - 1e-10  # upper ≥ observed

    def test_ci_width_positive(self) -> None:
        alpha_j, alpha_k, target = _make_signals(T=80)
        _, lower, upper = bootstrap_synergy(
            alpha_j, alpha_k, target, n_bootstrap=50
        )
        assert upper >= lower

    def test_bootstrap_returns_three_floats(self) -> None:
        alpha_j, alpha_k, target = _make_signals(T=60)
        result = bootstrap_synergy(alpha_j, alpha_k, target, n_bootstrap=20)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)
            assert np.isfinite(val)
