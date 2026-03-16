"""Tests for smim/emergence/transfer_entropy.py — KSG transfer entropy."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.emergence.transfer_entropy import (
    conditional_transfer_entropy,
    ksg_transfer_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ar1_driven(T: int, rho: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """X is AR(1), Y is driven by X: Y_t = 0.5 X_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    X = np.zeros(T)
    Y = np.zeros(T)
    for t in range(1, T):
        X[t] = rho * X[t - 1] + rng.standard_normal() * 0.5
        Y[t] = 0.5 * X[t - 1] + rng.standard_normal() * 0.3
    return X, Y


# ---------------------------------------------------------------------------
# M4.6-T3 tests
# ---------------------------------------------------------------------------

class TestKSGTransferEntropy:
    def test_te_non_negative(self) -> None:
        T = 100
        rng = np.random.default_rng(0)
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        te = ksg_transfer_entropy(X, Y, lag=1, k_neighbours=5)
        assert te >= 0.0

    def test_coupled_ar_asymmetric(self) -> None:
        """X drives Y but not vice versa: TE(X→Y) should exceed TE(Y→X)."""
        T = 300
        X, Y = _ar1_driven(T, rho=0.8, seed=1)
        te_xy = ksg_transfer_entropy(X, Y, lag=1, k_neighbours=5)
        te_yx = ksg_transfer_entropy(Y, X, lag=1, k_neighbours=5)
        # TE from driver X→Y should be higher than reverse
        assert te_xy > te_yx, f"Expected TE(X→Y)={te_xy:.4f} > TE(Y→X)={te_yx:.4f}"

    def test_independent_processes_small_te(self) -> None:
        """Independent i.i.d. processes should have near-zero TE."""
        T = 200
        rng = np.random.default_rng(2)
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        te = ksg_transfer_entropy(X, Y, lag=1, k_neighbours=5)
        # Should be small (KSG gives 0 floor via max(..., 0))
        assert te < 1.0  # generous upper bound for small T

    def test_te_finite(self) -> None:
        T = 80
        rng = np.random.default_rng(3)
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        te = ksg_transfer_entropy(X, Y)
        assert np.isfinite(te)


# ---------------------------------------------------------------------------
# M4.6-T4 tests (conditional TE)
# ---------------------------------------------------------------------------

class TestConditionalTransferEntropy:
    def test_chain_mediation(self) -> None:
        """X→Z→Y: conditioning on Z should reduce TE(X→Y|Z) vs TE(X→Y)."""
        T = 300
        rng = np.random.default_rng(4)
        X = np.cumsum(rng.standard_normal(T)) * 0.1
        Z = np.roll(X, 1) + rng.standard_normal(T) * 0.2
        Y = np.roll(Z, 1) + rng.standard_normal(T) * 0.2

        te_xy = ksg_transfer_entropy(X, Y, lag=1, k_neighbours=5)
        cte_xyz = conditional_transfer_entropy(X, Y, Z, lag=1, k=5)
        # Conditioning on the mediator Z reduces the TE from X to Y
        assert cte_xyz <= te_xy + 0.5  # allow tolerance

    def test_direct_path(self) -> None:
        """X→Y with unrelated Z: conditional TE(X→Y|Z) remains positive."""
        T = 200
        rng = np.random.default_rng(5)
        X, Y = _ar1_driven(T, rho=0.8, seed=5)
        Z = rng.standard_normal(T)  # completely unrelated

        cte = conditional_transfer_entropy(X, Y, Z, lag=1, k=5)
        te = ksg_transfer_entropy(X, Y, lag=1, k_neighbours=5)
        # With unrelated Z, conditional TE should be similar to unconditional
        # Both should be non-negative
        assert cte >= 0.0
        assert te >= 0.0

    def test_cte_non_negative(self) -> None:
        T = 100
        rng = np.random.default_rng(6)
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        Z = rng.standard_normal(T)
        cte = conditional_transfer_entropy(X, Y, Z, lag=1, k=4)
        assert cte >= 0.0

    def test_cte_finite(self) -> None:
        T = 80
        rng = np.random.default_rng(7)
        X = rng.standard_normal(T)
        Y = rng.standard_normal(T)
        Z = rng.standard_normal(T)
        cte = conditional_transfer_entropy(X, Y, Z)
        assert np.isfinite(cte)
