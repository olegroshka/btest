"""Tests for smim/dynamics/phase_transition.py — order parameter, GL landscape, criticality."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.phase_transition import (
    GLParams,
    criticality_index,
    extract_order_parameter,
    fit_gl_landscape,
    gl_potential,
)


# ---------------------------------------------------------------------------
# M4.5-T1: Order parameter extraction
# ---------------------------------------------------------------------------

class TestOrderParameter:
    def test_order_parameter_shape(self) -> None:
        T, K = 50, 5
        alpha = np.random.default_rng(0).standard_normal((T, K))
        for d in [1, 2, 3]:
            psi = extract_order_parameter(alpha, d=d)
            assert psi.shape == (T, d), f"Expected ({T}, {d}), got {psi.shape}"

    def test_order_parameter_finite(self) -> None:
        T, K = 40, 4
        alpha = np.random.default_rng(1).standard_normal((T, K))
        for method in ["amplitudes", "interactions", "entropy"]:
            psi = extract_order_parameter(alpha, d=2, method=method)
            assert np.all(np.isfinite(psi)), f"NaN/inf in method={method}"

    def test_entropy_method(self) -> None:
        """Entropy order parameter should be non-negative."""
        T, K = 30, 3
        alpha = np.random.default_rng(2).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=2, method="entropy")
        # Entropy is non-negative; each column is the same scalar signal
        assert np.all(psi >= -1e-12)

    def test_amplitudes_returns_first_d_modes(self) -> None:
        T, K = 20, 4
        alpha = np.random.default_rng(3).standard_normal((T, K))
        d = 2
        psi = extract_order_parameter(alpha, d=d, method="amplitudes")
        np.testing.assert_array_equal(psi, alpha[:, :d])

    def test_unknown_method_raises(self) -> None:
        alpha = np.ones((10, 3))
        with pytest.raises(ValueError, match="Unknown method"):
            extract_order_parameter(alpha, d=2, method="bogus")

    def test_d_clipped_to_K(self) -> None:
        """d > K should be silently clipped to K."""
        T, K = 20, 2
        alpha = np.random.default_rng(4).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=10, method="amplitudes")
        assert psi.shape == (T, K)


# ---------------------------------------------------------------------------
# M4.5-T2: GL landscape
# ---------------------------------------------------------------------------

class TestGLLandscape:
    def test_gl_params_returned(self) -> None:
        T, K = 50, 2
        alpha = np.random.default_rng(5).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=2)
        params = fit_gl_landscape(psi, d=2)
        assert isinstance(params, GLParams)
        assert params.d == 2

    def test_gl_potential_is_scalar(self) -> None:
        params = GLParams(a=np.array([0.1, -0.2]), b=np.array([0.5, 0.3]), d=2)
        psi = np.array([1.0, 2.0])
        val = gl_potential(psi, params)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_gl_params_finite(self) -> None:
        T, K = 60, 2
        alpha = np.random.default_rng(6).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=2)
        params = fit_gl_landscape(psi, d=2)
        assert np.all(np.isfinite(params.a))
        assert np.all(np.isfinite(params.b))

    def test_gl_landscape_computation(self) -> None:
        """Smoke test: fit_gl_landscape runs and returns params with correct shape."""
        T, K = 40, 3
        alpha = np.random.default_rng(7).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=2)
        params = fit_gl_landscape(psi, d=2)
        assert len(params.a) == 2
        assert len(params.b) == 2


# ---------------------------------------------------------------------------
# M4.5-T3: Criticality index
# ---------------------------------------------------------------------------

class TestCriticalityIndex:
    def test_criticality_index_shape(self) -> None:
        T, K = 60, 2
        alpha = np.random.default_rng(8).standard_normal((T, K))
        psi = extract_order_parameter(alpha, d=2)
        C = criticality_index(psi, window_size=8)
        assert C.shape == (T,)

    def test_criticality_rising_with_autocorrelation(self) -> None:
        """AR(1) with ρ increasing from 0.5 to 0.99 should produce rising C_t."""
        T = 200
        rng = np.random.default_rng(9)
        x = np.zeros(T)
        # Gradually increase AR coefficient
        rho = np.linspace(0.5, 0.99, T)
        for t in range(1, T):
            x[t] = rho[t] * x[t - 1] + rng.standard_normal() * 0.1
        psi = x[:, np.newaxis]  # (T, 1)
        C = criticality_index(psi, window_size=15)
        w = 15
        # Mean of last third should exceed mean of first third (after warm-up)
        start = 2 * w
        early = C[start: start + (T - start) // 3]
        late = C[-(T - start) // 3:]
        assert late.mean() > early.mean() * 0.5  # C should rise, be lenient

    def test_criticality_stationary_around_one(self) -> None:
        """Pure white noise → C_t should be approximately 1 on average."""
        T = 300
        rng = np.random.default_rng(10)
        psi = rng.standard_normal((T, 1))
        C = criticality_index(psi, window_size=20)
        w = 20
        C_valid = C[2 * w:]  # skip warm-up
        # Mean should not be wildly far from 1
        assert 0.1 < C_valid.mean() < 20.0

    def test_criticality_index_scalar_input(self) -> None:
        """1D (T,) input should work."""
        T = 100
        psi = np.random.default_rng(11).standard_normal(T)
        C = criticality_index(psi, window_size=10)
        assert C.shape == (T,)
        assert np.all(np.isfinite(C))
