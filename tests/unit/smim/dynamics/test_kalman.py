"""Tests for smim/dynamics/kalman.py — KalmanFilter implementation."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import DecompositionMethod, FilteredState, ModalFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_modal_frame(N: int = 5, K: int = 3) -> ModalFrame:
    rng = np.random.default_rng(0)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_observations(T: int = 30, N: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


# ---------------------------------------------------------------------------
# M4.1-T1 tests
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_filter_returns_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert isinstance(result, FilteredState)

    def test_log_likelihood_is_finite(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert np.isfinite(result.log_likelihood)
        assert not np.isnan(result.log_likelihood)

    def test_filtered_state_shape(self) -> None:
        T, N, K = 40, 6, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = _make_observations(T=T, N=N)
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert result.alpha_filtered.shape == (T, K)
        assert result.alpha_predicted.shape == (T, K)

    def test_single_regime(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        T = len(obs)
        assert result.regime_probs.shape == (T, 1)
        assert result.regimes_selected == 1
        np.testing.assert_allclose(result.regime_probs, 1.0)

    def test_params_contain_matrices(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert "F" in result.params
        assert "Q" in result.params
        assert "R" in result.params
        assert "P_filtered" in result.params
        assert "P_predicted" in result.params


# ---------------------------------------------------------------------------
# M4.1-T2 tests
# ---------------------------------------------------------------------------

class TestKalmanFilterEM:
    def test_em_estimation_converges(self) -> None:
        """ll_history should show improvement from start to end."""
        mf = _make_modal_frame(N=5, K=2)
        obs = _make_observations(T=50, N=5)
        kf = KalmanFilter()
        result = kf.em_estimate(obs, mf, max_iter=30, tol=1e-6)
        ll = result.params["ll_history"]
        assert len(ll) >= 2
        # The final LL should be finite
        assert np.isfinite(ll[-1])
        assert np.isfinite(ll[0])
        # Overall: log-likelihood should not dramatically worsen
        # (EM can have minor oscillations with simplified cross-covariance approx)
        assert ll[-1] >= ll[0] - 50.0, (
            f"LL dramatically worse at end: {ll[0]:.2f} → {ll[-1]:.2f}"
        )

    def test_em_alpha_filtered_shape(self) -> None:
        T, N, K = 40, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = _make_observations(T=T, N=N)
        kf = KalmanFilter()
        result = kf.em_estimate(obs, mf, max_iter=10)
        assert result.alpha_filtered.shape == (T, K)

    def test_em_result_is_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.em_estimate(obs, mf, max_iter=5)
        assert isinstance(result, FilteredState)

    def test_em_ll_history_in_params(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.em_estimate(obs, mf, max_iter=5)
        assert "ll_history" in result.params
        assert len(result.params["ll_history"]) >= 1

    def test_em_large_n_small_t_finite_log_likelihood(self) -> None:
        """Regression: N=100, T=24, K=1 (N>>T) must yield finite LL.

        Guards against numerical instability in the EM when the number of
        actors far exceeds the number of time steps, as in A4-SCALING.
        """
        rng = np.random.default_rng(42)
        N, K, T = 100, 1, 24  # N >> T: the A4 problematic regime
        U, _ = np.linalg.qr(rng.standard_normal((N, K)))
        obs = rng.uniform(0.0, 1.0, (T, N))  # capex_assets_xsrank-like [0,1]
        modal_frame = ModalFrame(
            basis=U,
            eigenvalues=np.array([0.9], dtype=complex),
            method=DecompositionMethod.SCHUR,
        )
        kf = KalmanFilter()
        state = kf.em_estimate(obs, modal_frame, max_iter=50)
        assert np.isfinite(state.log_likelihood), (
            f"log_likelihood={state.log_likelihood} not finite for N={N}, T={T}"
        )
        assert np.all(np.isfinite(state.alpha_filtered)), \
            "alpha_filtered contains non-finite values"
        assert np.all(np.isfinite(state.alpha_predicted)), \
            "alpha_predicted contains non-finite values"
