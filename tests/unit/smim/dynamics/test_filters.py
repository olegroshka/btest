"""Integration tests for smim/dynamics/ state-space filters (M4.x tasks).

The dedicated tests are in test_kalman.py and test_kim_filter.py.
This file provides smoke tests verifying both filters can run together.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


def _make_modal_frame(N: int = 5, K: int = 2) -> ModalFrame:
    rng = np.random.default_rng(42)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_observations(T: int = 40, N: int = 5) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((T, N))


class TestKalmanFilter:
    def test_filter_returns_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert result.alpha_filtered.shape == (40, 2)

    def test_single_regime(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert result.regimes_selected == 1
        assert result.regime_probs.shape == (40, 1)

    def test_alpha_filtered_shape(self) -> None:
        T, N, K = 30, 5, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = _make_observations(T=T, N=N)
        kf = KalmanFilter()
        result = kf.filter(obs, mf)
        assert result.alpha_filtered.shape == (T, K)

    def test_em_estimation_converges(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KalmanFilter()
        result = kf.em_estimate(obs, mf, max_iter=10)
        assert "ll_history" in result.params
        assert len(result.params["ll_history"]) >= 1


class TestKimFilter:
    def test_filter_returns_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        assert result.alpha_filtered.shape == (40, 2)
        assert result.regimes_selected == 2

    def test_regime_probs_sum_to_one(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        np.testing.assert_allclose(result.regime_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_a5_regime_persistence_minimum_8_quarters(self) -> None:
        """A5: average regime duration > 8 quarters with p_ii=0.9."""
        mf = _make_modal_frame(N=4, K=2)
        obs = _make_observations(N=4)
        kf = KimFilter(n_regimes=2)
        P_trans = np.array([[0.9, 0.1], [0.1, 0.9]])
        state = kf.filter(obs, mf, P_trans=P_trans)
        state.params["P_trans"] = P_trans
        diag = kf.regime_diagnostics(state)
        assert diag.a5_passes
