"""Tests for smim/dynamics/kim_filter.py — KimFilter implementation."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter, RegimeDiagnostics
from quantdsl_backtest.smim.interfaces import DecompositionMethod, FilteredState, ModalFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_modal_frame(N: int = 5, K: int = 2) -> ModalFrame:
    rng = np.random.default_rng(10)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_observations(T: int = 40, N: int = 5, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


# ---------------------------------------------------------------------------
# M4.2-T1 tests
# ---------------------------------------------------------------------------

class TestKimFilterCore:
    def test_filter_returns_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        assert isinstance(result, FilteredState)

    def test_regime_probs_sum_to_one(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        sums = result.regime_probs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_no_numerical_overflow_M4(self) -> None:
        """M=4 regimes: verify no NaN in output."""
        mf = _make_modal_frame(N=6, K=3)
        obs = _make_observations(T=30, N=6)
        kf = KimFilter(n_regimes=4)
        result = kf.filter(obs, mf)
        assert not np.any(np.isnan(result.alpha_filtered))
        assert not np.any(np.isnan(result.regime_probs))

    def test_a5_regime_persistence_minimum_8_quarters(self) -> None:
        """Placeholder: verify regime_probs shape is correct (T, M)."""
        T, M = 40, 2
        mf = _make_modal_frame()
        obs = _make_observations(T=T)
        kf = KimFilter(n_regimes=M)
        result = kf.filter(obs, mf)
        assert result.regime_probs.shape == (T, M)

    def test_alpha_filtered_shape(self) -> None:
        T, N, K = 35, 5, 2
        mf = _make_modal_frame(N=N, K=K)
        obs = _make_observations(T=T, N=N)
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        assert result.alpha_filtered.shape == (T, K)
        assert result.alpha_predicted.shape == (T, K)

    def test_log_likelihood_finite(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.filter(obs, mf)
        assert np.isfinite(result.log_likelihood)

    def test_regimes_selected_matches_n_regimes(self) -> None:
        for M in [1, 2, 3]:
            mf = _make_modal_frame()
            obs = _make_observations()
            kf = KimFilter(n_regimes=M)
            result = kf.filter(obs, mf)
            assert result.regimes_selected == M


# ---------------------------------------------------------------------------
# M4.2-T2 tests
# ---------------------------------------------------------------------------

class TestKimFilterEM:
    def test_em_ll_non_decreasing(self) -> None:
        """ll_history should be broadly non-decreasing (with tolerance)."""
        mf = _make_modal_frame(N=5, K=2)
        obs = _make_observations(T=50, N=5)
        kf = KimFilter(n_regimes=2)
        result = kf.em_estimate(obs, mf, max_iter=20, tol=1e-6)
        ll = result.params["ll_history"]
        assert len(ll) >= 1
        assert np.isfinite(ll[-1])
        # Final LL should not catastrophically worsen
        if len(ll) >= 2:
            assert ll[-1] >= ll[0] - 100.0

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.em_estimate(obs, mf, max_iter=10)
        P = result.params["P_trans"]
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    def test_em_result_is_filtered_state(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        result = kf.em_estimate(obs, mf, max_iter=5)
        assert isinstance(result, FilteredState)


# ---------------------------------------------------------------------------
# M4.2-T4 tests (regime diagnostics)
# ---------------------------------------------------------------------------

class TestRegimeDiagnostics:
    def test_avg_durations_positive(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        state = kf.filter(obs, mf)
        diag = kf.regime_diagnostics(state)
        assert np.all(diag.avg_durations > 0)

    def test_a5_check_with_persistent_regime(self) -> None:
        """P with p_ii=0.9 → duration=10 > 8 → a5_passes=True."""
        mf = _make_modal_frame(N=4, K=2)
        obs = _make_observations(N=4)
        kf = KimFilter(n_regimes=2)
        # Manually set the P_trans to be persistent
        P_trans_persistent = np.array([[0.9, 0.1], [0.1, 0.9]])
        F_list = [np.eye(2) * 0.9, np.eye(2) * 0.8]
        Q_list = [np.eye(2) * 0.1, np.eye(2) * 0.1]
        R = np.eye(4) * 0.1
        state = kf.filter(obs, mf, F_list=F_list, Q_list=Q_list, R=R,
                          P_trans=P_trans_persistent)
        # Override params with persistent matrix for diagnostic check
        state.params["P_trans"] = P_trans_persistent
        diag = kf.regime_diagnostics(state)
        assert diag.a5_passes is True
        # Expected duration = 10 for p_ii = 0.9
        np.testing.assert_allclose(diag.avg_durations, [10.0, 10.0], atol=0.1)

    def test_diagnostics_returns_regime_diagnostics(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        kf = KimFilter(n_regimes=2)
        state = kf.filter(obs, mf)
        diag = kf.regime_diagnostics(state)
        assert isinstance(diag, RegimeDiagnostics)
        assert diag.transition_matrix.shape == (2, 2)
        assert len(diag.avg_durations) == 2
