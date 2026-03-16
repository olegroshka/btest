"""Tests for smim/dynamics/evaluation.py — OOS prediction evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.evaluation import (
    SSMEvaluationResult,
    oos_prediction_evaluation,
)
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_modal_frame(N: int = 5, K: int = 2) -> ModalFrame:
    rng = np.random.default_rng(7)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_observations(T: int = 50, N: int = 5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOOSEvaluation:
    def test_result_fields_are_finite(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = oos_prediction_evaluation(obs, mf, holdout_fraction=0.2, max_em_iter=10)
        assert np.isfinite(result.r2_oos)
        assert np.isfinite(result.rmse_oos)
        assert np.isfinite(result.mae_oos)
        assert np.isfinite(result.r2_vs_rw)
        assert np.isfinite(result.r2_vs_mean)

    def test_holdout_size(self) -> None:
        T = 50
        frac = 0.2
        mf = _make_modal_frame()
        obs = _make_observations(T=T)
        result = oos_prediction_evaluation(obs, mf, holdout_fraction=frac, max_em_iter=5)
        expected = T - int(T * (1 - frac))
        assert result.n_holdout == expected

    def test_runs_without_error(self) -> None:
        mf = _make_modal_frame(N=4, K=2)
        obs = _make_observations(T=60, N=4)
        result = oos_prediction_evaluation(obs, mf, holdout_fraction=0.25, max_em_iter=5)
        assert isinstance(result, SSMEvaluationResult)

    def test_rmse_non_negative(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = oos_prediction_evaluation(obs, mf, max_em_iter=5)
        assert result.rmse_oos >= 0.0

    def test_mae_non_negative(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = oos_prediction_evaluation(obs, mf, max_em_iter=5)
        assert result.mae_oos >= 0.0
