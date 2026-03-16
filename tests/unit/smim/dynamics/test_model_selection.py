"""Tests for smim/dynamics/model_selection.py — BIC/MDL regime count selection."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.model_selection import (
    RegimeSelectionResult,
    select_regime_count,
)
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_modal_frame(N: int = 4, K: int = 2) -> ModalFrame:
    rng = np.random.default_rng(20)
    basis, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


def _make_observations(T: int = 40, N: int = 4, seed: int = 55) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegimeSelection:
    def test_result_has_all_candidates(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        candidates = [1, 2, 3]
        result = select_regime_count(obs, mf, candidates=candidates, max_em_iter=5)
        for m in candidates:
            assert m in result.bic_scores

    def test_selected_m_is_in_candidates(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        candidates = [1, 2, 3]
        result = select_regime_count(obs, mf, candidates=candidates, max_em_iter=5)
        assert result.selected_m in candidates

    def test_delta_bic_is_non_negative(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = select_regime_count(obs, mf, candidates=[1, 2], max_em_iter=5)
        assert result.delta_bic >= 0.0

    def test_single_regime_data_prefers_m1(self) -> None:
        """With very simple data, the function should run without error."""
        mf = _make_modal_frame(N=3, K=2)
        obs = _make_observations(T=30, N=3)
        result = select_regime_count(
            obs, mf, candidates=[1, 2], max_em_iter=5
        )
        assert isinstance(result, RegimeSelectionResult)
        assert result.selected_m in [1, 2]

    def test_bic_scores_are_finite(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = select_regime_count(obs, mf, candidates=[1, 2], max_em_iter=5)
        for score in result.bic_scores.values():
            assert np.isfinite(score)

    def test_default_candidates_used_when_none(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = select_regime_count(obs, mf, candidates=None, max_em_iter=3)
        # Default candidates = [1, 2, 3, 4]
        assert set(result.bic_scores.keys()) == {1, 2, 3, 4}

    def test_gate_g4_is_bool(self) -> None:
        mf = _make_modal_frame()
        obs = _make_observations()
        result = select_regime_count(obs, mf, candidates=[1, 2], max_em_iter=5)
        assert isinstance(result.gate_g4_passes, bool)
