"""Tests for rolling OOS reconstruction evaluation (M3.4-T1)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.spectral.oos_evaluation import (
    OOSReconstructionResult,
    rolling_oos_reconstruction,
)
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer


def _make_signals(T: int = 50, N: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


class TestRollingOOSReconstruction:
    def test_returns_oos_reconstruction_result(self) -> None:
        signals = _make_signals(T=30, N=4)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=10, step=2, k=3)
        assert isinstance(result, OOSReconstructionResult)

    def test_correct_number_of_windows(self) -> None:
        T, N, window_size, step = 30, 4, 10, 2
        signals = _make_signals(T=T, N=N)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=window_size, step=step, k=3)
        # Windows: t=0,2,4,...,T-window_size-1
        expected = len(range(0, T - window_size, step))
        assert result.n_windows == expected

    def test_mean_r2_is_finite(self) -> None:
        signals = _make_signals(T=40, N=5)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=15, step=3, k=4)
        assert np.isfinite(result.mean_r2)

    def test_rank_correlations_length(self) -> None:
        """rank_correlations should have n_windows - 1 entries."""
        signals = _make_signals(T=40, N=5)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=10, step=3, k=4)
        assert len(result.rank_correlations) == result.n_windows - 1

    def test_per_window_r2_length_matches_n_windows(self) -> None:
        signals = _make_signals(T=30, N=3)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=8, step=2, k=2)
        assert len(result.per_window_r2) == result.n_windows

    def test_r2_can_be_negative(self) -> None:
        """R2 < 0 is valid (worse than mean predictor)."""
        signals = _make_signals(T=30, N=4)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=8, step=2, k=2)
        # Just verify no assertion error for negative R2
        for r2 in result.per_window_r2:
            assert np.isfinite(r2)

    def test_with_mode_selector(self) -> None:
        from quantdsl_backtest.smim.spectral.mode_selection import CombinedModeSelector
        signals = _make_signals(T=50, N=5)
        decomposer = SchurDecomposer()
        selector = CombinedModeSelector(mdl=False, min_compressibility=0.0, rg_relevance=False)
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=selector, window_size=15, step=5, k=4)
        assert result.n_windows > 0

    def test_empty_when_window_too_large(self) -> None:
        signals = _make_signals(T=10, N=3)
        decomposer = SchurDecomposer()
        result = rolling_oos_reconstruction(signals, decomposer, mode_selector=None, window_size=15, step=1, k=2)
        assert result.n_windows == 0
        assert np.isnan(result.mean_r2)
