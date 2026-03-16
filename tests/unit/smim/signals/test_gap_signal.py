"""Tests for SMIM bridge signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.interfaces import BenchmarkClass, GapResult
from quantdsl_backtest.smim.signals.gap_signal import (
    CriticalitySignal,
    GapSignal,
    RegimeSignal,
)


def _make_gap_result(gaps: np.ndarray) -> GapResult:
    N, T = gaps.shape
    benchmarks = np.zeros_like(gaps)
    return GapResult(
        gaps=gaps,
        benchmark_class=BenchmarkClass.PREDICTIVE,
        benchmarks=benchmarks,
    )


class TestGapSignal:
    def test_positive_gap_gives_negative_signal(self) -> None:
        """Large positive gap → over-invested → short signal (-1)."""
        rng = np.random.default_rng(0)
        # Actor 0 has consistently large positive gaps
        gaps = np.zeros((3, 20))
        gaps[0, :] = 10.0  # way above threshold
        gr = _make_gap_result(gaps)
        sig = GapSignal(gr, threshold=0.5, scale=False)
        df = sig.to_signal_matrix()
        # actor 0 should be -1 (or negative)
        assert (df.iloc[0] < 0).all()

    def test_negative_gap_gives_positive_signal(self) -> None:
        """Large negative gap → under-invested → long signal (+1)."""
        gaps = np.zeros((3, 20))
        gaps[1, :] = -10.0  # way below threshold
        gr = _make_gap_result(gaps)
        sig = GapSignal(gr, threshold=0.5, scale=False)
        df = sig.to_signal_matrix()
        assert (df.iloc[1] > 0).all()

    def test_small_gap_gives_zero_signal(self) -> None:
        """Gap within threshold → 0."""
        gaps = np.zeros((3, 20))
        gaps[2, :] = 0.01  # tiny gap
        gaps[2, 0] = 10.0  # set some variance so std > 0
        gaps[2, 0] = 0.0
        # Make a gap with std=1, threshold=2 → gap=0.01 < 2*1 → zero
        gaps = np.zeros((1, 50))
        gaps[0, :] = 0.0
        gaps[0, 0] = 1.0  # std ≈ 0.14
        # threshold=10 * std → nearly everything is zero
        gr = _make_gap_result(gaps)
        sig = GapSignal(gr, threshold=10.0, scale=False)
        df = sig.to_signal_matrix()
        # Most signals should be 0
        assert (df.iloc[0, 1:] == 0).all()

    def test_signal_matrix_shape(self) -> None:
        """Returns (N_actors, T_dates) DataFrame."""
        N, T = 5, 30
        gaps = np.random.randn(N, T)
        gr = _make_gap_result(gaps)
        sig = GapSignal(gr, threshold=0.5, scale=True)
        df = sig.to_signal_matrix()
        assert df.shape == (N, T)

    def test_scale_true_clips_to_minus1_plus1(self) -> None:
        """With scale=True, values are in [-1, 1]."""
        gaps = np.random.randn(4, 40) * 100  # large gaps
        gr = _make_gap_result(gaps)
        sig = GapSignal(gr, threshold=0.1, scale=True)
        df = sig.to_signal_matrix()
        assert (df.values >= -1.0).all()
        assert (df.values <= 1.0).all()


class TestRegimeSignal:
    def test_regime_signal_dominant_regime(self) -> None:
        """dominant_regime returns argmax across M regimes."""
        T, M = 20, 3
        regime_probs = np.zeros((T, M))
        # First 10 steps: regime 0 dominant; last 10: regime 2 dominant
        regime_probs[:10, 0] = 0.9
        regime_probs[:10, 1] = 0.05
        regime_probs[:10, 2] = 0.05
        regime_probs[10:, 0] = 0.05
        regime_probs[10:, 1] = 0.05
        regime_probs[10:, 2] = 0.9

        actors = ["A", "B"]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        rs = RegimeSignal(regime_probs, actors, list(dates))

        dom = rs.dominant_regime()
        assert (dom[:10] == 0).all()
        assert (dom[10:] == 2).all()

    def test_regime_signal_matrix_shape(self) -> None:
        T, M, N = 15, 2, 4
        regime_probs = np.random.dirichlet([1.0] * M, size=T)
        actors = [f"A{i}" for i in range(N)]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        rs = RegimeSignal(regime_probs, actors, list(dates))
        df = rs.to_signal_matrix()
        assert df.shape == (N, T)

    def test_regime_signal_binary(self) -> None:
        """Signal matrix contains only 0 or 1."""
        T, M = 20, 3
        rng = np.random.default_rng(1)
        regime_probs = rng.dirichlet([1.0] * M, size=T)
        actors = ["X", "Y", "Z"]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        rs = RegimeSignal(regime_probs, actors, list(dates))
        df = rs.to_signal_matrix()
        unique_vals = set(df.values.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})


class TestCriticalitySignal:
    def test_criticality_signal_normalized(self) -> None:
        """Values in [-1, 1]."""
        T = 30
        criticality = np.sin(np.linspace(0, 2 * np.pi, T)) * 5
        actors = ["A", "B", "C"]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        cs = CriticalitySignal(criticality, actors, list(dates))
        df = cs.to_signal_matrix()
        assert df.values.min() >= -1.0 - 1e-9
        assert df.values.max() <= 1.0 + 1e-9

    def test_criticality_signal_shape(self) -> None:
        T, N = 25, 6
        criticality = np.random.randn(T)
        actors = [f"A{i}" for i in range(N)]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        cs = CriticalitySignal(criticality, actors, list(dates))
        df = cs.to_signal_matrix()
        assert df.shape == (N, T)

    def test_criticality_constant_gives_zero(self) -> None:
        """Constant criticality → all zeros after normalization."""
        T = 10
        criticality = np.ones(T) * 3.0
        actors = ["A"]
        dates = pd.date_range("2020-01-01", periods=T, freq="QS")
        cs = CriticalitySignal(criticality, actors, list(dates))
        df = cs.to_signal_matrix()
        assert (df.values == 0.0).all()
