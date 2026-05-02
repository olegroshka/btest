"""Tests for the btest-owned SMIM bridge helpers in `quantdsl_backtest.dsl.smim`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantdsl_backtest.dsl.smim import CriticalitySignal, GapSignal, RegimeSignal


@dataclass(slots=True)
class DummyGapResult:
    gaps: NDArray[np.float64]
    benchmark_class: str
    benchmarks: NDArray[np.float64]
    actors: list[str] | None = None
    dates: list[pd.Timestamp] | list[int] | None = None


def _make_gap_result(gaps: NDArray[np.float64]) -> DummyGapResult:
    n_actors, n_dates = gaps.shape
    return DummyGapResult(
        gaps=gaps,
        benchmark_class="predictive",
        benchmarks=np.zeros_like(gaps),
        actors=[f"A{i}" for i in range(n_actors)],
        dates=list(pd.date_range("2020-01-01", periods=n_dates, freq="QS")),
    )


class TestGapSignal:
    def test_positive_gap_gives_negative_signal(self) -> None:
        gaps = np.zeros((3, 20))
        gaps[0, :] = 10.0
        signal = GapSignal(_make_gap_result(gaps), threshold=0.5, scale=False)
        matrix = signal.to_signal_matrix()
        assert (matrix.iloc[0] < 0).all()

    def test_negative_gap_gives_positive_signal(self) -> None:
        gaps = np.zeros((3, 20))
        gaps[1, :] = -10.0
        signal = GapSignal(_make_gap_result(gaps), threshold=0.5, scale=False)
        matrix = signal.to_signal_matrix()
        assert (matrix.iloc[1] > 0).all()

    def test_small_gap_gives_zero_signal(self) -> None:
        gaps = np.zeros((1, 50))
        gaps[0, 0] = 1.0
        signal = GapSignal(_make_gap_result(gaps), threshold=10.0, scale=False)
        matrix = signal.to_signal_matrix()
        assert (matrix.iloc[0, 1:] == 0).all()

    def test_signal_matrix_shape(self) -> None:
        gaps: NDArray[np.float64] = np.asarray(
            np.random.default_rng(0).normal(size=(5, 30)),
            dtype=float,
        )
        signal = GapSignal(_make_gap_result(gaps), threshold=0.5, scale=True)
        matrix = signal.to_signal_matrix()
        assert matrix.shape == (5, 30)

    def test_scale_true_clips_to_minus1_plus1(self) -> None:
        gaps: NDArray[np.float64] = np.asarray(
            np.random.default_rng(1).normal(size=(4, 40)) * 100,
            dtype=float,
        )
        signal = GapSignal(_make_gap_result(gaps), threshold=0.1, scale=True)
        matrix = signal.to_signal_matrix()
        assert (matrix.values >= -1.0).all()
        assert (matrix.values <= 1.0).all()

    def test_benchmark_metadata_preserved(self) -> None:
        gap_result = _make_gap_result(np.random.randn(2, 10))
        signal = GapSignal(gap_result)
        assert signal.gap_result.benchmark_class == "predictive"


class TestRegimeSignal:
    def test_regime_signal_dominant_regime(self) -> None:
        regime_probs = np.zeros((20, 3))
        regime_probs[:10, 0] = 0.9
        regime_probs[:10, 1:] = 0.05
        regime_probs[10:, 2] = 0.9
        regime_probs[10:, :2] = 0.05
        actors = ["A", "B"]
        dates = list(pd.date_range("2020-01-01", periods=20, freq="QS"))
        signal = RegimeSignal(regime_probs, actors, dates)
        dominant = signal.dominant_regime()
        assert (dominant[:10] == 0).all()
        assert (dominant[10:] == 2).all()

    def test_regime_signal_matrix_shape(self) -> None:
        regime_probs = np.random.dirichlet([1.0, 1.0], size=15)
        signal = RegimeSignal(
            regime_probs=regime_probs,
            actors=[f"A{i}" for i in range(4)],
            dates=list(pd.date_range("2020-01-01", periods=15, freq="QS")),
        )
        assert signal.to_signal_matrix().shape == (4, 15)

    def test_regime_signal_binary(self) -> None:
        rng = np.random.default_rng(1)
        regime_probs = rng.dirichlet([1.0, 1.0, 1.0], size=20)
        signal = RegimeSignal(
            regime_probs=regime_probs,
            actors=["X", "Y", "Z"],
            dates=list(pd.date_range("2020-01-01", periods=20, freq="QS")),
        )
        unique_vals = set(signal.to_signal_matrix().values.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})


class TestCriticalitySignal:
    def test_criticality_signal_normalized(self) -> None:
        criticality = np.sin(np.linspace(0, 2 * np.pi, 30)) * 5
        signal = CriticalitySignal(
            criticality=criticality,
            actors=["A", "B", "C"],
            dates=list(pd.date_range("2020-01-01", periods=30, freq="QS")),
        )
        matrix = signal.to_signal_matrix()
        assert matrix.values.min() >= -1.0 - 1e-9
        assert matrix.values.max() <= 1.0 + 1e-9

    def test_criticality_signal_shape(self) -> None:
        signal = CriticalitySignal(
            criticality=np.random.default_rng(2).normal(size=25),
            actors=[f"A{i}" for i in range(6)],
            dates=list(pd.date_range("2020-01-01", periods=25, freq="QS")),
        )
        assert signal.to_signal_matrix().shape == (6, 25)

    def test_criticality_constant_gives_zero(self) -> None:
        signal = CriticalitySignal(
            criticality=np.ones(10) * 3.0,
            actors=["A"],
            dates=list(pd.date_range("2020-01-01", periods=10, freq="QS")),
        )
        assert (signal.to_signal_matrix().values == 0.0).all()

