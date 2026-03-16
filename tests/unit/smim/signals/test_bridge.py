"""Tests for smim/signals/gap_signal.py — GapSignal bridge to btest DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.interfaces import BenchmarkClass, GapResult
from quantdsl_backtest.smim.signals.gap_signal import GapSignal


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_gap_result(N: int = 4, T: int = 20) -> GapResult:
    rng = np.random.default_rng(0)
    gaps = rng.standard_normal((N, T))
    return GapResult(
        gaps=gaps,
        benchmark_class=BenchmarkClass.PREDICTIVE,
        benchmarks=np.zeros((N, T)),
    )


# ═══════════════════════════════════════════════════════════
# TestGapSignalBridge
# ═══════════════════════════════════════════════════════════

class TestGapSignalBridge:
    def test_converts_gap_result_to_btest_signal(self) -> None:
        gap_result = _make_gap_result()
        signal = GapSignal(gap_result)
        matrix = signal.to_signal_matrix()
        assert isinstance(matrix, pd.DataFrame)

    def test_output_conforms_to_dsl_signal_convention(self) -> None:
        """Signal values must be in [-1, 1] range."""
        gap_result = _make_gap_result(N=4, T=20)
        signal = GapSignal(gap_result, scale=True)
        matrix = signal.to_signal_matrix()
        vals = matrix.values
        assert (vals >= -1.0).all() and (vals <= 1.0).all()

    def test_output_shape_matches_n_and_t(self) -> None:
        N, T = 4, 20
        gap_result = _make_gap_result(N=N, T=T)
        signal = GapSignal(gap_result)
        matrix = signal.to_signal_matrix()
        assert matrix.shape == (N, T)

    def test_benchmark_label_preserved_in_metadata(self) -> None:
        """GapSignal must preserve access to the benchmark_class from GapResult."""
        gap_result = _make_gap_result()
        signal = GapSignal(gap_result)
        assert signal.gap_result.benchmark_class == BenchmarkClass.PREDICTIVE

    def test_no_scale_produces_ternary_values(self) -> None:
        """With scale=False, signal values must be exactly -1, 0, or 1."""
        gap_result = _make_gap_result(N=4, T=20)
        signal = GapSignal(gap_result, scale=False)
        matrix = signal.to_signal_matrix()
        unique_vals = set(matrix.values.ravel())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
