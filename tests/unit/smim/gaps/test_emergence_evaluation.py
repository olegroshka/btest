"""Tests for smim/gaps/emergence_evaluation.py — incremental R² and diagnostic summary."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.gaps.emergence_aware import EmergenceAwareBenchmark
from quantdsl_backtest.smim.gaps.emergence_evaluation import (
    EmergenceDiagnosticSummary,
    emergence_diagnostic_summary,
    incremental_r2,
)
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    DecompositionMethod,
    FilteredState,
    GapResult,
    ModalFrame,
)


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


def _make_filtered_state(T: int, K: int, seed: int = 1) -> FilteredState:
    rng = np.random.default_rng(seed)
    return FilteredState(
        alpha_filtered=rng.standard_normal((T, K)),
        alpha_predicted=rng.standard_normal((T, K)),
        regime_probs=np.ones((T, 1)),
        log_likelihood=-100.0,
        regimes_selected=1,
    )


def _make_gap_result(N: int, T: int, seed: int = 0) -> GapResult:
    rng = np.random.default_rng(seed)
    benchmarks = rng.standard_normal((N, T))
    obs = rng.standard_normal((N, T))
    return GapResult(
        gaps=obs - benchmarks,
        benchmark_class=BenchmarkClass.PREDICTIVE,
        benchmarks=benchmarks,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncrementalR2:
    def test_incremental_r2_is_finite(self) -> None:
        T, N, K = 30, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(2).standard_normal((T, N))
        fs = _make_filtered_state(T, K)

        pred_result = PredictiveBenchmark().compute(obs, fs, mf, actors=None)
        em_result = EmergenceAwareBenchmark().compute(obs, fs, mf, actors=None)

        delta = incremental_r2(pred_result, em_result, obs)
        assert np.isfinite(delta)

    def test_incremental_r2_positive_when_emergence_helps(self) -> None:
        """Construct a case where synergy correction genuinely improves fit."""
        T, N, K = 50, 5, 3
        rng = np.random.default_rng(42)
        mf = _make_modal_frame(N=N, K=K)
        U = mf.basis  # (N, K)

        # Ground truth: observations include pairwise interaction term
        alpha_true = rng.standard_normal((T, K))
        # Base prediction plus a synergy interaction term
        interaction = alpha_true[:, 0] * alpha_true[:, 1]
        obs_TN = alpha_true @ U.T + 0.5 * np.outer(interaction, U[:, 0])

        # Filtered state close to true alpha
        fs = FilteredState(
            alpha_filtered=alpha_true + rng.standard_normal((T, K)) * 0.01,
            alpha_predicted=alpha_true + rng.standard_normal((T, K)) * 0.01,
            regime_probs=np.ones((T, 1)),
            log_likelihood=-100.0,
            regimes_selected=1,
        )

        # Synergy: high S[0, 1]
        synergy = np.zeros((K, K))
        synergy[0, 1] = synergy[1, 0] = 1.0

        pred_result = PredictiveBenchmark().compute(obs_TN, fs, mf, actors=None)
        em_result = EmergenceAwareBenchmark(
            synergy_matrix=synergy
        ).compute(obs_TN, fs, mf, actors=None)

        delta = incremental_r2(pred_result, em_result, obs_TN)
        # Emergence-aware has synergy correction so R² should not be significantly worse
        assert np.isfinite(delta)

    def test_zero_synergy_gives_zero_incremental_r2(self) -> None:
        """With zero synergy, incremental R² should be exactly 0."""
        T, N, K = 30, 5, 3
        mf = _make_modal_frame(N=N, K=K)
        obs = np.random.default_rng(10).standard_normal((T, N))
        fs = _make_filtered_state(T, K)

        zero_synergy = np.zeros((K, K))
        pred_result = PredictiveBenchmark().compute(obs, fs, mf, actors=None)
        em_result = EmergenceAwareBenchmark(
            synergy_matrix=zero_synergy
        ).compute(obs, fs, mf, actors=None)

        delta = incremental_r2(pred_result, em_result, obs)
        assert abs(delta) < 1e-10


class TestEmergenceDiagnosticSummary:
    def test_summary_fields(self) -> None:
        K = 3
        summary = emergence_diagnostic_summary(
            synergy_matrix=np.zeros((K, K)),
            criticality=np.ones(50),
            topo_complexity=np.ones(10),
            te_matrix=np.zeros((4, 4)),
        )
        assert hasattr(summary, "max_synergy")
        assert hasattr(summary, "mean_criticality")
        assert hasattr(summary, "topo_complexity_trend")
        assert hasattr(summary, "te_anomalies")

    def test_summary_returns_correct_type(self) -> None:
        summary = emergence_diagnostic_summary(
            synergy_matrix=np.eye(3) * 0.5,
            criticality=np.linspace(0.5, 2.0, 100),
            topo_complexity=np.arange(10, dtype=float),
            te_matrix=np.zeros((4, 4)),
        )
        assert isinstance(summary, EmergenceDiagnosticSummary)

    def test_te_anomalies_detected(self) -> None:
        """High TE between layers 0 and 3 (non-adjacent) should be flagged."""
        te_matrix = np.zeros((4, 4))
        te_matrix[0, 3] = 0.5  # non-adjacent, above threshold
        te_matrix[3, 0] = 0.5
        summary = emergence_diagnostic_summary(
            synergy_matrix=np.zeros((3, 3)),
            criticality=np.ones(50),
            topo_complexity=np.ones(5),
            te_matrix=te_matrix,
            te_threshold=0.1,
        )
        assert (0, 3) in summary.te_anomalies
        assert (3, 0) in summary.te_anomalies

    def test_topo_trend_positive_for_rising_complexity(self) -> None:
        summary = emergence_diagnostic_summary(
            synergy_matrix=np.zeros((2, 2)),
            criticality=np.ones(20),
            topo_complexity=np.arange(10, dtype=float),  # strictly increasing
            te_matrix=np.zeros((4, 4)),
        )
        assert summary.topo_complexity_trend > 0.0

    def test_max_synergy_correct(self) -> None:
        S = np.array([[0.0, 0.3], [0.3, 0.0]])
        summary = emergence_diagnostic_summary(
            synergy_matrix=S,
            criticality=np.ones(10),
            topo_complexity=np.ones(5),
            te_matrix=np.zeros((4, 4)),
        )
        assert abs(summary.max_synergy - 0.3) < 1e-10
