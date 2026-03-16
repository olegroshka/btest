"""Tests for model comparison suite."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.validation.model_comparison import (
    ComparisonTableRow,
    ModelComparisonResult,
    ModelComparisonSuite,
)


def _make_suite_with_perfect_smim(n: int = 200) -> ModelComparisonSuite:
    """Create a suite where SMIM has perfect predictions and baselines are noisy."""
    rng = np.random.default_rng(42)
    actual = rng.standard_normal(n)

    suite = ModelComparisonSuite()
    suite.add_result("smim", actual.copy(), actual)  # perfect

    for baseline in ModelComparisonSuite.BASELINE_NAMES:
        noisy = actual + rng.standard_normal(n) * 3.0
        suite.add_result(baseline, noisy, actual)

    return suite


class TestPerfectModelWinsDMTest:
    def test_perfect_model_wins_dm_test(self) -> None:
        """SMIM = perfect predictions, baselines = noisy → beats all baselines."""
        suite = _make_suite_with_perfect_smim(n=200)
        assert suite.smim_beats_baselines(alpha=0.05)

    def test_metrics_computed_correctly(self) -> None:
        """Known predictions → known R², RMSE."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual.copy()

        suite = ModelComparisonSuite()
        suite.add_result("smim", predicted, actual)

        result = suite._results["smim"]
        assert result.r2 == pytest.approx(1.0)
        assert result.rmse == pytest.approx(0.0)


class TestComparisonTable:
    def test_comparison_table_has_all_baselines(self) -> None:
        """Comparison table includes one row per registered baseline."""
        suite = _make_suite_with_perfect_smim(n=100)
        metrics_df, comparisons = suite.compare_all()

        baseline_names = {c.model_b for c in comparisons}
        for b in ModelComparisonSuite.BASELINE_NAMES:
            assert b in baseline_names

    def test_metrics_df_has_smim_row(self) -> None:
        """Metrics DataFrame includes a row for SMIM."""
        suite = _make_suite_with_perfect_smim(n=50)
        metrics_df, _ = suite.compare_all()
        assert "smim" in metrics_df["model"].values

    def test_comparison_rows_have_correct_structure(self) -> None:
        """Each comparison row has required fields."""
        suite = _make_suite_with_perfect_smim(n=100)
        _, comparisons = suite.compare_all()
        for c in comparisons:
            assert isinstance(c, ComparisonTableRow)
            assert c.model_a == "smim"
            assert c.winner in {"A", "B", "tie"}
            assert 0.0 <= c.dm_pvalue <= 1.0


class TestSmimBeatsBaselines:
    def test_smim_beats_baselines_counts_correctly(self) -> None:
        """smim_beats_baselines returns True when SMIM beats ≥3 baselines."""
        suite = _make_suite_with_perfect_smim(n=300)
        assert suite.smim_beats_baselines() is True

    def test_smim_does_not_beat_baselines_when_worse(self) -> None:
        """SMIM worse than baselines → smim_beats_baselines returns False."""
        rng = np.random.default_rng(99)
        actual = rng.standard_normal(200)

        suite = ModelComparisonSuite()
        # SMIM is very noisy
        suite.add_result("smim", actual + rng.standard_normal(200) * 5, actual)
        # Baselines are perfect
        for baseline in ModelComparisonSuite.BASELINE_NAMES:
            suite.add_result(baseline, actual.copy(), actual)

        result = suite.smim_beats_baselines()
        assert result is False

    def test_no_smim_registered_returns_false(self) -> None:
        """No SMIM registered → no comparisons → False."""
        suite = ModelComparisonSuite()
        rng = np.random.default_rng(0)
        actual = rng.standard_normal(50)
        suite.add_result("historical_mean", actual * 0.5, actual)
        assert suite.smim_beats_baselines() is False

    def test_metrics_df_columns(self) -> None:
        """metrics_df has model, r2, rmse, spearman_rho columns."""
        suite = _make_suite_with_perfect_smim(n=50)
        metrics_df, _ = suite.compare_all()
        for col in ["model", "r2", "rmse", "spearman_rho"]:
            assert col in metrics_df.columns
