"""Tests for evidence synthesis report generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.validation.evidence_report import (
    EvidenceReportData,
    build_evidence_report,
    gate_g5_passes,
    write_evidence_report,
)
from quantdsl_backtest.smim.validation.event_studies import EventStudySuiteResult
from quantdsl_backtest.smim.validation.falsification import FalsificationResult
from quantdsl_backtest.smim.validation.model_comparison import ComparisonTableRow
from quantdsl_backtest.smim.validation.rolling_oos import OOSEvaluationResult


def _make_passing_oos() -> OOSEvaluationResult:
    return OOSEvaluationResult(
        windows=[],
        per_window_metrics=[{"r2": 0.15, "rmse": 0.1, "n_obs": 10}],
        mean_r2=0.15,
        mean_rmse=0.1,
        n_windows=1,
        strategy="expanding",
    )


def _make_passing_falsification() -> dict[str, FalsificationResult]:
    results = {}
    names = [
        "shuffled_edge", "lag_destroyed", "randomised_type",
        "frozen_regime", "symmetric_operator", "block_preserving", "no_network_dfm"
    ]
    for i, name in enumerate(names):
        results[name] = FalsificationResult(
            test_name=name,
            observed_metric=1.0,
            null_distribution=np.zeros(10),
            p_value=0.01,
            passes=True,  # all pass
            n_instances=10,
        )
    return results


def _make_passing_event_study() -> EventStudySuiteResult:
    from quantdsl_backtest.smim.validation.event_studies import EventStudyResult

    results = []
    for i in range(5):
        results.append(EventStudyResult(
            event_name=f"e{i}",
            event_date=pd.Timestamp("2010-01-01"),
            expected_direction=1,
            pre_mean=0.1,
            post_mean=0.8,
            t_stat=3.0,
            p_value=0.01,
            significant=True,
            top_modal_channels=[],
        ))
    return EventStudySuiteResult(
        results=results,
        significant_count=5,
        total_count=5,
        passes=True,
    )


def _make_passing_comparison() -> tuple[pd.DataFrame, list]:
    metrics_df = pd.DataFrame([
        {"model": "smim", "r2": 0.5, "rmse": 0.1, "spearman_rho": 0.7},
        {"model": "historical_mean", "r2": 0.1, "rmse": 0.5, "spearman_rho": 0.2},
        {"model": "random_walk", "r2": 0.0, "rmse": 0.6, "spearman_rho": 0.1},
        {"model": "dfm", "r2": 0.15, "rmse": 0.45, "spearman_rho": 0.3},
        {"model": "graph_free", "r2": 0.05, "rmse": 0.55, "spearman_rho": 0.15},
        {"model": "symmetric_laplacian", "r2": 0.08, "rmse": 0.52, "spearman_rho": 0.18},
    ])
    comparisons = [
        ComparisonTableRow("smim", "historical_mean", -3.0, 0.001, 0.4, "A"),
        ComparisonTableRow("smim", "random_walk", -4.0, 0.0001, 0.5, "A"),
        ComparisonTableRow("smim", "dfm", -2.5, 0.01, 0.35, "A"),
        ComparisonTableRow("smim", "graph_free", -3.2, 0.002, 0.45, "A"),
        ComparisonTableRow("smim", "symmetric_laplacian", -2.8, 0.005, 0.42, "A"),
    ]
    return metrics_df, comparisons


def _make_passing_data() -> EvidenceReportData:
    return EvidenceReportData(
        oos_result=_make_passing_oos(),
        falsification_results=_make_passing_falsification(),
        event_study_result=_make_passing_event_study(),
        comparison_result=_make_passing_comparison(),
    )


class TestReportContainsExecutiveSummary:
    def test_report_contains_executive_summary(self) -> None:
        """Report contains executive summary section."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        assert "Executive Summary" in report
        assert "OOS" in report

    def test_report_contains_gate_g5_status(self) -> None:
        """Report includes pass/fail status."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        assert "PASS" in report or "FAIL" in report


class TestReportContainsFalsificationScorecard:
    def test_report_contains_falsification_scorecard(self) -> None:
        """Report contains falsification scorecard section."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        assert "Falsification" in report
        assert "shuffled_edge" in report

    def test_falsification_scorecard_shows_pass_count(self) -> None:
        """Falsification section shows how many tests pass."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        assert "7" in report  # all 7 tests


class TestReportContainsAppendixJ:
    def test_report_contains_appendix_j(self) -> None:
        """Report contains Appendix J compliance checklist."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        assert "Appendix J" in report

    def test_appendix_j_has_11_items(self) -> None:
        """Appendix J checklist has all 11 items."""
        data = _make_passing_data()
        report = build_evidence_report(data)
        # Count checklist items
        checked = report.count("- [x]") + report.count("- [ ]")
        assert checked == 11


class TestGateG5Passes:
    def test_gate_g5_passes_all_criteria_met(self) -> None:
        """All criteria met → gate passes."""
        data = _make_passing_data()
        assert gate_g5_passes(data) is True

    def test_gate_g5_fails_missing_oos(self) -> None:
        """No OOS result → gate fails."""
        data = _make_passing_data()
        data.oos_result = None
        assert gate_g5_passes(data) is False

    def test_gate_g5_fails_negative_oos_r2(self) -> None:
        """OOS mean R² ≤ 0 → gate fails."""
        data = _make_passing_data()
        data.oos_result = OOSEvaluationResult(
            windows=[], per_window_metrics=[], mean_r2=-0.1, mean_rmse=1.0,
            n_windows=0, strategy="expanding"
        )
        assert gate_g5_passes(data) is False

    def test_gate_g5_fails_insufficient_falsifications(self) -> None:
        """Only 3/7 falsification tests pass → gate fails."""
        data = _make_passing_data()
        falsif = _make_passing_falsification()
        # Make 4 tests fail
        keys = list(falsif.keys())
        for k in keys[:4]:
            falsif[k].passes = False
        data.falsification_results = falsif
        assert gate_g5_passes(data) is False

    def test_gate_g5_fails_no_event_study(self) -> None:
        """No event study result → gate fails."""
        data = _make_passing_data()
        data.event_study_result = None
        assert gate_g5_passes(data) is False

    def test_gate_g5_fails_model_comparison(self) -> None:
        """SMIM doesn't beat baselines → gate fails."""
        data = _make_passing_data()
        # Make all comparisons ties
        metrics_df = pd.DataFrame([{"model": "smim", "r2": 0.1, "rmse": 0.5, "spearman_rho": 0.1}])
        comparisons = [
            ComparisonTableRow("smim", "historical_mean", 0.0, 0.5, 0.0, "tie"),
        ]
        data.comparison_result = (metrics_df, comparisons)
        assert gate_g5_passes(data) is False


class TestWriteCreatesFile:
    def test_write_creates_file(self) -> None:
        """write_evidence_report creates a file at the specified path."""
        data = _make_passing_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            ret = write_evidence_report(data, output_path)
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert len(content) > 100
            assert ret == 0  # gate passes

    def test_write_returns_1_when_gate_fails(self) -> None:
        """Returns 1 when gate fails."""
        data = EvidenceReportData()  # empty data → gate fails
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            ret = write_evidence_report(data, output_path)
            assert ret == 1
            assert output_path.exists()
