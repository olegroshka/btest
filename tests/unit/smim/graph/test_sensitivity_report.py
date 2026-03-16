"""Tests for sensitivity report and Gate G2 (M2.5-T2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import EdgeConfig
from quantdsl_backtest.smim.graph.ablation import AblationResult
from quantdsl_backtest.smim.graph.null_comparison import NullComparisonResult
from quantdsl_backtest.smim.graph.sensitivity_report import (
    build_sensitivity_report,
    gate_g2_passes,
)
from quantdsl_backtest.smim.interfaces import RelationChannel
from quantdsl_backtest.smim.validation.baselines import BaselineResult


def _random_sparse(n: int, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    dense = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(dense, 0.0)
    return csr_matrix(dense)


def _make_null_result(p_value: float) -> NullComparisonResult:
    return NullComparisonResult(
        observed_stat=1.0,
        null_distribution=np.array([0.5, 0.8, 1.2]),
        p_value=p_value,
        passes=p_value < 0.05,
    )


def _make_ablation_result(delta: float = 0.5) -> AblationResult:
    import pandas as pd
    return AblationResult(
        channel_ablation={"financial": delta, "narrative": delta * 0.5},
        density_sweep={0.1: 0.9, 0.3: 0.7, 0.5: 0.5},
        summary=pd.DataFrame([
            {"analysis": "channel_ablation", "key": "financial", "value": delta},
        ]),
    )


def _make_baseline_results() -> list[BaselineResult]:
    return [
        BaselineResult(
            model_name="DynamicFactorModel",
            r2_oos=0.45,
            rmse_oos=0.12,
            n_params=15,
        ),
        BaselineResult(
            model_name="VAR",
            r2_oos=0.30,
            rmse_oos=0.18,
            n_params=48,
        ),
    ]


class TestBuildSensitivityReport:
    def test_returns_non_empty_string(self):
        n = 4
        matrices = {
            RelationChannel.FINANCIAL: _random_sparse(n, seed=0),
            RelationChannel.NARRATIVE: _random_sparse(n, seed=1),
        }
        null_results = {
            RelationChannel.FINANCIAL: _make_null_result(0.02),
            RelationChannel.NARRATIVE: _make_null_result(0.03),
        }
        config = EdgeConfig()
        report = build_sensitivity_report(
            all_matrices=matrices,
            null_comparison_results=null_results,
            baseline_results=_make_baseline_results(),
            ablation_result=_make_ablation_result(),
            config=config,
        )
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_gate_g2(self):
        n = 4
        matrices = {RelationChannel.FINANCIAL: _random_sparse(n, seed=0)}
        null_results = {RelationChannel.FINANCIAL: _make_null_result(0.01)}
        config = EdgeConfig()
        report = build_sensitivity_report(
            all_matrices=matrices,
            null_comparison_results=null_results,
            baseline_results=[],
            ablation_result=_make_ablation_result(),
            config=config,
        )
        assert "Gate G2" in report

    def test_report_contains_channel_names(self):
        n = 4
        matrices = {
            RelationChannel.FINANCIAL: _random_sparse(n, seed=0),
        }
        null_results = {RelationChannel.FINANCIAL: _make_null_result(0.04)}
        config = EdgeConfig()
        report = build_sensitivity_report(
            all_matrices=matrices,
            null_comparison_results=null_results,
            baseline_results=_make_baseline_results(),
            ablation_result=_make_ablation_result(),
            config=config,
        )
        assert "financial" in report

    def test_report_contains_baseline_section(self):
        n = 3
        matrices = {RelationChannel.NARRATIVE: _random_sparse(n, seed=2)}
        null_results = {RelationChannel.NARRATIVE: _make_null_result(0.04)}
        config = EdgeConfig()
        report = build_sensitivity_report(
            all_matrices=matrices,
            null_comparison_results=null_results,
            baseline_results=_make_baseline_results(),
            ablation_result=_make_ablation_result(),
            config=config,
        )
        assert "Baseline" in report
        assert "DynamicFactorModel" in report


class TestGateG2Passes:
    def test_passes_when_all_p_values_below_threshold(self):
        null_results = {
            RelationChannel.FINANCIAL: _make_null_result(0.01),
            RelationChannel.NARRATIVE: _make_null_result(0.02),
            RelationChannel.SUPPLY_CHAIN: _make_null_result(0.04),
        }
        assert gate_g2_passes(null_results) is True

    def test_fails_when_any_p_value_above_threshold(self):
        null_results = {
            RelationChannel.FINANCIAL: _make_null_result(0.01),
            RelationChannel.NARRATIVE: _make_null_result(0.10),  # > 0.05
        }
        assert gate_g2_passes(null_results) is False

    def test_fails_for_empty_results(self):
        assert gate_g2_passes({}) is False

    def test_passes_for_single_channel_low_p(self):
        null_results = {RelationChannel.FINANCIAL: _make_null_result(0.001)}
        assert gate_g2_passes(null_results) is True

    def test_boundary_exactly_0_05_fails(self):
        """p_value == 0.05 is NOT < 0.05, so gate fails."""
        null_results = {RelationChannel.FINANCIAL: _make_null_result(0.05)}
        assert gate_g2_passes(null_results) is False
