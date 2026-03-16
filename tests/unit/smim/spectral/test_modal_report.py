"""Tests for build_modal_report and gate_g3_passes (M3.4-T2)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sparse

from quantdsl_backtest.smim.config import SmimConfig
from quantdsl_backtest.smim.spectral.modal_report import build_modal_report, gate_g3_passes
from quantdsl_backtest.smim.spectral.oos_evaluation import OOSReconstructionResult


def _make_config_with_methods(methods: list[str]) -> SmimConfig:
    return SmimConfig.model_validate(
        {
            "scope": {"domain": "test", "geographies": ["US"]},
            "spectral": {
                "methods": methods,
                "max_modes": 5,
                "rolling_window_quarters": 4,
            },
        }
    )


class TestGateG3:
    def test_passes_when_one_method_positive_r2(self) -> None:
        results = {
            "schur": OOSReconstructionResult(per_window_r2=[0.1], mean_r2=0.1, rank_correlations=[], n_windows=1),
            "polar": OOSReconstructionResult(per_window_r2=[-0.5], mean_r2=-0.5, rank_correlations=[], n_windows=1),
        }
        assert gate_g3_passes(results) is True

    def test_fails_when_all_negative(self) -> None:
        results = {
            "schur": OOSReconstructionResult(per_window_r2=[-0.1], mean_r2=-0.1, rank_correlations=[], n_windows=1),
            "polar": OOSReconstructionResult(per_window_r2=[-0.5], mean_r2=-0.5, rank_correlations=[], n_windows=1),
        }
        assert gate_g3_passes(results) is False

    def test_fails_when_all_nan(self) -> None:
        results = {
            "schur": OOSReconstructionResult(per_window_r2=[], mean_r2=float("nan"), rank_correlations=[], n_windows=0),
        }
        assert gate_g3_passes(results) is False

    def test_empty_dict_fails(self) -> None:
        assert gate_g3_passes({}) is False


class TestBuildModalReport:
    def setup_method(self) -> None:
        rng = np.random.default_rng(42)
        N = 5
        T = 30
        self.signals = rng.standard_normal((T, N))
        A = rng.standard_normal((N, N))
        A = A + A.T  # symmetric
        self.operator = sparse.csr_matrix(A)
        self.config = _make_config_with_methods(["schur", "polar"])

    def test_returns_non_empty_string(self) -> None:
        report = build_modal_report(self.operator, self.signals, self.config)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_gate_g3(self) -> None:
        report = build_modal_report(self.operator, self.signals, self.config)
        assert "Gate G3" in report

    def test_report_contains_method_names(self) -> None:
        report = build_modal_report(self.operator, self.signals, self.config)
        assert "schur" in report
        assert "polar" in report

    def test_report_is_markdown(self) -> None:
        report = build_modal_report(self.operator, self.signals, self.config)
        assert "#" in report  # markdown headings

    def test_report_with_layer_assignments(self) -> None:
        layers = np.array([0, 1, 2, 3, 3])
        report = build_modal_report(self.operator, self.signals, self.config, layer_assignments=layers)
        assert "Gate G3" in report
