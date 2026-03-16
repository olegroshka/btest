"""Tests for extension evaluation report for Gate G6 (M6.4-T1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantdsl_backtest.smim.validation.extension_report import (
    ExtensionReportData,
    ExtensionResult,
    build_extension_report,
    gate_g6_passes,
    write_extension_report,
)
from quantdsl_backtest.smim.validation.transfer import (
    TransferExperimentResult,
    TransferabilityResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_transfer_result(passes: bool) -> TransferExperimentResult:
    rate = 0.75 if passes else 0.25
    n_trans = 3 if passes else 1
    return TransferExperimentResult(
        source_name="source",
        target_name="target",
        n_transferable=n_trans,
        n_total=4,
        transfer_rate=rate,
        parameter_results=[],
        passes=passes,
    )


def _make_extension_result(name: str, significant: bool) -> ExtensionResult:
    return ExtensionResult(
        name=name,
        baseline_metric=0.5,
        extension_metric=0.6 if significant else 0.51,
        delta=0.1 if significant else 0.01,
        p_value=0.05 if significant else 0.50,
        significant=significant,
        effect_size=0.8 if significant else 0.1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensionReport:

    def test_report_contains_structural_section(self):
        """build_extension_report includes structural benchmark section."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
            transfer_result=_make_transfer_result(passes=True),
        )
        report = build_extension_report(data)
        assert "Structural Benchmark" in report or "structural" in report.lower()
        assert "2. Structural" in report

    def test_report_contains_actor_level_section(self):
        """build_extension_report includes actor-level SSM section."""
        data = ExtensionReportData(
            actor_level_result=_make_extension_result("actor_level", significant=True),
            transfer_result=_make_transfer_result(passes=True),
        )
        report = build_extension_report(data)
        assert "Actor-Level SSM" in report or "actor" in report.lower()
        assert "3. Actor-Level" in report

    def test_report_contains_gate_g6_status(self):
        """build_extension_report includes Gate G6 status."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
            transfer_result=_make_transfer_result(passes=True),
        )
        report = build_extension_report(data)
        assert "G6" in report
        assert "PASS" in report or "FAIL" in report

    def test_gate_g6_passes_with_one_significant_extension(self):
        """Gate G6 passes when at least 1 extension is significant and transfer >= 50%."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
            actor_level_result=_make_extension_result("actor_level", significant=False),
            transfer_result=_make_transfer_result(passes=True),
        )
        assert gate_g6_passes(data) is True

    def test_gate_g6_fails_no_significant_extensions(self):
        """Gate G6 fails when no extensions are significant."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=False),
            actor_level_result=_make_extension_result("actor_level", significant=False),
            transfer_result=_make_transfer_result(passes=True),
        )
        assert gate_g6_passes(data) is False

    def test_gate_g6_fails_when_transfer_fails(self):
        """Gate G6 fails when transfer rate < 50% even if extension is significant."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
            transfer_result=_make_transfer_result(passes=False),
        )
        assert gate_g6_passes(data) is False

    def test_gate_g6_fails_no_transfer_result(self):
        """Gate G6 fails when no transfer result is provided."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
        )
        assert gate_g6_passes(data) is False

    def test_write_creates_file(self, tmp_path):
        """write_extension_report creates the output file."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=True),
            transfer_result=_make_transfer_result(passes=True),
        )
        out = tmp_path / "report.md"
        rc = write_extension_report(data, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert len(content) > 0
        assert rc == 0  # gate passes

    def test_write_returns_1_when_gate_fails(self, tmp_path):
        """write_extension_report returns 1 when Gate G6 fails."""
        data = ExtensionReportData(
            structural_result=_make_extension_result("structural", significant=False),
            transfer_result=_make_transfer_result(passes=True),
        )
        out = tmp_path / "report_fail.md"
        rc = write_extension_report(data, out)
        assert rc == 1

    def test_report_both_extensions_absent(self):
        """Report handles missing extension results gracefully."""
        data = ExtensionReportData(
            transfer_result=_make_transfer_result(passes=True),
        )
        report = build_extension_report(data)
        assert "No structural benchmark" in report or "not available" in report.lower()
        assert "No actor-level" in report or "not available" in report.lower()
