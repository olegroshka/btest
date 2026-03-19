"""
SMIM Acceptance Report plugin.

Collects pass/fail per test and prints a structured acceptance report at the
end of the test session (via pytest_terminal_summary hook).

Usage — just import this module in your conftest or load it as a plugin.
The report is printed automatically when running acceptance tests.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Iterator

import pytest


# ═══════════════════════════════════════════════════════════
# Section → test-ID prefix mapping
# (based on test naming conventions in ACCEPTANCE_TESTS.md)
# ═══════════════════════════════════════════════════════════

_SECTION_PREFIXES: list[tuple[str, list[str]]] = [
    ("Graph Construction",       ["A_GR", "I_GR", "D_GR", "A_NE", "I_NE", "D_NE",
                                   "A_AO", "I_AO", "A_SP", "I_SP",
                                   "I_NM", "A_NM"]),
    ("Spectral Decomposition",   ["A_SC", "I_SC", "R_SC", "D_SC",
                                   "A_PL", "I_PL", "R_PL", "D_PL",
                                   "A_HD", "I_HD", "R_HD", "D_HD",
                                   "I_DV", "R_DV",
                                   "A_DMD", "I_DMD", "R_DMD", "D_DMD"]),
    ("Mode Selection",           ["A_MDL", "I_MDL", "A_LZ", "I_LZ", "A_RG"]),
    ("Kalman Filter + EM",       ["A_KF", "I_KF", "R_KF", "D_KF",
                                   "A_EM", "I_EM", "D_EM"]),
    ("Kim Filter",               ["A_KIM", "I_KIM", "R_KIM", "D_KIM"]),
    ("Observability",            ["A_OB", "I_OB"]),
    ("Phase Transition",         ["I_OP", "A_GL", "I_GL", "A_CI", "I_CI"]),
    ("PID",                      ["A_PID", "I_PID"]),
    ("Transfer Entropy",         ["A_TE", "I_TE", "R_TE"]),
    ("TDA",                      ["A_TDA", "I_TDA", "R_TDA"]),
    ("Benchmarks/Gaps",          ["A_PB", "I_PB", "I_MB", "A_EB"]),
    ("Pipeline Sanity",          ["P_"]),
]

# Total expected tests per section (from ACCEPTANCE_TESTS.md)
_SECTION_EXPECTED: dict[str, int] = {
    "Graph Construction":       20,
    "Spectral Decomposition":   35,
    "Mode Selection":            9,
    "Kalman Filter + EM":       14,
    "Kim Filter":                9,
    "Observability":             3,
    "Phase Transition":          8,
    "PID":                       6,
    "Transfer Entropy":          6,
    "TDA":                       7,
    "Benchmarks/Gaps":           7,
    "Pipeline Sanity":           4,
}


def _section_for_test(node_id: str) -> str:
    """Return the section name for a given test node ID."""
    # node_id looks like:  tests/acceptance/smim/test_graph.py::TestGranger::test_A_GR_1_known_var
    # Extract the test function/class name part
    parts = node_id.split("::")
    test_name = parts[-1] if parts else node_id

    for section, prefixes in _SECTION_PREFIXES:
        for prefix in prefixes:
            # Match prefix in test name (case-insensitive)
            if prefix.lower() in test_name.lower():
                return section

    return "Uncategorised"


class AcceptanceReport:
    """Collects acceptance test results and generates a formatted report."""

    def __init__(self) -> None:
        self._passed: dict[str, list[str]] = defaultdict(list)
        self._failed: dict[str, list[str]] = defaultdict(list)

    def record(self, node_id: str, passed: bool, skipped: bool = False) -> None:
        if skipped:
            return  # skipped tests do not count against pass/fail
        section = _section_for_test(node_id)
        if passed:
            self._passed[section].append(node_id)
        else:
            self._failed[section].append(node_id)

    @property
    def all_sections(self) -> Iterator[str]:
        sections_seen = set(self._passed) | set(self._failed)
        for section, _ in _SECTION_PREFIXES:
            if section in sections_seen:
                yield section
        for s in sorted(sections_seen - {s for s, _ in _SECTION_PREFIXES}):
            yield s

    def format_report(self) -> str:
        date_str = datetime.date.today().isoformat()
        lines: list[str] = [
            "",
            f"SMIM Acceptance Report — {date_str}",
            "===========================================",
        ]

        total_passed = 0
        total_expected = 0
        all_green = True

        for section, _ in _SECTION_PREFIXES:
            passed = len(self._passed.get(section, []))
            failed = len(self._failed.get(section, []))
            ran = passed + failed
            if ran == 0:
                continue
            expected = _SECTION_EXPECTED.get(section, ran)
            total_passed += passed
            total_expected += expected
            icon = "✅" if failed == 0 else "❌"
            if failed > 0:
                all_green = False
            label = f"{section}:"
            lines.append(f"  {label:<26} {passed}/{expected} passed {icon}")
            for fid in self._failed.get(section, []):
                lines.append(f"      FAIL: {fid}")

        # Uncategorised
        unc_passed = len(self._passed.get("Uncategorised", []))
        unc_failed = len(self._failed.get("Uncategorised", []))
        if unc_passed + unc_failed > 0:
            icon = "✅" if unc_failed == 0 else "❌"
            if unc_failed > 0:
                all_green = False
            lines.append(
                f"  {'Uncategorised:':<26} {unc_passed}/{unc_passed+unc_failed} passed {icon}"
            )
            total_passed += unc_passed
            total_expected += unc_passed + unc_failed

        ran_total = sum(len(v) for v in self._passed.values()) + \
                    sum(len(v) for v in self._failed.values())
        lines.append("-------------------------------------------")
        lines.append(f"  {'TOTAL:':<26} {total_passed}/{total_expected} passed")
        status = "READY FOR EXPERIMENTS" if all_green else "BLOCKED"
        lines.append(f"  STATUS: {status}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Pytest plugin
# ═══════════════════════════════════════════════════════════

class _AcceptanceReportPlugin:
    def __init__(self) -> None:
        self.report = AcceptanceReport()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call):
        outcome = yield
        rep = outcome.get_result()
        # Only count acceptance-marked tests in the final summary phase
        if (
            rep.when == "call"
            and item.get_closest_marker("acceptance") is not None
        ):
            self.report.record(item.nodeid, rep.passed, skipped=rep.skipped)

    def pytest_terminal_summary(
        self, terminalreporter, exitstatus: int, config: pytest.Config
    ) -> None:
        total = sum(len(v) for v in self.report._passed.values()) + \
                sum(len(v) for v in self.report._failed.values())
        if total == 0:
            return
        terminalreporter.write_sep("=", "SMIM Acceptance Report", bold=True)
        terminalreporter.write_line(self.report.format_report())


def pytest_configure(config: pytest.Config) -> None:
    """Register the acceptance report plugin and custom markers."""
    config.addinivalue_line(
        "markers",
        "acceptance: marks a test as part of the SMIM acceptance suite",
    )
    config.addinivalue_line(
        "markers",
        "section(name): tag a test with its acceptance section name",
    )
    # Only register the plugin if we are not a worker process (xdist)
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(_AcceptanceReportPlugin(), "smim_acceptance_report")
