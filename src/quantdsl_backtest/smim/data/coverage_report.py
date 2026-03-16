"""Coverage report generator for SMIM Gate G1.

Generates a markdown report summarising data coverage, quality, and
A1 compliance for a PointInTimeStore snapshot. The report is used as
evidence at the Gate G1 milestone review.

Gate G1 criteria (as defined in data_source_audit.md):
  - All required signals present for ≥ 80% of actor-date cells
  - Zero A1 leak violations in the store
  - All actor_ids resolved to the registry

CLI usage::

    uv run python -m quantdsl_backtest.smim.data.coverage_report \\
        --store-path smim_pit_store \\
        --as-of 2023-01-01 \\
        --output docs/smim/gate_g1_coverage.md

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from quantdsl_backtest.smim.data.pit_store import PointInTimeStore
from quantdsl_backtest.smim.data.quality_checks import (
    LeakDetectionResult,
    MissingnessResult,
    leak_detection_test,
    missingness_check,
)


# ── report dataclass ──────────────────────────────────────────────────────────

@dataclass
class CoverageReport:
    """Structured coverage report for Gate G1.

    Attributes:
        as_of: Point-in-time snapshot date.
        generated_at: UTC timestamp when the report was created.
        sources: List of sources found in the store.
        actor_count: Total unique actors.
        signal_count: Total unique signals.
        total_rows: Total observation rows (post A1 filter).
        leak_result: Result of leak detection scan.
        missingness: Per-signal missingness statistics.
        g1_passed: Whether all Gate G1 criteria are met.
    """

    as_of: pd.Timestamp
    generated_at: datetime = field(default_factory=datetime.utcnow)
    sources: list[str] = field(default_factory=list)
    actor_count: int = 0
    signal_count: int = 0
    total_rows: int = 0
    leak_result: LeakDetectionResult = field(default_factory=LeakDetectionResult)
    missingness: list[MissingnessResult] = field(default_factory=list)
    g1_passed: bool = False

    _MISSINGNESS_THRESHOLD = 0.20  # allow up to 20% missing


def build_report(
    store: PointInTimeStore,
    as_of: pd.Timestamp,
    signal_ids: list[str] | None = None,
) -> CoverageReport:
    """Build a CoverageReport from a PointInTimeStore snapshot.

    Args:
        store: PointInTimeStore to analyse.
        as_of: PIT snapshot date (A1 filter applied).
        signal_ids: Signals to include in missingness check.
            If None, uses all signals present in the store.

    Returns:
        Populated CoverageReport.
    """
    report = CoverageReport(as_of=as_of)
    report.sources = store.list_sources()

    all_data = store.query(as_of=as_of)
    report.total_rows = len(all_data)

    if not all_data.empty:
        report.actor_count = all_data["actor_id"].nunique()
        report.signal_count = all_data["signal_id"].nunique()

    # Leak detection
    report.leak_result = leak_detection_test(store)

    # Missingness
    sids = signal_ids if signal_ids else store.list_signals()
    if sids:
        report.missingness = missingness_check(store, sids, as_of=as_of)

    # Gate G1 pass/fail
    leak_ok = report.leak_result.passed
    coverage_ok = all(
        m.missing_fraction <= report._MISSINGNESS_THRESHOLD
        for m in report.missingness
    ) if report.missingness else True

    report.g1_passed = leak_ok and coverage_ok
    return report


def render_markdown(report: CoverageReport) -> str:
    """Render a CoverageReport as a markdown string.

    Args:
        report: Populated CoverageReport.

    Returns:
        Markdown-formatted string suitable for writing to a ``.md`` file.
    """
    lines: list[str] = []
    ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    status = "PASS" if report.g1_passed else "FAIL"
    status_badge = "✅" if report.g1_passed else "❌"

    lines += [
        "# SMIM Gate G1 — Data Coverage Report",
        "",
        f"**As-of date:** {report.as_of.date()}  ",
        f"**Generated:** {ts}  ",
        f"**Gate G1 status:** {status_badge} {status}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sources | {', '.join(report.sources) or '—'} |",
        f"| Actors | {report.actor_count} |",
        f"| Signals | {report.signal_count} |",
        f"| Total rows (post A1) | {report.total_rows} |",
        f"| Leak violations | {report.leak_result.violation_count} |",
        "",
    ]

    # Leak section
    lines += [
        "## A1 Leak Detection",
        "",
    ]
    if report.leak_result.passed:
        lines.append(
            f"✅ No violations found in {report.leak_result.rows_checked} rows."
        )
    else:
        lines.append(
            f"❌ **{report.leak_result.violation_count} violation(s)** "
            f"in {report.leak_result.rows_checked} rows."
        )
        lines.append("")
        lines.append("| Source | Actor | Signal | event_date | pub_date |")
        lines.append("|--------|-------|--------|------------|----------|")
        for v in report.leak_result.violations[:20]:
            lines.append(
                f"| {v.source} | {v.actor_id} | {v.signal_id} "
                f"| {v.event_date.date()} | {v.pub_date.date()} |"
            )
        if report.leak_result.violation_count > 20:
            lines.append(
                f"| … | … | … | … | … | *(and "
                f"{report.leak_result.violation_count - 20} more)* |"
            )

    lines += ["", "## Signal Coverage (Missingness)", ""]

    if not report.missingness:
        lines.append("*No signals checked.*")
    else:
        threshold_pct = int(report._MISSINGNESS_THRESHOLD * 100)
        lines.append(
            f"Threshold: ≤ {threshold_pct}% missing (G1 requirement)."
        )
        lines.append("")
        lines.append("| Signal | Actors | Dates | Missing % | Status |")
        lines.append("|--------|--------|-------|-----------|--------|")
        for m in sorted(report.missingness, key=lambda x: -x.missing_fraction):
            pct = m.missing_fraction * 100
            ok = m.missing_fraction <= report._MISSINGNESS_THRESHOLD
            badge = "✅" if ok else "❌"
            lines.append(
                f"| {m.signal_id} | {m.actor_count} | {m.dates_checked} "
                f"| {pct:.1f}% | {badge} |"
            )

    lines += [
        "",
        "---",
        f"*Auto-generated by `smim.data.coverage_report` — do not edit manually.*",
    ]

    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coverage_report",
        description="Generate SMIM Gate G1 data coverage report.",
    )
    p.add_argument("--store-path", required=True, help="PointInTimeStore root dir")
    p.add_argument("--as-of", required=True, help="Snapshot date YYYY-MM-DD")
    p.add_argument(
        "--output",
        default=None,
        help="Output markdown file path (default: print to stdout)",
    )
    p.add_argument(
        "--signal",
        action="append",
        dest="signals",
        metavar="SIGNAL_ID",
        help="Signals to check (repeatable); default: all signals in store",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = PointInTimeStore(root_dir=Path(args.store_path))
    as_of = pd.Timestamp(args.as_of)

    report = build_report(store, as_of=as_of, signal_ids=args.signals)
    md = render_markdown(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Report written to {out}")
    else:
        print(md)

    return 0 if report.g1_passed else 1


if __name__ == "__main__":
    sys.exit(main())
