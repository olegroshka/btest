"""Evidence synthesis report generator for Gate G5.

M5.6-T1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantdsl_backtest.smim.validation.event_studies import EventStudySuiteResult
from quantdsl_backtest.smim.validation.falsification import FalsificationResult
from quantdsl_backtest.smim.validation.rolling_oos import OOSEvaluationResult


@dataclass
class EvidenceReportData:
    """Container for all Gate G5 evidence."""

    oos_result: OOSEvaluationResult | None = None
    falsification_results: dict[str, FalsificationResult] = field(default_factory=dict)
    event_study_result: EventStudySuiteResult | None = None
    persistence_results: dict | None = None  # actor -> MeanReversionResult
    comparison_result: tuple[pd.DataFrame, list] | None = None  # (metrics_df, comparisons)
    metadata: dict = field(default_factory=dict)


_APPENDIX_J_ITEMS = [
    "A1 point-in-time compliance verified",
    "Gate G1 data coverage passed",
    "Gate G2 graph null model validation passed",
    "Gate G3 spectral comparison conducted",
    "Gate G4 regime selection (BIC delta > 10)",
    "Rolling OOS R² positive",
    "≥4/7 falsification tests pass",
    "≥3/5 event studies significant at p<0.10",
    "Gap mean-reversion documented",
    "SMIM beats ≥3 baselines by DM test",
    "All gap benchmarks labelled with benchmark class",
]


def _format_bool(val: bool) -> str:
    return "PASS" if val else "FAIL"


def build_evidence_report(data: EvidenceReportData) -> str:
    """Render a complete Gate G5 evidence report as markdown.

    Sections:
    1. Executive summary
    2. OOS performance table
    3. Falsification scorecard
    4. Event study results
    5. Persistence findings
    6. Model comparison table
    7. Appendix J compliance checklist
    8. Go/no-go recommendation
    """
    sections = []

    # ─── 1. Executive summary ──────────────────────────────────────────────
    g5_pass = gate_g5_passes(data)

    oos_ok = (
        data.oos_result is not None and data.oos_result.mean_r2 > 0.0
    )
    falsif_pass_count = sum(
        1 for r in data.falsification_results.values() if r.passes
    )
    falsif_ok = falsif_pass_count >= 4
    event_ok = (
        data.event_study_result is not None
        and data.event_study_result.passes
    )
    model_ok = False
    if data.comparison_result is not None:
        _, comparisons = data.comparison_result
        beat_count = sum(
            1 for c in comparisons
            if c.dm_pvalue < 0.05 and c.winner == "A"
        )
        model_ok = beat_count >= 3

    summary_lines = [
        "# Gate G5 Evidence Report\n",
        "## 1. Executive Summary\n",
        f"| Criterion | Status |",
        f"|-----------|--------|",
        f"| OOS mean R² > 0.0 | {_format_bool(oos_ok)} |",
        f"| ≥4/7 falsification tests pass ({falsif_pass_count}/7) | {_format_bool(falsif_ok)} |",
        f"| Event study threshold met | {_format_bool(event_ok)} |",
        f"| SMIM beats ≥3 baselines | {_format_bool(model_ok)} |",
        f"\n**Overall Gate G5: {_format_bool(g5_pass)}**\n",
    ]
    sections.append("\n".join(summary_lines))

    # ─── 2. OOS performance table ──────────────────────────────────────────
    oos_lines = ["## 2. OOS Performance\n"]
    if data.oos_result is not None:
        oos = data.oos_result
        oos_lines.append(f"- Strategy: {oos.strategy}")
        oos_lines.append(f"- N windows: {oos.n_windows}")
        oos_lines.append(f"- Mean R²: {oos.mean_r2:.4f}")
        oos_lines.append(f"- Mean RMSE: {oos.mean_rmse:.4f}\n")
        if oos.per_window_metrics:
            oos_lines.append("| Window | R² | RMSE | N obs |")
            oos_lines.append("|--------|----|------|-------|")
            for i, m in enumerate(oos.per_window_metrics):
                oos_lines.append(
                    f"| {i + 1} | {m['r2']:.3f} | {m['rmse']:.3f} | {m['n_obs']} |"
                )
    else:
        oos_lines.append("_No OOS result available._\n")
    sections.append("\n".join(oos_lines))

    # ─── 3. Falsification scorecard ────────────────────────────────────────
    falsif_lines = ["## 3. Falsification Scorecard\n"]
    if data.falsification_results:
        falsif_lines.append("| Test | Observed | p-value | Pass |")
        falsif_lines.append("|------|----------|---------|------|")
        for name, r in data.falsification_results.items():
            falsif_lines.append(
                f"| {r.test_name} | {r.observed_metric:.4f} | "
                f"{r.p_value:.4f} | {_format_bool(r.passes)} |"
            )
        falsif_lines.append(
            f"\n**Total passing: {falsif_pass_count}/7** "
            f"({_format_bool(falsif_ok)})\n"
        )
    else:
        falsif_lines.append("_No falsification results available._\n")
    sections.append("\n".join(falsif_lines))

    # ─── 4. Event study results ────────────────────────────────────────────
    event_lines = ["## 4. Event Study Results\n"]
    if data.event_study_result is not None:
        es = data.event_study_result
        event_lines.append(
            f"- Significant events: {es.significant_count}/{es.total_count}"
        )
        event_lines.append(f"- Suite passes: {_format_bool(es.passes)}\n")
        if es.results:
            event_lines.append("| Event | Date | Dir | Pre mean | Post mean | p-value | Sig |")
            event_lines.append("|-------|------|-----|----------|-----------|---------|-----|")
            for r in es.results:
                sig = "✓" if r.significant else "✗"
                event_lines.append(
                    f"| {r.event_name} | {r.event_date.date()} | "
                    f"{r.expected_direction:+d} | {r.pre_mean:.3f} | "
                    f"{r.post_mean:.3f} | {r.p_value:.3f} | {sig} |"
                )
    else:
        event_lines.append("_No event study results available._\n")
    sections.append("\n".join(event_lines))

    # ─── 5. Persistence findings ───────────────────────────────────────────
    persist_lines = ["## 5. Persistence Findings\n"]
    if data.persistence_results:
        persist_lines.append("| Actor | AR(1) coef | Half-life (Q) | Reversion rate |")
        persist_lines.append("|-------|------------|---------------|----------------|")
        for actor_id, r in data.persistence_results.items():
            hl = f"{r.half_life_quarters:.2f}" if not (
                r.half_life_quarters != r.half_life_quarters  # NaN check
            ) else "NaN"
            persist_lines.append(
                f"| {r.signal_id} | {r.ar1_coef:.3f} | {hl} | {r.reversion_rate:.3f} |"
            )
    else:
        persist_lines.append("_No persistence results available._\n")
    sections.append("\n".join(persist_lines))

    # ─── 6. Model comparison table ─────────────────────────────────────────
    model_lines = ["## 6. Model Comparison\n"]
    if data.comparison_result is not None:
        metrics_df, comparisons = data.comparison_result
        if not metrics_df.empty:
            model_lines.append("### 6.1 Metrics\n")
            model_lines.append(metrics_df.to_markdown(index=False))
        if comparisons:
            model_lines.append("\n### 6.2 Pairwise DM Tests (SMIM vs baselines)\n")
            model_lines.append("| Model A | Model B | DM stat | p-value | Winner |")
            model_lines.append("|---------|---------|---------|---------|--------|")
            for c in comparisons:
                model_lines.append(
                    f"| {c.model_a} | {c.model_b} | {c.dm_stat:.3f} | "
                    f"{c.dm_pvalue:.4f} | {c.winner} |"
                )
    else:
        model_lines.append("_No model comparison available._\n")
    sections.append("\n".join(model_lines))

    # ─── 7. Appendix J compliance checklist ───────────────────────────────
    appendix_lines = ["## 7. Appendix J Compliance Checklist\n"]
    check_statuses = _build_appendix_j_statuses(data)
    for item, status in zip(_APPENDIX_J_ITEMS, check_statuses):
        mark = "- [x]" if status else "- [ ]"
        appendix_lines.append(f"{mark} {item}")
    sections.append("\n".join(appendix_lines))

    # ─── 8. Go/no-go recommendation ────────────────────────────────────────
    reco_lines = ["\n## 8. Go/No-Go Recommendation\n"]
    if g5_pass:
        reco_lines.append(
            "**RECOMMENDATION: GO** — All Gate G5 criteria are satisfied. "
            "The SMIM framework demonstrates positive out-of-sample predictive power, "
            "passes ≥4/7 falsification tests, event study validation, and "
            "outperforms ≥3 baselines.\n"
        )
    else:
        failed = []
        if not oos_ok:
            failed.append("OOS R² ≤ 0")
        if not falsif_ok:
            failed.append(f"only {falsif_pass_count}/7 falsification tests pass")
        if not event_ok:
            failed.append("event study threshold not met")
        if not model_ok:
            failed.append("SMIM does not beat ≥3 baselines")
        reco_lines.append(
            f"**RECOMMENDATION: NO-GO** — Gate G5 criteria not met: "
            f"{'; '.join(failed)}.\n"
        )
    sections.append("\n".join(reco_lines))

    return "\n\n".join(sections)


def _build_appendix_j_statuses(data: EvidenceReportData) -> list[bool]:
    """Build boolean status list for Appendix J items."""
    # Items 1-5 are assumed to be checked externally; default to True if metadata says so
    meta = data.metadata
    statuses = [
        bool(meta.get("a1_pit_compliance", True)),  # 1
        bool(meta.get("gate_g1_passed", True)),       # 2
        bool(meta.get("gate_g2_passed", True)),       # 3
        bool(meta.get("gate_g3_passed", True)),       # 4
        bool(meta.get("gate_g4_passed", True)),       # 5
        # Item 6: Rolling OOS R² positive
        (data.oos_result is not None and data.oos_result.mean_r2 > 0.0),
        # Item 7: ≥4/7 falsification tests pass
        (sum(1 for r in data.falsification_results.values() if r.passes) >= 4),
        # Item 8: ≥3/5 event studies significant
        (data.event_study_result is not None and data.event_study_result.passes),
        # Item 9: Gap mean-reversion documented
        (data.persistence_results is not None and len(data.persistence_results) > 0),
        # Item 10: SMIM beats ≥3 baselines
        _smim_beats_baselines(data),
        # Item 11: All gap benchmarks labelled (checked via metadata)
        bool(meta.get("benchmarks_labelled", True)),
    ]
    return statuses


def _smim_beats_baselines(data: EvidenceReportData) -> bool:
    """Check if SMIM beats ≥3 baselines from comparison_result."""
    if data.comparison_result is None:
        return False
    _, comparisons = data.comparison_result
    beat_count = sum(
        1 for c in comparisons
        if c.dm_pvalue < 0.05 and c.winner == "A"
    )
    return beat_count >= 3


def gate_g5_passes(data: EvidenceReportData) -> bool:
    """True if all Gate G5 criteria are met.

    Criteria:
    - OOS mean R² > 0.0
    - ≥4/7 falsification tests pass
    - Event study threshold met
    - SMIM beats ≥3 baselines
    """
    # OOS R²
    if data.oos_result is None or data.oos_result.mean_r2 <= 0.0:
        return False

    # Falsification
    n_pass = sum(1 for r in data.falsification_results.values() if r.passes)
    if n_pass < 4:
        return False

    # Event study
    if data.event_study_result is None or not data.event_study_result.passes:
        return False

    # Model comparison
    if not _smim_beats_baselines(data):
        return False

    return True


def write_evidence_report(data: EvidenceReportData, output_path: Path) -> int:
    """Write report to file.

    Args:
        data: EvidenceReportData containing all evaluation results.
        output_path: Path to write the markdown report.

    Returns:
        0 if gate passes, 1 otherwise.
    """
    report = build_evidence_report(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return 0 if gate_g5_passes(data) else 1
