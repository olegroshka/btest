"""Extension evaluation report for Gate G6."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from quantdsl_backtest.smim.validation.transfer import TransferExperimentResult


@dataclass
class ExtensionResult:
    name: str
    baseline_metric: float
    extension_metric: float
    delta: float
    p_value: float
    significant: bool
    effect_size: float


@dataclass
class ExtensionReportData:
    structural_result: ExtensionResult | None = None
    actor_level_result: ExtensionResult | None = None
    transfer_result: TransferExperimentResult | None = None
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _format_bool(val: bool) -> str:
    return "PASS" if val else "FAIL"


def build_extension_report(data: ExtensionReportData) -> str:
    """Render Gate G6 extension evaluation report as markdown.

    Sections:
    1. Overview
    2. Structural benchmark
    3. Actor-level SSM
    4. Cross-geography transfer
    5. Summary table
    6. Gate G6 pass/fail
    """
    sections = []

    g6 = gate_g6_passes(data)

    # ─── 1. Overview ─────────────────────────────────────────────────────────
    overview_lines = [
        "# Gate G6 Extension Evaluation Report\n",
        "## 1. Overview\n",
        "Extensions evaluated:\n",
        f"- Structural benchmark: {'yes' if data.structural_result is not None else 'no'}",
        f"- Actor-level SSM: {'yes' if data.actor_level_result is not None else 'no'}",
        f"- Cross-geography transfer: {'yes' if data.transfer_result is not None else 'no'}",
        f"\n**Overall Gate G6: {_format_bool(g6)}**\n",
    ]
    sections.append("\n".join(overview_lines))

    # ─── 2. Structural benchmark ─────────────────────────────────────────────
    struct_lines = ["## 2. Structural Benchmark\n"]
    if data.structural_result is not None:
        r = data.structural_result
        struct_lines.append(f"- Baseline R²: {r.baseline_metric:.4f}")
        struct_lines.append(f"- Extension R²: {r.extension_metric:.4f}")
        struct_lines.append(f"- Delta: {r.delta:+.4f}")
        struct_lines.append(f"- p-value: {r.p_value:.4f}")
        struct_lines.append(f"- Significant (p<0.10): {_format_bool(r.significant)}")
        struct_lines.append(f"- Effect size: {r.effect_size:.4f}")
        decision = "RETAIN" if r.significant else "REJECT"
        struct_lines.append(f"- Decision: **{decision}**\n")
    else:
        struct_lines.append("_No structural benchmark result available._\n")
    sections.append("\n".join(struct_lines))

    # ─── 3. Actor-level SSM ──────────────────────────────────────────────────
    actor_lines = ["## 3. Actor-Level SSM\n"]
    if data.actor_level_result is not None:
        r = data.actor_level_result
        actor_lines.append(f"- Baseline R²: {r.baseline_metric:.4f}")
        actor_lines.append(f"- Extension R²: {r.extension_metric:.4f}")
        actor_lines.append(f"- Incremental R²: {r.delta:+.4f}")
        actor_lines.append(f"- p-value: {r.p_value:.4f}")
        actor_lines.append(f"- Significant (p<0.10): {_format_bool(r.significant)}")
        actor_lines.append(f"- Effect size: {r.effect_size:.4f}")
        decision = "RETAIN" if r.significant else "REJECT"
        actor_lines.append(f"- Decision: **{decision}**\n")
    else:
        actor_lines.append("_No actor-level SSM result available._\n")
    sections.append("\n".join(actor_lines))

    # ─── 4. Cross-geography transfer ─────────────────────────────────────────
    transfer_lines = ["## 4. Cross-Geography Transfer\n"]
    if data.transfer_result is not None:
        tr = data.transfer_result
        transfer_lines.append(f"- Source: {tr.source_name}")
        transfer_lines.append(f"- Target: {tr.target_name}")
        transfer_lines.append(f"- Transferable: {tr.n_transferable}/{tr.n_total}")
        transfer_lines.append(f"- Transfer rate: {tr.transfer_rate:.2%}")
        transfer_lines.append(f"- Result: **{_format_bool(tr.passes)}**\n")
        if tr.parameter_results:
            transfer_lines.append("| Parameter | Source | Target | Rel. Diff | Transfers |")
            transfer_lines.append("|-----------|--------|--------|-----------|-----------|")
            for pr in tr.parameter_results:
                mark = "yes" if pr.transfers else "no"
                transfer_lines.append(
                    f"| {pr.parameter_name} | {pr.source_value:.4f} | "
                    f"{pr.target_value:.4f} | {pr.relative_diff:.4f} | {mark} |"
                )
    else:
        transfer_lines.append("_No transfer experiment result available._\n")
    sections.append("\n".join(transfer_lines))

    # ─── 5. Summary table ────────────────────────────────────────────────────
    summary_lines = ["## 5. Summary Table\n"]
    summary_lines.append("| Extension | Delta | p-value | Significant | Decision |")
    summary_lines.append("|-----------|-------|---------|-------------|----------|")

    for ext_result in [data.structural_result, data.actor_level_result]:
        if ext_result is not None:
            decision = "RETAIN" if ext_result.significant else "REJECT"
            summary_lines.append(
                f"| {ext_result.name} | {ext_result.delta:+.4f} | "
                f"{ext_result.p_value:.4f} | {_format_bool(ext_result.significant)} | {decision} |"
            )

    if data.transfer_result is not None:
        tr = data.transfer_result
        summary_lines.append(
            f"| cross-geography transfer | — | — | — | {_format_bool(tr.passes)} |"
        )
    sections.append("\n".join(summary_lines))

    # ─── 6. Gate G6 pass/fail ────────────────────────────────────────────────
    gate_lines = ["\n## 6. Gate G6\n"]
    if g6:
        gate_lines.append(
            "**GATE G6: PASS** — At least one extension is statistically significant "
            "(p < 0.10) and parameter transfer rate is ≥ 50%.\n"
        )
    else:
        reasons = []
        has_significant = any(
            r.significant
            for r in [data.structural_result, data.actor_level_result]
            if r is not None
        )
        if not has_significant:
            reasons.append("no extension is statistically significant (p < 0.10)")
        if data.transfer_result is not None and not data.transfer_result.passes:
            reasons.append(f"transfer rate {data.transfer_result.transfer_rate:.0%} < 50%")
        elif data.transfer_result is None:
            reasons.append("no transfer experiment provided")
        gate_lines.append(
            f"**GATE G6: FAIL** — Criteria not met: {'; '.join(reasons)}.\n"
        )
    sections.append("\n".join(gate_lines))

    return "\n\n".join(sections)


def gate_g6_passes(data: ExtensionReportData) -> bool:
    """Gate G6 passes if at least 1 extension is significant (p < 0.10)
    AND transfer rate >= 0.5.
    """
    has_significant = any(
        r.significant
        for r in [data.structural_result, data.actor_level_result]
        if r is not None
    )
    transfer_ok = (
        data.transfer_result is not None and data.transfer_result.passes
    )
    return has_significant and transfer_ok


def write_extension_report(data: ExtensionReportData, output_path: Path) -> int:
    """Write to file. Returns 0 if gate passes, 1 otherwise."""
    report = build_extension_report(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return 0 if gate_g6_passes(data) else 1
