"""
Sensitivity report builder for WP2 Gate G2 verification.

M2.5-T2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import EdgeConfig
from quantdsl_backtest.smim.graph.ablation import AblationResult
from quantdsl_backtest.smim.graph.null_comparison import NullComparisonResult
from quantdsl_backtest.smim.interfaces import RelationChannel
from quantdsl_backtest.smim.validation.baselines import BaselineResult


def gate_g2_passes(
    null_results: dict[RelationChannel, NullComparisonResult],
) -> bool:
    """Gate G2: passes when all null-model p-values < 0.05.

    Args:
        null_results: Mapping from RelationChannel to NullComparisonResult.

    Returns:
        True if all p_values < 0.05, False otherwise.
    """
    if not null_results:
        return False
    return all(r.p_value < 0.05 for r in null_results.values())


def build_sensitivity_report(
    all_matrices: dict[RelationChannel, csr_matrix],
    null_comparison_results: dict[RelationChannel, NullComparisonResult],
    baseline_results: list[BaselineResult],
    ablation_result: AblationResult,
    config: EdgeConfig,
) -> str:
    """Build a markdown sensitivity report for WP2.

    Includes:
    - Table: channel × metric (null p-value, ablation delta)
    - Baseline comparison section
    - Gate G2 pass/fail verdict

    Args:
        all_matrices: Per-channel adjacency matrices.
        null_comparison_results: Null model comparison results per channel.
        baseline_results: List of BaselineResult from baseline models.
        ablation_result: AblationResult from channel ablation analysis.
        config: EdgeConfig for metadata.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []
    lines.append("# SMIM WP2 Sensitivity Report\n")
    lines.append("## Graph Construction Validation\n")

    # ── Channel Summary Table ──────────────────────────────────────────
    lines.append("## Channel Summary\n")
    lines.append("| Channel | Null p-value | Ablation Delta | Passes G2 |")
    lines.append("|---------|--------------|----------------|-----------|")

    channel_names = {
        RelationChannel.FINANCIAL: "financial",
        RelationChannel.NARRATIVE: "narrative",
        RelationChannel.SUPPLY_CHAIN: "supply_chain",
        RelationChannel.REGULATORY: "regulatory",
        RelationChannel.FISCAL: "fiscal",
        RelationChannel.IMITATION: "imitation",
        RelationChannel.MARKET_IMPLIED: "market_implied",
    }

    for channel in all_matrices:
        name = channel_names.get(channel, channel.value)
        null_res = null_comparison_results.get(channel)
        ablation_delta = ablation_result.channel_ablation.get(name, float("nan"))

        p_val_str = f"{null_res.p_value:.4f}" if null_res else "N/A"
        passes_str = "YES" if (null_res and null_res.passes) else "NO"
        delta_str = f"{ablation_delta:.4f}" if not np.isnan(ablation_delta) else "N/A"

        lines.append(f"| {name} | {p_val_str} | {delta_str} | {passes_str} |")

    lines.append("")

    # ── Density Sweep Section ──────────────────────────────────────────
    lines.append("## Density Sweep (Spectral Energy Retained)\n")
    lines.append("| Target Density | Spectral Energy Retained |")
    lines.append("|----------------|--------------------------|")
    for density, retained in sorted(ablation_result.density_sweep.items()):
        lines.append(f"| {density:.2f} | {retained:.4f} |")
    lines.append("")

    # ── Baseline Comparison ────────────────────────────────────────────
    lines.append("## Baseline Model Comparison\n")
    if baseline_results:
        lines.append("| Model | OOS R² | OOS RMSE | N Params |")
        lines.append("|-------|--------|----------|----------|")
        for br in baseline_results:
            r2_str = f"{br.r2_oos:.4f}" if not np.isnan(br.r2_oos) else "NaN"
            rmse_str = f"{br.rmse_oos:.4f}" if not np.isnan(br.rmse_oos) else "NaN"
            lines.append(f"| {br.model_name} | {r2_str} | {rmse_str} | {br.n_params} |")
    else:
        lines.append("_No baseline results provided._")
    lines.append("")

    # ── Gate G2 Verdict ───────────────────────────────────────────────
    lines.append("## Gate G2 Verdict\n")
    g2_pass = gate_g2_passes(null_comparison_results)
    has_positive_ablation = any(
        v > 0 for v in ablation_result.channel_ablation.values()
    )
    if g2_pass and has_positive_ablation:
        verdict = "**PASS** — All null-model p-values < 0.05 and at least one channel has positive ablation delta."
    elif g2_pass:
        verdict = "**PARTIAL PASS** — All null-model p-values < 0.05, but no positive ablation delta found."
    else:
        verdict = "**FAIL** — Not all null-model p-values < 0.05."

    lines.append(f"Gate G2: {verdict}\n")

    # Config snapshot
    lines.append("## Configuration Snapshot\n")
    lines.append(f"- Channels: {', '.join(config.channels)}")
    lines.append(f"- Null model instances: {config.null_model_instances}")
    lines.append(f"- Granger p-threshold: {config.granger.p_threshold}")
    lines.append(f"- Narrative similarity threshold: {config.narrative.similarity_threshold}")
    lines.append(f"- Supply chain coefficient threshold: {config.supply_chain.coefficient_threshold}")
    lines.append("")

    return "\n".join(lines)


def write_sensitivity_report(
    all_matrices: dict[RelationChannel, csr_matrix],
    null_comparison_results: dict[RelationChannel, NullComparisonResult],
    baseline_results: list[BaselineResult],
    ablation_result: AblationResult,
    config: EdgeConfig,
    output_path: Path,
) -> Path:
    """Write sensitivity report to a file.

    Creates docs/smim/reports/ directory structure and writes the markdown.

    Args:
        all_matrices: Per-channel adjacency matrices.
        null_comparison_results: Null model comparison results.
        baseline_results: Baseline model results.
        ablation_result: Ablation analysis results.
        config: EdgeConfig for metadata.
        output_path: Path where the markdown file will be written.

    Returns:
        Path to the written file.
    """
    report = build_sensitivity_report(
        all_matrices=all_matrices,
        null_comparison_results=null_comparison_results,
        baseline_results=baseline_results,
        ablation_result=ablation_result,
        config=config,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
