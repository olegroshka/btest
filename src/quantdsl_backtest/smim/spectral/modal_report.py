"""Modal decomposition report generation and Gate G3 (M3.4-T2)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from quantdsl_backtest.smim.config import SmimConfig
from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.comparison import SpectralComparisonHarness
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer
from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer
from quantdsl_backtest.smim.spectral.mode_selection import CombinedModeSelector
from quantdsl_backtest.smim.spectral.oos_evaluation import (
    OOSReconstructionResult,
    rolling_oos_reconstruction,
)
from quantdsl_backtest.smim.spectral.polar import PolarDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer


_DECOMPOSER_MAP = {
    DecompositionMethod.SCHUR: SchurDecomposer,
    DecompositionMethod.POLAR: PolarDecomposer,
    DecompositionMethod.HERMITIAN_DILATION: HermitianDilationDecomposer,
    DecompositionMethod.DIRECTED_VARIATION: DirectedVariationDecomposer,
    DecompositionMethod.DMD: ExactDMDDecomposer,
}


def build_modal_report(
    operator: sparse.csr_matrix,
    signals: np.ndarray,          # (T, N)
    config: SmimConfig,
    layer_assignments: np.ndarray | None = None,
) -> str:
    """Run all configured decomposers and generate markdown comparison report.

    Steps:
    1. SpectralComparisonHarness for reconstruction metrics.
    2. CombinedModeSelector on each decomposer result.
    3. rolling_oos_reconstruction for each method.
    4. Return markdown with tables and Gate G3 verdict.
    """
    signals = np.asarray(signals, dtype=float)
    T, N = signals.shape
    spec_cfg = config.spectral
    k = spec_cfg.max_modes

    # Step 1: Comparison harness
    harness = SpectralComparisonHarness(methods=spec_cfg.methods, k=k)
    comparison_df = harness.run(operator)

    # Step 2 + 3: OOS reconstruction per method
    mode_sel_cfg = spec_cfg.mode_selection
    oos_results: dict[str, OOSReconstructionResult] = {}

    window_size = min(spec_cfg.rolling_window_quarters * 3, max(T // 3, 5))
    step = max(1, window_size // 4)

    for method_str in spec_cfg.methods:
        try:
            dm = DecompositionMethod(method_str)
        except ValueError:
            continue
        if dm not in _DECOMPOSER_MAP:
            continue
        decomposer = _DECOMPOSER_MAP[dm]()
        mode_selector = CombinedModeSelector(
            mdl=mode_sel_cfg.mdl,
            min_compressibility=mode_sel_cfg.compressibility_min,
            rg_relevance=mode_sel_cfg.rg_relevance,
            layer_assignments=layer_assignments,
        )
        if T > window_size + 1:
            oos_result = rolling_oos_reconstruction(
                signals=signals,
                decomposer=decomposer,
                mode_selector=mode_selector,
                window_size=window_size,
                step=step,
                k=k,
            )
        else:
            oos_result = OOSReconstructionResult(
                per_window_r2=[],
                mean_r2=float("nan"),
                rank_correlations=[],
                n_windows=0,
            )
        oos_results[method_str] = oos_result

    # Gate G3
    g3 = gate_g3_passes(oos_results)

    # Build markdown
    lines = [
        "# Modal Representation Report",
        "",
        f"**Signals shape**: T={T}, N={N}",
        f"**Max modes (k)**: {k}",
        f"**Methods evaluated**: {', '.join(spec_cfg.methods)}",
        "",
        "## Spectral Method Comparison",
        "",
    ]

    if not comparison_df.empty:
        lines.append(comparison_df.to_markdown(index=False))
    else:
        lines.append("_No comparison results available._")

    lines += [
        "",
        "## OOS Reconstruction Summary",
        "",
        "| Method | n_windows | mean_R2 | rank_corr_mean |",
        "|--------|-----------|---------|----------------|",
    ]
    for method_str, oos in oos_results.items():
        rc_mean = float(np.mean(oos.rank_correlations)) if oos.rank_correlations else float("nan")
        lines.append(
            f"| {method_str} | {oos.n_windows} | {oos.mean_r2:.4f} | {rc_mean:.4f} |"
        )

    lines += [
        "",
        "## Gate G3: Modal Basis OOS Validity",
        "",
        f"**Gate G3 passes**: {'YES' if g3 else 'NO'}",
        "",
        "> Gate G3 requires at least one method to have mean OOS R² > 0.",
        "",
    ]

    return "\n".join(lines)


def gate_g3_passes(oos_results: dict[str, OOSReconstructionResult]) -> bool:
    """Gate G3: at least one method has mean OOS R² > 0."""
    return any(
        np.isfinite(r.mean_r2) and r.mean_r2 > 0
        for r in oos_results.values()
    )
