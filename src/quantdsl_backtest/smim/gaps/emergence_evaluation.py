"""Emergence evaluation utilities for SMIM gap analysis.

M4.8-T2: Incremental R² and emergence diagnostic summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantdsl_backtest.smim.interfaces import GapResult


def incremental_r2(
    gap_pred: GapResult,
    gap_em: GapResult,
    observations: np.ndarray,
) -> float:
    """Compute the incremental R² gained by the emergence-aware benchmark.

    Δ R² = R²(emergence-aware) - R²(predictive)

    Args:
        gap_pred:     GapResult from the predictive benchmark.
        gap_em:       GapResult from the emergence-aware benchmark.
        observations: (N, T) or (T, N) observation matrix.

    Returns:
        Incremental R² (can be negative if emergence-aware benchmark is worse).
    """
    # Normalise observations to (N, T)
    if observations.shape[0] == gap_pred.gaps.shape[0]:
        obs = observations
    else:
        obs = observations.T

    ss_tot = np.sum((obs - obs.mean()) ** 2) + 1e-12

    def _r2(benchmarks: np.ndarray) -> float:
        ss_res = np.sum((obs - benchmarks) ** 2)
        return float(1.0 - ss_res / ss_tot)

    return _r2(gap_em.benchmarks) - _r2(gap_pred.benchmarks)


@dataclass
class EmergenceDiagnosticSummary:
    """High-level summary of emergence diagnostics.

    Attributes:
        max_synergy:           Maximum off-diagonal synergy value.
        mean_criticality:      Time-average of criticality index.
        topo_complexity_trend: Linear slope of topological complexity over windows.
        te_anomalies:          List of (layer_i, layer_j) pairs with anomalously high TE.
    """

    max_synergy: float
    mean_criticality: float
    topo_complexity_trend: float
    te_anomalies: list[tuple[int, int]] = field(default_factory=list)


def emergence_diagnostic_summary(
    synergy_matrix: np.ndarray,
    criticality: np.ndarray,
    topo_complexity: np.ndarray,
    te_matrix: np.ndarray,
    te_threshold: float = 0.1,
) -> EmergenceDiagnosticSummary:
    """Summarise emergence diagnostics into a single structured result.

    Args:
        synergy_matrix:    (K, K) pairwise synergy matrix.
        criticality:       (T,) criticality index time series.
        topo_complexity:   (n_windows,) topological complexity.
        te_matrix:         (4, 4) layer-to-layer transfer entropy matrix.
        te_threshold:      TE value above which non-adjacent pairs are flagged.

    Returns:
        EmergenceDiagnosticSummary with scalar summaries.
    """
    max_syn = float(synergy_matrix.max()) if synergy_matrix.size > 0 else 0.0
    mean_crit = float(criticality.mean()) if len(criticality) > 0 else 0.0

    # Trend: linear slope of topological complexity
    if len(topo_complexity) > 1:
        t = np.arange(len(topo_complexity))
        slope = float(np.polyfit(t, topo_complexity, 1)[0])
    else:
        slope = 0.0

    # Anomalies: non-adjacent layers with high TE
    anomalies: list[tuple[int, int]] = []
    n = te_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and te_matrix[i, j] > te_threshold:
                anomalies.append((i, j))

    return EmergenceDiagnosticSummary(
        max_synergy=max_syn,
        mean_criticality=mean_crit,
        topo_complexity_trend=slope,
        te_anomalies=anomalies,
    )
