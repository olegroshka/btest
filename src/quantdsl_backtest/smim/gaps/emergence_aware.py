"""Emergence-aware benchmark for SMIM gap analysis.

M4.8-T1: EmergenceAwareBenchmark — extends predictive benchmark with
         synergy-based corrections and criticality scaling.
"""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    FilteredState,
    GapResult,
    ModalFrame,
)


class EmergenceAwareBenchmark:
    """Computes investment gaps against an emergence-corrected benchmark.

    The benchmark is:
        y*_{i,t} = (U α_{t|t-1} + Σ_{j<k} S[j,k] α_j α_k U) × C_t_scale

    where S is the pairwise synergy matrix and C_t is the criticality index.

    Args:
        synergy_matrix:       Optional (K, K) synergy matrix from PID.
        criticality:          Optional (T',) criticality index time series.
        topological_complexity: Optional (n_windows,) TDA complexity series (unused in
                              benchmark computation, available for metadata).
    """

    def __init__(
        self,
        synergy_matrix: np.ndarray | None = None,
        criticality: np.ndarray | None = None,
        topological_complexity: np.ndarray | None = None,
    ) -> None:
        self.synergy_matrix = synergy_matrix
        self.criticality = criticality
        self.topological_complexity = topological_complexity

    @property
    def benchmark_class(self) -> BenchmarkClass:
        return BenchmarkClass.EMERGENCE_AWARE

    def compute(
        self,
        observations: np.ndarray,
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors,
    ) -> GapResult:
        """Compute emergence-aware benchmark and gaps.

        Args:
            observations:   (N, T) or (T, N) observation matrix.
            filtered_state: FilteredState from Kalman/Kim filter.
            modal_frame:    ModalFrame with basis U of shape (N, K).
            actors:         ActorRegistry (unused here).

        Returns:
            GapResult with gaps and benchmarks of shape (N, T).
        """
        U = modal_frame.basis  # (N, K)
        alpha_p = filtered_state.alpha_predicted  # (T, K)
        T, K = alpha_p.shape

        # Base benchmark: linear prediction
        benchmarks_TN = alpha_p @ U.T  # (T, N)

        # Synergy correction: pairwise α_j × α_k interactions weighted by S[j,k]
        if (
            self.synergy_matrix is not None
            and self.synergy_matrix.shape[0] >= K
        ):
            S = self.synergy_matrix[:K, :K]
            synergy_correction = np.zeros((T, K))
            for j in range(K):
                for k in range(j + 1, K):
                    interaction = alpha_p[:, j] * alpha_p[:, k] * S[j, k]
                    synergy_correction[:, j] += interaction * 0.5
                    synergy_correction[:, k] += interaction * 0.5
            correction_obs = synergy_correction @ U.T  # (T, N)
            # Damp: limit correction magnitude to 5% of base prediction RMS.
            # Synergy estimation from T=40 is noisy; large corrections overshoot.
            base_rms = max(np.sqrt(np.mean(benchmarks_TN ** 2)), 1e-8)
            corr_rms = np.sqrt(np.mean(correction_obs ** 2))
            if corr_rms > 0.05 * base_rms:
                correction_obs *= (0.05 * base_rms) / corr_rms
            benchmarks_TN = benchmarks_TN + correction_obs

        # Criticality scaling
        if self.criticality is not None:
            crit = np.asarray(self.criticality)
            if len(crit) >= T:
                C_t = crit[-T:]
            else:
                C_t = crit
            # Pad if needed
            C_t = np.pad(C_t, (0, max(0, T - len(C_t))))[:T]
            scale = np.clip(C_t[:, np.newaxis], 0.5, 2.0)
            benchmarks_TN = benchmarks_TN * scale

        # Normalise observations to (N, T)
        if observations.shape[0] == modal_frame.N:
            obs_NT = observations  # already (N, T)
        else:
            obs_NT = observations.T  # (T, N) → (N, T)

        benchmarks = benchmarks_TN.T  # (N, T)
        gaps = obs_NT - benchmarks

        return GapResult(
            gaps=gaps,
            benchmark_class=self.benchmark_class,
            benchmarks=benchmarks,
        )
