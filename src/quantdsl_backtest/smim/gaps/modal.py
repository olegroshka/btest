"""Modal benchmark with per-mode gap attribution for SMIM gap analysis.

M4.4-T2: Benchmark y*_{i,t} = U α_{t|t} (filtered mean) with modal attribution.
"""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    FilteredState,
    GapResult,
    ModalFrame,
)


class ModalBenchmark:
    """Computes investment gaps against the filtered state benchmark.

    The benchmark is y*_{i,t} = U_{i,·} α_{t|t} (filtered, not predicted).
    Modal attribution decomposes each gap into per-mode contributions:

        Δ_{i,t,k} = U_{ik} (α_{t|t-1,k} - α_{t|t,k})

    summing over k gives the total gap.
    """

    @property
    def benchmark_class(self) -> BenchmarkClass:
        return BenchmarkClass.MODAL

    def compute(
        self,
        observations: np.ndarray,
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors,
    ) -> GapResult:
        """Compute modal benchmark and per-mode gap attribution.

        Args:
            observations:   Either (N, T) or (T, N) observation matrix.
            filtered_state: FilteredState from Kalman/Kim filter.
            modal_frame:    ModalFrame with basis U of shape (N, K).
            actors:         ActorRegistry (unused here).

        Returns:
            GapResult with gaps, benchmarks (N, T) and modal_attribution (N, T, K).
        """
        U = modal_frame.basis  # (N, K)
        alpha_s = filtered_state.alpha_filtered   # (T, K) — smoothed/filtered states
        alpha_p = filtered_state.alpha_predicted  # (T, K) — predicted states

        # Modal attribution: per-mode contribution to prediction-vs-filtered gap
        # diff[t, k] = alpha_p[t, k] - alpha_s[t, k]
        diff = alpha_p - alpha_s  # (T, K)
        # attribution[i, t, k] = U[i, k] * diff[t, k]
        attribution = U[:, np.newaxis, :] * diff[np.newaxis, :, :]  # (N, T, K)

        # Benchmark: filtered state projected into observation space
        benchmarks_TN = alpha_s @ U.T  # (T, N)

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
            modal_attribution=attribution,
        )
