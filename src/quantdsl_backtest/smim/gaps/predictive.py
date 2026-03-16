"""Predictive benchmark computation for SMIM gap analysis.

M4.4-T1: One-step-ahead prediction benchmark y*_{i,t} = U α_{t|t-1}.
"""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    FilteredState,
    GapResult,
    ModalFrame,
)


class PredictiveBenchmark:
    """Computes investment gaps against the one-step-ahead predicted benchmark.

    The benchmark for actor i at time t is:
        y*_{i,t} = U_{i,·} α_{t|t-1}

    where α_{t|t-1} is the Kalman predicted state (prior to observing s_t).
    """

    @property
    def benchmark_class(self) -> BenchmarkClass:
        return BenchmarkClass.PREDICTIVE

    def compute(
        self,
        observations: np.ndarray,
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors,
    ) -> GapResult:
        """Compute predictive benchmark and gaps.

        Args:
            observations:   Either (N, T) or (T, N) observation matrix.
            filtered_state: FilteredState from Kalman/Kim filter.
            modal_frame:    ModalFrame with basis U of shape (N, K).
            actors:         ActorRegistry (unused here).

        Returns:
            GapResult with gaps and benchmarks of shape (N, T).
        """
        U = modal_frame.basis  # (N, K)
        # alpha_predicted: (T, K) → benchmark in observation space: (T, N)
        benchmarks_TN = filtered_state.alpha_predicted @ U.T  # (T, N)

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
