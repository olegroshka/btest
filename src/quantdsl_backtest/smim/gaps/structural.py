"""Structural benchmark: stable vs distortionary channel decomposition (M6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantdsl_backtest.smim.interfaces import (
    BenchmarkClass,
    FilteredState,
    GapResult,
    ModalFrame,
)


@dataclass
class ChannelStabilityResult:
    channel: str
    mean_contribution: float
    std_contribution: float
    stability_ratio: float
    is_stable: bool


class ChannelDecomposer:
    """Decomposes channels into stable vs distortionary based on R² consistency."""

    def __init__(self, stability_threshold: float = 1.0, n_windows: int = 5):
        self.stability_threshold = stability_threshold
        self.n_windows = n_windows

    def decompose(
        self,
        channel_matrices: dict[str, np.ndarray],
        observations: np.ndarray,
        n_modes: int = 3,
    ) -> list[ChannelStabilityResult]:
        """For each channel, compute R² contribution across rolling windows.

        For each window w:
          - project observations onto top n_modes eigenvectors of channel_matrix
          - compute R² of this projection vs observations

        Stability = mean_R2 / (std_R2 + eps)
        is_stable = stability_ratio > threshold
        """
        T, N = observations.shape
        window_size = T // self.n_windows
        if window_size < 2:
            window_size = 2

        results = []
        for channel_name, matrix in channel_matrices.items():
            r2_values = []

            for w in range(self.n_windows):
                start = w * window_size
                end = min(start + window_size, T)
                if end <= start:
                    continue
                obs_window = observations[start:end]  # (w_size, N)

                # Get top n_modes eigenvectors of channel_matrix
                k = min(n_modes, matrix.shape[0])
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                    # Take top k by magnitude
                    idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
                    U = eigenvectors[:, idx]  # (N, k)

                    # Project observations onto U
                    proj = obs_window @ U @ U.T  # (w_size, N)

                    # Compute R²
                    ss_res = float(np.sum((obs_window - proj) ** 2))
                    ss_tot = float(np.sum((obs_window - np.mean(obs_window)) ** 2))
                    r2 = 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
                    r2_values.append(r2)
                except np.linalg.LinAlgError:
                    r2_values.append(0.0)

            r2_arr = np.array(r2_values) if r2_values else np.array([0.0])
            mean_c = float(np.mean(r2_arr))
            std_c = float(np.std(r2_arr, ddof=0))
            ratio = mean_c / (std_c + 1e-9)
            is_stable = ratio > self.stability_threshold

            results.append(
                ChannelStabilityResult(
                    channel=channel_name,
                    mean_contribution=mean_c,
                    std_contribution=std_c,
                    stability_ratio=ratio,
                    is_stable=is_stable,
                )
            )

        return results

    def stable_channels(self, results: list[ChannelStabilityResult]) -> list[str]:
        return [r.channel for r in results if r.is_stable]

    def distortionary_channels(self, results: list[ChannelStabilityResult]) -> list[str]:
        return [r.channel for r in results if not r.is_stable]


class StructuralBenchmarkComputer:
    """BenchmarkComputer for BenchmarkClass.STRUCTURAL.

    Uses only stable-channel operator to compute benchmark.
    y*_str = alpha_filtered @ U_stable^T
    where U_stable = eigenvectors of the stable-channel aggregate operator
    """

    benchmark_class = BenchmarkClass.STRUCTURAL

    def __init__(
        self,
        channel_decomposer: ChannelDecomposer | None = None,
        n_modes: int = 3,
    ):
        self.channel_decomposer = channel_decomposer or ChannelDecomposer()
        self.n_modes = n_modes

    def compute(
        self,
        observations: np.ndarray,
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors,
        stable_operator: np.ndarray | None = None,
    ) -> GapResult:
        """Compute structural benchmark and gaps.

        If stable_operator provided:
          U_stable = top n_modes eigenvectors of stable_operator
          benchmark = filtered_state.alpha_filtered @ U_stable.T  (T, N)
        Else:
          benchmark = filtered_state.alpha_filtered @ modal_frame.basis.T

        gaps = observations - benchmark  (T, N) → transposed to (N, T) for GapResult
        """
        N = modal_frame.N

        if stable_operator is not None:
            k = min(self.n_modes, stable_operator.shape[0])
            eigenvalues, eigenvectors = np.linalg.eigh(stable_operator)
            idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
            U_stable = eigenvectors[:, idx]  # (N, k)
        else:
            U_stable = modal_frame.basis  # (N, K)

        # alpha_filtered: (T, K), U_stable: (N, k)
        # For reconstruction we need matching dims — use as projection basis
        # benchmark shape: (T, N)
        alpha = filtered_state.alpha_filtered  # (T, K)

        # Project via U_stable: need alpha to have same dim as U_stable columns
        k_stable = U_stable.shape[1]
        k_alpha = alpha.shape[1]

        if k_stable <= k_alpha:
            alpha_proj = alpha[:, :k_stable]
        else:
            # Pad alpha with zeros
            pad = np.zeros((alpha.shape[0], k_stable - k_alpha))
            alpha_proj = np.concatenate([alpha, pad], axis=1)

        benchmarks_TN = alpha_proj @ U_stable.T  # (T, N)

        # Normalise observations to (T, N)
        if observations.ndim == 2 and observations.shape[0] == N:
            obs_TN = observations.T  # (N, T) → (T, N)
        else:
            obs_TN = observations  # already (T, N)

        gaps_TN = obs_TN - benchmarks_TN  # (T, N)
        gaps = gaps_TN.T       # (N, T)
        benchmarks = benchmarks_TN.T  # (N, T)

        return GapResult(
            gaps=gaps,
            benchmark_class=BenchmarkClass.STRUCTURAL,
            benchmarks=benchmarks,
        )
