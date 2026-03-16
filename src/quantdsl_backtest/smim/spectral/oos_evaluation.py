"""Out-of-sample reconstruction evaluation (M3.4-T1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer
from quantdsl_backtest.smim.spectral.mode_selection import CombinedModeSelector


@dataclass
class OOSReconstructionResult:
    per_window_r2: list[float]
    mean_r2: float
    rank_correlations: list[float]  # eigenmode rank correlation between consecutive windows
    n_windows: int


def rolling_oos_reconstruction(
    signals: np.ndarray,              # (T, N) — T time steps, N actors
    decomposer: AbstractSpectralDecomposer,
    mode_selector: CombinedModeSelector | None,
    window_size: int,
    step: int = 1,
    k: int = 10,
) -> OOSReconstructionResult:
    """Rolling out-of-sample reconstruction evaluation.

    For each window [t, t+window_size]:
      1. Fit decomposer on signals[t : t+window_size].
      2. Predict next step by projecting signals[t+window_size] onto basis.
      3. Compare prediction to actual via R^2.
      4. Compute Spearman rank correlation of eigenvalue magnitudes vs previous window.
    """
    signals = np.asarray(signals, dtype=float)
    T, N = signals.shape

    per_window_r2: list[float] = []
    rank_correlations: list[float] = []
    prev_eig_mags: np.ndarray | None = None

    t = 0
    while t + window_size < T:
        window = signals[t : t + window_size]  # (window_size, N)
        # Fit decomposer on window (treat transposed as operator)
        # For spectral decomposers expecting square operators, form correlation/covariance matrix
        # Use windowed data: build N x N covariance matrix as proxy operator
        # Note: window is (window_size, N); form (N, N) operator
        W = window - window.mean(axis=0)
        cov = (W.T @ W) / max(window_size - 1, 1)  # (N, N)

        import scipy.sparse as sparse
        op = sparse.csr_matrix(cov)
        try:
            frame = decomposer.decompose(op, k)
        except Exception:
            t += step
            continue

        if mode_selector is not None:
            try:
                frame, _ = mode_selector.select(frame, window)
            except Exception:
                pass

        # Predict next observation by projecting onto modal basis
        y_actual = signals[t + window_size]  # (N,)
        U = frame.basis  # (N, k*)
        # Projection: y_pred = U @ U^T @ y_actual
        y_pred = U @ (U.T @ y_actual)

        # R^2
        y_mean = y_actual.mean()
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - y_mean) ** 2)
        if ss_tot < 1e-15:
            r2 = 0.0
        else:
            r2 = float(1.0 - ss_res / ss_tot)
        per_window_r2.append(r2)

        # Rank correlation with previous window
        eig_mags = np.abs(frame.eigenvalues)
        if prev_eig_mags is not None:
            # Align lengths by taking min
            min_len = min(len(eig_mags), len(prev_eig_mags))
            if min_len >= 2:
                rho, _ = spearmanr(eig_mags[:min_len], prev_eig_mags[:min_len])
                rank_correlations.append(float(rho) if np.isfinite(rho) else 0.0)
            else:
                rank_correlations.append(0.0)

        prev_eig_mags = eig_mags
        t += step

    n_windows = len(per_window_r2)
    mean_r2 = float(np.mean(per_window_r2)) if per_window_r2 else float("nan")

    return OOSReconstructionResult(
        per_window_r2=per_window_r2,
        mean_r2=mean_r2,
        rank_correlations=rank_correlations,
        n_windows=n_windows,
    )
