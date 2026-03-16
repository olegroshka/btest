"""Gap persistence and mean-reversion analysis.

M5.4-T1
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from scipy.sparse import csr_matrix


@dataclass
class MeanReversionResult:
    """Mean-reversion statistics for one actor's gap series."""

    signal_id: str  # actor or aggregate
    half_life_quarters: float  # from AR(1): HL = -log(2)/log(rho)
    ar1_coef: float
    reversion_rate: float  # fraction of large gaps that revert within horizon
    predictive_r2: float  # R² of top-decile gap at t predicting correction at t+h
    metadata: dict = field(default_factory=dict)


def gap_mean_reversion(
    gaps: np.ndarray,  # (N, T)
    threshold_sigma: float = 1.5,
    horizon_quarters: int = 8,
    h: int = 4,  # prediction horizon
) -> dict[str, MeanReversionResult]:
    """For each actor, fit AR(1) and compute reversion stats.

    Args:
        gaps: (N, T) array of investment gaps
        threshold_sigma: Multiple of std for defining "large" gaps
        horizon_quarters: Horizon for reversion rate calculation
        h: Prediction horizon for predictive R²

    Returns:
        Dict mapping actor index string to MeanReversionResult
    """
    gaps = np.asarray(gaps, dtype=float)
    N, T = gaps.shape
    results = {}

    for i in range(N):
        series = gaps[i, :]
        actor_id = str(i)

        if T < 4:
            results[actor_id] = MeanReversionResult(
                signal_id=actor_id,
                half_life_quarters=float("nan"),
                ar1_coef=0.0,
                reversion_rate=0.0,
                predictive_r2=0.0,
                metadata={"reason": "insufficient_data"},
            )
            continue

        std = float(np.std(series))
        if std < 1e-10:
            results[actor_id] = MeanReversionResult(
                signal_id=actor_id,
                half_life_quarters=float("nan"),
                ar1_coef=0.0,
                reversion_rate=0.0,
                predictive_r2=0.0,
                metadata={"reason": "constant_series"},
            )
            continue

        # AR(1) fit: series[1:] ~ rho * series[:-1]
        x = series[:-1]
        y = series[1:]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        ar1_coef = float(slope)

        # Half-life: HL = -log(2) / log(|rho|)
        if abs(ar1_coef) >= 1.0 or abs(ar1_coef) < 1e-10:
            half_life = float("inf") if abs(ar1_coef) >= 1.0 else 0.0
        else:
            half_life = float(-np.log(2) / np.log(abs(ar1_coef)))

        # Reversion rate: fraction of large gaps that revert within horizon_quarters
        large_gap_mask = np.abs(series) > threshold_sigma * std
        reversion_count = 0
        total_large = 0
        for t in range(T - horizon_quarters):
            if large_gap_mask[t]:
                total_large += 1
                # Check if sign reverses within horizon
                future = series[t + 1: t + horizon_quarters + 1]
                if len(future) > 0 and np.any(np.sign(future) != np.sign(series[t])):
                    reversion_count += 1

        reversion_rate = float(reversion_count / total_large) if total_large > 0 else 0.0

        # Predictive R²: top decile of |gap| at t predicts correction at t+h
        # Positive gap at t → correction = series[t+h] < series[t]
        # Simple rule: sign(series[t]) predicts -sign(series[t+h])
        if T > h + 1:
            threshold_decile = np.percentile(np.abs(series[:T - h]), 90)
            top_decile_mask = np.abs(series[:T - h]) >= threshold_decile

            if top_decile_mask.sum() >= 2:
                actual_change = series[h:] - series[:T - h]  # (T-h,)
                # Predicted: correction = -series[t] (mean-reversion prediction)
                predicted_change = -series[:T - h]
                # Filter to top decile
                a = actual_change[top_decile_mask]
                p = predicted_change[top_decile_mask]
                ss_res = np.sum((a - p) ** 2)
                ss_tot = np.sum((a - np.mean(a)) ** 2)
                predictive_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                predictive_r2 = 0.0
        else:
            predictive_r2 = 0.0

        results[actor_id] = MeanReversionResult(
            signal_id=actor_id,
            half_life_quarters=half_life,
            ar1_coef=ar1_coef,
            reversion_rate=reversion_rate,
            predictive_r2=predictive_r2,
            metadata={"std": std, "large_gap_count": total_large},
        )

    return results


@dataclass
class DiffusionResult:
    """Diffusion statistics for one actor."""

    actor_id: str
    mean_neighbor_lag: float  # mean lag-1 cross-correlation with neighbors
    diffusion_coefficient: float  # slope of regression: neighbor_gap[t+1] ~ actor_gap[t]
    metadata: dict = field(default_factory=dict)


def gap_diffusion(
    gaps: np.ndarray,  # (N, T)
    adjacency: csr_matrix,  # (N, N)
    actors: list[str],
    h: int = 1,
) -> list[DiffusionResult]:
    """For each actor, test if their large gap predicts neighbor gaps at t+h.

    Args:
        gaps: (N, T) array of investment gaps
        adjacency: (N, N) sparse adjacency matrix
        actors: List of N actor identifiers
        h: Prediction horizon (default 1)

    Returns:
        List of DiffusionResult, one per actor
    """
    gaps = np.asarray(gaps, dtype=float)
    N, T = gaps.shape
    adj_dense = adjacency.toarray()
    results = []

    for i in range(N):
        actor_id = actors[i] if i < len(actors) else str(i)

        # Find neighbors (actors j where adj[j, i] > 0 or adj[i, j] > 0)
        neighbor_idx = list(
            set(np.where(adj_dense[i, :] > 0)[0].tolist())
            | set(np.where(adj_dense[:, i] > 0)[0].tolist())
        )
        neighbor_idx = [j for j in neighbor_idx if j != i]

        if not neighbor_idx or T <= h:
            results.append(DiffusionResult(
                actor_id=actor_id,
                mean_neighbor_lag=0.0,
                diffusion_coefficient=0.0,
                metadata={"n_neighbors": 0},
            ))
            continue

        # actor i's gap at t predicts neighbors' gaps at t+h
        actor_gaps_t = gaps[i, :T - h]  # (T-h,)

        slopes = []
        lag_corrs = []
        for j in neighbor_idx:
            neighbor_gaps_th = gaps[j, h:]  # (T-h,)
            # Lag-1 cross-correlation
            if np.std(actor_gaps_t) > 1e-10 and np.std(neighbor_gaps_th) > 1e-10:
                lag_corr = float(np.corrcoef(actor_gaps_t, neighbor_gaps_th)[0, 1])
                if not np.isnan(lag_corr):
                    lag_corrs.append(lag_corr)
            # Regression slope
            if np.std(actor_gaps_t) > 1e-10:
                result = scipy.stats.linregress(actor_gaps_t, neighbor_gaps_th)
                slopes.append(float(result.slope))

        mean_lag = float(np.mean(lag_corrs)) if lag_corrs else 0.0
        diff_coef = float(np.mean(slopes)) if slopes else 0.0

        results.append(DiffusionResult(
            actor_id=actor_id,
            mean_neighbor_lag=mean_lag,
            diffusion_coefficient=diff_coef,
            metadata={"n_neighbors": len(neighbor_idx)},
        ))

    return results
