"""OOS evaluation metrics for SMIM validation."""

from __future__ import annotations

import numpy as np
import scipy.stats


def oos_r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """R² = 1 - SS_res/SS_tot. Returns 0.0 if SS_tot == 0."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root Mean Squared Error."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def spearman_rho(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation. Returns 0.0 on error."""
    try:
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        result = scipy.stats.spearmanr(predicted, actual)
        rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
        if np.isnan(rho):
            return 0.0
        return rho
    except Exception:
        return 0.0


def ndcg(scores: np.ndarray, relevance: np.ndarray, k: int | None = None) -> float:
    """Normalized Discounted Cumulative Gain.

    Args:
        scores: Predicted scores used to rank items.
        relevance: Ground-truth relevance values.
        k: Number of top items to consider. None = all items.

    Returns:
        NDCG value in [0, 1].
    """
    scores = np.asarray(scores, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    n = len(scores)
    if k is None:
        k = n

    # Sort by predicted scores descending
    order = np.argsort(scores)[::-1][:k]
    gains = relevance[order]

    # DCG
    positions = np.arange(1, len(gains) + 1)
    discounts = np.log2(positions + 1)
    dcg = float(np.sum(gains / discounts))

    # IDCG: sort relevance descending
    ideal_order = np.argsort(relevance)[::-1][:k]
    ideal_gains = relevance[ideal_order]
    idcg = float(np.sum(ideal_gains / np.log2(np.arange(1, len(ideal_gains) + 1) + 1)))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Harvey, Leybourne & Newbold (1997) modified Diebold-Mariano test.

    d_t = e1_t^2 - e2_t^2; test stat = mean(d) / sqrt(var(d)/n)
    p-value from t(n-1) distribution (two-sided).

    Args:
        errors_1: Forecast errors for model 1 (T,)
        errors_2: Forecast errors for model 2 (T,)
        h: Forecast horizon (default 1)

    Returns:
        (statistic, p_value)
    """
    errors_1 = np.asarray(errors_1, dtype=float)
    errors_2 = np.asarray(errors_2, dtype=float)
    d = errors_1 ** 2 - errors_2 ** 2
    n = len(d)
    if n < 2:
        return 0.0, 1.0

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0.0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d / n)

    # HLN correction factor
    # correction = sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = float(dm_stat * correction)

    p_value = float(2 * scipy.stats.t.sf(abs(dm_stat_corrected), df=n - 1))
    return dm_stat_corrected, p_value


class MetricSuite:
    """Compute a suite of evaluation metrics."""

    def compute_all(self, predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        """Returns dict with keys: r2, rmse, spearman_rho."""
        return {
            "r2": oos_r_squared(predicted, actual),
            "rmse": rmse(predicted, actual),
            "spearman_rho": spearman_rho(predicted, actual),
        }

    def compute_comparative(
        self,
        errors_1: np.ndarray,
        errors_2: np.ndarray,
    ) -> dict[str, float]:
        """Returns dict with keys: dm_stat, dm_pvalue."""
        stat, pval = diebold_mariano_test(errors_1, errors_2)
        return {"dm_stat": stat, "dm_pvalue": pval}
