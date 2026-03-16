"""
Baseline comparison models for SMIM validation.

M2.3-T1 and M2.3-T2

Two baselines:
- DynamicFactorBaseline: PCA-based dynamic factor model
- VARBaseline: BIC-selected VAR model
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


@dataclass
class BaselineResult:
    """Result from a baseline model evaluation.

    Attributes:
        model_name: Human-readable model identifier.
        r2_oos: Out-of-sample R².
        rmse_oos: Out-of-sample RMSE.
        n_params: Number of estimated model parameters.
        metadata: Additional method-specific information.
    """

    model_name: str
    r2_oos: float
    rmse_oos: float
    n_params: int
    metadata: dict = field(default_factory=dict)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


class DynamicFactorBaseline:
    """PCA-based dynamic factor model baseline.

    Fit on training set, project test set into factor space, reconstruct Y,
    compute OOS R² and RMSE.

    Args:
        n_factors: Number of PCA components / latent factors.
    """

    def __init__(self, n_factors: int = 3) -> None:
        self.n_factors = n_factors

    def fit_and_evaluate(
        self,
        data: pd.DataFrame,
        holdout_fraction: float = 0.2,
    ) -> BaselineResult:
        """Fit PCA on training split, evaluate reconstruction on test split.

        Args:
            data: DataFrame of shape (T, N) — time rows, actor columns.
            holdout_fraction: Fraction of rows reserved for OOS evaluation.

        Returns:
            BaselineResult with finite r2_oos.
        """
        T, N = data.shape
        split = int(T * (1.0 - holdout_fraction))
        if split < 2 or (T - split) < 1:
            raise ValueError("Not enough data for train/test split.")

        train = data.iloc[:split].values.astype(float)
        test = data.iloc[split:].values.astype(float)

        # Fit PCA on training data (mean-center per feature)
        mean = np.nanmean(train, axis=0)
        train_centered = train - mean

        n_components = min(self.n_factors, min(train_centered.shape) - 1)
        n_components = max(n_components, 1)

        pca = PCA(n_components=n_components)
        pca.fit(train_centered)

        # Reconstruct test data
        test_centered = test - mean
        factors = pca.transform(test_centered)
        reconstructed = pca.inverse_transform(factors) + mean

        y_true = test.flatten()
        y_pred = reconstructed.flatten()

        # Handle NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            r2 = float("nan")
            rmse = float("nan")
        else:
            r2 = _r2_score(y_true_clean, y_pred_clean)
            rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

        n_params = n_components * N + n_components  # loadings + factor variances

        return BaselineResult(
            model_name="DynamicFactorModel",
            r2_oos=r2,
            rmse_oos=rmse,
            n_params=n_params,
            metadata={"n_factors": n_components, "explained_variance": list(pca.explained_variance_ratio_)},
        )


class VARBaseline:
    """BIC-selected Vector Autoregression baseline.

    Uses statsmodels VAR with BIC-selected lag order.

    Args:
        max_lag: Maximum lag order for BIC selection.
    """

    def __init__(self, max_lag: int = 4) -> None:
        self.max_lag = max_lag

    def fit_and_evaluate(
        self,
        data: pd.DataFrame,
        holdout_fraction: float = 0.2,
    ) -> BaselineResult:
        """Fit VAR on training split, evaluate one-step OOS predictions.

        Args:
            data: DataFrame of shape (T, N) — time rows, actor columns.
            holdout_fraction: Fraction of rows reserved for OOS evaluation.

        Returns:
            BaselineResult with finite rmse_oos.
        """
        from statsmodels.tsa.vector_ar.var_model import VAR

        T, N = data.shape
        split = int(T * (1.0 - holdout_fraction))
        if split < 5 or (T - split) < 1:
            raise ValueError("Not enough data for train/test split.")

        train = data.iloc[:split].values.astype(float)
        test = data.iloc[split:].values.astype(float)

        # Drop NaN rows from training
        train_clean = train[~np.any(np.isnan(train), axis=1)]
        if len(train_clean) < self.max_lag + 2:
            raise ValueError("Not enough clean training rows for VAR.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(train_clean)
            fit_result = model.fit(ic="bic", maxlags=self.max_lag, trend="c")
            k_ar = max(1, fit_result.k_ar)

        # OOS one-step-ahead predictions
        preds = []
        # Build rolling context starting from last k_ar rows of training
        context = list(train_clean[-k_ar:])

        for t in range(len(test)):
            context_arr = np.array(context[-k_ar:])
            forecast = fit_result.forecast(context_arr, steps=1)[0]
            preds.append(forecast)
            context.append(test[t])

        preds_arr = np.array(preds)
        y_true = test.flatten()
        y_pred = preds_arr.flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            r2 = float("nan")
            rmse = float("nan")
        else:
            r2 = _r2_score(y_true_clean, y_pred_clean)
            rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

        n_params = k_ar * N * N + N  # VAR coefficients + intercepts

        return BaselineResult(
            model_name="VAR",
            r2_oos=r2,
            rmse_oos=rmse,
            n_params=n_params,
            metadata={"k_ar": k_ar},
        )
