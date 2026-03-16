"""Out-of-sample prediction evaluation for the SMIM state-space model.

M4.1-T3: OOS evaluation — R², RMSE, MAE vs. random-walk and historical-mean baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame


@dataclass
class SSMEvaluationResult:
    """Out-of-sample prediction performance of the state-space model.

    Attributes:
        r2_oos:    R² on holdout set vs own mean.
        rmse_oos:  Root mean squared error on holdout set.
        mae_oos:   Mean absolute error on holdout set.
        r2_vs_rw:  R² on holdout vs random-walk (last training value).
        r2_vs_mean: R² on holdout vs historical training mean.
        n_holdout: Number of holdout time steps.
    """

    r2_oos: float
    rmse_oos: float
    mae_oos: float
    r2_vs_rw: float
    r2_vs_mean: float
    n_holdout: int


def oos_prediction_evaluation(
    observations: np.ndarray,
    modal_frame: ModalFrame,
    holdout_fraction: float = 0.2,
    max_em_iter: int = 50,
) -> SSMEvaluationResult:
    """Evaluate OOS prediction quality of a Kalman filter fitted by EM.

    The model is fitted on the training split, then one-step predictions
    from the full-sequence filter pass are compared to the holdout actuals.

    Args:
        observations:    (T, N) observation matrix.
        modal_frame:     ModalFrame with basis (N, K).
        holdout_fraction: Fraction of T reserved for evaluation.
        max_em_iter:     Maximum EM iterations on training data.

    Returns:
        SSMEvaluationResult with performance metrics.
    """
    T = len(observations)
    n_train = int(T * (1.0 - holdout_fraction))
    train = observations[:n_train]
    test = observations[n_train:]

    kf = KalmanFilter()
    kf.em_estimate(train, modal_frame, max_iter=max_em_iter)

    # One-step predictions over the full sequence; use holdout portion
    state = kf.filter(observations, modal_frame)
    pred = state.alpha_predicted[n_train:] @ modal_frame.basis.T  # (n_test, N)
    actual = test  # (n_test, N)

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))

    # vs random walk (last training value repeated)
    rw_pred = np.tile(train[-1], (len(test), 1))
    r2_rw = 1.0 - np.sum((actual - rw_pred) ** 2) / (ss_tot + 1e-12)

    # vs historical mean
    mean_pred = np.tile(train.mean(axis=0), (len(test), 1))
    r2_mean = 1.0 - np.sum((actual - mean_pred) ** 2) / (ss_tot + 1e-12)

    return SSMEvaluationResult(
        r2_oos=float(r2),
        rmse_oos=rmse,
        mae_oos=mae,
        r2_vs_rw=float(r2_rw),
        r2_vs_mean=float(r2_mean),
        n_holdout=len(test),
    )
