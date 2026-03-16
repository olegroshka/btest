"""BIC/MDL-based regime count selection for SMIM state-space models.

M4.2-T3: Select optimal number of regimes using BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.interfaces import ModalFrame


@dataclass
class RegimeSelectionResult:
    """Result of regime count selection.

    Attributes:
        selected_m:    The optimal number of regimes (minimises BIC).
        bic_scores:    BIC score for each candidate M.
        mdl_scores:    MDL scores (currently same as BIC).
        delta_bic:     Difference between best and second-best BIC scores.
        gate_g4_passes: True if delta_bic > 10 (strong evidence for selection).
    """

    selected_m: int
    bic_scores: dict[int, float]
    mdl_scores: dict[int, float]
    delta_bic: float
    gate_g4_passes: bool


def select_regime_count(
    observations: np.ndarray,
    modal_frame: ModalFrame,
    candidates: list[int] | None = None,
    method: str = "bic",
    max_em_iter: int = 30,
) -> RegimeSelectionResult:
    """Select the optimal number of Markov regimes using BIC.

    For M=1 a standard KalmanFilter is used; for M>1 a KimFilter is used.
    The number of free parameters is approximated as:
        n_params = M * (K² + K²) + M² + N²

    Args:
        observations: (T, N) observation matrix.
        modal_frame:  ModalFrame with basis (N, K).
        candidates:   List of regime counts to evaluate (default [1,2,3,4]).
        method:       Selection criterion — currently only "bic" is supported.
        max_em_iter:  Maximum EM iterations per candidate.

    Returns:
        RegimeSelectionResult with selected M and all BIC scores.
    """
    if candidates is None:
        candidates = [1, 2, 3, 4]

    T, N = observations.shape
    K = modal_frame.K
    bic_scores: dict[int, float] = {}

    for M in candidates:
        if M == 1:
            kf: KalmanFilter | KimFilter = KalmanFilter()
            state = kf.em_estimate(observations, modal_frame, max_iter=max_em_iter)
        else:
            kf = KimFilter(n_regimes=M)
            state = kf.em_estimate(observations, modal_frame, max_iter=max_em_iter)

        # Parameter count: M × (K² transition + K² noise) + M² (Markov trans) + N² obs noise
        n_params = M * (K ** 2 + K ** 2) + M ** 2 + N ** 2
        bic = -2.0 * state.log_likelihood + n_params * np.log(T)
        bic_scores[M] = float(bic)

    # Select M with minimum BIC
    selected_m = min(bic_scores, key=lambda m: bic_scores[m])
    sorted_scores = sorted(bic_scores.values())
    delta_bic = sorted_scores[1] - sorted_scores[0] if len(sorted_scores) > 1 else 0.0

    return RegimeSelectionResult(
        selected_m=selected_m,
        bic_scores=bic_scores,
        mdl_scores=dict(bic_scores),  # same for now
        delta_bic=float(delta_bic),
        gate_g4_passes=(delta_bic > 10.0),
    )
