"""Kim filter for regime-switching state-space models.

M4.2-T1: Core Kim filter with moment-matching collapse.
M4.2-T2: EM estimation for regime-switching model.
M4.2-T4: Regime diagnostics and A5 verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantdsl_backtest.smim.interfaces import FilteredState, ModalFrame


# ---------------------------------------------------------------------------
# Regime diagnostics result
# ---------------------------------------------------------------------------

@dataclass
class RegimeDiagnostics:
    """Diagnostics for the estimated Markov-switching model.

    Attributes:
        transition_matrix: (M, M) estimated transition probability matrix.
        avg_durations:     (M,) expected regime durations = 1/(1-p_ii).
        a5_passes:         True if all avg_durations >= 8 quarters (Assumption A5).
    """

    transition_matrix: np.ndarray
    avg_durations: np.ndarray
    a5_passes: bool


# ---------------------------------------------------------------------------
# Kim filter
# ---------------------------------------------------------------------------

@dataclass
class KimFilter:
    """Kim (1994) approximate filter for Markov-switching state-space models.

    Uses moment-matching collapse to keep the number of components tractable.

    Attributes:
        n_regimes: Number of Markov regimes M.
    """

    n_regimes: int = 2

    def filter(
        self,
        observations: np.ndarray,
        modal_frame: ModalFrame,
        F_list: list[np.ndarray] | None = None,
        Q_list: list[np.ndarray] | None = None,
        R: np.ndarray | None = None,
        P_trans: np.ndarray | None = None,
    ) -> FilteredState:
        """Run Kim filter over the full observation sequence.

        Args:
            observations: (T, N) observation matrix.
            modal_frame:  ModalFrame with basis (N, K).
            F_list:       List of M (K, K) transition matrices, one per regime.
            Q_list:       List of M (K, K) state noise covariances.
            R:            (N, N) observation noise covariance.
            P_trans:      (M, M) Markov transition probability matrix.

        Returns:
            FilteredState with regime_probs shape (T, M).
        """
        T, N = observations.shape
        K = modal_frame.K
        M = self.n_regimes
        U = modal_frame.basis  # (N, K)

        # Default parameters
        if F_list is None:
            F_list = [np.eye(K) * 0.9 for _ in range(M)]
        if Q_list is None:
            Q_list = [np.eye(K) * 0.1 for _ in range(M)]
        if R is None:
            R = np.eye(N) * 0.1
        if P_trans is None:
            P_trans = np.full((M, M), 1.0 / M)

        # Precompute Woodbury components — amortises O(N^3) R^{-1} over T × M^2 steps,
        # reducing per-step cost from O(N^3) to O(N K^2 + K^3).
        _use_woodbury = False
        _R_inv: np.ndarray | None = None
        _R_inv_U: np.ndarray | None = None
        _Ut_Rinv_U: np.ndarray | None = None
        _logdet_R = 0.0
        try:
            _R_inv = np.linalg.solve(R, np.eye(N))
            # Fall back to direct for near-singular R to avoid catastrophic cancellation.
            if np.max(np.abs(_R_inv)) <= 1e8:
                _R_inv_U = _R_inv @ U          # (N, K)
                _Ut_Rinv_U = U.T @ _R_inv_U   # (K, K)
                _, _logdet_R = np.linalg.slogdet(R)
                _use_woodbury = True
        except np.linalg.LinAlgError:
            pass

        # Per-regime filtered state
        alpha = [np.zeros(K) for _ in range(M)]
        P_cov = [np.eye(K) for _ in range(M)]

        regime_probs = np.zeros((T, M))
        alpha_filt = np.zeros((T, K))
        alpha_pred_arr = np.zeros((T, K))
        log_lik = 0.0

        # Initial regime probabilities: uniform
        xi = np.ones(M) / M

        for t in range(T):
            # For each (i, j) pair: predict from regime i, assume regime j
            alpha_ij = np.zeros((M, M, K))
            P_ij = np.zeros((M, M, K, K))
            lik_ij = np.zeros((M, M))

            for j in range(M):
                for i in range(M):
                    a_p = F_list[j] @ alpha[i]
                    P_p = F_list[j] @ P_cov[i] @ F_list[j].T + Q_list[j]
                    alpha_ij[i, j] = a_p
                    P_ij[i, j] = P_p

                    v = observations[t] - U @ a_p
                    try:
                        if _use_woodbury:
                            # Woodbury: replace O(N^3) inversion with O(K^3 + N K^2).
                            P_inv = np.linalg.solve(P_p, np.eye(K))
                            inner = P_inv + _Ut_Rinv_U  # (K, K)
                            cond = np.linalg.cond(inner)
                            sign_inner, logdet_inner = np.linalg.slogdet(inner)
                            sign_P, logdet_P = np.linalg.slogdet(P_p)

                            if cond <= 1e8 and sign_inner > 0 and sign_P > 0:
                                logdet_S = _logdet_R + logdet_P + logdet_inner
                                # Quadratic form without forming S^{-1}:
                                # v^T S^{-1} v = v^T R^{-1} v
                                #              - (U^T R^{-1} v)^T inner^{-1} (U^T R^{-1} v)
                                Ut_Rinv_v = _R_inv_U.T @ v  # type: ignore[union-attr]  # (K,)
                                R_inv_v = _R_inv @ v         # type: ignore[union-attr]  # (N,)
                                quad = float(v @ R_inv_v) - float(
                                    Ut_Rinv_v @ np.linalg.solve(inner, Ut_Rinv_v)
                                )
                                exponent = -0.5 * (logdet_S + quad)
                                lik_ij[i, j] = np.exp(np.clip(exponent, -700, 700))
                                continue  # skip direct fallback below

                        # Direct O(N^3) path (original, or Woodbury fallback).
                        S = U @ P_p @ U.T + R
                        sign, logdet = np.linalg.slogdet(S)
                        if sign <= 0:
                            lik_ij[i, j] = 1e-300
                        else:
                            S_inv = np.linalg.solve(S, np.eye(N))
                            exponent = -0.5 * (logdet + v @ S_inv @ v)
                            lik_ij[i, j] = np.exp(np.clip(exponent, -700, 700))
                    except np.linalg.LinAlgError:
                        lik_ij[i, j] = 1e-300

            lik_ij = np.clip(lik_ij, 1e-300, None)

            # Joint probability P(z_{t-1}=i, z_t=j | Y_t)
            joint = np.outer(xi, np.ones(M)) * P_trans * lik_ij
            denom = joint.sum()
            if denom < 1e-300:
                denom = 1e-300
            log_lik += np.log(denom)
            joint /= denom

            # Marginal: P(z_t=j | Y_t)
            xi_new = joint.sum(axis=0)
            xi_new = np.clip(xi_new, 0.0, None)
            total = xi_new.sum()
            if total > 1e-300:
                xi_new /= total
            else:
                xi_new = np.ones(M) / M

            # Moment-matching collapse
            for j in range(M):
                if xi_new[j] < 1e-300:
                    continue
                cond_i = joint[:, j] / xi_new[j]
                a_j = sum(cond_i[i] * alpha_ij[i, j] for i in range(M))
                P_j = sum(
                    cond_i[i] * (
                        P_ij[i, j]
                        + np.outer(
                            alpha_ij[i, j] - a_j,
                            alpha_ij[i, j] - a_j,
                        )
                    )
                    for i in range(M)
                )
                alpha[j] = a_j
                P_cov[j] = P_j

            xi = xi_new
            regime_probs[t] = xi

            # Overall filtered mean
            alpha_filt[t] = sum(xi[j] * alpha[j] for j in range(M))
            alpha_pred_arr[t] = alpha_filt[t]  # approximate

        return FilteredState(
            alpha_filtered=alpha_filt,
            alpha_predicted=alpha_pred_arr,
            regime_probs=regime_probs,
            log_likelihood=log_lik,
            regimes_selected=M,
            params={
                "F_list": F_list,
                "Q_list": Q_list,
                "R": R,
                "P_trans": P_trans,
            },
        )

    # ------------------------------------------------------------------ #
    # EM estimation                                                         #
    # ------------------------------------------------------------------ #

    def em_estimate(
        self,
        observations: np.ndarray,
        modal_frame: ModalFrame,
        max_iter: int = 50,
        tol: float = 1e-3,
    ) -> FilteredState:
        """EM estimation for regime-switching state-space model.

        Args:
            observations: (T, N) observation matrix.
            modal_frame:  ModalFrame with basis (N, K).
            max_iter:     Maximum EM iterations.
            tol:          Convergence tolerance on log-likelihood.

        Returns:
            FilteredState from the final filter pass, with ll_history in params.
        """
        T, N = observations.shape
        K = modal_frame.K
        M = self.n_regimes
        U = modal_frame.basis

        # Initialise
        F_list = [np.eye(K) * 0.9 for _ in range(M)]
        Q_list = [np.eye(K) * 0.1 for _ in range(M)]
        R = np.eye(N) * 0.1
        P_trans = np.full((M, M), 1.0 / M)

        prev_ll = -np.inf
        ll_history: list[float] = []

        for iteration in range(max_iter):
            state = self.filter(observations, modal_frame, F_list, Q_list, R, P_trans)
            ll_history.append(state.log_likelihood)

            if abs(state.log_likelihood - prev_ll) < tol and iteration > 0:
                break
            prev_ll = state.log_likelihood

            xi = state.regime_probs  # (T, M)
            alpha_s = state.alpha_filtered

            # Update transition matrix
            for j in range(M):
                for k_reg in range(M):
                    P_trans[j, k_reg] = max(xi[:-1, j].sum() * (1.0 / M), 1e-6)
            row_sums = P_trans.sum(axis=1, keepdims=True)
            P_trans /= np.where(row_sums > 0, row_sums, 1.0)

            # Update F, Q per regime
            for m in range(M):
                w = xi[1:, m]
                w_sum = w.sum()
                if w_sum < 1e-6:
                    continue
                Sxy = sum(
                    w[t] * np.outer(alpha_s[t + 1], alpha_s[t]) for t in range(T - 1)
                )
                Sxx = sum(
                    w[t] * np.outer(alpha_s[t], alpha_s[t]) for t in range(T - 1)
                )
                Sxx += np.eye(K) * 1e-6
                F_list[m] = Sxy @ np.linalg.inv(Sxx)

                resid = np.array(
                    [alpha_s[t + 1] - F_list[m] @ alpha_s[t] for t in range(T - 1)]
                )
                Q_list[m] = (
                    sum(w[t] * np.outer(resid[t], resid[t]) for t in range(T - 1))
                    / (w_sum + 1e-12)
                )
                Q_list[m] = (Q_list[m] + Q_list[m].T) / 2
                Q_list[m] += np.eye(K) * 1e-8

            # Update R
            pred = alpha_s @ U.T
            resid_obs = observations - pred
            R = (resid_obs.T @ resid_obs) / T
            R = (R + R.T) / 2
            R += np.eye(N) * 1e-8

        final = self.filter(observations, modal_frame, F_list, Q_list, R, P_trans)
        final.params["ll_history"] = ll_history
        return final

    # ------------------------------------------------------------------ #
    # Regime diagnostics (M4.2-T4)                                         #
    # ------------------------------------------------------------------ #

    def regime_diagnostics(self, state: FilteredState) -> RegimeDiagnostics:
        """Compute regime persistence diagnostics from a filtered state.

        Args:
            state: FilteredState produced by filter() or em_estimate().

        Returns:
            RegimeDiagnostics with transition matrix, durations, and A5 flag.
        """
        P = state.params["P_trans"]
        durations = np.array(
            [1.0 / max(1.0 - P[i, i], 1e-6) for i in range(len(P))]
        )
        return RegimeDiagnostics(
            transition_matrix=P,
            avg_durations=durations,
            a5_passes=bool(np.all(durations >= 8)),
        )
