"""Kalman filter and EM estimation for the SMIM state-space model.

M4.1-T1: Standard linear Gaussian Kalman filter.
M4.1-T2: RTS smoother + EM estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantdsl_backtest.smim.interfaces import FilteredState, ModalFrame


@dataclass
class KalmanFilter:
    """Linear Gaussian Kalman filter for modal state-space dynamics.

    State equation:   α_t = F α_{t-1} + η,  η ~ N(0, Q)
    Observation eq.:  s_t = U α_t   + ε,   ε ~ N(0, R)

    Attributes:
        F: (K, K) state transition matrix. Default = identity.
        Q: (K, K) state noise covariance. Default = 0.1 * I.
        R: (N, N) observation noise covariance. Default = 0.1 * I.
    """

    F: np.ndarray | None = None
    Q: np.ndarray | None = None
    R: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Core filter                                                          #
    # ------------------------------------------------------------------ #

    def predict(
        self, alpha: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-step prediction.

        Returns:
            (alpha_pred, P_pred)
        """
        alpha_pred = F @ alpha
        P_pred = F @ P @ F.T + Q
        return alpha_pred, P_pred

    def update(
        self,
        alpha_pred: np.ndarray,
        P_pred: np.ndarray,
        obs: np.ndarray,
        U: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Kalman update step.

        Returns:
            (alpha_filt, P_filt, log_lik_contribution)
        """
        N = len(obs)
        K = len(alpha_pred)
        v = obs - U @ alpha_pred                      # innovation
        S = U @ P_pred @ U.T + R                      # innovation covariance
        S_inv = np.linalg.solve(S, np.eye(N))
        K_gain = P_pred @ U.T @ S_inv

        alpha_filt = alpha_pred + K_gain @ v
        P_filt = (np.eye(K) - K_gain @ U) @ P_pred

        sign, logdet = np.linalg.slogdet(S)
        log_lik_contrib = -0.5 * (N * np.log(2.0 * np.pi) + logdet + v @ S_inv @ v)

        return alpha_filt, P_filt, float(log_lik_contrib)

    def filter(self, observations: np.ndarray, modal_frame: ModalFrame) -> FilteredState:
        """Run Kalman filter over full observation sequence.

        Args:
            observations: (T, N) observation matrix.
            modal_frame: ModalFrame with basis (N, K).

        Returns:
            FilteredState containing filtered/predicted states and log-likelihood.
        """
        T, N = observations.shape
        K = modal_frame.K
        U = modal_frame.basis  # (N, K)

        F = self.F if self.F is not None else np.eye(K)
        Q = self.Q if self.Q is not None else np.eye(K) * 0.1
        R = self.R if self.R is not None else np.eye(N) * 0.1

        alpha_filt = np.zeros((T, K))
        alpha_pred = np.zeros((T, K))
        P_filt = np.zeros((T, K, K))
        P_pred_all = np.zeros((T, K, K))
        log_lik = 0.0

        alpha_curr = np.zeros(K)
        P_curr = np.eye(K)

        for t in range(T):
            a_p, P_p = self.predict(alpha_curr, P_curr, F, Q)
            a_f, P_f, ll_contrib = self.update(a_p, P_p, observations[t], U, R)

            alpha_pred[t] = a_p
            alpha_filt[t] = a_f
            P_pred_all[t] = P_p
            P_filt[t] = P_f
            log_lik += ll_contrib

            alpha_curr = a_f
            P_curr = P_f

        return FilteredState(
            alpha_filtered=alpha_filt,
            alpha_predicted=alpha_pred,
            regime_probs=np.ones((T, 1)),
            log_likelihood=log_lik,
            regimes_selected=1,
            params={
                "F": F,
                "Q": Q,
                "R": R,
                "P_filtered": P_filt,
                "P_predicted": P_pred_all,
            },
        )

    # ------------------------------------------------------------------ #
    # RTS smoother                                                          #
    # ------------------------------------------------------------------ #

    def rts_smooth(
        self,
        alpha_filt: np.ndarray,
        P_filt: np.ndarray,
        alpha_pred: np.ndarray,
        P_pred: np.ndarray,
        F: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """RTS backward smoother.

        Args:
            alpha_filt: (T, K) filtered states.
            P_filt:     (T, K, K) filtered covariances.
            alpha_pred: (T, K) predicted states.
            P_pred:     (T, K, K) predicted covariances.
            F:          (K, K) transition matrix.

        Returns:
            (alpha_smooth (T, K), P_smooth (T, K, K))
        """
        T, K = alpha_filt.shape
        alpha_s = alpha_filt.copy()
        P_s = P_filt.copy()

        for t in range(T - 2, -1, -1):
            G = P_filt[t] @ F.T @ np.linalg.solve(P_pred[t + 1], np.eye(K))
            alpha_s[t] += G @ (alpha_s[t + 1] - alpha_pred[t + 1])
            P_s[t] += G @ (P_s[t + 1] - P_pred[t + 1]) @ G.T

        return alpha_s, P_s

    # ------------------------------------------------------------------ #
    # EM estimation                                                         #
    # ------------------------------------------------------------------ #

    def em_estimate(
        self,
        observations: np.ndarray,
        modal_frame: ModalFrame,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> FilteredState:
        """EM estimation of F, Q, R followed by a final filter pass.

        Args:
            observations: (T, N) observation matrix.
            modal_frame: ModalFrame with basis (N, K).
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood.

        Returns:
            FilteredState from the final filter pass, with ll_history in params.
        """
        T, N = observations.shape
        K = modal_frame.K
        U = modal_frame.basis

        # Initialise parameters
        self.F = np.eye(K) * 0.9
        self.Q = np.eye(K) * 0.1
        self.R = np.eye(N) * 0.1

        prev_ll = -np.inf
        ll_history: list[float] = []

        for iteration in range(max_iter):
            # E-step: Kalman filter
            state = self.filter(observations, modal_frame)
            P_filt = state.params["P_filtered"]
            P_pred = state.params["P_predicted"]

            ll_history.append(state.log_likelihood)
            if abs(state.log_likelihood - prev_ll) < tol and iteration > 0:
                break
            prev_ll = state.log_likelihood

            # RTS smoother
            alpha_s, P_s = self.rts_smooth(
                state.alpha_filtered,
                P_filt,
                state.alpha_predicted,
                P_pred,
                self.F,
            )

            # M-step: update F
            cross = sum(
                np.outer(alpha_s[t], alpha_s[t - 1]) + P_s[t] @ self.F
                for t in range(1, T)
            )
            lag0_prev = sum(
                np.outer(alpha_s[t - 1], alpha_s[t - 1]) + P_s[t - 1]
                for t in range(1, T)
            )
            self.F = cross @ np.linalg.solve(lag0_prev + np.eye(K) * 1e-8, np.eye(K))

            # M-step: update Q
            lag0_curr = sum(
                np.outer(alpha_s[t], alpha_s[t]) + P_s[t] for t in range(1, T)
            )
            self.Q = (lag0_curr - self.F @ cross.T) / (T - 1)
            self.Q = (self.Q + self.Q.T) / 2  # symmetrize
            self.Q += np.eye(K) * 1e-8

            # M-step: update R
            resid = observations - alpha_s @ U.T
            self.R = (resid.T @ resid) / T
            self.R = (self.R + self.R.T) / 2
            self.R += np.eye(N) * 1e-8

        final_state = self.filter(observations, modal_frame)
        final_state.params["ll_history"] = ll_history
        return final_state
