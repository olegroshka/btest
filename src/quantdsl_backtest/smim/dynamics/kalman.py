"""Kalman filter and EM estimation for the SMIM state-space model.

M4.1-T1: Standard linear Gaussian Kalman filter.
M4.1-T2: RTS smoother + EM estimation.
"""

from __future__ import annotations

import warnings
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

    def _woodbury_gain(
        self,
        U: np.ndarray,
        P_pred: np.ndarray,
        R_inv: np.ndarray,
        R_inv_U: np.ndarray,
        Ut_Rinv_U: np.ndarray,
        logdet_R: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute Kalman gain and S^{-1} via the Woodbury identity.

        Applies the matrix inversion lemma:
            S^{-1} = R^{-1} − R^{-1} U (P^{-1} + U^T R^{-1} U)^{-1} U^T R^{-1}

        This reduces the per-step cost from O(N^3) to O(N K^2 + K^3), since the
        inner matrix (P^{-1} + U^T R^{-1} U) is K×K (typically 15×15).

        Args:
            U:          (N, K) modal frame basis.
            P_pred:     (K, K) predicted state covariance.
            R_inv:      (N, N) precomputed R^{-1} (constant across time steps).
            R_inv_U:    (N, K) precomputed R^{-1} U (constant across time steps).
            Ut_Rinv_U:  (K, K) precomputed U^T R^{-1} U (constant across time steps).
            logdet_R:   Precomputed log|R| (constant across time steps).

        Returns:
            (K_gain, S_inv, logdet_S) where K_gain is (K, N), S_inv is (N, N), and
            logdet_S = log|S| computed via the matrix determinant lemma.

        Notes:
            If cond(P^{-1} + U^T R^{-1} U) > 1e8, falls back to least-squares
            for the K×K solve and emits a RuntimeWarning.
        """
        K = U.shape[1]

        P_inv = np.linalg.solve(P_pred, np.eye(K))
        inner = P_inv + Ut_Rinv_U  # (K, K)

        # Numerical safety: if inner is ill-conditioned use lstsq instead of solve.
        cond = np.linalg.cond(inner)
        if cond > 1e8:
            warnings.warn(
                f"Woodbury: inner matrix ill-conditioned (cond={cond:.2e}); "
                "using least-squares fallback.",
                RuntimeWarning,
                stacklevel=4,
            )
            inner_inv_UtRinv, _, _, _ = np.linalg.lstsq(inner, R_inv_U.T, rcond=None)
        else:
            inner_inv_UtRinv = np.linalg.solve(inner, R_inv_U.T)  # (K, N)

        # S^{-1} = R^{-1} − R^{-1} U inner^{-1} U^T R^{-1}
        S_inv = R_inv - R_inv_U @ inner_inv_UtRinv  # (N, N)
        K_gain = P_pred @ U.T @ S_inv               # (K, N)

        # log|S| via the matrix determinant lemma:
        # det(R + U P U^T) = det(R) · det(P) · det(P^{-1} + U^T R^{-1} U)
        sign_inner, logdet_inner = np.linalg.slogdet(inner)
        sign_P, logdet_P = np.linalg.slogdet(P_pred)
        logdet_S = logdet_R + logdet_P + logdet_inner

        return K_gain, S_inv, logdet_S

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

        # Precompute Woodbury components — amortises O(N^3) R^{-1} computation
        # over T time steps, reducing per-step cost from O(N^3) to O(N K^2 + K^3).
        use_woodbury = False
        R_inv: np.ndarray | None = None
        R_inv_U: np.ndarray | None = None
        Ut_Rinv_U: np.ndarray | None = None
        logdet_R = 0.0
        try:
            R_inv = np.linalg.solve(R, np.eye(N))
            # Fall back to direct for near-singular R (e.g. R = 1e-12 * I),
            # where Woodbury suffers catastrophic cancellation.
            if np.max(np.abs(R_inv)) <= 1e8:
                R_inv_U = R_inv @ U          # (N, K)
                Ut_Rinv_U = U.T @ R_inv_U   # (K, K)
                _, logdet_R = np.linalg.slogdet(R)
                use_woodbury = True
        except np.linalg.LinAlgError:
            pass

        alpha_filt = np.zeros((T, K))
        alpha_pred = np.zeros((T, K))
        P_filt = np.zeros((T, K, K))
        P_pred_all = np.zeros((T, K, K))
        log_lik = 0.0

        alpha_curr = np.zeros(K)
        P_curr = np.eye(K)

        for t in range(T):
            a_p, P_p = self.predict(alpha_curr, P_curr, F, Q)

            if use_woodbury:
                v = observations[t] - U @ a_p
                K_gain, S_inv, logdet_S = self._woodbury_gain(
                    U, P_p, R_inv, R_inv_U, Ut_Rinv_U, logdet_R  # type: ignore[arg-type]
                )
                a_f = a_p + K_gain @ v
                P_f = (np.eye(K) - K_gain @ U) @ P_p
                ll_contrib = -0.5 * (
                    N * np.log(2.0 * np.pi) + logdet_S + float(v @ S_inv @ v)
                )
            else:
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
            # Correct lag-one smoothed cross-covariance via RTS gain:
            #   P_{t,t-1|T} = P_s[t] @ G_{t-1}^T
            # where G_{t-1} = P_filt[t-1] @ F^T @ P_pred[t]^{-1}.
            cross = np.zeros((K, K))
            lag0_prev = np.zeros((K, K))
            for t in range(1, T):
                G_tm1 = P_filt[t - 1] @ self.F.T @ np.linalg.solve(
                    P_pred[t] + np.eye(K) * 1e-8, np.eye(K)
                )
                P_lag_t = P_s[t] @ G_tm1.T           # P_{t, t-1 | T}
                cross += np.outer(alpha_s[t], alpha_s[t - 1]) + P_lag_t
                lag0_prev += np.outer(alpha_s[t - 1], alpha_s[t - 1]) + P_s[t - 1]
            self.F = cross @ np.linalg.solve(lag0_prev + np.eye(K) * 1e-8, np.eye(K))

            # M-step: update Q
            lag0_curr = sum(
                np.outer(alpha_s[t], alpha_s[t]) + P_s[t] for t in range(1, T)
            )
            self.Q = (lag0_curr - self.F @ cross.T) / max(T - 1, 1)
            self.Q = (self.Q + self.Q.T) / 2  # symmetrize
            self.Q += np.eye(K) * 1e-8

            # M-step: update R (Shumway-Stoffer formula)
            # R_new = (1/T) Σ_t [r_t r_t^T + U P_s[t] U^T]
            # The correction U P_s_mean U^T accounts for smoothing uncertainty;
            # omitting it biases R downward by exactly U @ P_s @ U^T.
            resid = observations - alpha_s @ U.T
            P_s_mean = P_s.mean(axis=0)   # (K, K)
            self.R = (resid.T @ resid) / T + U @ P_s_mean @ U.T
            self.R = (self.R + self.R.T) / 2
            self.R += np.eye(N) * 1e-8

        final_state = self.filter(observations, modal_frame)
        final_state.params["ll_history"] = ll_history
        return final_state
