"""Actor-level state-space model for small actor sets (N' <= 50)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


@dataclass
class ActorLevelConfig:
    state_dim: int = 3
    n_iter_em: int = 10
    max_actors: int = 50


@dataclass
class ActorLevelResult:
    fitted: bool
    residual_variance: float
    modal_residual_variance: float
    incremental_r2: float
    n_actors: int
    n_timesteps: int
    converged: bool
    metadata: dict = field(default_factory=dict)


class ActorLevelSSM:
    """Simplified actor-level SSM for N' <= 50 actors.

    State equation (vectorised):
        H_t = F_full H_{t-1} + w_t,  w_t ~ N(0, Q_full)
    Observation equation:
        Y_t = C_full H_t + v_t,       v_t ~ N(0, R_full)

    H_t = [h_{1,t}; ...; h_{N',t}] shape (N'*d,)
    Y_t shape (N',)
    """

    def __init__(self, config: ActorLevelConfig | None = None):
        self.config = config or ActorLevelConfig()

    def fit(
        self,
        observations: np.ndarray,
        modal_frame: ModalFrame,
        adjacency: np.ndarray | None = None,
    ) -> ActorLevelResult:
        """Fit actor-level SSM and return ActorLevelResult.

        Args:
            observations: (T, N') observation matrix.
            modal_frame: ModalFrame for comparison.
            adjacency: (N', N') adjacency matrix. If None, use identity.

        Returns:
            ActorLevelResult with performance metrics.
        """
        T, N = observations.shape
        d = self.config.state_dim

        if N > self.config.max_actors:
            raise ValueError(
                f"Actor count {N} exceeds max_actors={self.config.max_actors}. "
                "ActorLevelSSM is only tractable for small actor sets."
            )

        if adjacency is None:
            adjacency = np.eye(N)

        # Build vectorised system matrices
        F_full, C_full = self._build_system_matrices(N, d, adjacency)

        # Create a ModalFrame-like object for the vectorised system
        # The "basis" is C_full.T so that C_full @ state = C_full @ I_state
        # KalmanFilter uses: obs = U @ alpha + noise, so U = C_full (N, N*d)
        vect_modal_frame = ModalFrame(
            basis=C_full.T,  # (N*d, N) — KalmanFilter expects (state_dim, obs_dim).T = (obs_dim, state_dim)
            eigenvalues=np.ones(N * d, dtype=complex),
            method=DecompositionMethod.SCHUR,
        )
        # KalmanFilter expects observations (T, N) and basis (N, K)
        # Here N_obs = N, K_state = N*d, basis = C_full shape (N, N*d)
        vect_modal_frame_correct = ModalFrame(
            basis=C_full,  # (N, N*d)
            eigenvalues=np.ones(N * d, dtype=complex),
            method=DecompositionMethod.SCHUR,
        )

        kf = KalmanFilter(F=F_full, Q=np.eye(N * d) * 0.1, R=np.eye(N) * 0.1)

        # Run EM to fit parameters
        converged = False
        prev_ll = -np.inf
        for _ in range(self.config.n_iter_em):
            state = kf.filter(observations, vect_modal_frame_correct)
            if abs(state.log_likelihood - prev_ll) < 1e-4 and prev_ll != -np.inf:
                converged = True
                break
            prev_ll = state.log_likelihood

            # Simple M-step: update R only (keep F fixed for tractability)
            reconstructed = state.alpha_filtered @ C_full.T  # (T, N)
            resid = observations - reconstructed
            kf.R = (resid.T @ resid) / T + np.eye(N) * 1e-8

        # Final filter pass
        state = kf.filter(observations, vect_modal_frame_correct)
        reconstructed = state.alpha_filtered @ C_full.T  # (T, N)

        # Residual variance of actor-level model
        resid_actor = observations - reconstructed
        residual_var = float(np.mean(resid_actor ** 2))

        # Modal-only reconstruction: project via modal_frame
        K_modal = modal_frame.K
        U_modal = modal_frame.basis  # (N, K_modal)
        # Use least-squares projection of observations onto modal basis
        # alpha_modal = obs @ U_modal  (T, K_modal) — simple projection
        alpha_modal = observations @ U_modal  # (T, K_modal)
        modal_reconstructed = alpha_modal @ U_modal.T  # (T, N)
        resid_modal = observations - modal_reconstructed
        modal_residual_var = float(np.mean(resid_modal ** 2))

        # R² values
        ss_tot = float(np.var(observations))
        r2_actor = max(0.0, 1.0 - residual_var / ss_tot) if ss_tot > 0 else 0.0
        r2_modal = max(0.0, 1.0 - modal_residual_var / ss_tot) if ss_tot > 0 else 0.0
        incremental_r2 = r2_actor - r2_modal

        return ActorLevelResult(
            fitted=True,
            residual_variance=residual_var,
            modal_residual_variance=modal_residual_var,
            incremental_r2=incremental_r2,
            n_actors=N,
            n_timesteps=T,
            converged=converged,
            metadata={
                "r2_actor": r2_actor,
                "r2_modal": r2_modal,
                "log_likelihood": state.log_likelihood,
            },
        )

    def _build_system_matrices(
        self, N: int, d: int, adjacency: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (F_full, C_full) for the vectorised system.

        F_full shape: (N*d, N*d) — block matrix encoding local dynamics + interactions
        C_full shape: (N, N*d)   — block-diagonal observation matrix

        Each actor block M_i is a d×d local transition.
        Interaction term: A_{ji} * W where W is a d×d coupling matrix.
        """
        state_dim = N * d

        # Local transition M: d×d random stable matrix (scaled identity for simplicity)
        M = np.eye(d) * 0.9

        # Coupling weight matrix W: d×d
        W = np.eye(d) * 0.1

        # Build F_full as (N*d, N*d) block matrix
        F_full = np.zeros((state_dim, state_dim))
        for i in range(N):
            # Diagonal block: M
            F_full[i * d:(i + 1) * d, i * d:(i + 1) * d] = M
            # Off-diagonal blocks: A[i, j] * W
            for j in range(N):
                if i != j and abs(adjacency[i, j]) > 1e-12:
                    F_full[i * d:(i + 1) * d, j * d:(j + 1) * d] += (
                        adjacency[i, j] * W
                    )

        # C_full: block-diagonal observation matrix
        # Each actor observes via U_i = e_1^T (first element of state)
        # For simplicity use U_i = [1, 0, ..., 0] (1×d)
        C_full = np.zeros((N, state_dim))
        for i in range(N):
            C_full[i, i * d] = 1.0

        return F_full, C_full
