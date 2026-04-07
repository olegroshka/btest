"""
Kalman Smoother — lightweight scalar/vector version for Index Directional.

Stripped from smim/dynamics/kalman.py — no ModalFrame, no EM, no Woodbury.
Works directly on pd.Series or 1-D / 2-D np.ndarray.

Typical uses in this strategy
─────────────────────────────
  1. Denoise TKAN cumulative prediction before thresholding
     → replaces raw 5-day sum with a smoother trend estimate

  2. Denoise IVol before z-scoring
     → reduces regime flip-flop on noisy vol spikes

  3. Track a dynamic "bias" term on top of TKAN raw output
     → state = [level, trend], observation = tkan_pred

API
───
  ks = ScalarKalman(obs_noise=0.5, state_noise=0.05)
  smoothed = ks.smooth(series)          # offline — uses full future data
  online   = ks.filter_online(series)   # causal — safe for backtesting
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ScalarKalman:
    """Lightweight scalar (or low-dim) Kalman filter/smoother.

    State-space model
    -----------------
      α_t  = F α_{t-1} + η,   η ~ N(0, Q)      [state transition]
      y_t  = H α_t    + ε,   ε ~ N(0, R)       [observation]

    Defaults are single-state (K=1, H=[[1]]):
      - F = [[persistence]]      (AR(1) coefficient)
      - Q = [[state_noise]]      (process noise variance)
      - R = [[obs_noise]]        (observation noise variance)

    For a level+trend model, pass K=2 (see make_level_trend()).

    Parameters
    ----------
    obs_noise : float
        Observation noise variance R. Higher → trust data less → smoother output.
    state_noise : float
        State process noise Q. Higher → state tracks fast changes.
    persistence : float
        Diagonal of F (AR coefficient). 1.0 = random walk, <1 = mean-reverting.
    K : int
        State dimension. 1 = level only, 2 = level + trend.
    """

    def __init__(
        self,
        obs_noise: float = 0.5,
        state_noise: float = 0.05,
        persistence: float = 0.98,
        K: int = 1,
    ) -> None:
        self.R = np.array([[obs_noise]])          # (1, 1) — scalar obs
        self.K = K

        if K == 1:
            self.F = np.array([[persistence]])
            self.Q = np.array([[state_noise]])
            self.H = np.array([[1.0]])
        elif K == 2:
            # level + trend model: α = [level, trend]
            # level_{t} = level_{t-1} + trend_{t-1}
            # trend_{t} = persistence * trend_{t-1}
            self.F = np.array([[1.0, 1.0], [0.0, persistence]])
            self.Q = np.diag([state_noise, state_noise * 0.1])
            self.H = np.array([[1.0, 0.0]])   # observe level only
        else:
            raise ValueError("K must be 1 or 2 for ScalarKalman. For K>2 use smim KalmanFilter.")

    # ------------------------------------------------------------------ #
    # Core predict / update (same math as smim KalmanFilter)              #
    # ------------------------------------------------------------------ #

    def _predict(
        self,
        alpha: np.ndarray,
        P: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_p = self.F @ alpha
        P_p = self.F @ P @ self.F.T + self.Q
        return a_p, P_p

    def _update(
        self,
        a_p: np.ndarray,
        P_p: np.ndarray,
        obs: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        v = obs - self.H @ a_p                           # scalar innovation
        S = self.H @ P_p @ self.H.T + self.R             # (1,1)
        K_gain = P_p @ self.H.T / S[0, 0]               # (K,1)
        a_f = a_p + K_gain[:, 0] * v[0]
        P_f = (np.eye(self.K) - K_gain @ self.H) @ P_p
        return a_f, P_f

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def filter_online(self, series: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        """Causal forward-only Kalman filter. Safe for backtesting.

        Each output value at time t uses only data up to t (no lookahead).

        Parameters
        ----------
        series : pd.Series or 1-D np.ndarray
            Input signal (NaN values are skipped — state propagated, not updated).

        Returns
        -------
        Same type as input, with filtered (denoised) values.
        """
        values = np.asarray(series, dtype=float)
        T = len(values)
        out = np.full(T, np.nan)

        alpha = np.zeros(self.K)
        P = np.eye(self.K)

        for t in range(T):
            a_p, P_p = self._predict(alpha, P)
            if np.isnan(values[t]):
                alpha, P = a_p, P_p
                continue
            alpha, P = self._update(a_p, P_p, values[t])
            out[t] = (self.H @ alpha)[0]   # extract observed-state component

        if isinstance(series, pd.Series):
            return pd.Series(out, index=series.index, name=series.name)
        return out

    def smooth(self, series: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        """Offline RTS smoother. Uses full future data — for research only.

        NOT safe for live backtesting (lookahead bias).
        Use for: comparing filter quality, parameter tuning, charts.

        Returns
        -------
        Same type as input, with smoothed values.
        """
        values = np.asarray(series, dtype=float)
        T = len(values)

        # Forward pass
        alphas_f = np.zeros((T, self.K))
        alphas_p = np.zeros((T, self.K))
        Ps_f = np.zeros((T, self.K, self.K))
        Ps_p = np.zeros((T, self.K, self.K))

        alpha = np.zeros(self.K)
        P = np.eye(self.K)

        for t in range(T):
            a_p, P_p = self._predict(alpha, P)
            alphas_p[t] = a_p
            Ps_p[t] = P_p
            if np.isnan(values[t]):
                alpha, P = a_p, P_p
            else:
                alpha, P = self._update(a_p, P_p, values[t])
            alphas_f[t] = alpha
            Ps_f[t] = P

        # RTS backward pass (identical to smim rts_smooth)
        alphas_s = alphas_f.copy()
        Ps_s = Ps_f.copy()
        for t in range(T - 2, -1, -1):
            G = Ps_f[t] @ self.F.T @ np.linalg.solve(
                Ps_p[t + 1] + np.eye(self.K) * 1e-10, np.eye(self.K)
            )
            alphas_s[t] += G @ (alphas_s[t + 1] - alphas_p[t + 1])
            Ps_s[t] += G @ (Ps_s[t + 1] - Ps_p[t + 1]) @ G.T

        out = (alphas_s @ self.H.T)[:, 0]   # project to observation space

        if isinstance(series, pd.Series):
            return pd.Series(out, index=series.index, name=series.name)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_tkan_smoother(obs_noise: float = 0.3, state_noise: float = 0.03) -> ScalarKalman:
    """Kalman tuned to denoise TKAN cumulative 5-day predictions.

    obs_noise=0.3  → moderate trust in raw TKAN output (~0.3 std units)
    state_noise=0.03 → slow-moving underlying trend

    Use::
        ks = make_tkan_smoother()
        tkan_smooth = ks.filter_online(df["tkan_pred"])
        signal = (tkan_smooth >= 0.0).astype(int)
    """
    return ScalarKalman(obs_noise=obs_noise, state_noise=state_noise, persistence=0.97, K=1)


def make_ivol_smoother(obs_noise: float = 1.0, state_noise: float = 0.1) -> ScalarKalman:
    """Kalman tuned to denoise IVol before z-scoring.

    obs_noise=1.0 → vol spikes are noisy, smooth aggressively
    state_noise=0.1 → allow regime shifts within days, not hours

    Use::
        ks = make_ivol_smoother()
        ivol_smooth = ks.filter_online(df["ivol"])
        # then z-score ivol_smooth as usual
    """
    return ScalarKalman(obs_noise=obs_noise, state_noise=state_noise, persistence=0.99, K=1)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo (run this file directly to sanity-check)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    T = 200

    # Simulate a slow-moving "regime" signal with observation noise
    true_signal = np.cumsum(rng.normal(0, 0.05, T))   # random walk truth
    noisy_obs   = true_signal + rng.normal(0, 0.5, T) # add observation noise

    series = pd.Series(noisy_obs, name="raw")

    ks = make_tkan_smoother(obs_noise=0.5, state_noise=0.05)
    filtered  = ks.filter_online(series)
    smoothed  = ks.smooth(series)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(noisy_obs,        alpha=0.4, label="noisy obs")
    ax.plot(true_signal,      lw=2,      label="truth")
    ax.plot(filtered.values,  lw=1.5,    label="filter_online (causal)")
    ax.plot(smoothed.values,  lw=1.5,    label="smooth (RTS, lookahead)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.legend(); ax.set_title("ScalarKalman demo")
    plt.tight_layout()
    plt.savefig("kalman_demo.png", dpi=120)
    print("Saved kalman_demo.png")
    print(f"Filter lag (mean abs diff vs truth): {np.mean(np.abs(filtered.values - true_signal)):.4f}")
    print(f"Smooth lag (mean abs diff vs truth): {np.mean(np.abs(smoothed.values - true_signal)):.4f}")
