"""Phase transition detection and order parameter extraction for SMIM.

M4.5-T1: Order parameter extraction from modal states.
M4.5-T2: Ginzburg-Landau free-energy landscape fitting.
M4.5-T3: Criticality index for early-warning detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# M4.5-T1: Order parameter extraction
# ---------------------------------------------------------------------------

def extract_order_parameter(
    alpha_filtered: np.ndarray,
    d: int,
    method: str = "amplitudes",
) -> np.ndarray:
    """Extract a d-dimensional order parameter from modal state time series.

    Args:
        alpha_filtered: (T, K) filtered modal states.
        d:              Desired dimensionality of order parameter.
        method:         One of "amplitudes", "interactions", "entropy".

    Returns:
        (T, d) order parameter time series.

    Raises:
        ValueError: If method is not recognised.
    """
    T, K = alpha_filtered.shape
    d = min(d, K)

    if method == "amplitudes":
        return alpha_filtered[:, :d].copy()

    elif method == "interactions":
        # Quadratic pairwise products for top-d unique pairs (j < k)
        result = np.zeros((T, d))
        count = 0
        for j in range(K):
            for k in range(j + 1, K):
                if count >= d:
                    break
                result[:, count] = alpha_filtered[:, j] * alpha_filtered[:, k]
                count += 1
            if count >= d:
                break
        return result

    elif method == "entropy":
        # Entropy of mode amplitude distribution, broadcast to (T, d)
        eps = np.abs(alpha_filtered) + 1e-12
        probs = eps / eps.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs), axis=1)  # (T,)
        return np.tile(entropy[:, np.newaxis], (1, d))

    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# M4.5-T2: Ginzburg-Landau landscape
# ---------------------------------------------------------------------------

@dataclass
class GLParams:
    """Ginzburg-Landau free-energy landscape parameters.

    F(ψ) = Σ_k [a_k/2 ψ_k² + b_k/4 ψ_k⁴]

    Attributes:
        a: (d,) linear (quadratic) coefficients.
        b: (d,) quartic coefficients.
        d: Order-parameter dimensionality.
    """

    a: np.ndarray
    b: np.ndarray
    d: int


def gl_potential(psi: np.ndarray, params: GLParams) -> float:
    """Evaluate the GL free-energy potential at point ψ.

    Args:
        psi:    (d,) state vector.
        params: GLParams with coefficients a and b.

    Returns:
        Scalar potential value F(ψ).
    """
    return float(np.sum(params.a / 2.0 * psi ** 2 + params.b / 4.0 * psi ** 4))


def fit_gl_landscape(psi_t: np.ndarray, d: int) -> GLParams:
    """Fit a Ginzburg-Landau free-energy landscape to modal state dynamics.

    The model is:
        ψ_{t+1} - ψ_t + ∇F(ψ_t) = noise
        ∇F_k = a_k ψ_k + b_k ψ_k³

    Parameters are found by minimising the residual sum of squares.

    Args:
        psi_t: (T, d) order parameter time series.
        d:     Dimensionality to fit (clipped to available columns).

    Returns:
        GLParams with fitted coefficients.
    """
    d = min(d, psi_t.shape[1])
    psi = psi_t[:, :d]

    def loss(params: np.ndarray) -> float:
        a = params[:d]
        b = params[d:]
        grad_F = a * psi[:-1] + b * psi[:-1] ** 3
        resid = psi[1:] - psi[:-1] + grad_F
        return float(np.sum(resid ** 2))

    x0 = np.zeros(2 * d)
    result = minimize(
        loss,
        x0,
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0)] * (2 * d),
    )

    return GLParams(a=result.x[:d], b=result.x[d:], d=d)


# ---------------------------------------------------------------------------
# M4.5-T3: Criticality index
# ---------------------------------------------------------------------------

def criticality_index(psi_t: np.ndarray, window_size: int = 8) -> np.ndarray:
    """Compute a rolling criticality index from order parameter dynamics.

    The index is:
        C_t = (Var_{t-w:t} / Var_{t-2w:t-w}) × (ACF1_{t-w:t} / ACF1_{t-2w:t-w})

    where Var is the rolling variance and ACF1 is the lag-1 autocorrelation.

    Rising C_t signals approach to a critical transition.

    Args:
        psi_t:       (T, d) or (T,) order parameter.
        window_size: Rolling window length w.

    Returns:
        (T,) criticality index, initialised to 1 for t < 2w.
    """
    T = len(psi_t)
    # Collapse multi-dim order parameter to scalar
    psi = psi_t.mean(axis=1) if psi_t.ndim > 1 else psi_t.copy()
    C = np.ones(T)
    w = window_size

    def _acf1(x: np.ndarray) -> float:
        x = x - x.mean()
        if np.var(x) < 1e-12:
            return 0.0
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    for t in range(2 * w, T):
        window1 = psi[t - w: t]
        window2 = psi[t - 2 * w: t - w]

        var1 = np.var(window1) + 1e-12
        var2 = np.var(window2) + 1e-12
        var_ratio = var1 / var2

        acf_curr = abs(_acf1(window1)) + 1e-12
        acf_prev = abs(_acf1(window2)) + 1e-12
        acf_ratio = acf_curr / acf_prev

        C[t] = var_ratio * acf_ratio

    return C
