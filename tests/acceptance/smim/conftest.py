"""
Shared fixtures and synthetic data generators for SMIM acceptance tests.

The conftest_report plugin is loaded here so it is active for all acceptance tests.


Every generator is deterministic given a fixed seed. All tests that use random
data MUST pass a fixed seed so results are reproducible.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

# Load the acceptance report plugin — registered via pytest_plugins so that
# pytest calls its pytest_configure and hook implementations automatically.
pytest_plugins = ["tests.acceptance.smim.conftest_report"]

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.interfaces import Actor, ActorType, Layer


# ═══════════════════════════════════════════════════════════
# Device fixture — allows SMIM_DEVICE=cuda to run suite on GPU
# ═══════════════════════════════════════════════════════════

def pytest_configure(config: pytest.Config) -> None:
    """Register gpu marker."""
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


@pytest.fixture(autouse=True)
def configure_device_from_env() -> None:
    """Allow SMIM_DEVICE=cuda to run acceptance suite on GPU."""
    device = os.environ.get("SMIM_DEVICE")
    if device:
        from quantdsl_backtest.smim.compute.torch_ops import get_device
        get_device.cache_clear()
    yield  # type: ignore[misc]
    if device:
        from quantdsl_backtest.smim.compute.torch_ops import get_device
        get_device.cache_clear()


# ═══════════════════════════════════════════════════════════
# Tolerance constants
# ═══════════════════════════════════════════════════════════

EXACT: dict[str, float] = dict(atol=1e-12, rtol=1e-12)
TIGHT: dict[str, float] = dict(atol=1e-8, rtol=1e-8)
LOOSE: dict[str, float] = dict(atol=1e-4, rtol=1e-4)


# ═══════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════

class VAR1Result(NamedTuple):
    data: pd.DataFrame        # shape (T, N)
    A_true: NDArray[np.float64]  # shape (N, N)


def make_var1_system(
    N: int,
    T: int,
    A: NDArray[np.float64],
    noise_std: float = 0.1,
    seed: int = 42,
) -> VAR1Result:
    """Generate data from a VAR(1) system: y_{t+1} = A @ y_t + ε.

    Args:
        N: Number of actors (dimension).
        T: Number of time steps.
        A: True transition matrix (N, N).
        noise_std: Standard deviation of iid Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        VAR1Result with data DataFrame (T, N) and A_true matrix.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((T, N))
    data[0] = rng.standard_normal(N)
    for t in range(T - 1):
        data[t + 1] = A @ data[t] + noise_std * rng.standard_normal(N)
    df = pd.DataFrame(data, columns=[f"actor_{i}" for i in range(N)])
    return VAR1Result(data=df, A_true=A.copy())


class SwitchingResult(NamedTuple):
    observations: NDArray[np.float64]   # shape (T, N_obs)
    true_states: NDArray[np.float64]    # shape (T, N_state)
    true_regimes: NDArray[np.int64]     # shape (T,)


def make_switching_system(
    N: int,
    T: int,
    F_list: list[NDArray[np.float64]],
    Q_list: list[NDArray[np.float64]],
    R: NDArray[np.float64],
    P_transition: NDArray[np.float64],
    seed: int = 42,
) -> SwitchingResult:
    """Generate data from a regime-switching state-space model.

    State equation:  x_t = F^{(z_t)} @ x_{t-1} + η_t,  η_t ~ N(0, Q^{(z_t)})
    Observation eq:  y_t = H @ x_t + ε_t,               ε_t ~ N(0, R)
    where H = I (identity), so obs_dim == state_dim == N.

    Args:
        N: State / observation dimension.
        T: Number of time steps.
        F_list: List of M transition matrices, each (N, N).
        Q_list: List of M state noise covariances, each (N, N).
        R: Observation noise covariance (N, N).
        P_transition: Regime transition matrix (M, M), rows sum to 1.
        seed: Random seed.

    Returns:
        SwitchingResult with observations (T, N), true_states (T, N),
        true_regimes (T,) integer array.
    """
    rng = np.random.default_rng(seed)
    M = len(F_list)
    regimes = np.zeros(T, dtype=np.int64)
    # Initial regime: stationary distribution (approximate as uniform)
    regimes[0] = rng.integers(0, M)
    for t in range(1, T):
        regimes[t] = rng.choice(M, p=P_transition[regimes[t - 1]])

    states = np.zeros((T, N))
    obs = np.zeros((T, N))
    noise_R = np.linalg.cholesky(R)
    states[0] = np.zeros(N)
    obs[0] = states[0] + noise_R @ rng.standard_normal(N)
    for t in range(1, T):
        z = regimes[t]
        noise_Q = np.linalg.cholesky(Q_list[z])
        states[t] = F_list[z] @ states[t - 1] + noise_Q @ rng.standard_normal(N)
        obs[t] = states[t] + noise_R @ rng.standard_normal(N)

    return SwitchingResult(
        observations=obs,
        true_states=states,
        true_regimes=regimes,
    )


class ModalResult(NamedTuple):
    observations: NDArray[np.float64]   # shape (T, N)
    true_alphas: NDArray[np.float64]    # shape (T, K)
    U_true: NDArray[np.float64]         # shape (N, K)


def make_modal_system(
    N: int,
    K: int,
    T: int,
    U_true: NDArray[np.float64],
    F_true: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    seed: int = 42,
) -> ModalResult:
    """Generate data from the modal state-space model.

    State equation:  α_t = F @ α_{t-1} + η_t,  η_t ~ N(0, Q)
    Observation eq:  s_t = U @ α_t + ε_t,       ε_t ~ N(0, R)

    Args:
        N: Observation dimension.
        K: Number of modes (latent state dimension).
        T: Number of time steps.
        U_true: True modal basis (N, K).
        F_true: True mode dynamics (K, K).
        Q: State noise covariance (K, K).
        R: Observation noise covariance (N, N).
        seed: Random seed.

    Returns:
        ModalResult with observations (T, N), true_alphas (T, K), U_true (N, K).
    """
    rng = np.random.default_rng(seed)
    alphas = np.zeros((T, K))
    noise_Q = np.linalg.cholesky(Q)
    noise_R = np.linalg.cholesky(R)
    alphas[0] = rng.standard_normal(K)
    for t in range(1, T):
        alphas[t] = F_true @ alphas[t - 1] + noise_Q @ rng.standard_normal(K)
    obs = alphas @ U_true.T + (noise_R @ rng.standard_normal((N, T))).T
    return ModalResult(observations=obs, true_alphas=alphas, U_true=U_true.copy())


def make_point_cloud(
    shape: str,
    N_points: int,
    noise_std: float = 0.05,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate a labelled point cloud for TDA tests.

    Args:
        shape: One of "circle", "two_clusters", "sphere".
        N_points: Number of points.
        noise_std: Gaussian noise added to ideal positions.
        seed: Random seed.

    Returns:
        Array of shape (N_points, dim) where dim is 2 or 3.
    """
    rng = np.random.default_rng(seed)
    if shape == "circle":
        angles = rng.uniform(0, 2 * math.pi, N_points)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        pts += noise_std * rng.standard_normal(pts.shape)
    elif shape == "two_clusters":
        half = N_points // 2
        c1 = rng.standard_normal((half, 2)) * 0.3 + np.array([0.0, 0.0])
        c2 = rng.standard_normal((N_points - half, 2)) * 0.3 + np.array([5.0, 0.0])
        pts = np.vstack([c1, c2])
        pts += noise_std * rng.standard_normal(pts.shape)
    elif shape == "sphere":
        raw = rng.standard_normal((N_points, 3))
        pts = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        pts += noise_std * rng.standard_normal(pts.shape)
    else:
        raise ValueError(f"Unknown shape: {shape!r}")
    return pts.astype(np.float64)


class PIDSources(NamedTuple):
    X1: NDArray[np.float64]  # shape (T,)
    X2: NDArray[np.float64]  # shape (T,)
    Y: NDArray[np.float64]   # shape (T,)


def make_pid_sources(
    scenario: str,
    T: int,
    seed: int = 42,
) -> PIDSources:
    """Generate source data for PID / transfer entropy tests.

    Args:
        scenario: One of "redundant", "synergistic", "independent".
        T: Number of samples.
        seed: Random seed.

    Returns:
        PIDSources with X1, X2, Y arrays of length T.
    """
    rng = np.random.default_rng(seed)
    if scenario == "redundant":
        Z = rng.standard_normal(T)
        X1 = Z.copy()
        X2 = Z.copy()
        Y = Z + 0.1 * rng.standard_normal(T)
    elif scenario == "synergistic":
        X1 = rng.standard_normal(T)
        X2 = rng.standard_normal(T)
        Y = X1 * X2 + 0.1 * rng.standard_normal(T)
    elif scenario == "independent":
        X1 = rng.standard_normal(T)
        X2 = rng.standard_normal(T)
        Y = rng.standard_normal(T)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    return PIDSources(X1=X1, X2=X2, Y=Y)


class CoupledARResult(NamedTuple):
    X: NDArray[np.float64]  # shape (T,)
    Y: NDArray[np.float64]  # shape (T,)
    Z: NDArray[np.float64] | None  # shape (T,) or None (chain coupling only)


def make_coupled_ar(
    coupling: str,
    T: int,
    seed: int = 42,
) -> CoupledARResult:
    """Generate coupled AR(1) processes for Granger / transfer entropy tests.

    Args:
        coupling: One of "x_causes_y", "bidirectional", "independent", "chain".
            "chain" creates X→Z→Y (for conditional TE tests); Z is returned.
        T: Number of time steps.
        seed: Random seed.

    Returns:
        CoupledARResult with X, Y (and Z for "chain").
    """
    rng = np.random.default_rng(seed)
    X = np.zeros(T)
    Y = np.zeros(T)
    Z: NDArray[np.float64] | None = None

    if coupling == "x_causes_y":
        X[0] = rng.standard_normal()
        Y[0] = rng.standard_normal()
        for t in range(1, T):
            X[t] = 0.8 * X[t - 1] + rng.standard_normal()
            Y[t] = 0.5 * Y[t - 1] + 0.4 * X[t - 1] + rng.standard_normal()
    elif coupling == "bidirectional":
        X[0] = rng.standard_normal()
        Y[0] = rng.standard_normal()
        for t in range(1, T):
            X[t] = 0.6 * X[t - 1] + 0.3 * Y[t - 1] + rng.standard_normal()
            Y[t] = 0.5 * Y[t - 1] + 0.3 * X[t - 1] + rng.standard_normal()
    elif coupling == "independent":
        for t in range(1, T):
            X[t] = 0.7 * X[t - 1] + rng.standard_normal()
            Y[t] = 0.7 * Y[t - 1] + rng.standard_normal()
    elif coupling == "chain":
        Z_arr = np.zeros(T)
        X[0] = rng.standard_normal()
        Z_arr[0] = rng.standard_normal()
        Y[0] = rng.standard_normal()
        for t in range(1, T):
            X[t] = 0.7 * X[t - 1] + rng.standard_normal()
            Z_arr[t] = 0.6 * Z_arr[t - 1] + 0.4 * X[t - 1] + rng.standard_normal()
            Y[t] = 0.5 * Y[t - 1] + 0.4 * Z_arr[t - 1] + rng.standard_normal()
        Z = Z_arr
    else:
        raise ValueError(f"Unknown coupling: {coupling!r}")

    return CoupledARResult(X=X, Y=Y, Z=Z)


# Cycle over the four actor types in a round-robin to give diversity
_LAYER_TYPES: list[tuple[Layer, ActorType]] = [
    (Layer.EXOGENOUS, ActorType.GLOBAL_SHOCK),
    (Layer.UPSTREAM, ActorType.CENTRAL_BANK),
    (Layer.TRANSMISSION, ActorType.LARGE_FIRM),
    (Layer.DOWNSTREAM, ActorType.SME),
]


def make_actor_registry(N: int, n_layers: int = 4) -> ActorRegistry:
    """Create a synthetic ActorRegistry with N actors spread across layers.

    Args:
        N: Total number of actors.
        n_layers: Number of layers to distribute actors across (max 4).

    Returns:
        An ActorRegistry with N actors.
    """
    n_layers = min(n_layers, len(_LAYER_TYPES))
    actors: list[Actor] = []
    for i in range(N):
        layer, actor_type = _LAYER_TYPES[i % n_layers]
        actors.append(
            Actor(
                actor_id=f"actor_{i:04d}",
                name=f"Actor {i}",
                actor_type=actor_type,
                layer=layer,
                geography="US",
                sector="energy",
            )
        )
    return ActorRegistry(actors=actors)


# ═══════════════════════════════════════════════════════════
# Assertion helpers
# ═══════════════════════════════════════════════════════════

def assert_upper_triangular(T: NDArray[np.complexfloating], tol: float = 1e-10) -> None:
    """Assert that T is upper triangular (all below-diagonal entries < tol)."""
    n = T.shape[0]
    for i in range(1, n):
        for j in range(i):
            assert abs(T[i, j]) < tol, (
                f"T[{i},{j}] = {T[i,j]:.3e} is not upper triangular (tol={tol})"
            )


def assert_unitary(Q: NDArray[np.complexfloating], tol: float = 1e-10) -> None:
    """Assert that Q is unitary: ||Q^H Q - I||_F < tol."""
    n = Q.shape[0]
    err = np.linalg.norm(Q.conj().T @ Q - np.eye(n))
    assert err < tol, f"||Q^H Q - I||_F = {err:.3e} (tol={tol})"


def assert_symmetric(M: NDArray[np.floating], tol: float = 1e-12) -> None:
    """Assert that M is symmetric: ||M - M^T||_F < tol."""
    err = np.linalg.norm(M - M.T)
    assert err < tol, f"||M - M^T||_F = {err:.3e} (tol={tol})"


def assert_psd(M: NDArray[np.floating], tol: float = -1e-10) -> None:
    """Assert that M is positive semi-definite: all eigenvalues ≥ tol."""
    eigs = np.linalg.eigvalsh(M)
    min_eig = eigs.min()
    assert min_eig >= tol, f"Minimum eigenvalue = {min_eig:.3e} < {tol}"


def assert_sparse_density(M: NDArray[np.floating], max_density: float) -> None:
    """Assert that the fraction of non-zero entries ≤ max_density."""
    density = np.count_nonzero(M) / M.size
    assert density <= max_density, (
        f"Matrix density = {density:.4f} > max_density = {max_density:.4f}"
    )


def assert_no_nan_inf(arr: NDArray) -> None:
    """Assert that arr contains no NaN or Inf values."""
    assert not np.any(np.isnan(arr)), "Array contains NaN"
    assert not np.any(np.isinf(arr)), "Array contains Inf"


def assert_probabilities_valid(probs: NDArray[np.floating]) -> None:
    """Assert that probs contains valid probability values in [0,1] summing to 1."""
    assert np.all(probs >= 0), f"Negative probability: min={probs.min():.3e}"
    assert np.all(probs <= 1), f"Probability > 1: max={probs.max():.3e}"
    total = probs.sum()
    assert abs(total - 1.0) < 1e-10, f"Probabilities sum to {total:.10f} ≠ 1"


def assert_monotonic_nondecreasing(seq: NDArray[np.floating]) -> None:
    """Assert that seq is monotonically non-decreasing."""
    diffs = np.diff(seq)
    assert np.all(diffs >= -1e-12), (
        f"Sequence is not non-decreasing: min diff = {diffs.min():.3e}"
    )


# ═══════════════════════════════════════════════════════════
# Pytest fixtures (convenience wrappers)
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def simple_var1_4d() -> VAR1Result:
    """4-actor VAR(1) with known causal edges 0→1 and 3→2."""
    A = np.zeros((4, 4))
    A[1, 0] = 0.6   # actor 0 causes actor 1
    A[2, 3] = 0.5   # actor 3 causes actor 2
    # Diagonal for stability
    np.fill_diagonal(A, 0.3)
    return make_var1_system(N=4, T=2000, A=A, noise_std=0.5, seed=0)


@pytest.fixture
def actor_registry_20() -> ActorRegistry:
    """20-actor registry for pipeline tests."""
    return make_actor_registry(20)
