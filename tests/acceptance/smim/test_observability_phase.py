"""
SMIM Acceptance Tests — Section 8: Observability and Phase Transition (11 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Section 8.

Observability (3):     A-OB-1, A-OB-2, I-OB-1
Order Parameter (2):   I-OP-1, I-OP-2
Ginzburg-Landau (3):   A-GL-1, A-GL-2, I-GL-1
Criticality Index (3): A-CI-1, A-CI-2, I-CI-1
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.observability import (
    compute_observability_matrix,
    observability_diagnostics,
)
from quantdsl_backtest.smim.dynamics.phase_transition import (
    GLParams,
    criticality_index,
    extract_order_parameter,
    fit_gl_landscape,
    gl_potential,
)

pytestmark = [
    pytest.mark.acceptance,
]


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("observability")
def test_A_OB_1_observable_system():
    """A-OB-1: Coupled 2-mode system is observable — cond < 1e6, passes=True.

    F = [[0.9, 0.1], [0, 0.8]], U_K = [[1, 0]] (N=1, K=2).
    Observability matrix O = [[1, 0], [0.9, 0.1]] has rank 2.
    """
    F = np.array([[0.9, 0.1], [0.0, 0.8]])
    U_K = np.array([[1.0, 0.0]])  # (N=1, K=2)

    diag = observability_diagnostics(F, U_K, condition_max=1e6)

    assert diag["passes"], (
        f"Observable system failed condition check: cond={diag['condition_number']:.2e}"
    )
    assert diag["condition_number"] < 1e6, (
        f"Condition number {diag['condition_number']:.2e} ≥ 1e6 for observable system"
    )
    assert diag["suggested_k_star"] == 2, (
        f"Expected suggested_k_star=2, got {diag['suggested_k_star']}"
    )


@pytest.mark.acceptance
@pytest.mark.section("observability")
def test_A_OB_2_unobservable_system():
    """A-OB-2: Decoupled 2-mode system is unobservable — cond ≥ 1e6, passes=False.

    F = [[0.9, 0], [0, 0.8]], U_K = [[1, 0]] (N=1, K=2).
    Observability matrix O = [[1, 0], [0.9, 0]] has rank 1 (col-1 is zero).
    """
    F = np.array([[0.9, 0.0], [0.0, 0.8]])
    U_K = np.array([[1.0, 0.0]])  # (N=1, K=2)

    diag = observability_diagnostics(F, U_K, condition_max=1e6)

    assert not diag["passes"], (
        f"Unobservable system passed condition check: cond={diag['condition_number']:.2e}"
    )
    assert diag["condition_number"] >= 1e6, (
        f"Condition number {diag['condition_number']:.2e} < 1e6 for unobservable system"
    )
    assert diag["suggested_k_star"] == 1, (
        f"Expected suggested_k_star=1 (rank-1 system), got {diag['suggested_k_star']}"
    )


@pytest.mark.acceptance
@pytest.mark.section("observability")
def test_I_OB_1_structural_invariants():
    """I-OB-1: Structural invariants of the observability matrix.

    For any valid (F, U_K):
      - observability_matrix has shape (N*K, K)
      - singular_values are non-negative and in descending order
      - suggested_k_star is in [1, K]
      - condition_number ≥ 1
    """
    K, N = 3, 5
    rng = np.random.default_rng(801)
    F = rng.standard_normal((K, K)) * 0.5  # stable-ish
    U_K = rng.standard_normal((N, K))

    diag = observability_diagnostics(F, U_K)

    O = diag["observability_matrix"]
    sv = diag["singular_values"]
    cond = diag["condition_number"]
    k_star = diag["suggested_k_star"]

    assert O.shape == (N * K, K), (
        f"Expected observability_matrix shape ({N*K}, {K}), got {O.shape}"
    )
    assert len(sv) == K, f"Expected {K} singular values, got {len(sv)}"
    assert np.all(sv >= 0), f"Negative singular values: {sv}"
    # Descending order
    for i in range(K - 1):
        assert sv[i] >= sv[i + 1] - 1e-10, (
            f"Singular values not descending: sv[{i}]={sv[i]:.4e} < sv[{i+1}]={sv[i+1]:.4e}"
        )
    assert 1 <= k_star <= K, (
        f"suggested_k_star={k_star} not in [1, {K}]"
    )
    assert cond >= 1.0, f"condition_number={cond:.4e} < 1"


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER PARAMETER (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("observability")
def test_I_OP_1_amplitudes_method():
    """I-OP-1: 'amplitudes' method returns first d columns of alpha_filtered exactly."""
    T, K, d = 100, 5, 3
    rng = np.random.default_rng(802)
    alpha = rng.standard_normal((T, K))

    psi = extract_order_parameter(alpha, d, method="amplitudes")

    assert psi.shape == (T, d), f"Expected ({T}, {d}), got {psi.shape}"
    np.testing.assert_array_equal(
        psi, alpha[:, :d],
        err_msg="amplitudes method must return exact copy of first d columns",
    )


@pytest.mark.acceptance
@pytest.mark.section("observability")
def test_I_OP_2_interactions_and_entropy_methods():
    """I-OP-2: 'interactions' returns pairwise products; 'entropy' returns (T, d) broadcast.

    Checks:
      - interactions: column 0 = alpha[:, 0] * alpha[:, 1]
      - entropy: all d columns are identical (broadcast of scalar entropy)
      - both return shape (T, d)
    """
    T, K, d = 50, 4, 3
    rng = np.random.default_rng(803)
    alpha = rng.standard_normal((T, K))

    psi_int = extract_order_parameter(alpha, d, method="interactions")
    psi_ent = extract_order_parameter(alpha, d, method="entropy")

    # Shape
    assert psi_int.shape == (T, d), f"interactions shape: expected ({T},{d}), got {psi_int.shape}"
    assert psi_ent.shape == (T, d), f"entropy shape: expected ({T},{d}), got {psi_ent.shape}"

    # First interactions column = alpha[:,0] * alpha[:,1]
    expected_col0 = alpha[:, 0] * alpha[:, 1]
    np.testing.assert_allclose(
        psi_int[:, 0], expected_col0, rtol=1e-12,
        err_msg="interactions column 0 must equal alpha[:,0]*alpha[:,1]",
    )

    # Entropy: all columns identical
    for col in range(1, d):
        np.testing.assert_array_equal(
            psi_ent[:, 0], psi_ent[:, col],
            err_msg=f"entropy columns 0 and {col} must be identical",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GINZBURG-LANDAU (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


def _simulate_gl(a_true: float, b_true: float, T_sim: int, dt: float, sigma: float,
                 seed: int) -> np.ndarray:
    """Euler-Maruyama simulation of ψ_{t+1} = ψ_t + (-a·ψ_t - b·ψ_t³)·dt + σ·√dt·ε."""
    rng = np.random.default_rng(seed)
    psi = np.zeros(T_sim)
    psi[0] = 0.1
    for i in range(T_sim - 1):
        psi[i + 1] = (
            psi[i]
            + (-a_true * psi[i] - b_true * psi[i] ** 3) * dt
            + sigma * np.sqrt(dt) * rng.standard_normal()
        )
    return psi


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_A_GL_1_double_well_recovery():
    """A-GL-1: fit_gl_landscape recovers negative a and positive b for double-well.

    Double-well: a_true=-1, b_true=1. Minima at ψ=±1.
    Fitted coefficients must satisfy: a_hat < 0 and b_hat > 0.
    """
    a_true, b_true = -1.0, 1.0
    dt, sigma = 0.01, 0.3
    T_sim = 10000

    psi = _simulate_gl(a_true, b_true, T_sim, dt, sigma, seed=901)
    psi_2d = psi[:, np.newaxis]  # (T, 1)

    params = fit_gl_landscape(psi_2d, d=1)

    assert params.a[0] < 0, (
        f"Double-well: expected a_hat < 0 (double-well), got a_hat={params.a[0]:.4f}"
    )
    assert params.b[0] > 0, (
        f"Double-well: expected b_hat > 0, got b_hat={params.b[0]:.4f}"
    )


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_A_GL_2_single_well_recovery():
    """A-GL-2: fit_gl_landscape recovers positive a and positive b for single-well.

    Single-well: a_true=1, b_true=1. Minimum at ψ=0.
    Fitted coefficients must satisfy: a_hat > 0 and b_hat > 0.
    """
    a_true, b_true = 1.0, 1.0
    dt, sigma = 0.01, 0.3
    T_sim = 10000

    psi = _simulate_gl(a_true, b_true, T_sim, dt, sigma, seed=902)
    psi_2d = psi[:, np.newaxis]

    params = fit_gl_landscape(psi_2d, d=1)

    assert params.a[0] > 0, (
        f"Single-well: expected a_hat > 0 (single-well), got a_hat={params.a[0]:.4f}"
    )
    assert params.b[0] > 0, (
        f"Single-well: expected b_hat > 0, got b_hat={params.b[0]:.4f}"
    )


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_I_GL_1_potential_structural_invariants():
    """I-GL-1: gl_potential structural invariants.

    For any GLParams:
      - F(0) = 0 exactly
      - F is symmetric: F(ψ) == F(-ψ) for any ψ
    For double-well (a<0, b>0):
      - Two off-zero minima at ψ*=±sqrt(-a/b)
      - F(ψ*) < F(0) = 0 (potential wells below origin)
    """
    # Generic params
    params_gen = GLParams(a=np.array([0.5]), b=np.array([1.0]), d=1)

    # F(0) == 0
    val_zero = gl_potential(np.array([0.0]), params_gen)
    assert abs(val_zero) < 1e-12, f"gl_potential(0) = {val_zero:.3e} ≠ 0"

    # Symmetry: F(x) == F(-x)
    x_test = np.array([1.5])
    val_pos = gl_potential(x_test, params_gen)
    val_neg = gl_potential(-x_test, params_gen)
    assert abs(val_pos - val_neg) < 1e-12, (
        f"gl_potential not symmetric: F(+x)={val_pos:.6f} ≠ F(-x)={val_neg:.6f}"
    )

    # Double-well: minima below 0
    a_dw, b_dw = -1.0, 1.0
    params_dw = GLParams(a=np.array([a_dw]), b=np.array([b_dw]), d=1)
    psi_min = np.sqrt(-a_dw / b_dw)  # analytic minimum location = 1.0
    val_at_min = gl_potential(np.array([psi_min]), params_dw)
    val_at_zero = gl_potential(np.array([0.0]), params_dw)

    assert val_at_min < val_at_zero, (
        f"Double-well: F(ψ_min)={val_at_min:.4f} ≥ F(0)={val_at_zero:.4f}; "
        "expected minima below origin"
    )
    # Check the other minimum is equal
    val_at_neg_min = gl_potential(np.array([-psi_min]), params_dw)
    assert abs(val_at_min - val_at_neg_min) < 1e-12, (
        f"Double-well minima asymmetric: F(+ψ*)={val_at_min:.6f} ≠ F(-ψ*)={val_at_neg_min:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CRITICALITY INDEX (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_A_CI_1_rising_variance_triggers_criticality():
    """A-CI-1: Stationary phase then explosive variance — C_t spikes at transition.

    The criticality index C_t measures the CHANGE between adjacent windows:
    C_t = (Var_curr/Var_prev) × (ACF1_curr/ACF1_prev).  It is highest when
    variance and/or ACF1 are rapidly increasing.

    Design: T=1000 with AR(1).
      - Phase 1 (t < T//2): stable, ρ=0.5, σ=1   → Var ≈ 1.33
      - Phase 2 (t ≥ T//2): near-unit-root, ρ=0.95, σ=1 → Var ≈ 10.3

    The windows straddling the phase boundary should show large C_t, so
    max(C[T//2 : T//2 + 5*w]) must be substantially > 5.

    Implementation note: window_size=30 balances stable ACF1 estimates against
    sensitivity to the phase change.
    """
    T = 1000
    w = 30
    rng = np.random.default_rng(1001)

    switch = T // 2
    psi = np.zeros(T)
    psi[0] = rng.standard_normal()
    for t in range(1, T):
        rho_t = 0.5 if t < switch else 0.95
        psi[t] = rho_t * psi[t - 1] + rng.standard_normal()

    C = criticality_index(psi, window_size=w)

    # Stable phase C_t median (well before transition)
    stable_median = float(np.median(C[2 * w : switch - w]))
    # Peak C_t in the transition and post-transition region
    peak_transition = float(C[switch: switch + 5 * w].max())

    assert peak_transition > 5.0, (
        f"Expected C_t peak > 5 at phase boundary, got {peak_transition:.2f}"
    )
    assert peak_transition > stable_median * 2, (
        f"C_t peak ({peak_transition:.2f}) not > 2× stable median ({stable_median:.2f})"
    )


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_A_CI_2_stationary_series_stays_near_one():
    """A-CI-2: Stationary AR(1) with ρ=0.7 — median C_t ∈ (0.5, 2.0).

    With window_size=50, ACF1 and variance estimates are stable window-to-window,
    so C_t ≈ 1 throughout. Band calibrated from 100-seed Monte Carlo:
      mean=0.997, 1st pct=0.913, 99th pct=1.073 (all seeds in [0.90, 1.09])
    Acceptance band (0.5, 2.0) gives generous margin while catching broken
    implementations — the old band (0.1, 20) was too wide to be meaningful.

    Implementation note: median is used instead of mean to avoid sensitivity to
    rare large spikes when ACF1_prev is close to zero.
    """
    T = 2000
    w = 50
    rng = np.random.default_rng(1002)

    # Stationary AR(1) with fixed rho=0.7
    psi = np.zeros(T)
    psi[0] = rng.standard_normal()
    for t in range(1, T):
        psi[t] = 0.7 * psi[t - 1] + rng.standard_normal()

    C = criticality_index(psi, window_size=w)

    burn = 2 * w
    median_C = float(np.median(C[burn:]))
    assert 0.5 < median_C < 2.0, (
        f"Stationary AR(1) ρ=0.7: median C_t = {median_C:.3f}; expected in (0.5, 2.0) "
        f"[Monte Carlo 1st–99th pct: 0.913–1.073]"
    )


@pytest.mark.acceptance
@pytest.mark.section("phase_transition")
def test_I_CI_1_output_invariants():
    """I-CI-1: criticality_index output invariants.

    - Output shape is (T,)
    - First 2w entries are exactly 1 (burn-in initialisation)
    - All entries are non-negative (Var and ACF ratios are non-negative + 1e-12)
    """
    T = 200
    w = 8
    rng = np.random.default_rng(1003)
    psi = rng.standard_normal(T)

    C = criticality_index(psi, window_size=w)

    assert C.shape == (T,), f"Expected shape ({T},), got {C.shape}"

    # First 2w entries should be 1 (initialised to ones, loop starts at 2w)
    np.testing.assert_array_equal(
        C[: 2 * w],
        np.ones(2 * w),
        err_msg=f"First {2*w} criticality values must be 1 (burn-in)",
    )

    # All entries non-negative
    assert np.all(C >= 0), (
        f"Negative criticality values: min={C.min():.4e}"
    )
