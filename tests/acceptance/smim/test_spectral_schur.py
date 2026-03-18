"""
SMIM Acceptance Tests — Section 2.1: Schur Decomposition (9 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Section 2.1.

API contract (SchurDecomposer):
    frame = decomposer.decompose(csr_matrix(A), k=N)
    T = frame.metadata["T"]   # full complex Schur triangular matrix
    Q = frame.metadata["Q"]   # full unitary transformation: A = Q T Q^H
    eigenvalues = frame.metadata["full_eigenvalues"]  # diag(T), shape (N,)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

from tests.acceptance.smim.conftest import (
    EXACT,
    TIGHT,
    assert_no_nan_inf,
    assert_unitary,
    assert_upper_triangular,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("spectral_decomposition"),
]

# ─── shared helper ────────────────────────────────────────────────────────────

def _decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SchurDecomposer on A and return (T, Q, eigenvalues) from metadata."""
    N = A.shape[0]
    decomposer = SchurDecomposer()
    frame = decomposer.decompose(csr_matrix(A), k=N)
    T = frame.metadata["T"]
    Q = frame.metadata["Q"]
    eigenvalues = frame.metadata["full_eigenvalues"]
    return T, Q, eigenvalues


def _sort_complex_by_angle(v: np.ndarray) -> np.ndarray:
    """Sort complex array by angle in [0, 2π), for stable eigenvalue matching."""
    angles = np.angle(v) % (2 * math.pi)
    return v[np.argsort(angles)]


# ═══════════════════════════════════════════════════════════
# Analytical tests
# ═══════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_SC_1_known_2x2_upper_triangular():
    """A-SC-1: A = [[2,1],[0,3]] — eigenvalues {2,3}, T upper triangular,
    Q unitary, reconstruction QTQ^H = A, all to machine precision."""
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    T, Q, eigenvalues = _decompose(A)

    # Eigenvalues from diagonal of T are {2, 3}
    eig_real_sorted = sorted(np.diag(T).real)
    np.testing.assert_allclose(
        eig_real_sorted, [2.0, 3.0], **EXACT,
        err_msg="Diagonal of T must equal {2, 3} to machine precision",
    )

    # T is upper triangular
    assert_upper_triangular(T, tol=1e-12)

    # Q is unitary: ||Q^H Q - I|| < 1e-12
    assert_unitary(Q, tol=1e-12)

    # Reconstruction: ||QTQ^H - A||_F < 1e-12
    A_reconstructed = Q @ T @ Q.conj().T
    err = np.linalg.norm(A_reconstructed - A, "fro")
    assert err < 1e-12, f"Reconstruction error ||QTQ^H - A||_F = {err:.3e} >= 1e-12"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_SC_2_symmetric_reduces_to_eigendecomp():
    """A-SC-2: A = [[4,1],[1,3]] — eigenvalues (7±√5)/2, T diagonal for symmetric input."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    expected_eigs = sorted([
        (7.0 + math.sqrt(5.0)) / 2.0,
        (7.0 - math.sqrt(5.0)) / 2.0,
    ])

    T, Q, eigenvalues = _decompose(A)

    # Schur eigenvalues match known analytical eigenvalues (TIGHT)
    eig_real_sorted = sorted(np.diag(T).real)
    np.testing.assert_allclose(
        eig_real_sorted, expected_eigs, **TIGHT,
        err_msg="Schur eigenvalues must match known analytical values",
    )

    # For symmetric (normal) input the Schur form is diagonal (T = Λ)
    off_diag_max = max(
        abs(T[i, j]) for i in range(2) for j in range(2) if i != j
    )
    assert off_diag_max < 1e-12, (
        f"Off-diagonal max |T[i,j]| = {off_diag_max:.3e}: "
        "T should be diagonal for symmetric input"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_SC_3_companion_matrix_5th_roots_of_unity():
    """A-SC-3: 5×5 companion matrix for x⁵-1 — Schur eigenvalues are 5th roots of unity."""
    # Frobenius companion matrix for x^5 - 1:
    #   sub-diagonal of ones + last column [1,0,0,0,0]^T
    C = np.zeros((5, 5))
    C[1:, :-1] = np.eye(4)   # sub-diagonal
    C[0, -1] = 1.0            # from -c_0 = -(-1) = 1

    T, Q, eigenvalues = _decompose(C)

    # True 5th roots of unity
    true_eigs = np.exp(2j * math.pi * np.arange(5) / 5)

    # Sort both sets by angle for stable comparison
    comp_sorted = _sort_complex_by_angle(eigenvalues)
    true_sorted = _sort_complex_by_angle(true_eigs)

    # Moduli must all be ≈ 1 (TIGHT)
    np.testing.assert_allclose(
        np.abs(comp_sorted), np.ones(5), **TIGHT,
        err_msg="All eigenvalue moduli must equal 1 (5th roots of unity lie on unit circle)",
    )

    # Angles must match to TIGHT tolerance
    computed_angles = np.angle(comp_sorted) % (2 * math.pi)
    true_angles = np.angle(true_sorted) % (2 * math.pi)
    np.testing.assert_allclose(
        computed_angles, true_angles, **TIGHT,
        err_msg="Eigenvalue angles must match 5th roots of unity to tight tolerance",
    )


# ═══════════════════════════════════════════════════════════
# Invariant tests
# ═══════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_SC_1_unitarity_of_Q():
    """I-SC-1: Random 100×100 non-symmetric — Q^H Q = I and Q Q^H = I."""
    rng = np.random.default_rng(101)
    A = rng.standard_normal((100, 100))

    T, Q, _ = _decompose(A)

    N = 100
    err_right = np.linalg.norm(Q.conj().T @ Q - np.eye(N), "fro")
    err_left  = np.linalg.norm(Q @ Q.conj().T - np.eye(N), "fro")

    assert err_right < 1e-10, f"||Q^H Q - I||_F = {err_right:.3e} >= 1e-10"
    assert err_left  < 1e-10, f"||Q Q^H - I||_F = {err_left:.3e} >= 1e-10"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_SC_2_triangularity_of_T():
    """I-SC-2: Random 100×100 — T is upper triangular (max below-diagonal entry < 1e-10)."""
    rng = np.random.default_rng(202)
    A = rng.standard_normal((100, 100))

    T, _, _ = _decompose(A)

    assert_upper_triangular(T, tol=1e-10)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_SC_3_reconstruction():
    """I-SC-3: Random 100×100 — ||QTQ^H - A||_F / ||A||_F < 1e-10."""
    rng = np.random.default_rng(303)
    A = rng.standard_normal((100, 100))

    T, Q, _ = _decompose(A)

    A_reconstructed = Q @ T @ Q.conj().T
    rel_err = np.linalg.norm(A_reconstructed - A, "fro") / np.linalg.norm(A, "fro")
    assert rel_err < 1e-10, f"Relative reconstruction error = {rel_err:.3e} >= 1e-10"


# ═══════════════════════════════════════════════════════════
# Reference test
# ═══════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_R_SC_1_scipy_eigenvalue_agreement():
    """R-SC-1: Random 50×50 — our eigenvalues match scipy.linalg.schur to machine precision.

    Note: Q and T themselves may differ by a unitary transformation; only
    the eigenvalues (diagonal of T) are the comparison invariant.
    """
    rng = np.random.default_rng(404)
    A = rng.standard_normal((50, 50))

    _, _, our_eigs = _decompose(A)

    T_scipy, _ = scipy.linalg.schur(A, output="complex")
    scipy_eigs = np.diag(T_scipy)

    # Sort both by magnitude descending for stable comparison
    our_sorted   = our_eigs[np.argsort(-np.abs(our_eigs))]
    scipy_sorted = scipy_eigs[np.argsort(-np.abs(scipy_eigs))]

    np.testing.assert_allclose(
        our_sorted, scipy_sorted, **EXACT,
        err_msg="Our Schur eigenvalues must match scipy.linalg.schur to machine precision",
    )


# ═══════════════════════════════════════════════════════════
# Adversarial tests
# ═══════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_SC_1_near_defective_matrix():
    """D-SC-1: [[1,1],[0,1+1e-10]] — no crash, eigenvalues close to 1,
    condition number is computable from metadata Q."""
    A = np.array([[1.0, 1.0], [0.0, 1.0 + 1e-10]])

    # Must not crash
    T, Q, eigenvalues = _decompose(A)

    # No NaN or Inf
    assert_no_nan_inf(np.abs(eigenvalues))
    assert_no_nan_inf(np.abs(T))
    assert_no_nan_inf(np.abs(Q))

    # Both eigenvalues are close to 1
    for eig in eigenvalues:
        assert abs(eig - 1.0) < 1e-6, (
            f"Eigenvalue {eig} is not close to 1 for near-defective input"
        )

    # Condition number is finite and computable from the Q in metadata
    cond_Q = np.linalg.cond(Q)
    assert np.isfinite(cond_Q), f"Condition number of Q is not finite: {cond_Q}"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_SC_2_zero_matrix():
    """D-SC-2: Zero 5×5 matrix — all eigenvalues exactly 0, no NaN, no crash."""
    A = np.zeros((5, 5))

    T, Q, eigenvalues = _decompose(A)

    # No NaN or Inf
    assert_no_nan_inf(np.abs(eigenvalues))
    assert_no_nan_inf(np.abs(T))

    # All eigenvalues must be exactly 0
    np.testing.assert_allclose(
        eigenvalues, np.zeros(5, dtype=complex), **EXACT,
        err_msg="All eigenvalues of the zero matrix must be exactly 0",
    )
