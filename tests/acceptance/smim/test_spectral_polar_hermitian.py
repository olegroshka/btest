"""
SMIM Acceptance Tests — Sections 2.2 & 2.3: Polar Decomposition + Hermitian Dilation
(14 tests total: 8 polar, 6 Hermitian dilation)

References: docs/smim/ACCEPTANCE_TESTS.md, Sections 2.2–2.3.

Polar API:
    frame = PolarDecomposer().decompose(csr_matrix(A), k=N)
    U = frame.metadata["U"]    # orthogonal/unitary factor
    P = frame.metadata["P"]    # symmetric PSD factor; A = U @ P

Hermitian dilation API (square input only via _validate_operator):
    frame = HermitianDilationDecomposer().decompose(csr_matrix(A), k=N)
    H = frame.metadata["H"]                     # 2N×2N dilation
    all_eigenvalues = frame.metadata["all_eigenvalues"]  # from eigh, ascending
    frame.eigenvalues                            # positive eigenvalues (SVs)

For rectangular inputs (test_A_HD_2, test_D_HD_1) H is constructed directly
because the decomposer interface enforces square matrices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer
from quantdsl_backtest.smim.spectral.polar import PolarDecomposer

from tests.acceptance.smim.conftest import (
    EXACT,
    TIGHT,
    assert_no_nan_inf,
    assert_psd,
    assert_symmetric,
    assert_unitary,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("spectral_decomposition"),
]

# ─── helpers ──────────────────────────────────────────────────────────────────

def _polar(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run PolarDecomposer and return (U, P)."""
    N = A.shape[0]
    frame = PolarDecomposer().decompose(csr_matrix(A), k=N)
    return frame.metadata["U"], frame.metadata["P"]


def _hdilate_square(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run HermitianDilationDecomposer and return (H, all_eigenvalues)."""
    N = A.shape[0]
    frame = HermitianDilationDecomposer().decompose(csr_matrix(A), k=N)
    return frame.metadata["H"], frame.metadata["all_eigenvalues"]


def _hdilate_rect(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build H and compute its eigenvalues directly for non-square A.

    The decomposer enforces square input via _validate_operator, so rectangular
    cases are tested at the mathematical level: H = [[0,A],[A^T,0]].
    """
    N, M = A.shape
    H = np.block([[np.zeros((N, N)), A], [A.T, np.zeros((M, M))]])
    eigenvalues = np.linalg.eigvalsh(H)  # ascending
    return H, eigenvalues


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# ═══════════════════════════════════════════════════════════════════════════════
# POLAR DECOMPOSITION (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_PL_1_rotation_matrix():
    """A-PL-1: Rotation matrix → U = A (already orthogonal), P = I."""
    theta = math.pi / 4
    A = _rotation(theta)

    U, P = _polar(A)

    np.testing.assert_allclose(U, A, **TIGHT, err_msg="U must equal A for a rotation matrix")
    np.testing.assert_allclose(P, np.eye(2), **TIGHT, err_msg="P must be I for a rotation matrix")

    # Reconstruction sanity
    np.testing.assert_allclose(U @ P, A, **TIGHT)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_PL_2_diagonal_scaling():
    """A-PL-2: A = diag(3,5) → U = I, P = A."""
    A = np.diag([3.0, 5.0])

    U, P = _polar(A)

    np.testing.assert_allclose(U, np.eye(2), **TIGHT, err_msg="U must be I for a positive diagonal")
    np.testing.assert_allclose(P, A, **TIGHT, err_msg="P must equal A for a positive diagonal")


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_PL_3_rotation_times_scaling():
    """A-PL-3: A = R @ S (known rotation × symmetric PSD) → recover U≈R, P≈S."""
    theta = math.pi / 6
    R = _rotation(theta)
    S = np.array([[2.0, 0.5], [0.5, 3.0]])   # symmetric PSD (eigenvalues > 0)
    A = R @ S

    U, P = _polar(A)

    # Recovered U should match the rotation R (up to sign / axis ambiguity)
    # Check ||UP - A||_F first, then compare factors
    np.testing.assert_allclose(U @ P, A, **TIGHT, err_msg="Reconstruction U@P must equal A")

    # Factor comparison: allow for the fact that scipy uses the canonical form
    # || U - R ||_F < LOOSE tolerance (polar decomposition is unique for invertible A)
    err_U = min(
        np.linalg.norm(U - R, "fro"),
        np.linalg.norm(U + R, "fro"),  # handle sign freedom
    )
    err_P = np.linalg.norm(P - S, "fro")
    assert err_U < 1e-10, f"||U - R||_F = {err_U:.3e}: recovered rotation factor differs from true R"
    assert err_P < 1e-10, f"||P - S||_F = {err_P:.3e}: recovered scaling factor differs from true S"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_PL_1_U_is_orthogonal():
    """I-PL-1: Random 100×100 → U is orthogonal: ||U^T U - I||_F < 1e-10."""
    rng = np.random.default_rng(111)
    A = rng.standard_normal((100, 100))

    U, _ = _polar(A)

    assert_unitary(U, tol=1e-10)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_PL_2_P_is_symmetric_psd():
    """I-PL-2: Random 100×100 → P is symmetric (||P - P^T||_F < 1e-12) and PSD (eigs ≥ -1e-10)."""
    rng = np.random.default_rng(222)
    A = rng.standard_normal((100, 100))

    _, P = _polar(A)

    assert_symmetric(P, tol=1e-12)
    assert_psd(P, tol=-1e-10)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_PL_3_reconstruction():
    """I-PL-3: Random 100×100 → ||UP - A||_F / ||A||_F < 1e-10."""
    rng = np.random.default_rng(333)
    A = rng.standard_normal((100, 100))

    U, P = _polar(A)

    rel_err = np.linalg.norm(U @ P - A, "fro") / np.linalg.norm(A, "fro")
    assert rel_err < 1e-10, f"Relative reconstruction error = {rel_err:.3e} >= 1e-10"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_R_PL_1_scipy_agreement():
    """R-PL-1: Our (U, P) matches scipy.linalg.polar to tight tolerance."""
    rng = np.random.default_rng(444)
    A = rng.standard_normal((40, 40))

    U_ours, P_ours = _polar(A)
    U_scipy, P_scipy = scipy.linalg.polar(A)

    np.testing.assert_allclose(
        U_ours, U_scipy, **TIGHT,
        err_msg="U from our decomposer must match scipy.linalg.polar",
    )
    np.testing.assert_allclose(
        P_ours, P_scipy, **TIGHT,
        err_msg="P from our decomposer must match scipy.linalg.polar",
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_PL_1_singular_matrix():
    """D-PL-1: Singular [[1,0],[0,0]] → P has zero eigenvalue, U still orthogonal."""
    A = np.array([[1.0, 0.0], [0.0, 0.0]])

    U, P = _polar(A)

    assert_no_nan_inf(U)
    assert_no_nan_inf(P)

    # U must still be orthogonal
    assert_unitary(U, tol=1e-10)

    # P must be symmetric PSD with at least one zero eigenvalue
    assert_symmetric(P, tol=1e-12)
    eigs_P = np.linalg.eigvalsh(P)
    assert np.any(np.abs(eigs_P) < 1e-10), (
        f"P has no near-zero eigenvalue for rank-deficient input; eigenvalues = {eigs_P}"
    )
    assert np.all(eigs_P >= -1e-10), f"P has negative eigenvalue: {eigs_P.min():.3e}"


# ═══════════════════════════════════════════════════════════════════════════════
# HERMITIAN DILATION (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_HD_1_diagonal_known_svs():
    """A-HD-1: A = diag(3,5) → eigenvalues of H are {-5,-3,3,5}."""
    A = np.diag([3.0, 5.0])
    H, all_eigs = _hdilate_square(A)

    sorted_eigs = np.sort(all_eigs.real)
    np.testing.assert_allclose(
        sorted_eigs, [-5.0, -3.0, 3.0, 5.0], **TIGHT,
        err_msg="Eigenvalues of H(diag(3,5)) must be {-5,-3,3,5}",
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_HD_2_rank1_matrix():
    """A-HD-2: Rank-1 A = [1,2,3]^T [4,5] → H has eigenvalues ±σ plus zeros.

    A is 3×2 so H is constructed directly (decomposer enforces square input).
    Analytical singular value: σ = sqrt(||[1,2,3]||^2 * ||[4,5]||^2) = sqrt(14*41).
    """
    A = np.array([[1], [2], [3]]) @ np.array([[4, 5]])  # 3×2, rank 1
    sigma_true = math.sqrt(14.0) * math.sqrt(41.0)  # = sqrt(574)

    H, all_eigs = _hdilate_rect(A)

    # H should be 5×5
    assert H.shape == (5, 5), f"H shape {H.shape} != (5,5)"

    # Only two non-zero eigenvalues (±σ)
    nonzero_eigs = all_eigs[np.abs(all_eigs) > 1e-10]
    assert len(nonzero_eigs) == 2, (
        f"Expected 2 non-zero eigenvalues, got {len(nonzero_eigs)}: {nonzero_eigs}"
    )

    sorted_nonzero = np.sort(nonzero_eigs)
    np.testing.assert_allclose(
        sorted_nonzero, [-sigma_true, sigma_true], **TIGHT,
        err_msg=f"Non-zero eigenvalues must be ±sigma (sigma={sigma_true:.6f})",
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_HD_1_eigenvalue_pairing():
    """I-HD-1: Random 50×50 → for every λ in H, -λ is also present; magnitudes come in ±pairs."""
    rng = np.random.default_rng(555)
    A = rng.standard_normal((50, 50))

    H, all_eigs = _hdilate_square(A)

    # Separate positive and (absolute) negative eigenvalues, ignore numerical zeros
    tol_zero = 1e-9
    pos_eigs = np.sort(all_eigs[all_eigs >  tol_zero])[::-1]  # descending
    neg_abs   = np.sort(np.abs(all_eigs[all_eigs < -tol_zero]))[::-1]  # |neg| descending

    # Must have the same number of positive and negative eigenvalues
    assert len(pos_eigs) == len(neg_abs), (
        f"Asymmetric spectrum: {len(pos_eigs)} positive vs {len(neg_abs)} negative eigenvalues"
    )

    # Each positive eigenvalue must match its negative counterpart in magnitude
    np.testing.assert_allclose(
        pos_eigs, neg_abs, **TIGHT,
        err_msg="Positive and |negative| eigenvalues of H must be paired (±σ structure)",
    )

    # Sorted by magnitude the sequence is σ_1,σ_1,σ_2,σ_2,... (each doubled)
    by_mag = np.sort(np.abs(all_eigs))[::-1]  # all magnitudes descending
    for k in range(0, len(pos_eigs) * 2, 2):
        assert abs(by_mag[k] - by_mag[k + 1]) < 1e-9 * max(by_mag[k], 1.0), (
            f"Pair at position [{k},{k+1}]: magnitudes {by_mag[k]:.6f}, {by_mag[k+1]:.6f} differ"
        )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_HD_2_H_is_hermitian():
    """I-HD-2: H = [[0,A],[A^T,0]] is symmetric (Hermitian): ||H - H^T||_F < 1e-12."""
    rng = np.random.default_rng(666)
    A = rng.standard_normal((30, 30))

    H, _ = _hdilate_square(A)

    assert_symmetric(H, tol=1e-12)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_HD_3_singular_value_recovery():
    """I-HD-3: Positive eigenvalues of H match SVD singular values of A (TIGHT)."""
    rng = np.random.default_rng(777)
    A = rng.standard_normal((40, 40))

    _, all_eigs = _hdilate_square(A)

    pos_eigs = np.sort(all_eigs[all_eigs > 1e-12])[::-1]   # descending
    sv_svd   = np.linalg.svd(A, compute_uv=False)           # already descending

    np.testing.assert_allclose(
        pos_eigs, sv_svd, **TIGHT,
        err_msg="Positive eigenvalues of H must match singular values of A",
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_R_HD_1_numpy_svd_agreement():
    """R-HD-1: Singular values recovered from H match np.linalg.svd(A) to machine precision."""
    rng = np.random.default_rng(888)
    A = rng.standard_normal((50, 50))

    _, all_eigs = _hdilate_square(A)

    pos_eigs = np.sort(all_eigs[all_eigs > 1e-12])[::-1]
    sv_numpy = np.linalg.svd(A, compute_uv=False)

    np.testing.assert_allclose(
        pos_eigs, sv_numpy, **EXACT,
        err_msg="Singular values from H must match np.linalg.svd to machine precision",
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_HD_1_rectangular_matrix():
    """D-HD-1: A is 30×50 (non-square) → H is 80×80 with 30 non-zero ±pairs + 20 zeros.

    The decomposer enforces square input, so H is constructed directly here.
    """
    rng = np.random.default_rng(999)
    A = rng.standard_normal((30, 50))   # 30 rows, 50 cols
    N, M = A.shape                       # 30, 50

    H, all_eigs = _hdilate_rect(A)

    # H must be (N+M) × (N+M) = 80×80
    assert H.shape == (N + M, N + M), f"H shape {H.shape} != ({N+M},{N+M})"

    # H must be symmetric
    assert_symmetric(H, tol=1e-12)

    # A is 30×50 with rank ≤ 30: expect 30 positive + 30 negative eigenvalues
    # and 20 near-zero eigenvalues (from the null space of the 50-dimensional factor)
    tol_zero = 1e-9 * np.linalg.norm(A, "fro")
    pos_eigs  = all_eigs[all_eigs >  tol_zero]
    neg_eigs  = all_eigs[all_eigs < -tol_zero]
    zero_eigs = all_eigs[np.abs(all_eigs) <= tol_zero]

    assert len(pos_eigs) == N, (
        f"Expected {N} positive eigenvalues, got {len(pos_eigs)}"
    )
    assert len(neg_eigs) == N, (
        f"Expected {N} negative eigenvalues, got {len(neg_eigs)}"
    )
    assert len(zero_eigs) == M - N, (
        f"Expected {M - N} zero eigenvalues, got {len(zero_eigs)}"
    )

    # ±pairs must match in magnitude
    pos_sorted = np.sort(pos_eigs)[::-1]
    neg_sorted = np.sort(np.abs(neg_eigs))[::-1]
    np.testing.assert_allclose(
        pos_sorted, neg_sorted, **TIGHT,
        err_msg="Positive and |negative| eigenvalues must be matched ±pairs",
    )
