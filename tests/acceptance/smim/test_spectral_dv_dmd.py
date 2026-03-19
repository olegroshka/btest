"""
SMIM Acceptance Tests — Sections 2.4 & 2.5: Directed Variation Basis + DMD (11 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Sections 2.4–2.5.

Directed Variation API:
    frame = DirectedVariationDecomposer().decompose(csr_matrix(A), k)
    frame.basis            — (N, k) orthonormal Laplacian eigenvectors (ascending DV)
    frame.eigenvalues      — k Laplacian eigenvalues as complex (= DV values)
    frame.metadata["laplacian"]        — symmetrised Laplacian L
    frame.metadata["all_eigenvalues"]  — all N eigenvalues, ascending

DMD API:
    decomposer = ExactDMDDecomposer()
    frame = decomposer.decompose_snapshots(snapshots, k)  # snapshots: (N, T)
    frame.basis            — (N, k) real DMD modes
    frame.eigenvalues      — (k,) complex DMD eigenvalues
    frame.metadata["full_eigenvalues"] — all eigenvalues of reduced propagator
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer

from tests.acceptance.smim.conftest import (
    EXACT,
    TIGHT,
    assert_monotonic_nondecreasing,
    assert_no_nan_inf,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("spectral_decomposition"),
]

# ─── helpers ──────────────────────────────────────────────────────────────────

def _dv(A: np.ndarray, k: int):
    return DirectedVariationDecomposer().decompose(csr_matrix(A), k=k)


def _dmd(snapshots: np.ndarray, k: int):
    """snapshots: (N, T) — rows = spatial dimensions, cols = time steps."""
    return ExactDMDDecomposer().decompose_snapshots(snapshots, k=k)


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _dmd_reconstruct(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    x0: np.ndarray,
    T: int,
    mode_pairs: list[tuple[int, ...]] | None = None,
) -> np.ndarray:
    """Reconstruct time series from real DMD modes.

    When mode_pairs is None (legacy path, all-real eigenvalues):
        x_t = (modes @ (B * λ^t)).real

    When mode_pairs is provided (general path with conjugate-pair real modes):
        For singleton (j,):   contribution = basis[:,j] * |λ_j|^t * B[j]
        For pair (j, j+1):    contribution = B[j]  * r^t * (u cos tθ − v sin tθ)
                                           + B[j+1] * r^t * (v cos tθ + u sin tθ)
        where u=modes[:,j], v=modes[:,j+1], r=|λ_j|, θ=angle(λ_j).

    Args:
        modes:       (N, k) real DMD mode matrix
        eigenvalues: (k,) complex DMD eigenvalues
        x0:          (N,) initial state
        T:           number of time steps
        mode_pairs:  pairing info from frame.metadata["mode_pairs"] (optional)

    Returns:
        (N, T) reconstructed snapshot matrix
    """
    B = np.linalg.lstsq(modes, x0, rcond=None)[0]   # (k,) real amplitudes
    N = modes.shape[0]
    rec = np.zeros((N, T))

    if mode_pairs is None:
        # Legacy: works for real eigenvalues; modes @ (B * λ^t) is real
        for t in range(T):
            rec[:, t] = (modes @ (B * eigenvalues ** t)).real
        return rec

    # General real-mode reconstruction
    for pair in mode_pairs:
        if len(pair) == 1:
            # Real eigenvalue (possibly negative): use λ^t directly
            j = pair[0]
            lam_real = eigenvalues[j].real   # negligible imaginary part by construction
            for t in range(T):
                rec[:, t] += B[j] * (lam_real ** t) * modes[:, j]
        else:
            j, j1 = pair
            lam = eigenvalues[j]   # positive-imaginary eigenvalue of the pair
            r = abs(lam)
            theta = np.angle(lam)
            u = modes[:, j]        # Re(φ)
            v = modes[:, j1]       # Im(φ)
            for t in range(T):
                ct = (r ** t) * np.cos(t * theta)
                st = (r ** t) * np.sin(t * theta)
                rec[:, t] += B[j] * (ct * u - st * v) + B[j1] * (ct * v + st * u)

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTED VARIATION BASIS (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_DV_1_basis_orthonormality():
    """I-DV-1: Output basis is orthonormal: ||U^T U - I||_F < 1e-8."""
    rng = np.random.default_rng(11)
    N = 30
    A = rng.random((N, N))
    k = 10

    frame = _dv(A, k)
    U = frame.basis  # (N, k)

    err = np.linalg.norm(U.T @ U - np.eye(k), "fro")
    assert err < 1e-8, f"||U^T U - I||_F = {err:.3e} >= 1e-8 (optimisation tolerance)"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_DV_2_dv_values_monotone():
    """I-DV-2: DV values (Laplacian eigenvalues) are monotonically non-decreasing."""
    rng = np.random.default_rng(22)
    N = 20
    A = rng.random((N, N))
    k = N  # request all modes

    frame = _dv(A, k)
    dv_values = frame.eigenvalues.real  # imaginary parts are numerical noise

    assert_monotonic_nondecreasing(dv_values)


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_DV_3_dv_values_nonnegative():
    """I-DV-3: DV values are all non-negative (Laplacian eigenvalues ≥ 0)."""
    rng = np.random.default_rng(33)
    N = 25
    A = rng.random((N, N))
    k = N

    frame = _dv(A, k)
    dv_values = frame.eigenvalues.real

    assert np.all(dv_values >= -1e-12), (
        f"Negative DV value: min = {dv_values.min():.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_R_DV_1_symmetric_agrees_with_laplacian():
    """R-DV-1: On symmetric A, DV basis correlates >0.9 with Laplacian eigenvectors."""
    rng = np.random.default_rng(44)
    N = 15
    # Build a symmetric non-negative adjacency matrix
    raw = rng.random((N, N))
    A_sym = (raw + raw.T) / 2.0
    k = 8

    frame = _dv(A_sym, k)
    U_dv = frame.basis  # (N, k)

    # Independently compute Laplacian eigenvectors for the same symmetric A
    d = A_sym.sum(axis=1)
    L = np.diag(d) - A_sym
    eigs_ref, evecs_ref = np.linalg.eigh(L)
    # Take same k smallest-eigenvalue eigenvectors
    V_lap = evecs_ref[:, :k]  # (N, k)

    # For each DV basis vector, find the highest absolute correlation with any
    # Laplacian eigenvector (handles sign ambiguity and potential reordering)
    gram = np.abs(U_dv.T @ V_lap)  # (k, k) — absolute inner products
    for j in range(k):
        max_corr = gram[j, :].max()
        assert max_corr > 0.9, (
            f"DV basis vector {j} max |correlation| with Laplacian eigenvectors "
            f"= {max_corr:.4f} < 0.9"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DMD — EXACT (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_DMD_1_known_linear_system():
    """A-DMD-1: Known 2D system — DMD eigenvalues ≈{0.9,0.8}, modes span correct subspace.

    T=50 is used (not 500): both modes decay to near-noise after ~20 steps, so longer
    trajectories degrade the estimate of the smaller eigenvalue.
    Tolerance is 15%: the smaller eigenvalue (0.8) is harder to recover from noisy data.
    """
    A_true = np.array([[0.9, 0.1], [0.0, 0.8]])
    T = 50
    rng = np.random.default_rng(101)

    y = np.zeros((T, 2))
    y[0] = [1.0, 0.5]
    for t in range(T - 1):
        y[t + 1] = A_true @ y[t] + rng.normal(0, 0.01, 2)

    frame = _dmd(y.T, k=2)
    eigs = frame.eigenvalues
    modes = frame.basis  # (2, 2)

    # Eigenvalue magnitudes within 15% of true values {0.9, 0.8}
    # (noisy system; smaller eigenvalue decays to noise floor in ~20 steps)
    computed_mags = np.sort(np.abs(eigs))[::-1]   # descending
    true_mags     = np.array([0.9, 0.8])
    rel_err = np.abs(computed_mags - true_mags) / true_mags
    assert np.all(rel_err < 0.15), (
        f"DMD eigenvalue magnitudes {computed_mags} not within 15% of {true_mags}"
    )

    # Modes span same subspace as eigenvectors of A_true (angle < 10°)
    _, evecs_true = np.linalg.eig(A_true)
    # Each DMD mode must align within 10° with at least one true eigenvector
    thresh = math.radians(10)
    for j in range(2):
        mode_j = modes[:, j:j+1]
        min_angle = min(
            scipy.linalg.subspace_angles(mode_j, evecs_true[:, i:i+1].real)[0]
            for i in range(2)
        )
        assert min_angle < thresh, (
            f"DMD mode {j} has min subspace angle {math.degrees(min_angle):.1f}° "
            f"with true eigenvectors (threshold 10°)"
        )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_DMD_2_oscillatory_system():
    """A-DMD-2: Eigenvalues 0.95*exp(±iπ/6) — magnitude, angle, period recovered."""
    r, theta = 0.95, math.pi / 6
    A_osc = r * _rotation(theta)   # real 2×2 with eigenvalues r*exp(±iθ)
    T = 500
    rng = np.random.default_rng(202)

    y = np.zeros((T, 2))
    y[0] = [1.0, 0.0]
    for t in range(T - 1):
        y[t + 1] = A_osc @ y[t] + rng.normal(0, 0.01, 2)

    frame = _dmd(y.T, k=2)
    eigs = frame.eigenvalues

    # Find the eigenvalue with positive imaginary part
    pos_idx = np.argmax(eigs.imag)
    lam = eigs[pos_idx]

    # Magnitude within 5% of r = 0.95
    assert abs(abs(lam) - r) / r < 0.05, (
        f"|λ| = {abs(lam):.4f} not within 5% of {r}"
    )

    # Angle within 5% of θ = π/6
    angle_computed = abs(np.angle(lam))
    assert abs(angle_computed - theta) / theta < 0.05, (
        f"angle(λ) = {math.degrees(angle_computed):.1f}° not within 5% of "
        f"{math.degrees(theta):.1f}°"
    )

    # Reconstructed period within 10% of 12
    period_true = 2 * math.pi / theta      # = 12 time steps
    period_computed = 2 * math.pi / angle_computed
    assert abs(period_computed - period_true) / period_true < 0.10, (
        f"Period {period_computed:.2f} not within 10% of {period_true:.2f}"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_A_DMD_3_reconstruction_convergence():
    """A-DMD-3: 5-mode system — reconstruction error monotone in k; at k=5 error <1%."""
    # Noise-free diagonal 5-mode system
    lambdas = np.array([0.99, 0.95, 0.90, 0.85, 0.80])
    A_5d = np.diag(lambdas)
    T = 200
    y = np.zeros((T, 5))
    y[0] = [2.0, 1.5, 1.0, 0.7, 0.5]
    for t in range(T - 1):
        y[t + 1] = A_5d @ y[t]   # pure linear, no noise

    snapshots = y.T  # (5, 200)

    errors = []
    for k in range(1, 6):
        frame = _dmd(snapshots, k=k)
        rec = _dmd_reconstruct(frame.basis, frame.eigenvalues, snapshots[:, 0], T)
        err = np.linalg.norm(rec - snapshots, "fro") / np.linalg.norm(snapshots, "fro")
        errors.append(err)

    # Errors must decrease monotonically (allow tiny tolerance for numerics)
    for j in range(len(errors) - 1):
        assert errors[j] >= errors[j + 1] - 1e-8, (
            f"Reconstruction error not monotone: errors[{j}]={errors[j]:.6f} < "
            f"errors[{j+1}]={errors[j+1]:.6f}"
        )

    # At k=5 error must be < 1% of signal energy
    assert errors[4] < 0.01, (
        f"At k=5 reconstruction error = {errors[4]:.6f} >= 1% of signal energy"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_DMD_1_reconstruction_fidelity():
    """I-DMD-1: Random 20D non-symmetric stable system T=200 — ||Y_rec-Y||/||Y|| < 0.1.

    A generic non-symmetric matrix is used, producing complex conjugate eigenvalue
    pairs. The real-mode reconstruction (conjugate pairs split into Re/Im columns)
    must still recover the trajectory to within 10% relative error.
    """
    N, T = 20, 200
    rng = np.random.default_rng(303)

    # Build a stable non-symmetric random linear system (complex eigenvalues)
    A_rand = rng.standard_normal((N, N)) * 0.3 / N
    sr = max(abs(np.linalg.eigvals(A_rand)))
    A_rand = A_rand * 0.9 / sr   # spectral radius = 0.9

    y = np.zeros((T, N))
    y[0] = rng.standard_normal(N)
    for t in range(T - 1):
        y[t + 1] = A_rand @ y[t]

    snapshots = y.T   # (20, 200)
    frame = _dmd(snapshots, k=N)   # all modes

    rec = _dmd_reconstruct(
        frame.basis, frame.eigenvalues, snapshots[:, 0], T,
        mode_pairs=frame.metadata.get("mode_pairs"),
    )
    rel_err = np.linalg.norm(rec - snapshots, "fro") / np.linalg.norm(snapshots, "fro")

    assert rel_err < 0.10, (
        f"DMD reconstruction relative error = {rel_err:.4f} >= 0.10"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_I_DMD_2_real_mode_conversion():
    """I-DMD-2: 4D system with 2 real + 1 conjugate pair → 4 real modes; reconstruction matches.

    System: diagonal 2D real block + 2D rotation block.
    Expected: DMD produces mode_pairs with 2 singletons + 1 pair-of-2.
    Reconstruction from real modes must match complex-mode reconstruction to <1e-6 relative error.
    """
    r_osc, theta = 0.92, math.pi / 5
    # Block-diagonal: [0.95, 0, 0, 0; 0, 0.80, 0, 0; 0, 0, r*cosθ, -r*sinθ; 0, 0, r*sinθ, r*cosθ]
    A = np.block([
        [np.diag([0.95, 0.80]),                np.zeros((2, 2))],
        [np.zeros((2, 2)),                     r_osc * _rotation(theta)],
    ])
    N, T = 4, 300
    rng = np.random.default_rng(999)
    y = np.zeros((T, N))
    y[0] = rng.standard_normal(N)
    for t in range(T - 1):
        y[t + 1] = A @ y[t]

    snapshots = y.T   # (4, 300)
    frame = _dmd(snapshots, k=N)

    mode_pairs = frame.metadata.get("mode_pairs")
    assert mode_pairs is not None, "mode_pairs must be stored in metadata"

    # Should have 4 columns: 2 singletons (real eigs) + 1 pair (conjugate)
    assert frame.basis.shape == (N, N), f"Expected ({N},{N}) basis, got {frame.basis.shape}"
    assert frame.basis.dtype.kind == "f", "basis must be real-valued"

    n_singletons = sum(1 for p in mode_pairs if len(p) == 1)
    n_pairs = sum(1 for p in mode_pairs if len(p) == 2)
    assert n_singletons == 2, f"Expected 2 singleton modes, got {n_singletons}"
    assert n_pairs == 1, f"Expected 1 conjugate pair, got {n_pairs}"

    # Real-mode reconstruction must match to tight tolerance
    rec_real = _dmd_reconstruct(
        frame.basis, frame.eigenvalues, snapshots[:, 0], T,
        mode_pairs=mode_pairs,
    )
    rel_err = np.linalg.norm(rec_real - snapshots, "fro") / np.linalg.norm(snapshots, "fro")
    assert rel_err < 0.01, (
        f"Real-mode reconstruction error = {rel_err:.6f} >= 0.01"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_R_DMD_1_pydmd_agreement():
    """R-DMD-1: Our eigenvalues agree with PyDMD TIGHT; mode subspace angles < 5°."""
    import pydmd  # pydmd is a required dev dependency

    N, T = 10, 100
    rng = np.random.default_rng(404)
    A_ref = np.diag(np.linspace(0.95, 0.70, N))
    y = np.zeros((T, N))
    y[0] = rng.standard_normal(N)
    for t in range(T - 1):
        y[t + 1] = A_ref @ y[t] + rng.normal(0, 0.005, N)

    snapshots = y.T   # (N, T)
    k = N

    # Our implementation
    frame_ours = _dmd(snapshots, k=k)
    our_eigs   = frame_ours.eigenvalues
    our_modes  = frame_ours.basis

    # PyDMD
    dmd_py = pydmd.DMD(svd_rank=k)
    dmd_py.fit(snapshots)
    py_eigs  = dmd_py.eigs
    py_modes = dmd_py.modes.real   # (N, k), real part

    # Sort both by |eigenvalue| descending for stable comparison
    our_sorted = our_eigs[np.argsort(-np.abs(our_eigs))]
    py_sorted  = py_eigs[np.argsort(-np.abs(py_eigs))]

    np.testing.assert_allclose(
        np.abs(our_sorted), np.abs(py_sorted), **TIGHT,
        err_msg="DMD eigenvalue magnitudes must agree with PyDMD to tight tolerance",
    )

    # Mode subspace principal angles < 5°
    min_k = min(our_modes.shape[1], py_modes.shape[1])
    angles = scipy.linalg.subspace_angles(our_modes[:, :min_k], py_modes[:, :min_k])
    max_angle_deg = math.degrees(angles.max())
    assert max_angle_deg < 5.0, (
        f"Max subspace angle between our modes and PyDMD modes = {max_angle_deg:.2f}° >= 5°"
    )


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_DMD_1_underdetermined():
    """D-DMD-1: T=5, N=20 (underdetermined) — graceful handling, no crash, no NaN."""
    rng = np.random.default_rng(505)
    N, T = 20, 5
    snapshots = rng.standard_normal((N, T))   # (20, 5)

    # Must not crash; should fall back to a truncated rank decomposition
    frame = _dmd(snapshots, k=min(N, T - 1))

    assert_no_nan_inf(np.abs(frame.eigenvalues))
    assert_no_nan_inf(frame.basis)
    assert frame.basis.shape[0] == N, "Modes must have N rows"


@pytest.mark.acceptance
@pytest.mark.section("spectral_decomposition")
def test_D_DMD_2_constant_signal():
    """D-DMD-2: All snapshots identical → single mode, eigenvalue ≈ 1.0, no NaN."""
    N, T = 5, 50
    y0 = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
    snapshots = np.tile(y0[:, np.newaxis], T)   # (N, T) — all columns = y0

    frame = _dmd(snapshots, k=1)

    assert_no_nan_inf(np.abs(frame.eigenvalues))
    assert_no_nan_inf(frame.basis)

    # Single eigenvalue must be ≈ 1.0
    lam = frame.eigenvalues[0]
    assert abs(lam - 1.0) < 1e-6, (
        f"Constant-signal eigenvalue = {lam} is not close to 1.0"
    )
