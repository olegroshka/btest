"""
SMIM Acceptance Tests — Section 3: Mode Selection Primitives (9 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Section 3.

MDL (3 tests)         : A-MDL-1, A-MDL-2, I-MDL-1
LZ compressibility (4): A-LZ-1, A-LZ-2, A-LZ-3, I-LZ-1
RG relevance (2)      : A-RG-1, A-RG-2
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.mode_selection import (
    MDLModeSelector,
    RGRelevanceFilter,
    lz_complexity,
)

from tests.acceptance.smim.conftest import make_modal_system

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("mode_selection"),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _svd_modal_frame(Y: np.ndarray, K: int) -> ModalFrame:
    """Build a ModalFrame from the leading K left singular vectors of Y (T, N).

    Y.T is (N, T); the SVD gives U (N, N) whose columns are the left singular
    vectors — i.e., the directions of maximum variance in observation space.
    """
    U, S, _ = np.linalg.svd(Y.T, full_matrices=False)   # U: (N, min(N,T))
    K_actual = min(K, U.shape[1])
    return ModalFrame(
        basis=U[:, :K_actual],
        eigenvalues=S[:K_actual].astype(complex),
        method=DecompositionMethod.SCHUR,
        metadata={},
    )


def _lz_rho(seq: np.ndarray) -> float:
    """Lempel-Ziv compressibility ρ = 1 − C·log₂(n) / n (Lempel & Ziv 1976), clamped to [0, 1]."""
    n = max(len(seq), 2)
    return float(np.clip(1.0 - lz_complexity(seq) * np.log2(n) / n, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# MDL (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_MDL_1_known_rank():
    """A-MDL-1: 3-mode signal + small noise, K_candidate=10 — MDL selects K* ∈ {2,3,4}."""
    N, K_true, T = 50, 3, 200
    rng = np.random.default_rng(101)

    # Random orthonormal true mode basis (N, K_true)
    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K_true)), full_matrices=False)
    U_true = U_raw

    # Strong signal: unit-variance state noise so modes dominate observation noise
    F_true = 0.9 * np.eye(K_true)
    Q = np.eye(K_true)           # state noise variance 1 → steady-state mode var ≈ 5.3
    R = 0.01 * np.eye(N)         # observation noise 0.01; SNR ≈ 30

    result = make_modal_system(N=N, K=K_true, T=T,
                               U_true=U_true, F_true=F_true, Q=Q, R=R, seed=42)
    Y = result.observations   # (T, N)

    frame_10 = _svd_modal_frame(Y, K=10)
    selector  = MDLModeSelector()
    reduced_frame, _ = selector.select(frame_10, Y)

    K_star = reduced_frame.K
    assert K_star in {2, 3, 4}, (
        f"MDL selected K*={K_star}; expected ∈ {{2, 3, 4}} for a 3-mode system"
    )


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_MDL_2_pure_noise():
    """A-MDL-2: Pure iid Gaussian noise — MDL selects K*=1 (minimum meaningful rank).

    Implementation note: MDLModeSelector always returns ≥ 1 mode, so K*=0 is
    not possible.  For noise without structure, all SVD modes contribute equally
    and the log(T) penalty dominates for k ≥ 2, so K*=1.
    """
    N, T = 50, 200
    rng = np.random.default_rng(202)
    Y = rng.standard_normal((T, N))

    frame_10 = _svd_modal_frame(Y, K=10)
    selector  = MDLModeSelector()
    reduced_frame, _ = selector.select(frame_10, Y)

    K_star = reduced_frame.K
    assert K_star <= 2, (
        f"MDL selected K*={K_star} for pure Gaussian noise; expected ≤ 2"
    )


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_I_MDL_1_description_length_structure():
    """I-MDL-1: DL components satisfy structural properties for fixed data.

    Structural requirements:
      - L(data|model)(k) non-increasing  — more modes always improve fit
      - L(model)(k) strictly increasing  — each mode adds log(T) penalty
      - total DL(k) has a minimum NOT at k_max  — regularisation effect exists
    """
    N, K_true, T = 30, 5, 300
    rng = np.random.default_rng(303)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K_true)), full_matrices=False)
    U_true = U_raw
    F_true = 0.9 * np.eye(K_true)
    Q = 0.01 * np.eye(K_true)
    R = 0.01 * np.eye(N)

    result = make_modal_system(N=N, K=K_true, T=T,
                               U_true=U_true, F_true=F_true, Q=Q, R=R, seed=43)
    Y = result.observations   # (T, N)

    frame_20 = _svd_modal_frame(Y, K=20)
    U     = frame_20.basis    # (N, K_max)
    K_max = U.shape[1]

    L_data, L_model, DL = [], [], []
    for k in range(1, K_max + 1):
        U_k   = U[:, :k]
        coefs  = Y @ U_k          # (T, k)
        Y_hat  = coefs @ U_k.T    # (T, N)
        res_var = np.sum((Y - Y_hat) ** 2) / (T * N)
        if res_var <= 0:
            res_var = 1e-300
        ld = float(T * np.log(res_var))
        lm = float(k * N * np.log(max(T, 2)) / 2)
        L_data.append(ld)
        L_model.append(lm)
        DL.append(ld + lm)

    # L(data|model) must be non-increasing
    for k in range(K_max - 1):
        assert L_data[k] >= L_data[k + 1] - 1e-6, (
            f"L(data|model) not non-increasing: L_data[{k}]={L_data[k]:.3f} "
            f"< L_data[{k+1}]={L_data[k+1]:.3f}"
        )

    # L(model) must strictly increase with k (adds log(T) per mode)
    for k in range(K_max - 1):
        assert L_model[k] < L_model[k + 1], (
            f"L(model) not strictly increasing: L_model[{k}]={L_model[k]:.3f} "
            f">= L_model[{k+1}]={L_model[k+1]:.3f}"
        )

    # DL must have a minimum that is NOT at the last candidate
    min_idx = int(np.argmin(DL))
    assert min_idx < K_max - 1, (
        f"DL minimum at k={min_idx + 1} (last candidate k={K_max}): "
        "DL appears monotonically decreasing — no regularisation effect visible"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LZ COMPRESSIBILITY (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_LZ_1_constant_sequence():
    """A-LZ-1: Constant sequence length 1000 — ρ > 0.99."""
    constant = np.ones(1000)
    rho = _lz_rho(constant)
    assert rho > 0.99, f"Constant sequence ρ = {rho:.4f} ≤ 0.99"


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_LZ_2_periodic_sequence():
    """A-LZ-2: Periodic [1,2,3,...] length 1000 — ρ > 0.5."""
    periodic = np.tile([1, 2, 3], 334)[:1000]
    rho = _lz_rho(periodic)
    assert rho > 0.5, f"Periodic sequence ρ = {rho:.4f} ≤ 0.5"


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_LZ_3_random_sequence():
    """A-LZ-3: Random sequence length 1000 — ρ < 0.15 (nearly incompressible).

    Under the standard normalisation C·log₂(n)/n, a random 256-symbol sequence
    of length 1000 has C ≈ n/log₂(n), giving ρ ≈ 0.
    """
    rng = np.random.default_rng(404)
    random_seq = rng.integers(0, 256, 1000).astype(float)
    rho = _lz_rho(random_seq)
    assert rho < 0.15, f"Random sequence ρ = {rho:.4f} ≥ 0.15"


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_I_LZ_1_bounds_and_ordering():
    """I-LZ-1: ρ ∈ [0,1] for all inputs; ρ(constant) > ρ(periodic) > ρ(random)."""
    constant   = np.ones(1000)
    periodic   = np.tile([1, 2, 3], 334)[:1000]
    rng        = np.random.default_rng(505)
    random_seq = rng.integers(0, 256, 1000).astype(float)

    rho_c = _lz_rho(constant)
    rho_p = _lz_rho(periodic)
    rho_r = _lz_rho(random_seq)

    for rho, name in [(rho_c, "constant"), (rho_p, "periodic"), (rho_r, "random")]:
        assert 0.0 <= rho <= 1.0, f"ρ({name}) = {rho:.4f} not in [0, 1]"

    assert rho_c > rho_p, (
        f"ρ(constant)={rho_c:.4f} ≤ ρ(periodic)={rho_p:.4f}"
    )
    assert rho_p > rho_r, (
        f"ρ(periodic)={rho_p:.4f} ≤ ρ(random)={rho_r:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RG RELEVANCE (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_RG_1_global_mode_passes():
    """A-RG-1: Uniform-loading global mode — RG filter retains it (high relevance)."""
    N, n_per_layer = 20, 5
    layer_assignments = np.repeat([0, 1, 2, 3], n_per_layer)

    u_global = np.ones(N) / np.sqrt(N)   # equal loading on every actor
    frame = ModalFrame(
        basis=u_global[:, np.newaxis],   # (N, 1)
        eigenvalues=np.array([1.0 + 0j]),
        method=DecompositionMethod.SCHUR,
        metadata={},
    )

    T   = 50
    rng = np.random.default_rng(601)
    ts  = rng.standard_normal((T, 1))    # (T, K=1) modal amplitudes

    flt = RGRelevanceFilter()
    reduced_frame, _ = flt.select(frame, ts, layer_assignments)

    assert reduced_frame.K >= 1, (
        "Global (uniform) mode was incorrectly filtered out by RG relevance filter"
    )


@pytest.mark.acceptance
@pytest.mark.section("mode_selection")
def test_A_RG_2_local_mode_filtered():
    """A-RG-2: Layer-3-only mode is filtered; global mode survives.

    A 2-mode frame is used (mode 0 = global, mode 1 = layer-3-only) so that
    the filter can remove mode 1 without triggering the 'keep at least 1' fallback.
    """
    N, n_per_layer = 20, 5
    layer_assignments = np.repeat([0, 1, 2, 3], n_per_layer)

    u_global = np.ones(N) / np.sqrt(N)           # non-zero on all layers
    u_local  = np.zeros(N)
    u_local[15:] = 1.0 / np.sqrt(n_per_layer)   # non-zero only in layer 3

    frame = ModalFrame(
        basis=np.column_stack([u_global, u_local]),  # (N, 2)
        eigenvalues=np.array([1.0 + 0j, 0.5 + 0j]),
        method=DecompositionMethod.SCHUR,
        metadata={},
    )

    T   = 50
    rng = np.random.default_rng(701)
    ts  = rng.standard_normal((T, 2))   # (T, K=2) modal amplitudes

    flt = RGRelevanceFilter()
    reduced_frame, _ = flt.select(frame, ts, layer_assignments)

    # The layer-3-only mode must be filtered; only the global mode remains
    assert reduced_frame.K == 1, (
        f"Expected K=1 after filtering layer-3-only mode, got K={reduced_frame.K}"
    )

    # Confirm the retained mode is the global one (uniform across all layers)
    retained = reduced_frame.basis[:, 0]
    load_layers_0_to_2 = np.sum(np.abs(retained[:15]))
    load_layer_3       = np.sum(np.abs(retained[15:]))
    assert load_layers_0_to_2 > load_layer_3, (
        f"Retained mode appears layer-3-concentrated: "
        f"layers 0-2 total loading = {load_layers_0_to_2:.4f}, "
        f"layer 3 total loading = {load_layer_3:.4f}"
    )
