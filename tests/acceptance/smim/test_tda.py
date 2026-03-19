"""
SMIM Acceptance Tests — Section 6.3: TDA (7 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Section 6.3.

A-TDA-1: circle — 1 long H1 feature, β_0=1
A-TDA-2: two clusters — β_0=2 at small scale, β_0=1 at large scale, no H1
A-TDA-3: sphere — 1 significant H2 feature, β_0=1
I-TDA-1: stability theorem (d_B < 2ε for ε-perturbation)
I-TDA-2: topological complexity non-negativity
I-TDA-3: Wasserstein distance properties (self=0, symmetry, non-negativity)
R-TDA-1: ripser wrapper agreement with direct ripser call
"""

from __future__ import annotations

import numpy as np
import pytest
from ripser import ripser
from persim import bottleneck, wasserstein as persim_wasserstein

from quantdsl_backtest.smim.emergence.tda import (
    sliding_window_persistence,
    topological_complexity,
    wasserstein_distances,
)
from tests.acceptance.smim.conftest import make_point_cloud

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("tda"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_A_TDA_1_circle():
    """A-TDA-1: Unit circle point cloud — exactly 1 long-lived H1 feature, β_0=1.

    Expected persistent homology of a noisy unit circle:
      H0: all components merge quickly except one (β_0=1 at large scale → 1 infinite bar)
      H1: 1 bar with large persistence (the loop) ≈ 1.25; all others < 0.01
    """
    pts = make_point_cloud("circle", N_points=100, noise_std=0.05, seed=42)
    dgms = ripser(pts, maxdim=1)["dgms"]

    h0, h1 = dgms[0], dgms[1]
    h1_pers = h1[:, 1] - h1[:, 0]

    # Exactly 1 long-lived H1 feature
    long_lived = h1_pers[h1_pers > 0.5]
    assert len(long_lived) == 1, (
        f"Expected 1 H1 feature with persistence > 0.5, "
        f"got {len(long_lived)}: {sorted(h1_pers, reverse=True)[:5]}"
    )

    # β_0 = 1 at large scale: exactly one H0 bar with death = ∞
    inf_bars = np.sum(np.isinf(h0[:, 1]))
    assert inf_bars == 1, (
        f"Expected exactly 1 infinite H0 bar (β_0=1), got {inf_bars}"
    )


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_A_TDA_2_two_clusters():
    """A-TDA-2: Two well-separated clusters — β_0=2 at small scale, β_0=1 at large.

    Clusters at (0,0) and (5,0) with σ=0.3.  Inter-cluster gap ≈ 3.2 units.
      H0: 1 large finite-death bar (cluster merge at ~3.3 units) + 1 infinite bar
      H1: no significant loops (max persistence < 0.3 for tight convex clusters)
    """
    pts = make_point_cloud("two_clusters", N_points=100, noise_std=0.3, seed=42)
    dgms = ripser(pts, maxdim=1)["dgms"]

    h0, h1 = dgms[0], dgms[1]

    # β_0=2 at small scale: the second-largest finite-death H0 bar should be
    # the cluster-merge event (death > 1.0, well above within-cluster scale)
    finite_deaths = np.sort(h0[~np.isinf(h0[:, 1]), 1])
    cluster_merge = finite_deaths[-1]  # largest finite death = cluster merge
    assert cluster_merge > 1.0, (
        f"Cluster merge expected at scale > 1.0, got {cluster_merge:.3f}"
    )

    # β_0=1 at large scale: exactly 1 infinite H0 bar
    assert np.sum(np.isinf(h0[:, 1])) == 1, (
        "Expected exactly 1 infinite H0 bar (β_0=1 at large scale)"
    )

    # No significant H1 features (convex clusters have no 1-cycles)
    if len(h1) > 0:
        h1_pers = h1[:, 1] - h1[:, 0]
        assert np.all(h1_pers < 0.3), (
            f"Unexpected H1 feature: max persistence = {h1_pers.max():.4f} ≥ 0.3"
        )


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_A_TDA_3_sphere():
    """A-TDA-3: Sphere point cloud — 1 significant H2 feature (the void), β_0=1.

    Uniform sampling of S² with 200 points and noise σ=0.05.
    Expected: H2 bar with persistence > 0.5 (the enclosed hollow sphere).
    """
    pts = make_point_cloud("sphere", N_points=200, noise_std=0.05, seed=42)
    dgms = ripser(pts, maxdim=2)["dgms"]

    h0, h2 = dgms[0], dgms[2]

    # 1 significant H2 feature
    h2_pers = h2[:, 1] - h2[:, 0] if len(h2) > 0 else np.array([])
    long_h2 = h2_pers[h2_pers > 0.5]
    assert len(long_h2) >= 1, (
        f"Expected ≥1 H2 feature with persistence > 0.5 (the void), "
        f"got {sorted(h2_pers, reverse=True)[:3] if len(h2_pers) else []}"
    )

    # β_0=1 at large scale
    assert np.sum(np.isinf(h0[:, 1])) == 1, (
        "Expected exactly 1 infinite H0 bar (β_0=1 for connected sphere)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Invariant tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_I_TDA_1_stability_theorem():
    """I-TDA-1: Stability theorem — bottleneck distance < 2ε for ε-perturbation.

    For a point cloud perturbation where each point moves by at most ε (L∞ per
    coordinate), the VR stability theorem gives:
      d_B(Dgm(X), Dgm(Y)) ≤ 2 · d_Hausdorff(X, Y) ≤ 2ε

    Test uses ε=0.01 with uniform ±ε perturbation (L∞ per coordinate).
    Empirically: d_B ≈ 0.006 ≪ 2ε = 0.02 for the unit circle.
    """
    pts = make_point_cloud("circle", N_points=100, noise_std=0.05, seed=42)

    eps = 0.01
    rng = np.random.default_rng(1101)
    # Uniform ±ε per coordinate → L∞ per-coordinate ≤ ε → d_H ≤ ε·sqrt(2)
    perturbation = rng.uniform(-eps, eps, pts.shape)
    pts_perturbed = pts + perturbation

    h1_orig = ripser(pts, maxdim=1)["dgms"][1]
    h1_pert = ripser(pts_perturbed, maxdim=1)["dgms"][1]

    d_b = bottleneck(h1_orig, h1_pert)

    assert d_b < 2 * eps, (
        f"Stability violated: d_B = {d_b:.6f} ≥ 2ε = {2*eps:.4f}"
    )


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_I_TDA_2_topological_complexity_non_negative():
    """I-TDA-2: Topological complexity T_t ≥ 0 for all windows.

    T_t = Σ_k (k+1) · total_persistence_k is a sum of non-negative persistence
    lengths weighted by (k+1), so T_t ≥ 0 exactly.
    """
    rng = np.random.default_rng(2001)
    T, K = 100, 3
    alpha = np.zeros((T, K))
    for t in range(1, T):
        alpha[t] = 0.8 * alpha[t - 1] + rng.standard_normal(K)

    diagrams = sliding_window_persistence(alpha, window_size=20, overlap=10)
    C = topological_complexity(diagrams)

    assert len(C) > 0, "Expected at least one window"
    assert np.all(C >= 0), (
        f"Topological complexity has negative values: min={C.min():.4e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_I_TDA_3_wasserstein_properties():
    """I-TDA-3: Wasserstein distance properties on H1 diagrams.

      W(D, D) = 0      (self-distance is zero)
      W(D1, D2) = W(D2, D1)  (symmetry, exact for persim)
      W(D1, D2) ≥ 0   (non-negativity)
    """
    pts_c = make_point_cloud("circle", N_points=100, noise_std=0.05, seed=42)
    pts_t = make_point_cloud("two_clusters", N_points=100, noise_std=0.3, seed=42)

    h1_c = ripser(pts_c, maxdim=1)["dgms"][1]
    h1_t = ripser(pts_t, maxdim=1)["dgms"][1]

    # Self-distance is zero
    d_self = persim_wasserstein(h1_c, h1_c)
    assert d_self < 1e-10, f"W(D, D) = {d_self:.4e} ≥ 1e-10"

    # Symmetry
    d_12 = persim_wasserstein(h1_c, h1_t)
    d_21 = persim_wasserstein(h1_t, h1_c)
    assert abs(d_12 - d_21) < 1e-10, (
        f"|W(D1,D2) - W(D2,D1)| = {abs(d_12-d_21):.4e} ≥ 1e-10"
    )

    # Non-negativity
    assert d_12 >= 0, f"W(D1,D2) = {d_12:.4f} < 0"


# ═══════════════════════════════════════════════════════════════════════════════
# Reference test
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.acceptance
@pytest.mark.section("tda")
def test_R_TDA_1_ripser_wrapper_agreement():
    """R-TDA-1: sliding_window_persistence wrapper produces identical diagrams to ripser.

    The wrapper calls ripser on each window.  For a single window covering the
    full array, the diagram must be identical to a direct ripser call on the same data.
    """
    rng = np.random.default_rng(3001)
    K = 2
    window = rng.standard_normal((20, K))  # 20 points in K=2 dimensions

    # Wrapper: one window = entire array (window_size = T, overlap = 0)
    T = len(window)
    diagrams = sliding_window_persistence(window, window_size=T, overlap=0)
    assert len(diagrams) == 1, f"Expected 1 window diagram, got {len(diagrams)}"
    wrapper_h0, wrapper_h1 = diagrams[0][0], diagrams[0][1]

    # Direct ripser call on the same data
    direct_dgms = ripser(window, maxdim=1)["dgms"]
    direct_h0, direct_h1 = direct_dgms[0], direct_dgms[1]

    np.testing.assert_allclose(
        wrapper_h0, direct_h0, atol=1e-6,
        err_msg="H0 diagrams differ between wrapper and direct ripser",
    )
    np.testing.assert_allclose(
        wrapper_h1, direct_h1, atol=1e-6,
        err_msg="H1 diagrams differ between wrapper and direct ripser",
    )
