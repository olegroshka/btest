"""Tests for ExactDMDDecomposer and ExtendedDMDDecomposer (M3.2-T1, M3.2-T2)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer, ExtendedDMDDecomposer


def _generate_linear_snapshots(
    n: int = 5, t: int = 30, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate snapshots from y_{k+1} = A y_k where A has known eigenvalues."""
    rng = np.random.default_rng(seed)
    # Build A with eigenvalues near 0.9, 0.8, 0.7, 0.6, 0.5
    # Diagonal matrix for simplicity
    eigenvalues = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    A = np.diag(eigenvalues)  # (n, n)
    # Generate trajectory
    y0 = rng.standard_normal(n)
    snapshots = np.zeros((n, t))
    snapshots[:, 0] = y0
    for i in range(1, t):
        snapshots[:, i] = A @ snapshots[:, i - 1]
    return snapshots, eigenvalues


class TestExactDMDDecomposer:
    def setup_method(self) -> None:
        self.decomposer = ExactDMDDecomposer()

    def test_method_is_dmd(self) -> None:
        assert self.decomposer.method == DecompositionMethod.DMD

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        snapshots, _ = _generate_linear_snapshots()
        frame = self.decomposer.decompose(snapshots, k=5)
        assert isinstance(frame, ModalFrame)

    def test_modes_shape(self) -> None:
        snapshots, _ = _generate_linear_snapshots(n=5, t=20)
        frame = self.decomposer.decompose(snapshots, k=5)
        assert frame.basis.shape == (5, 5)

    def test_modes_shape_k_limit(self) -> None:
        snapshots, _ = _generate_linear_snapshots(n=5, t=20)
        frame = self.decomposer.decompose(snapshots, k=3)
        assert frame.basis.shape == (5, 3)

    def test_eigenvalues_approximate_true_eigenvalues(self) -> None:
        """DMD eigenvalues should approximate A's eigenvalues."""
        snapshots, true_eigs = _generate_linear_snapshots(n=5, t=50, seed=0)
        frame = self.decomposer.decompose(snapshots, k=5)
        # Sort by magnitude
        dmd_eigs = np.sort(np.abs(frame.eigenvalues))[::-1]
        true_eigs_sorted = np.sort(np.abs(true_eigs))[::-1]
        # Allow some tolerance
        np.testing.assert_allclose(dmd_eigs[:5], true_eigs_sorted[:5], atol=0.05)

    def test_decompose_snapshots_method(self) -> None:
        snapshots, _ = _generate_linear_snapshots()
        frame = self.decomposer.decompose_snapshots(snapshots, k=3)
        assert frame.K == 3

    def test_accepts_non_square_input(self) -> None:
        """DMD should accept (N, T) non-square input."""
        snapshots = np.random.default_rng(0).standard_normal((4, 15))
        frame = self.decomposer.decompose(snapshots, k=3)
        assert frame.K == 3

    def test_metadata_contains_Atilde(self) -> None:
        snapshots, _ = _generate_linear_snapshots()
        frame = self.decomposer.decompose(snapshots, k=5)
        assert "Atilde" in frame.metadata


class TestExtendedDMDDecomposer:
    def setup_method(self) -> None:
        self.decomposer = ExtendedDMDDecomposer(dictionary="polynomial", degree=2)

    def test_method_is_extended_dmd(self) -> None:
        assert self.decomposer.method == DecompositionMethod.EXTENDED_DMD

    def test_lifted_data_has_more_features(self) -> None:
        """Lifting should increase the number of observables."""
        snapshots = np.random.default_rng(1).standard_normal((3, 20))
        X = snapshots[:, :-1]  # simulate X = snapshots[:, :-1]
        lifted = self.decomposer._lift(X)
        N, T_minus1 = X.shape
        P = lifted.shape[0]
        # With polynomial degree 2 and 3 features: 1 + 3 + 6 = 10 features
        assert P > N
        assert lifted.shape[1] == T_minus1

    def test_lift_output_shape(self) -> None:
        snapshots = np.random.default_rng(2).standard_normal((3, 15))
        X = snapshots[:, :-1]
        lifted = self.decomposer._lift(X)
        assert lifted.shape[1] == X.shape[1]

    def test_koopman_matrix_is_square(self) -> None:
        """K should be P x P."""
        snapshots = np.random.default_rng(3).standard_normal((3, 20))
        frame = self.decomposer.decompose(snapshots, k=5)
        K = frame.metadata["K"]
        assert K.shape[0] == K.shape[1]

    def test_decompose_returns_modal_frame(self) -> None:
        from quantdsl_backtest.smim.interfaces import ModalFrame
        snapshots = np.random.default_rng(4).standard_normal((3, 20))
        frame = self.decomposer.decompose(snapshots, k=3)
        assert isinstance(frame, ModalFrame)

    def test_basis_shape_is_N_by_k(self) -> None:
        N = 4
        snapshots = np.random.default_rng(5).standard_normal((N, 25))
        k = 3
        frame = self.decomposer.decompose(snapshots, k=k)
        assert frame.basis.shape == (N, k)
