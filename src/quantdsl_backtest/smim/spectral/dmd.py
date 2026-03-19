"""Dynamic Mode Decomposition (exact DMD and extended DMD) (M3.2-T1, M3.2-T2)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import PolynomialFeatures

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer


def _to_real_modes(
    eigenvalues: np.ndarray,
    modes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Convert complex DMD modes to a real-valued basis via conjugate-pair splitting.

    For each complex conjugate pair (λ, λ̄) with modes (φ, φ̄):
      - Store Re(φ) and Im(φ) as two consecutive real columns.
      - Record the pair in mode_pairs as (j, j+1).
    For real eigenvalues: store φ.real as a single column → singleton (j,).

    Returns:
        basis:      (N, k) real array of modes.
        eigs:       (k,) complex array — conjugate pairs kept as [λ, λ̄].
        mode_pairs: list of 1- or 2-tuples mapping output columns to mode type.
    """
    k = len(eigenvalues)
    imag_tol = 1e-8 * (float(np.abs(eigenvalues).max()) + 1e-12)

    real_cols: list[np.ndarray] = []
    real_eigs: list[complex] = []
    mode_pairs: list[tuple[int, ...]] = []
    col_idx = 0
    i = 0

    while i < k:
        lam = eigenvalues[i]
        phi = modes[:, i]

        # Try to identify a conjugate pair with the next eigenvalue
        if i + 1 < k and abs(lam.imag) > imag_tol:
            lam_next = eigenvalues[i + 1]
            avg_mag = (abs(lam) + abs(lam_next)) / 2.0 + 1e-12
            if abs(lam_next - np.conj(lam)) < 1e-4 * avg_mag:
                # Conjugate pair: ensure positive-imaginary eigenvalue is first
                if lam.imag < 0:
                    lam, lam_next = lam_next, lam
                    phi = modes[:, i + 1]
                real_cols.append(phi.real)
                real_cols.append(phi.imag)
                real_eigs.extend([complex(lam), complex(np.conj(lam))])
                mode_pairs.append((col_idx, col_idx + 1))
                col_idx += 2
                i += 2
                continue

        # Real eigenvalue or unpaired complex — keep real part only
        real_cols.append(phi.real)
        real_eigs.append(complex(lam))
        mode_pairs.append((col_idx,))
        col_idx += 1
        i += 1

    N = modes.shape[0]
    if real_cols:
        basis = np.column_stack(real_cols)
    else:
        basis = np.zeros((N, 0))
    return basis, np.array(real_eigs), mode_pairs


class ExactDMDDecomposer(AbstractSpectralDecomposer):
    """Exact DMD decomposer.

    The operator passed to decompose() is treated as an (N, T) snapshot matrix
    where each column is a state at time step t. For a square operator
    or sparse wrapper around snapshot data, columns are used as time steps.

    X = snapshots[:, :-1], Y = snapshots[:, 1:]
    SVD: X = U S V*
    Reduced propagator: Atilde = U* Y V S^{-1}
    Eigendecompose Atilde: Atilde W = W Lambda
    Exact modes: Phi = Y V S^{-1} W / Lambda (column-wise)
    """

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.DMD

    def _validate_operator(self, op) -> np.ndarray:
        """Override: accept non-square (N x T) matrices for snapshot data."""
        if sparse.issparse(op):
            A = op.toarray().astype(float)
        else:
            A = np.asarray(op, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"Operator must be 2-D, got shape {A.shape}")
        return A

    def decompose(self, operator, k: int) -> ModalFrame:
        snapshots = self._validate_operator(operator)
        return self._dmd_from_snapshots(snapshots, k)

    def decompose_snapshots(self, snapshots: np.ndarray, k: int) -> ModalFrame:
        """Primary interface: (N, T) snapshot matrix."""
        snapshots = np.asarray(snapshots, dtype=float)
        return self._dmd_from_snapshots(snapshots, k)

    def _dmd_from_snapshots(self, snapshots: np.ndarray, k: int) -> ModalFrame:
        N, T = snapshots.shape
        if T < 2:
            raise ValueError(f"Need at least 2 snapshots, got T={T}")
        X = snapshots[:, :-1]   # (N, T-1)
        Y = snapshots[:, 1:]    # (N, T-1)

        # SVD of X
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        # Truncate to k_svd modes
        k_svd = min(k, len(S))
        U_r = U[:, :k_svd]
        S_r = S[:k_svd]
        Vh_r = Vh[:k_svd, :]

        # Reduced propagator
        S_inv = np.diag(1.0 / S_r)
        Atilde = U_r.T @ Y @ Vh_r.T @ S_inv  # (k_svd, k_svd)

        # Eigendecompose Atilde
        eigenvalues, W = np.linalg.eig(Atilde)

        # Exact DMD modes: phi_k = Y V S^{-1} w_k / lambda_k
        # Avoid division by zero for near-zero eigenvalues
        safe_eigs = np.where(np.abs(eigenvalues) > 1e-12, eigenvalues, 1.0)
        modes = (Y @ Vh_r.T @ S_inv @ W) / safe_eigs[np.newaxis, :]  # (N, k_svd)

        # Sort by magnitude of eigenvalues descending
        order = np.argsort(-np.abs(eigenvalues))
        k_actual = min(k, k_svd)
        idx = order[:k_actual]

        basis, eigs, mode_pairs = _to_real_modes(eigenvalues[idx], modes[:, idx])

        return ModalFrame(
            basis=basis,
            eigenvalues=eigs,
            method=self.method,
            metadata={
                "Atilde": Atilde,
                "U": U_r,
                "S": S_r,
                "Vh": Vh_r,
                "full_eigenvalues": eigenvalues,
                "mode_pairs": mode_pairs,
            },
        )


class ExtendedDMDDecomposer(AbstractSpectralDecomposer):
    """Extended DMD (EDMD) with observable lifting via polynomial features.

    Lifts state-space data into a higher-dimensional observable dictionary
    then applies DMD in the lifted space to approximate the Koopman operator.
    """

    def __init__(self, dictionary: str = "polynomial", degree: int = 2) -> None:
        self.dictionary = dictionary
        self.degree = degree
        self._pf: PolynomialFeatures | None = None

    @property
    def method(self) -> DecompositionMethod:
        return DecompositionMethod.EXTENDED_DMD

    def _validate_operator(self, op) -> np.ndarray:
        """Override: accept non-square (N x T) matrices for snapshot data."""
        if sparse.issparse(op):
            A = op.toarray().astype(float)
        else:
            A = np.asarray(op, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"Operator must be 2-D, got shape {A.shape}")
        return A

    def _lift(self, X: np.ndarray) -> np.ndarray:
        """Lift (N, T) snapshot matrix into (P, T) observable space.

        Each column of X is a state vector. Each column of the output is
        the polynomial features of that state.
        """
        # sklearn PolynomialFeatures expects (n_samples, n_features)
        # X is (N, T) -> transpose to (T, N) for sklearn, then transpose back
        if self._pf is None:
            self._pf = PolynomialFeatures(degree=self.degree, include_bias=True)
        Xt = X.T  # (T, N)
        lifted = self._pf.fit_transform(Xt)  # (T, P)
        return lifted.T  # (P, T)

    def decompose(self, operator, k: int) -> ModalFrame:
        snapshots = self._validate_operator(operator)
        N, T = snapshots.shape
        if T < 2:
            raise ValueError(f"Need at least 2 snapshots, got T={T}")

        X = snapshots[:, :-1]   # (N, T-1)
        Y = snapshots[:, 1:]    # (N, T-1)

        # Lift
        Theta_X = self._lift(X)  # (P, T-1)
        Theta_Y = self._lift(Y)  # (P, T-1)
        P = Theta_X.shape[0]

        # Koopman approximation: K = pinv(Theta_X @ Theta_X.T) @ (Theta_X @ Theta_Y.T)
        # = (Theta_X @ Theta_X.T)^{-1} @ Theta_X @ Theta_Y.T
        # Shape: (P, P)
        G = Theta_X @ Theta_X.T  # (P, P)
        A_koopman = Theta_X @ Theta_Y.T  # (P, P)
        try:
            K = np.linalg.solve(G, A_koopman)
        except np.linalg.LinAlgError:
            K = np.linalg.lstsq(G, A_koopman, rcond=None)[0]

        # SVD of K for modes
        U, S, Vh = np.linalg.svd(K, full_matrices=False)
        eigenvalues_complex = S.astype(complex)

        k_actual = min(k, P)
        order = np.argsort(-np.abs(eigenvalues_complex))[:k_actual]
        basis = U[:k_actual, :].T  # Use top-k left singular vectors transposed back

        # Return in N-space by projecting back: use first N rows of U
        n_rows = min(N, U.shape[0])
        basis_n = U[:n_rows, order]  # (n_rows, k_actual) - may not be N if P < N
        # Pad or trim to N rows
        if basis_n.shape[0] < N:
            pad = np.zeros((N - basis_n.shape[0], k_actual))
            basis_n = np.vstack([basis_n, pad])
        else:
            basis_n = basis_n[:N, :]

        return ModalFrame(
            basis=basis_n,
            eigenvalues=eigenvalues_complex[order],
            method=self.method,
            metadata={
                "K": K,
                "Theta_X": Theta_X,
                "Theta_Y": Theta_Y,
                "P_lifted": P,
            },
        )
