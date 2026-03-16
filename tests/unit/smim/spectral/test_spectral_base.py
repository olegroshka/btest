"""Tests for AbstractSpectralDecomposer and SpectralDecomposerFactory (M3.1-T1)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sparse

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer, SpectralDecomposerFactory
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer


class TestSpectralDecomposerFactory:
    def test_factory_raises_key_error_for_unknown_method(self) -> None:
        factory = SpectralDecomposerFactory()
        with pytest.raises(KeyError, match="Unknown decomposition method"):
            factory.create("nonexistent_method")

    def test_factory_raises_key_error_for_valid_enum_not_registered(self) -> None:
        factory = SpectralDecomposerFactory()
        with pytest.raises(KeyError):
            factory.create(DecompositionMethod.SCHUR)

    def test_factory_creates_registered_decomposer(self) -> None:
        factory = SpectralDecomposerFactory()
        factory.register(DecompositionMethod.SCHUR, SchurDecomposer)
        decomposer = factory.create(DecompositionMethod.SCHUR)
        assert isinstance(decomposer, SchurDecomposer)

    def test_factory_create_from_string(self) -> None:
        factory = SpectralDecomposerFactory()
        factory.register(DecompositionMethod.SCHUR, SchurDecomposer)
        decomposer = factory.create("schur")
        assert isinstance(decomposer, SchurDecomposer)

    def test_factory_available_methods(self) -> None:
        factory = SpectralDecomposerFactory()
        factory.register(DecompositionMethod.SCHUR, SchurDecomposer)
        methods = factory.available_methods()
        assert DecompositionMethod.SCHUR in methods

    def test_factory_raises_value_error_for_invalid_string(self) -> None:
        factory = SpectralDecomposerFactory()
        with pytest.raises((KeyError, ValueError)):
            factory.create("totally_unknown")


class TestValidateOperator:
    """Tests for AbstractSpectralDecomposer._validate_operator."""

    def setup_method(self) -> None:
        self.decomposer = SchurDecomposer()

    def test_raises_value_error_for_non_square(self) -> None:
        A = np.ones((3, 4))
        with pytest.raises(ValueError, match="square"):
            self.decomposer._validate_operator(A)

    def test_raises_value_error_for_1d(self) -> None:
        A = np.ones(5)
        with pytest.raises(ValueError):
            self.decomposer._validate_operator(A)

    def test_returns_dense_array_for_sparse_input(self) -> None:
        A_sparse = sparse.eye(4, format="csr")
        A_dense = self.decomposer._validate_operator(A_sparse)
        assert isinstance(A_dense, np.ndarray)
        assert A_dense.shape == (4, 4)

    def test_returns_dense_array_for_dense_input(self) -> None:
        A = np.eye(5)
        result = self.decomposer._validate_operator(A)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 5)

    def test_converts_to_float(self) -> None:
        A = np.eye(3, dtype=int)
        result = self.decomposer._validate_operator(A)
        assert result.dtype == float


class TestTruncateModes:
    """Tests for AbstractSpectralDecomposer._truncate_modes."""

    def setup_method(self) -> None:
        self.decomposer = SchurDecomposer()

    def _make_frame(self, K: int) -> ModalFrame:
        N = 5
        basis = np.random.default_rng(42).standard_normal((N, K))
        eigenvalues = np.array([float(K - i) for i in range(K)], dtype=complex)
        return ModalFrame(basis=basis, eigenvalues=eigenvalues, method=DecompositionMethod.SCHUR)

    def test_truncate_retains_top_k_by_magnitude(self) -> None:
        rng = np.random.default_rng(0)
        N, K = 5, 6
        basis = rng.standard_normal((N, K))
        # Eigenvalues: [1, 5, 2, 4, 3, 6] — top-3 by magnitude should be [6, 5, 4]
        eigenvalues = np.array([1, 5, 2, 4, 3, 6], dtype=complex)
        frame = ModalFrame(basis=basis, eigenvalues=eigenvalues, method=DecompositionMethod.SCHUR)
        truncated = self.decomposer._truncate_modes(frame, k=3)
        assert truncated.K == 3
        # Check top eigenvalues retained
        retained_mags = sorted(np.abs(truncated.eigenvalues), reverse=True)
        assert retained_mags[0] == pytest.approx(6.0)
        assert retained_mags[1] == pytest.approx(5.0)
        assert retained_mags[2] == pytest.approx(4.0)

    def test_truncate_no_op_when_k_exceeds_K(self) -> None:
        frame = self._make_frame(K=4)
        truncated = self.decomposer._truncate_modes(frame, k=10)
        assert truncated.K == 4

    def test_truncate_preserves_method(self) -> None:
        frame = self._make_frame(K=4)
        truncated = self.decomposer._truncate_modes(frame, k=2)
        assert truncated.method == DecompositionMethod.SCHUR
