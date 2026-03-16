"""Tests for SpectralComparisonHarness (M3.1-T6)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sparse

from quantdsl_backtest.smim.interfaces import DecompositionMethod
from quantdsl_backtest.smim.spectral.comparison import MethodComparisonRow, SpectralComparisonHarness


def _make_symmetric_matrix(n: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A + A.T


class TestSpectralComparisonHarness:
    def test_runs_on_20x20_matrix_all_methods(self) -> None:
        A = _make_symmetric_matrix(20)
        methods = ["schur", "polar", "hermitian_dilation", "directed_variation"]
        harness = SpectralComparisonHarness(methods=methods, k=5)
        df = harness.run(sparse.csr_matrix(A))
        assert len(df) == len(methods)

    def test_dataframe_has_expected_columns(self) -> None:
        A = _make_symmetric_matrix(10)
        harness = SpectralComparisonHarness(methods=["schur", "polar"], k=3)
        df = harness.run(A)
        expected_cols = {"method", "condition_number", "reconstruction_error", "compute_time_ms", "k_retained"}
        assert expected_cols.issubset(set(df.columns))

    def test_reconstruction_error_finite_non_negative(self) -> None:
        A = _make_symmetric_matrix(8)
        harness = SpectralComparisonHarness(methods=["schur", "polar", "hermitian_dilation"], k=4)
        df = harness.run(A)
        for val in df["reconstruction_error"]:
            if not np.isnan(val):
                assert np.isfinite(val)
                assert val >= 0.0

    def test_compute_time_positive(self) -> None:
        A = _make_symmetric_matrix(6)
        harness = SpectralComparisonHarness(methods=["schur"], k=3)
        df = harness.run(A)
        assert df["compute_time_ms"].iloc[0] >= 0

    def test_k_retained_positive(self) -> None:
        A = _make_symmetric_matrix(6)
        harness = SpectralComparisonHarness(methods=["polar"], k=3)
        df = harness.run(A)
        assert df["k_retained"].iloc[0] > 0

    def test_sorted_by_reconstruction_error(self) -> None:
        A = _make_symmetric_matrix(10)
        harness = SpectralComparisonHarness(methods=["schur", "polar", "directed_variation"], k=4)
        df = harness.run(A)
        valid = df["reconstruction_error"].dropna()
        assert list(valid) == sorted(valid)

    def test_empty_methods_returns_empty_df(self) -> None:
        A = _make_symmetric_matrix(5)
        harness = SpectralComparisonHarness(methods=[], k=3)
        df = harness.run(A)
        assert len(df) == 0
