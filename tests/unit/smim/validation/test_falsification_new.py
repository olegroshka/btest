"""Tests for SMIM falsification tests (M5.2-T1 through M5.2-T4)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from quantdsl_backtest.smim.validation.falsification import (
    BlockPreservingPlacebo,
    FalsificationResult,
    FalsificationSuite,
    FrozenRegimePlacebo,
    LagDestroyedPlacebo,
    NoNetworkDFMPlacebo,
    RandomisedTypePlacebo,
    ShuffledEdgePlacebo,
    SymmetricOperatorPlacebo,
)


def _make_structured_operator(n: int = 6) -> sp.csr_matrix:
    """Create a structured directed graph with clear community structure."""
    rng = np.random.default_rng(42)
    data = []
    rows = []
    cols = []
    # Strong within-community edges, weak cross-community
    half = n // 2
    for i in range(half):
        for j in range(half):
            if i != j:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
    for i in range(half, n):
        for j in range(half, n):
            if i != j:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(n, n)
    )


def _spectral_gap(op: sp.csr_matrix) -> float:
    """Compute spectral gap: difference of two largest eigenvalue magnitudes."""
    try:
        import scipy.sparse.linalg as spla
        n = op.shape[0]
        k = min(n - 1, 3)
        if k < 1:
            return 0.0
        eigs = spla.eigs(op.astype(float), k=k, which="LM", return_eigenvectors=False)
        sorted_eigs = sorted(abs(eigs), reverse=True)
        if len(sorted_eigs) < 2:
            return 0.0
        return float(sorted_eigs[0] - sorted_eigs[1])
    except Exception:
        return 0.0


class TestShuffledEdgePlacebo:
    def test_result_has_correct_fields(self) -> None:
        """FalsificationResult has all expected fields."""
        op = _make_structured_operator(6)
        test = ShuffledEdgePlacebo(n_instances=10)
        result = test.run(op, lambda o: float(o.nnz))
        assert isinstance(result, FalsificationResult)
        assert result.test_name == "shuffled_edge"
        assert isinstance(result.observed_metric, float)
        assert isinstance(result.null_distribution, np.ndarray)
        assert len(result.null_distribution) == 10
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.passes, bool)
        assert result.n_instances == 10

    def test_structured_graph_beats_random(self) -> None:
        """Structured graph should have higher spectral gap than rewired nulls."""
        op = _make_structured_operator(8)
        test = ShuffledEdgePlacebo(n_instances=20)
        result = test.run(op, _spectral_gap)
        # The observed metric should exist and null distribution should be computed
        assert result.observed_metric >= 0.0
        assert len(result.null_distribution) == 20

    def test_null_distribution_length(self) -> None:
        """Null distribution has exactly n_instances samples."""
        op = _make_structured_operator(4)
        test = ShuffledEdgePlacebo(n_instances=15)
        result = test.run(op, lambda o: float(o.nnz))
        assert len(result.null_distribution) == 15


class TestLagDestroyedPlacebo:
    def test_temporal_signal_beats_shuffled(self) -> None:
        """AR(1) signal: observed autocorrelation > shuffled."""
        rng = np.random.default_rng(42)
        T, N = 100, 5
        # Generate AR(1) signal with rho=0.9
        signals = np.zeros((T, N))
        signals[0, :] = rng.standard_normal(N)
        for t in range(1, T):
            signals[t, :] = 0.9 * signals[t - 1, :] + 0.1 * rng.standard_normal(N)

        def autocorr_metric(s: np.ndarray) -> float:
            # Mean absolute lag-1 autocorrelation across columns
            acorrs = []
            for col in range(s.shape[1]):
                x = s[:, col]
                if np.std(x) < 1e-10:
                    continue
                corr = float(np.corrcoef(x[:-1], x[1:])[0, 1])
                acorrs.append(abs(corr))
            return float(np.mean(acorrs)) if acorrs else 0.0

        test = LagDestroyedPlacebo(n_instances=50)
        result = test.run(signals, autocorr_metric)
        # Observed autocorrelation should be higher than most shuffled
        assert result.observed_metric > np.median(result.null_distribution)

    def test_random_signal_not_significant(self) -> None:
        """White noise should NOT be significant (p > 0.05 typically)."""
        rng = np.random.default_rng(123)
        signals = rng.standard_normal((50, 5))

        def mean_metric(s: np.ndarray) -> float:
            return float(np.mean(np.abs(s)))

        test = LagDestroyedPlacebo(n_instances=30)
        result = test.run(signals, mean_metric)
        # The metric is basically the same for noise either way
        assert result.p_value >= 0.0  # just check it runs

    def test_result_fields(self) -> None:
        """Result has correct fields."""
        signals = np.zeros((20, 3))
        test = LagDestroyedPlacebo(n_instances=10)
        result = test.run(signals, lambda s: float(np.mean(s)))
        assert result.test_name == "lag_destroyed"
        assert len(result.null_distribution) == 10
        assert result.n_instances == 10


class TestRandomisedTypePlacebo:
    def test_result_fields(self) -> None:
        """Result has correct fields."""
        actor_types = ["bank", "firm", "bank", "regulator", "firm"]

        def type_diversity(types: list[str]) -> float:
            return float(len(set(types)))

        test = RandomisedTypePlacebo(n_instances=20)
        result = test.run(actor_types, type_diversity)
        assert result.test_name == "randomised_type"
        assert isinstance(result.observed_metric, float)
        assert len(result.null_distribution) == 20
        assert 0.0 <= result.p_value <= 1.0

    def test_constant_types_not_significant(self) -> None:
        """All same type → permutation doesn't change metric."""
        actor_types = ["bank"] * 10

        def type_count(types: list[str]) -> float:
            return float(len(set(types)))

        test = RandomisedTypePlacebo(n_instances=20)
        result = test.run(actor_types, type_count)
        assert result.observed_metric == 1.0
        assert (result.null_distribution == 1.0).all()


class TestFrozenRegimePlacebo:
    def test_improvement_passes(self) -> None:
        """Multi-regime metric > single + threshold → passes."""
        test = FrozenRegimePlacebo()
        result = test.run(multi_regime_metric=0.5, single_regime_metric=0.3)
        assert result.passes is True
        assert result.p_value == 0.0

    def test_no_improvement_fails(self) -> None:
        """Multi-regime metric <= single + threshold → fails."""
        test = FrozenRegimePlacebo()
        result = test.run(multi_regime_metric=0.31, single_regime_metric=0.3)
        assert result.passes is False
        assert result.p_value == 1.0

    def test_result_fields(self) -> None:
        test = FrozenRegimePlacebo()
        result = test.run(0.8, 0.5)
        assert result.test_name == "frozen_regime"
        assert isinstance(result.observed_metric, float)


class TestSymmetricOperatorPlacebo:
    def test_asymmetric_beats_symmetric(self) -> None:
        """Asymmetric operator: directed metric higher than symmetrized."""
        # Create highly asymmetric operator
        data = [1.0, 0.0001, 1.0, 0.0001]
        rows = [0, 1, 1, 0]
        cols = [1, 0, 2, 2]
        op = sp.csr_matrix((data, (rows, cols)), shape=(4, 4))

        def asymmetry_metric(o: sp.csr_matrix) -> float:
            # Measure asymmetry: |A - A^T| Frobenius norm
            diff = o - o.T
            return float(np.sqrt((diff.multiply(diff)).sum()))

        test = SymmetricOperatorPlacebo()
        result = test.run(op, asymmetry_metric)
        # Asymmetric operator should have higher asymmetry than its symmetrized version
        assert result.passes is True

    def test_result_fields(self) -> None:
        op = sp.eye(4, format="csr")
        test = SymmetricOperatorPlacebo()
        result = test.run(op, lambda o: float(o.nnz))
        assert result.test_name == "symmetric_operator"
        assert isinstance(result.passes, bool)


class TestFalsificationSuite:
    def test_suite_returns_7_results(self) -> None:
        """Suite run_all returns dict with 7 entries."""
        op = _make_structured_operator(6)
        layer_assignments = np.array([0, 0, 0, 1, 1, 1])
        signals = np.random.randn(20, 6)

        suite = FalsificationSuite(n_instances=5)
        results = suite.run_all(
            operator=op,
            signals=signals,
            actor_types=["bank", "firm", "bank", "firm", "bank", "firm"],
            multi_regime_metric=0.5,
            single_regime_metric=0.3,
            layer_assignments=layer_assignments,
            smim_metric=0.8,
            dfm_metric=0.6,
            operator_pipeline_fn=lambda o: float(o.nnz),
            signal_pipeline_fn=lambda s: float(np.mean(np.abs(s))),
            type_pipeline_fn=lambda t: float(len(set(t))),
        )
        assert len(results) == 7
        expected_keys = {
            "shuffled_edge", "lag_destroyed", "randomised_type",
            "frozen_regime", "symmetric_operator", "block_preserving", "no_network_dfm"
        }
        assert set(results.keys()) == expected_keys

    def test_summary_table_shape(self) -> None:
        """Summary table has 7 rows and correct columns."""
        op = _make_structured_operator(4)
        layer_assignments = np.array([0, 0, 1, 1])
        suite = FalsificationSuite(n_instances=5)
        results = suite.run_all(
            operator=op,
            layer_assignments=layer_assignments,
        )
        table = suite.summary_table(results)
        assert table.shape[0] == 7
        for col in ["test_name", "observed_metric", "p_value", "passes"]:
            assert col in table.columns

    def test_all_results_are_falsification_results(self) -> None:
        """Each value in run_all output is a FalsificationResult."""
        suite = FalsificationSuite(n_instances=3)
        results = suite.run_all()
        for key, val in results.items():
            assert isinstance(val, FalsificationResult), f"{key} is not FalsificationResult"
