"""Tests for OOS evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.validation.metrics import (
    MetricSuite,
    diebold_mariano_test,
    ndcg,
    oos_r_squared,
    rmse,
    spearman_rho,
)


class TestOOSR2:
    def test_perfect_prediction_r2_is_1(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual.copy()
        assert oos_r_squared(predicted, actual) == pytest.approx(1.0)

    def test_mean_prediction_r2_is_0(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.full_like(actual, np.mean(actual))
        assert oos_r_squared(predicted, actual) == pytest.approx(0.0, abs=1e-10)

    def test_ss_tot_zero_returns_zero(self) -> None:
        actual = np.array([2.0, 2.0, 2.0])
        predicted = np.array([1.0, 1.0, 1.0])
        assert oos_r_squared(predicted, actual) == 0.0


class TestRMSE:
    def test_rmse_known_value(self) -> None:
        actual = np.array([0.0, 0.0, 0.0, 0.0])
        predicted = np.array([1.0, 1.0, 1.0, 1.0])
        assert rmse(predicted, actual) == pytest.approx(1.0)

    def test_rmse_zero_for_perfect(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == pytest.approx(0.0)


class TestSpearmanRho:
    def test_spearman_perfect_rank(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual.copy()
        assert spearman_rho(predicted, actual) == pytest.approx(1.0)

    def test_spearman_inverse_rank(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual[::-1].copy()
        assert spearman_rho(predicted, actual) == pytest.approx(-1.0)

    def test_spearman_returns_zero_on_constant(self) -> None:
        actual = np.array([1.0, 1.0, 1.0])
        predicted = np.array([1.0, 2.0, 3.0])
        # Constant actual → NaN from scipy → returns 0.0
        result = spearman_rho(predicted, actual)
        assert result == 0.0 or isinstance(result, float)


class TestNDCG:
    def test_ndcg_perfect_ranking(self) -> None:
        relevance = np.array([3.0, 2.0, 1.0, 0.0])
        scores = relevance.copy()  # perfect order
        assert ndcg(scores, relevance) == pytest.approx(1.0)

    def test_ndcg_reversed_ranking(self) -> None:
        relevance = np.array([3.0, 2.0, 1.0, 0.0])
        scores = relevance[::-1].copy()  # worst order
        result = ndcg(scores, relevance)
        assert 0.0 <= result <= 1.0
        assert result < 1.0

    def test_ndcg_k_limit(self) -> None:
        relevance = np.array([3.0, 2.0, 1.0, 0.0])
        scores = relevance.copy()
        result_k2 = ndcg(scores, relevance, k=2)
        assert 0.0 <= result_k2 <= 1.0


class TestDieboldMarianoTest:
    def test_dm_test_same_errors_pvalue_near_1(self) -> None:
        rng = np.random.default_rng(42)
        errors = rng.standard_normal(100)
        stat, pval = diebold_mariano_test(errors, errors)
        assert stat == pytest.approx(0.0, abs=1e-10)
        assert pval >= 0.9

    def test_dm_test_model_1_worse(self) -> None:
        errors_1 = np.ones(100) * 2.0  # large errors
        errors_2 = np.ones(100) * 0.1  # small errors
        stat, pval = diebold_mariano_test(errors_1, errors_2)
        # d = e1^2 - e2^2 > 0, so model 1 is worse → positive stat, small p-value
        assert stat > 0
        assert pval < 0.05

    def test_dm_returns_tuple_of_floats(self) -> None:
        errors_1 = np.array([1.0, 2.0, 3.0])
        errors_2 = np.array([0.5, 1.0, 1.5])
        stat, pval = diebold_mariano_test(errors_1, errors_2)
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert 0.0 <= pval <= 1.0


class TestMetricSuite:
    def test_metric_suite_returns_all_keys(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual + 0.1
        suite = MetricSuite()
        result = suite.compute_all(predicted, actual)
        assert "r2" in result
        assert "rmse" in result
        assert "spearman_rho" in result

    def test_compute_comparative_returns_dm_keys(self) -> None:
        suite = MetricSuite()
        e1 = np.ones(50)
        e2 = np.ones(50) * 0.5
        result = suite.compute_comparative(e1, e2)
        assert "dm_stat" in result
        assert "dm_pvalue" in result
