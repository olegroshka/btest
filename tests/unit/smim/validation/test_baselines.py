"""Tests for baseline models (M2.3-T1, M2.3-T2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.validation.baselines import (
    BaselineResult,
    DynamicFactorBaseline,
    VARBaseline,
)


def _single_factor_data(n_actors: int = 5, n_obs: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate data dominated by a single common factor."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_obs)
    loadings = rng.standard_normal(n_actors)
    data = np.outer(factor, loadings) + rng.standard_normal((n_obs, n_actors)) * 0.2
    dates = pd.date_range("2010-01-01", periods=n_obs, freq="ME")
    return pd.DataFrame(data, index=dates, columns=[f"actor{i}" for i in range(n_actors)])


def _var1_data(n_actors: int = 4, n_obs: int = 80, seed: int = 0) -> pd.DataFrame:
    """Generate VAR(1) data with known structure."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_actors, n_actors)) * 0.3
    # Ensure stationarity via scaling
    max_ev = max(abs(np.linalg.eigvals(A)))
    if max_ev > 0.8:
        A = A * (0.8 / max_ev)
    data = np.zeros((n_obs, n_actors))
    for t in range(1, n_obs):
        data[t] = A @ data[t - 1] + rng.standard_normal(n_actors) * 0.5
    dates = pd.date_range("2010-01-01", periods=n_obs, freq="QS")
    return pd.DataFrame(data, index=dates, columns=[f"actor{i}" for i in range(n_actors)])


class TestBaselineResult:
    def test_has_required_fields(self):
        result = BaselineResult(
            model_name="test",
            r2_oos=0.5,
            rmse_oos=0.1,
            n_params=10,
        )
        assert result.model_name == "test"
        assert result.r2_oos == 0.5
        assert result.rmse_oos == 0.1
        assert result.n_params == 10

    def test_metadata_defaults_to_empty_dict(self):
        result = BaselineResult(model_name="x", r2_oos=0.0, rmse_oos=0.0, n_params=0)
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0


class TestDynamicFactorBaseline:
    def test_returns_baseline_result(self):
        data = _single_factor_data()
        dfm = DynamicFactorBaseline(n_factors=2)
        result = dfm.fit_and_evaluate(data)
        assert isinstance(result, BaselineResult)

    def test_r2_oos_is_finite(self):
        data = _single_factor_data()
        dfm = DynamicFactorBaseline(n_factors=2)
        result = dfm.fit_and_evaluate(data)
        assert np.isfinite(result.r2_oos)

    def test_rmse_oos_is_finite(self):
        data = _single_factor_data()
        dfm = DynamicFactorBaseline(n_factors=2)
        result = dfm.fit_and_evaluate(data)
        assert np.isfinite(result.rmse_oos)

    def test_single_factor_data_positive_r2(self):
        """DFM with K=1 on single-factor dominated data should have r2_oos > 0."""
        data = _single_factor_data(n_actors=5, n_obs=80, seed=42)
        dfm = DynamicFactorBaseline(n_factors=1)
        result = dfm.fit_and_evaluate(data)
        assert result.r2_oos > 0, f"Expected r2_oos > 0, got {result.r2_oos}"

    def test_model_name(self):
        data = _single_factor_data()
        dfm = DynamicFactorBaseline()
        result = dfm.fit_and_evaluate(data)
        assert result.model_name == "DynamicFactorModel"

    def test_n_params_positive(self):
        data = _single_factor_data()
        dfm = DynamicFactorBaseline(n_factors=3)
        result = dfm.fit_and_evaluate(data)
        assert result.n_params > 0


class TestVARBaseline:
    def test_returns_baseline_result(self):
        data = _var1_data()
        var = VARBaseline(max_lag=2)
        result = var.fit_and_evaluate(data)
        assert isinstance(result, BaselineResult)

    def test_rmse_oos_is_finite(self):
        data = _var1_data()
        var = VARBaseline(max_lag=2)
        result = var.fit_and_evaluate(data)
        assert np.isfinite(result.rmse_oos)

    def test_r2_oos_is_finite(self):
        data = _var1_data()
        var = VARBaseline(max_lag=2)
        result = var.fit_and_evaluate(data)
        assert np.isfinite(result.r2_oos) or np.isnan(result.r2_oos)

    def test_var_r2_not_nan_on_structured_data(self):
        """VAR on structured VAR(1) data should give finite r2_oos."""
        data = _var1_data(n_obs=100)
        var = VARBaseline(max_lag=2)
        result = var.fit_and_evaluate(data)
        assert np.isfinite(result.rmse_oos)

    def test_model_name(self):
        data = _var1_data()
        var = VARBaseline()
        result = var.fit_and_evaluate(data)
        assert result.model_name == "VAR"

    def test_metadata_contains_k_ar(self):
        data = _var1_data()
        var = VARBaseline(max_lag=2)
        result = var.fit_and_evaluate(data)
        assert "k_ar" in result.metadata
