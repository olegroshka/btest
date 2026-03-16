"""Tests for rolling OOS evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.smim.validation.rolling_oos import (
    OOSEvaluationResult,
    RollingOOSEvaluator,
)


def _make_quarterly_dates(n_years: int, start: str = "2000-01-01") -> list[pd.Timestamp]:
    """Generate quarterly dates for n_years."""
    idx = pd.date_range(start=start, periods=n_years * 4, freq="QS")
    return list(idx)


class TestExpandingWindowCount:
    def test_expanding_window_count(self) -> None:
        """With 10 years of quarterly dates and min_train_years=3, at least 5 windows."""
        dates = _make_quarterly_dates(10)
        evaluator = RollingOOSEvaluator(strategy="expanding")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        assert len(windows) >= 5

    def test_windows_cover_last_date(self) -> None:
        """Last holdout_end should be the last date."""
        dates = _make_quarterly_dates(8)
        evaluator = RollingOOSEvaluator(strategy="expanding")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        assert len(windows) > 0
        assert windows[-1][2] == dates[-1]


class TestRollingWindowFixedSize:
    def test_rolling_window_fixed_size(self) -> None:
        """rolling_5yr produces windows where train period is ~5 years."""
        dates = _make_quarterly_dates(12)
        evaluator = RollingOOSEvaluator(strategy="rolling_5yr")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        assert len(windows) > 0
        # Once we have enough data, each window should be at most 5yr + 1 period
        for train_start, train_end, holdout_end in windows:
            span_years = (train_end - train_start).days / 365.25
            assert span_years <= 5.5, f"Window too long: {span_years:.1f}yr"


class TestAsOfSetCorrectly:
    def test_as_of_set_correctly(self) -> None:
        """as_of passed to pipeline_fn equals train_end."""
        dates = _make_quarterly_dates(8)
        obs = np.random.randn(len(dates))

        recorded_as_of = []

        def pipeline_fn(train_obs, train_dates, as_of):
            recorded_as_of.append(as_of)
            return np.array([np.mean(train_obs)])

        evaluator = RollingOOSEvaluator(strategy="expanding")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        result = evaluator.evaluate(pipeline_fn, obs, dates)

        assert len(recorded_as_of) > 0
        for i, (_, train_end, _) in enumerate(windows):
            if i < len(recorded_as_of):
                assert recorded_as_of[i] == train_end


class TestHoldoutNonOverlapping:
    def test_holdout_non_overlapping(self) -> None:
        """Consecutive windows don't share holdout dates."""
        dates = _make_quarterly_dates(10)
        evaluator = RollingOOSEvaluator(strategy="expanding")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        holdout_ends = [w[2] for w in windows]
        # All holdout_ends should be unique
        assert len(holdout_ends) == len(set(holdout_ends))

    def test_holdout_end_after_train_end(self) -> None:
        """holdout_end is strictly after train_end for every window."""
        dates = _make_quarterly_dates(10)
        evaluator = RollingOOSEvaluator(strategy="expanding")
        windows = evaluator.generate_windows(dates, min_train_years=3)
        for train_start, train_end, holdout_end in windows:
            assert holdout_end > train_end


class TestEvaluateReturnsResult:
    def test_evaluate_returns_result(self) -> None:
        """Smoke test with identity pipeline_fn."""
        dates = _make_quarterly_dates(8)
        obs = np.ones(len(dates))

        def pipeline_fn(train_obs, train_dates, as_of):
            # Predict the mean of training
            return np.array([float(np.mean(train_obs))])

        evaluator = RollingOOSEvaluator(strategy="expanding")
        result = evaluator.evaluate(pipeline_fn, obs, dates)

        assert isinstance(result, OOSEvaluationResult)
        assert result.n_windows >= 1
        assert result.strategy == "expanding"
        assert len(result.per_window_metrics) == result.n_windows
        for m in result.per_window_metrics:
            assert "r2" in m
            assert "rmse" in m
            assert "n_obs" in m
