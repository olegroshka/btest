"""Rolling OOS evaluation harness for SMIM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class OOSEvaluationResult:
    """Result of rolling OOS evaluation."""

    windows: list[tuple]  # (train_start, train_end, holdout_end)
    per_window_metrics: list[dict]  # dict per window with keys r2, rmse, n_obs
    mean_r2: float
    mean_rmse: float
    n_windows: int
    strategy: str


class RollingOOSEvaluator:
    """Rolling out-of-sample evaluation harness.

    Args:
        strategy: One of "expanding", "rolling_5yr", "rolling_10yr"
        window_years: Used when strategy is "rolling_Xyr" (default 5)
    """

    def __init__(self, strategy: str = "expanding", window_years: int = 5) -> None:
        self.strategy = strategy
        self.window_years = window_years

    def generate_windows(
        self,
        dates: list[pd.Timestamp],
        min_train_years: int = 3,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, train_end, holdout_end) tuples.

        Args:
            dates: Sorted list of dates in the dataset.
            min_train_years: Minimum number of years required in training set.

        Returns:
            List of (train_start, train_end, holdout_end) tuples.
        """
        if len(dates) < 2:
            return []

        dates = sorted(dates)
        first_date = dates[0]
        last_date = dates[-1]

        # Minimum training period offset
        min_train_delta = pd.DateOffset(years=min_train_years)

        # Determine holdout step (one period = gap between consecutive dates)
        # We step holdout forward one period at a time
        windows = []

        # For each possible train_end, holdout_end is the next date
        # train_start depends on strategy
        for i in range(len(dates) - 1):
            train_end = dates[i]
            holdout_end = dates[i + 1]

            if self.strategy == "expanding":
                train_start = first_date
            elif self.strategy in ("rolling_5yr", "rolling_10yr"):
                yrs = 5 if self.strategy == "rolling_5yr" else 10
                train_start = train_end - pd.DateOffset(years=yrs)
                # Clamp to first available date
                if train_start < first_date:
                    train_start = first_date
            else:
                train_start = first_date - pd.DateOffset(years=self.window_years)
                if train_start < first_date:
                    train_start = first_date

            # Check minimum training period
            if train_end < train_start + min_train_delta:
                continue

            windows.append((train_start, train_end, holdout_end))

        return windows

    def evaluate(
        self,
        pipeline_fn: Callable,
        observations: np.ndarray,
        dates: list[pd.Timestamp],
        as_of_fn: Callable[[pd.Timestamp], pd.Timestamp] | None = None,
    ) -> OOSEvaluationResult:
        """Evaluate pipeline on rolling OOS windows.

        Args:
            pipeline_fn: fn(train_obs, train_dates, as_of) -> np.ndarray of predicted holdout
            observations: (T,) or (N, T) array of observations
            dates: list of T dates corresponding to observations
            as_of_fn: optional fn to map train_end -> as_of timestamp

        Returns:
            OOSEvaluationResult
        """
        dates = sorted(dates)
        windows = self.generate_windows(dates, min_train_years=3)

        per_window_metrics = []
        date_to_idx = {d: i for i, d in enumerate(dates)}

        for train_start, train_end, holdout_end in windows:
            # Gather train indices
            train_idx = [
                date_to_idx[d]
                for d in dates
                if train_start <= d <= train_end and d in date_to_idx
            ]
            holdout_idx = [
                date_to_idx[d]
                for d in dates
                if d == holdout_end and d in date_to_idx
            ]

            if not train_idx or not holdout_idx:
                continue

            # Slice observations
            if observations.ndim == 1:
                train_obs = observations[train_idx]
                actual = observations[holdout_idx]
            else:
                train_obs = observations[:, train_idx]
                actual = observations[:, holdout_idx]

            train_dates = [dates[i] for i in train_idx]
            as_of = as_of_fn(train_end) if as_of_fn is not None else train_end

            # Call pipeline
            predicted = pipeline_fn(train_obs, train_dates, as_of)

            predicted_flat = np.asarray(predicted).flatten()
            actual_flat = np.asarray(actual).flatten()

            n_obs = len(actual_flat)
            if n_obs == 0:
                continue

            # Compute metrics
            ss_res = np.sum((actual_flat - predicted_flat) ** 2)
            ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            rmse = float(np.sqrt(np.mean((actual_flat - predicted_flat) ** 2)))

            per_window_metrics.append({"r2": r2, "rmse": rmse, "n_obs": n_obs})

        if per_window_metrics:
            mean_r2 = float(np.mean([m["r2"] for m in per_window_metrics]))
            mean_rmse = float(np.mean([m["rmse"] for m in per_window_metrics]))
        else:
            mean_r2 = 0.0
            mean_rmse = 0.0

        return OOSEvaluationResult(
            windows=windows,
            per_window_metrics=per_window_metrics,
            mean_r2=mean_r2,
            mean_rmse=mean_rmse,
            n_windows=len(per_window_metrics),
            strategy=self.strategy,
        )
