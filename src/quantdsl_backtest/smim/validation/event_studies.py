"""Event study framework for SMIM historical shock validation.

M5.3-T1 and M5.3-T2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats


@dataclass
class EventStudyResult:
    """Result from one event study."""

    event_name: str
    event_date: pd.Timestamp
    expected_direction: int  # +1 or -1
    pre_mean: float
    post_mean: float
    t_stat: float
    p_value: float
    significant: bool  # p_value < 0.10
    top_modal_channels: list[int]  # indices of most-affected modal channels
    metadata: dict = field(default_factory=dict)


class EventStudy:
    """Run a single event study on gaps around a known event date.

    Args:
        event_name: Human-readable event identifier.
        event_date: Date of the economic event.
        expected_direction: +1 (gaps increase) or -1 (gaps decrease).
        window_quarters: Number of quarters before/after the event to compare.
    """

    def __init__(
        self,
        event_name: str,
        event_date: pd.Timestamp,
        expected_direction: int,
        window_quarters: int = 4,
    ) -> None:
        self.event_name = event_name
        self.event_date = pd.Timestamp(event_date)
        self.expected_direction = expected_direction
        self.window_quarters = window_quarters

    def run(
        self,
        gaps: np.ndarray,
        dates: list[pd.Timestamp],
        modal_frame=None,
    ) -> EventStudyResult:
        """Run event study on gaps array.

        Args:
            gaps: (N, T) array of investment gaps.
            dates: List of T timestamps.
            modal_frame: Optional ModalFrame for modal channel attribution.

        Returns:
            EventStudyResult
        """
        dates = [pd.Timestamp(d) for d in dates]
        event_date = self.event_date

        # Approximate quarters as ~91 days
        quarter_days = 91
        window_days = self.window_quarters * quarter_days

        # Find pre-event and post-event dates
        pre_mask = [
            (event_date - pd.Timedelta(days=window_days)) <= d < event_date
            for d in dates
        ]
        post_mask = [
            event_date <= d <= (event_date + pd.Timedelta(days=window_days))
            for d in dates
        ]

        pre_idx = [i for i, m in enumerate(pre_mask) if m]
        post_idx = [i for i, m in enumerate(post_mask) if m]

        if not pre_idx or not post_idx:
            return EventStudyResult(
                event_name=self.event_name,
                event_date=event_date,
                expected_direction=self.expected_direction,
                pre_mean=float("nan"),
                post_mean=float("nan"),
                t_stat=0.0,
                p_value=1.0,
                significant=False,
                top_modal_channels=[],
                metadata={"reason": "no_dates_in_window"},
            )

        # Compute mean |Δ_{i,t}| across all actors for each time point
        pre_values = np.mean(np.abs(gaps[:, pre_idx]), axis=0)   # (n_pre,)
        post_values = np.mean(np.abs(gaps[:, post_idx]), axis=0)  # (n_post,)

        pre_mean = float(np.mean(pre_values))
        post_mean = float(np.mean(post_values))

        # Paired t-test when possible, otherwise Welch t-test
        if len(pre_values) == len(post_values) and len(pre_values) >= 2:
            t_result = scipy.stats.ttest_rel(post_values, pre_values)
        elif len(pre_values) >= 2 and len(post_values) >= 2:
            t_result = scipy.stats.ttest_ind(post_values, pre_values, equal_var=False)
        else:
            # Not enough data for a proper t-test
            return EventStudyResult(
                event_name=self.event_name,
                event_date=event_date,
                expected_direction=self.expected_direction,
                pre_mean=pre_mean,
                post_mean=post_mean,
                t_stat=0.0,
                p_value=1.0,
                significant=False,
                top_modal_channels=[],
                metadata={"reason": "insufficient_observations"},
            )

        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)
        significant = p_value < 0.10

        # Modal channel attribution
        top_modal_channels: list[int] = []
        if modal_frame is not None:
            try:
                # Variance explained per mode during post-event window
                U = modal_frame.basis  # (N, K)
                if post_idx:
                    post_gaps = gaps[:, post_idx]  # (N, n_post)
                    mode_var = [
                        float(np.var(U[:, k] @ post_gaps))
                        for k in range(U.shape[1])
                    ]
                    top_k = min(3, len(mode_var))
                    top_modal_channels = list(
                        np.argsort(mode_var)[::-1][:top_k].astype(int)
                    )
            except Exception:
                pass

        return EventStudyResult(
            event_name=self.event_name,
            event_date=event_date,
            expected_direction=self.expected_direction,
            pre_mean=pre_mean,
            post_mean=post_mean,
            t_stat=t_stat,
            p_value=p_value,
            significant=significant,
            top_modal_channels=top_modal_channels,
        )


@dataclass
class EventStudySuiteResult:
    """Result from a suite of event studies."""

    results: list[EventStudyResult]
    significant_count: int
    total_count: int
    passes: bool  # significant_count >= min(3, total_count * 0.6)


class EventStudySuite:
    """Run multiple event studies and aggregate results.

    Args:
        events: List of (name, date, direction) tuples.
        window_quarters: Number of quarters around each event date.
    """

    def __init__(
        self,
        events: list[tuple[str, pd.Timestamp, int]],
        window_quarters: int = 4,
    ) -> None:
        self.events = events
        self.window_quarters = window_quarters

    def run(
        self,
        gaps: np.ndarray,
        dates: list[pd.Timestamp],
        modal_frame=None,
    ) -> EventStudySuiteResult:
        """Run all event studies.

        Args:
            gaps: (N, T) array of investment gaps.
            dates: List of T timestamps.
            modal_frame: Optional ModalFrame for modal attribution.

        Returns:
            EventStudySuiteResult
        """
        results = []
        for name, date, direction in self.events:
            study = EventStudy(
                event_name=name,
                event_date=date,
                expected_direction=direction,
                window_quarters=self.window_quarters,
            )
            result = study.run(gaps, dates, modal_frame)
            results.append(result)

        significant_count = sum(1 for r in results if r.significant)
        total_count = len(results)
        threshold = max(3, int(total_count * 0.6))
        passes = significant_count >= min(threshold, total_count)

        return EventStudySuiteResult(
            results=results,
            significant_count=significant_count,
            total_count=total_count,
            passes=passes,
        )

    def summary_table(self, result: EventStudySuiteResult) -> pd.DataFrame:
        """Build summary table from suite result.

        Returns:
            DataFrame with columns: event, date, expected_dir, pre_mean,
            post_mean, t_stat, p_value, significant
        """
        rows = []
        for r in result.results:
            rows.append({
                "event": r.event_name,
                "date": r.event_date,
                "expected_dir": r.expected_direction,
                "pre_mean": r.pre_mean,
                "post_mean": r.post_mean,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "significant": r.significant,
            })
        return pd.DataFrame(rows)
