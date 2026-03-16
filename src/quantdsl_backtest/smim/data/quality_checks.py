"""Data quality checks for SMIM PIT data.

Provides functions to validate that data stored in a PointInTimeStore
meets A1 (no look-ahead bias) requirements, coverage thresholds, and
consistency constraints.

CLI usage::

    uv run python -m quantdsl_backtest.smim.data.quality_checks \\
        --store-path smim_pit_store \\
        --as-of 2023-01-01

"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantdsl_backtest.smim.data.pit_store import PointInTimeStore

logger = logging.getLogger(__name__)


# ── result dataclasses ───────────────────────────────────────────────────────

@dataclass
class LeakViolation:
    """A row where pub_date < event_date (impossible unless data is fabricated)."""
    actor_id: str
    signal_id: str
    event_date: pd.Timestamp
    pub_date: pd.Timestamp
    source: str


@dataclass
class LeakDetectionResult:
    """Result of a leak-detection scan over the PIT store."""
    violations: list[LeakViolation] = field(default_factory=list)
    rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)


@dataclass
class MissingnessResult:
    """Per-signal missingness statistics."""
    signal_id: str
    actor_count: int
    missing_fraction: float
    dates_checked: int


@dataclass
class RangeViolation:
    """A value outside the expected numeric range."""
    actor_id: str
    signal_id: str
    event_date: pd.Timestamp
    value: float
    lower: float | None
    upper: float | None


@dataclass
class FrequencyResult:
    """Actual vs expected frequency for a signal."""
    signal_id: str
    expected_freq: str
    median_days_between_obs: float
    within_tolerance: bool


@dataclass
class CrossSourceDiff:
    """A large discrepancy between two sources for the same observation."""
    actor_id: str
    signal_id: str
    event_date: pd.Timestamp
    source_a: str
    value_a: float
    source_b: str
    value_b: float
    relative_diff: float


@dataclass
class EntityResolutionIssue:
    """An actor_id that appears in the store but not in the ActorRegistry."""
    actor_id: str
    source: str


# ── M1.3-T3: leak_detection_test ─────────────────────────────────────────────

def leak_detection_test(
    store: PointInTimeStore,
    sources: list[str] | None = None,
) -> LeakDetectionResult:
    """Scan the PIT store for A1 violations.

    A leak violation is any row where ``pub_date < event_date``.
    Such rows are impossible under correct A1 bookkeeping — if they exist,
    it means either the pub_date or event_date was recorded incorrectly.

    Note: ``pub_date == event_date`` is allowed (e.g. market data where the
    price is published on the same day it occurred).

    Args:
        store: PointInTimeStore instance to scan.
        sources: If provided, only check these sources.

    Returns:
        LeakDetectionResult with a list of violations and total rows checked.
    """
    result = LeakDetectionResult()

    for path in store._parquet_paths(sources):
        df = pd.read_parquet(path)
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
        df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
        result.rows_checked += len(df)

        violations_mask = df["pub_date"] < df["event_date"]
        for _, row in df[violations_mask].iterrows():
            result.violations.append(
                LeakViolation(
                    actor_id=str(row["actor_id"]),
                    signal_id=str(row["signal_id"]),
                    event_date=row["event_date"],
                    pub_date=row["pub_date"],
                    source=str(row["source"]),
                )
            )

    return result


# ── M1.4-T1: additional checks ───────────────────────────────────────────────

def missingness_check(
    store: PointInTimeStore,
    signal_ids: list[str],
    as_of: pd.Timestamp,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[MissingnessResult]:
    """Compute per-signal missingness across all actors.

    Args:
        store: PointInTimeStore to query.
        signal_ids: Signals to check.
        as_of: PIT cutoff (A1).
        date_range: Optional event_date window.

    Returns:
        List of MissingnessResult, one per signal.
    """
    df = store.query(as_of=as_of, signal_ids=signal_ids, date_range=date_range)
    results: list[MissingnessResult] = []

    for sid in signal_ids:
        subset = df[df["signal_id"] == sid]
        if subset.empty:
            results.append(MissingnessResult(
                signal_id=sid, actor_count=0, missing_fraction=1.0, dates_checked=0
            ))
            continue

        actors = subset["actor_id"].nunique()
        dates = subset["event_date"].nunique()
        total_cells = actors * dates
        present = len(subset)
        missing_frac = 1.0 - (present / total_cells) if total_cells > 0 else 1.0
        results.append(MissingnessResult(
            signal_id=sid,
            actor_count=actors,
            missing_fraction=missing_frac,
            dates_checked=dates,
        ))

    return results


def range_check(
    store: PointInTimeStore,
    signal_id: str,
    as_of: pd.Timestamp,
    lower: float | None = None,
    upper: float | None = None,
) -> list[RangeViolation]:
    """Find values outside [lower, upper] for a given signal.

    Args:
        store: PointInTimeStore to query.
        signal_id: Signal to check.
        as_of: PIT cutoff (A1).
        lower: Minimum acceptable value (inclusive). None = no lower bound.
        upper: Maximum acceptable value (inclusive). None = no upper bound.

    Returns:
        List of RangeViolation for out-of-bounds observations.
    """
    df = store.query(as_of=as_of, signal_ids=[signal_id])
    violations: list[RangeViolation] = []

    if df.empty:
        return violations

    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask = mask & (df["value"] >= lower)
    if upper is not None:
        mask = mask & (df["value"] <= upper)

    for _, row in df[~mask].iterrows():
        violations.append(RangeViolation(
            actor_id=str(row["actor_id"]),
            signal_id=str(row["signal_id"]),
            event_date=row["event_date"],
            value=float(row["value"]),
            lower=lower,
            upper=upper,
        ))

    return violations


def frequency_check(
    store: PointInTimeStore,
    signal_id: str,
    actor_id: str,
    as_of: pd.Timestamp,
    expected_freq: str = "Q",
    tolerance_days: int = 15,
) -> FrequencyResult:
    """Check that observations arrive at the expected frequency.

    Args:
        store: PointInTimeStore to query.
        signal_id: Signal to check.
        actor_id: Actor to check.
        as_of: PIT cutoff.
        expected_freq: Pandas offset alias ("Q" = quarterly, "M" = monthly, etc.).
        tolerance_days: Allowed deviation from expected period in days.

    Returns:
        FrequencyResult with actual median days between observations.
    """
    _EXPECTED_DAYS = {"D": 1, "W": 7, "M": 30, "Q": 91, "A": 365}
    df = store.query(as_of=as_of, actor_ids=[actor_id], signal_ids=[signal_id])
    df = df.sort_values("event_date")

    if len(df) < 2:
        return FrequencyResult(
            signal_id=signal_id,
            expected_freq=expected_freq,
            median_days_between_obs=float("nan"),
            within_tolerance=False,
        )

    diffs = df["event_date"].diff().dropna().dt.days
    median_gap = float(diffs.median())
    expected_days = _EXPECTED_DAYS.get(expected_freq.upper(), 91)
    within_tol = abs(median_gap - expected_days) <= tolerance_days

    return FrequencyResult(
        signal_id=signal_id,
        expected_freq=expected_freq,
        median_days_between_obs=median_gap,
        within_tolerance=within_tol,
    )


def cross_source_consistency(
    store: PointInTimeStore,
    signal_id: str,
    as_of: pd.Timestamp,
    source_a: str,
    source_b: str,
    threshold: float = 0.05,
) -> list[CrossSourceDiff]:
    """Find observations where two sources disagree by more than ``threshold``.

    Relative difference is ``|a - b| / max(|a|, |b|, 1e-9)``.

    Args:
        store: PointInTimeStore to query.
        signal_id: Signal to compare across sources.
        as_of: PIT cutoff.
        source_a: First source name.
        source_b: Second source name.
        threshold: Maximum allowed relative difference (default 5%).

    Returns:
        List of CrossSourceDiff for discrepant observations.
    """
    df = store.query(as_of=as_of, signal_ids=[signal_id], sources=[source_a, source_b])
    if df.empty:
        return []

    a = df[df["source"] == source_a].set_index(["actor_id", "event_date"])["value"]
    b = df[df["source"] == source_b].set_index(["actor_id", "event_date"])["value"]
    common = a.index.intersection(b.index)

    diffs: list[CrossSourceDiff] = []
    for idx in common:
        va, vb = float(a[idx]), float(b[idx])
        rel_diff = abs(va - vb) / max(abs(va), abs(vb), 1e-9)
        if rel_diff > threshold:
            actor, event_date = idx
            diffs.append(CrossSourceDiff(
                actor_id=str(actor),
                signal_id=signal_id,
                event_date=event_date,
                source_a=source_a,
                value_a=va,
                source_b=source_b,
                value_b=vb,
                relative_diff=rel_diff,
            ))

    return diffs


def entity_resolution_check(
    store: PointInTimeStore,
    known_actor_ids: list[str],
    source: str | None = None,
) -> list[EntityResolutionIssue]:
    """Find actor_ids in the store that are not in the known registry.

    Args:
        store: PointInTimeStore to scan.
        known_actor_ids: Complete list of valid actor_ids from ActorRegistry.
        source: If provided, only check this source.

    Returns:
        List of EntityResolutionIssue for unknown actors.
    """
    known = set(known_actor_ids)
    issues: list[EntityResolutionIssue] = []

    for path in store._parquet_paths([source] if source else None):
        df = pd.read_parquet(path)[["actor_id", "source"]]
        for actor_id in df["actor_id"].unique():
            if str(actor_id) not in known:
                issues.append(EntityResolutionIssue(
                    actor_id=str(actor_id),
                    source=str(df[df["actor_id"] == actor_id]["source"].iloc[0]),
                ))

    return issues


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quality_checks",
        description="Run SMIM PIT store quality checks.",
    )
    p.add_argument("--store-path", required=True, help="Path to PointInTimeStore root dir")
    p.add_argument("--as-of", required=False, help="PIT cutoff date (YYYY-MM-DD)")
    p.add_argument("--check", choices=["leak", "all"], default="all")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = PointInTimeStore(root_dir=Path(args.store_path))

    exit_code = 0

    if args.check in ("leak", "all"):
        result = leak_detection_test(store)
        if result.passed:
            print(f"[PASS] Leak detection: 0 violations in {result.rows_checked} rows.")
        else:
            print(
                f"[FAIL] Leak detection: {result.violation_count} violations "
                f"in {result.rows_checked} rows."
            )
            for v in result.violations[:10]:
                print(
                    f"  {v.source}/{v.actor_id}/{v.signal_id}: "
                    f"pub={v.pub_date.date()} < event={v.event_date.date()}"
                )
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
