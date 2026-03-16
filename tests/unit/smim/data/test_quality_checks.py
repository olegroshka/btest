"""Tests for smim/data/quality_checks.py — M1.3-T3 and M1.4-T1."""

from __future__ import annotations

import pandas as pd
import pytest

from quantdsl_backtest.smim.data.pit_store import PointInTimeStore
from quantdsl_backtest.smim.data.quality_checks import (
    CrossSourceDiff,
    EntityResolutionIssue,
    LeakDetectionResult,
    cross_source_consistency,
    entity_resolution_check,
    frequency_check,
    leak_detection_test,
    main,
    missingness_check,
    range_check,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_row(
    actor_id: str = "AAPL",
    signal_id: str = "CapEx",
    event_date: str = "2021-12-31",
    pub_date: str = "2022-03-15",
    value: float = 1.0,
    source: str = "edgar",
) -> dict:
    return dict(
        actor_id=actor_id,
        signal_id=signal_id,
        event_date=event_date,
        pub_date=pub_date,
        value=value,
        source=source,
    )


def _store_with(*rows: dict, tmp_path) -> PointInTimeStore:
    store = PointInTimeStore(root_dir=tmp_path)
    store.ingest(pd.DataFrame(list(rows)))
    return store


# ── M1.3-T3: leak_detection_test ─────────────────────────────────────────────

class TestLeakDetectionTest:
    def test_no_violations_passes(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2021-12-31", pub_date="2022-03-15"),
            tmp_path=tmp_path,
        )
        result = leak_detection_test(store)
        assert result.passed
        assert result.violation_count == 0
        assert result.rows_checked == 1

    def test_detects_pub_before_event(self, tmp_path) -> None:
        """pub_date < event_date is a leak violation."""
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path,
        )
        result = leak_detection_test(store)
        assert not result.passed
        assert result.violation_count == 1
        v = result.violations[0]
        assert v.actor_id == "AAPL"
        assert v.signal_id == "CapEx"

    def test_pub_equals_event_is_allowed(self, tmp_path) -> None:
        """Same-day publication (e.g. market data) is not a violation."""
        store = _store_with(
            _make_row(event_date="2022-01-03", pub_date="2022-01-03"),
            tmp_path=tmp_path,
        )
        result = leak_detection_test(store)
        assert result.passed

    def test_counts_all_rows(self, tmp_path) -> None:
        store = _store_with(
            _make_row(actor_id="A"),
            _make_row(actor_id="B"),
            _make_row(actor_id="C"),
            tmp_path=tmp_path,
        )
        result = leak_detection_test(store)
        assert result.rows_checked == 3

    def test_violation_source_captured(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31", source="fred"),
            tmp_path=tmp_path,
        )
        result = leak_detection_test(store)
        assert result.violations[0].source == "fred"

    def test_empty_store_passes(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        result = leak_detection_test(store)
        assert result.passed
        assert result.rows_checked == 0


# ── CLI test ──────────────────────────────────────────────────────────────────

class TestCli:
    def test_cli_passes_on_clean_store(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        code = main(["--store-path", str(tmp_path), "--check", "leak"])
        assert code == 0

    def test_cli_fails_on_violation(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path,
        )
        code = main(["--store-path", str(tmp_path), "--check", "leak"])
        assert code == 1


# ── M1.4-T1: additional checks ───────────────────────────────────────────────

class TestMissingnessCheck:
    def test_perfect_coverage_is_zero_missing(self, tmp_path) -> None:
        store = _store_with(
            _make_row(actor_id="AAPL", signal_id="CapEx"),
            tmp_path=tmp_path,
        )
        results = missingness_check(
            store, ["CapEx"], as_of=pd.Timestamp("2023-01-01")
        )
        assert len(results) == 1
        assert results[0].missing_fraction == pytest.approx(0.0)

    def test_absent_signal_is_fully_missing(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        # ingest a different signal
        store.ingest(pd.DataFrame([_make_row(signal_id="TotalAssets")]))
        results = missingness_check(
            store, ["CapEx"], as_of=pd.Timestamp("2023-01-01")
        )
        assert results[0].missing_fraction == pytest.approx(1.0)

    def test_returns_one_result_per_signal(self, tmp_path) -> None:
        store = _store_with(
            _make_row(signal_id="CapEx"),
            _make_row(signal_id="Revenue"),
            tmp_path=tmp_path,
        )
        results = missingness_check(
            store, ["CapEx", "Revenue"], as_of=pd.Timestamp("2023-01-01")
        )
        assert len(results) == 2
        assert {r.signal_id for r in results} == {"CapEx", "Revenue"}


class TestRangeCheck:
    def test_in_range_returns_no_violations(self, tmp_path) -> None:
        store = _store_with(_make_row(value=0.5), tmp_path=tmp_path)
        violations = range_check(
            store, "CapEx", as_of=pd.Timestamp("2023-01-01"), lower=0.0, upper=1.0
        )
        assert violations == []

    def test_below_lower_bound_is_violation(self, tmp_path) -> None:
        store = _store_with(_make_row(value=-0.1), tmp_path=tmp_path)
        violations = range_check(
            store, "CapEx", as_of=pd.Timestamp("2023-01-01"), lower=0.0
        )
        assert len(violations) == 1
        assert violations[0].value == pytest.approx(-0.1)

    def test_above_upper_bound_is_violation(self, tmp_path) -> None:
        store = _store_with(_make_row(value=1.1), tmp_path=tmp_path)
        violations = range_check(
            store, "CapEx", as_of=pd.Timestamp("2023-01-01"), upper=1.0
        )
        assert len(violations) == 1

    def test_no_bounds_returns_no_violations(self, tmp_path) -> None:
        store = _store_with(_make_row(value=1e9), tmp_path=tmp_path)
        violations = range_check(store, "CapEx", as_of=pd.Timestamp("2023-01-01"))
        assert violations == []


class TestFrequencyCheck:
    def test_quarterly_within_tolerance(self, tmp_path) -> None:
        rows = [
            _make_row(event_date=f"{y}-{m:02d}-01", pub_date=f"{y}-{m+1:02d}-15")
            for y in [2020, 2021]
            for m in [1, 4, 7, 10]
        ]
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(pd.DataFrame(rows))
        result = frequency_check(
            store, "CapEx", "AAPL", as_of=pd.Timestamp("2022-01-01"), expected_freq="Q"
        )
        assert result.within_tolerance

    def test_insufficient_data_not_within_tolerance(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        result = frequency_check(
            store, "CapEx", "AAPL", as_of=pd.Timestamp("2023-01-01"), expected_freq="Q"
        )
        assert not result.within_tolerance


class TestCrossSourceConsistency:
    def test_consistent_sources_returns_empty(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(pd.DataFrame([
            _make_row(value=100.0, source="fred"),
            _make_row(value=101.0, source="oecd"),  # 1% diff < 5% threshold
        ]))
        diffs = cross_source_consistency(
            store, "CapEx", pd.Timestamp("2023-01-01"), "fred", "oecd", threshold=0.05
        )
        assert diffs == []

    def test_large_diff_is_detected(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(pd.DataFrame([
            _make_row(value=100.0, source="fred"),
            _make_row(value=200.0, source="oecd"),  # 100% diff
        ]))
        diffs = cross_source_consistency(
            store, "CapEx", pd.Timestamp("2023-01-01"), "fred", "oecd", threshold=0.05
        )
        assert len(diffs) == 1
        assert diffs[0].relative_diff == pytest.approx(0.5, abs=0.01)

    def test_no_overlap_returns_empty(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(pd.DataFrame([
            _make_row(event_date="2020-01-01", source="fred"),
            _make_row(event_date="2021-01-01", source="oecd"),  # different dates
        ]))
        diffs = cross_source_consistency(
            store, "CapEx", pd.Timestamp("2023-01-01"), "fred", "oecd"
        )
        assert diffs == []


class TestEntityResolutionCheck:
    def test_known_actor_passes(self, tmp_path) -> None:
        store = _store_with(_make_row(actor_id="AAPL"), tmp_path=tmp_path)
        issues = entity_resolution_check(store, known_actor_ids=["AAPL"])
        assert issues == []

    def test_unknown_actor_detected(self, tmp_path) -> None:
        store = _store_with(_make_row(actor_id="UNKNOWN_CIK"), tmp_path=tmp_path)
        issues = entity_resolution_check(store, known_actor_ids=["AAPL"])
        assert len(issues) == 1
        assert issues[0].actor_id == "UNKNOWN_CIK"

    def test_filtered_by_source(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(pd.DataFrame([
            _make_row(actor_id="KNOWN", source="edgar"),
            _make_row(actor_id="FOREIGN", source="fred", signal_id="FEDFUNDS"),
        ]))
        # Only check "fred" source
        issues = entity_resolution_check(
            store, known_actor_ids=["KNOWN"], source="fred"
        )
        assert len(issues) == 1
        assert issues[0].actor_id == "FOREIGN"
