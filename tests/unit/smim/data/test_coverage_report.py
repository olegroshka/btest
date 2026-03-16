"""Tests for smim/data/coverage_report.py — M1.4-T2."""

from __future__ import annotations

import pandas as pd
import pytest

from quantdsl_backtest.smim.data.coverage_report import (
    CoverageReport,
    build_report,
    main,
    render_markdown,
)
from quantdsl_backtest.smim.data.pit_store import PointInTimeStore


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


_AS_OF = pd.Timestamp("2023-01-01")


# ── build_report ──────────────────────────────────────────────────────────────

class TestBuildReport:
    def test_returns_coverage_report(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        assert isinstance(report, CoverageReport)

    def test_actor_count(self, tmp_path) -> None:
        store = _store_with(
            _make_row(actor_id="AAPL"),
            _make_row(actor_id="MSFT"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        assert report.actor_count == 2

    def test_signal_count(self, tmp_path) -> None:
        store = _store_with(
            _make_row(signal_id="CapEx"),
            _make_row(signal_id="Revenue"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        assert report.signal_count == 2

    def test_total_rows_respects_a1(self, tmp_path) -> None:
        """Rows with pub_date > as_of must not be counted."""
        store = _store_with(
            _make_row(pub_date="2022-03-15"),   # included
            _make_row(actor_id="B", pub_date="2024-01-01"),  # excluded (future)
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        assert report.total_rows == 1

    def test_sources_listed(self, tmp_path) -> None:
        store = _store_with(
            _make_row(source="edgar"),
            _make_row(source="fred", signal_id="FEDFUNDS"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        assert set(report.sources) == {"edgar", "fred"}

    def test_g1_passed_on_clean_store(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        assert report.g1_passed is True

    def test_g1_fails_on_leak_violation(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        assert report.g1_passed is False

    def test_empty_store_g1_passes(self, tmp_path) -> None:
        """Empty store has no violations — G1 passes vacuously."""
        store = PointInTimeStore(root_dir=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        assert report.g1_passed is True

    def test_signal_ids_filter_applied(self, tmp_path) -> None:
        store = _store_with(
            _make_row(signal_id="CapEx"),
            _make_row(signal_id="Revenue"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF, signal_ids=["CapEx"])
        checked = {m.signal_id for m in report.missingness}
        assert checked == {"CapEx"}


# ── render_markdown ───────────────────────────────────────────────────────────

class TestRenderMarkdown:
    def test_returns_string(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert isinstance(md, str)

    def test_contains_gate_g1_header(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "Gate G1" in md

    def test_contains_as_of_date(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "2023-01-01" in md

    def test_pass_status_in_markdown(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "PASS" in md

    def test_fail_status_in_markdown(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "FAIL" in md

    def test_leak_violations_table_present(self, tmp_path) -> None:
        store = _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path,
        )
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "Leak Detection" in md

    def test_signal_coverage_table_present(self, tmp_path) -> None:
        store = _store_with(_make_row(signal_id="CapEx"), tmp_path=tmp_path)
        report = build_report(store, as_of=_AS_OF)
        md = render_markdown(report)
        assert "CapEx" in md


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCli:
    def test_cli_writes_output_file(self, tmp_path) -> None:
        store = _store_with(_make_row(), tmp_path=tmp_path / "store")
        out = tmp_path / "report.md"
        code = main([
            "--store-path", str(tmp_path / "store"),
            "--as-of", "2023-01-01",
            "--output", str(out),
        ])
        assert out.exists()
        content = out.read_text()
        assert "Gate G1" in content

    def test_cli_returns_0_on_pass(self, tmp_path) -> None:
        _store_with(_make_row(), tmp_path=tmp_path / "store")
        code = main([
            "--store-path", str(tmp_path / "store"),
            "--as-of", "2023-01-01",
        ])
        assert code == 0

    def test_cli_returns_1_on_fail(self, tmp_path) -> None:
        _store_with(
            _make_row(event_date="2022-03-15", pub_date="2021-12-31"),
            tmp_path=tmp_path / "store",
        )
        code = main([
            "--store-path", str(tmp_path / "store"),
            "--as-of", "2023-01-01",
        ])
        assert code == 1
