"""Tests for smim/data/adapters/gdelt.py (parquet-based file adapter).

The adapter was rewritten from an HTTP DOC-API client to a file-based parquet
reader (scripts/smim/smim_fetch_gdelt.py produces the canonical parquets).
Tests use a temporary parquet fixture to stay self-contained.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import GdeltConfig
from quantdsl_backtest.smim.data.adapters.gdelt import GdeltAdapter
from quantdsl_backtest.smim.interfaces import DateRange


def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-03-31"),
    )


def _make_weekly_parquet(rows: list[dict], path: Path) -> Path:
    """Write a minimal weekly-format parquet to *path* and return it."""
    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    """Parquet with two signals across two weeks."""
    rows = [
        {"week_start": "2020-01-06", "theme_or_actor": "sector_energy",
         "article_count": 100.0, "avg_tone": -2.5, "intensity": 0.012},
        {"week_start": "2020-01-06", "theme_or_actor": "sector_tech",
         "article_count": 80.0, "avg_tone": 1.0, "intensity": 0.008},
        {"week_start": "2020-01-13", "theme_or_actor": "sector_energy",
         "article_count": 120.0, "avg_tone": -3.0, "intensity": 0.015},
    ]
    p = tmp_path / "gdelt_narrative.parquet"
    return _make_weekly_parquet(rows, p)


# ── GdeltAdapter ──────────────────────────────────────────────────────────────

class TestGdeltAdapter:
    def _make_adapter(self, parquet_path: Path | None = None) -> GdeltAdapter:
        config = GdeltConfig(themes=["sector_energy"])
        return GdeltAdapter(config=config, parquet_path=parquet_path)

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "gdelt"

    def test_returns_count_and_tone_columns(self, tmp_parquet: Path) -> None:
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        assert "sector_energy_count" in df.columns
        assert "sector_energy_tone" in df.columns
        assert "sector_energy_intensity" in df.columns

    def test_index_is_event_date(self, tmp_parquet: Path) -> None:
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        assert df.index.name == "event_date"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_volume_is_correct(self, tmp_parquet: Path) -> None:
        """article_count for a given week should match the parquet value."""
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        val = df.loc[pd.Timestamp("2020-01-06"), "sector_energy_count"]
        assert val == pytest.approx(100.0)

    def test_tone_is_correct(self, tmp_parquet: Path) -> None:
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        val = df.loc[pd.Timestamp("2020-01-06"), "sector_energy_tone"]
        assert val == pytest.approx(-2.5)

    def test_as_of_restricts_rows(self, tmp_parquet: Path) -> None:
        """as_of cuts off rows with week_start > as_of."""
        as_of = pd.Timestamp("2020-01-10")
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range(), as_of=as_of)
        assert pd.Timestamp("2020-01-13") not in df.index
        assert pd.Timestamp("2020-01-06") in df.index

    def test_flat_timeline_format_parsed(self, tmp_parquet: Path) -> None:
        """Rows within the date_range window appear in the output."""
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        assert pd.Timestamp("2020-01-13") in df.index

    def test_empty_result_when_no_matching_rows(self, tmp_parquet: Path) -> None:
        narrow = DateRange(
            start=pd.Timestamp("2019-01-01"),
            end=pd.Timestamp("2019-12-31"),
        )
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], narrow)
        assert df.empty

    def test_empty_series_ids_returns_all_signals(self, tmp_parquet: Path) -> None:
        """fetch with empty list should return all signals in the parquet."""
        df = self._make_adapter(tmp_parquet).fetch([], _date_range())
        assert "sector_energy_count" in df.columns
        assert "sector_tech_count" in df.columns

    def test_multiple_themes(self, tmp_parquet: Path) -> None:
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy", "sector_tech"], _date_range())
        assert "sector_energy_count" in df.columns
        assert "sector_tech_count" in df.columns

    def test_output_dtype_is_float64(self, tmp_parquet: Path) -> None:
        df = self._make_adapter(tmp_parquet).fetch(["sector_energy"], _date_range())
        assert df["sector_energy_count"].dtype == "float64"
        assert df["sector_energy_tone"].dtype == "float64"

    def test_implements_data_adapter_protocol(self, tmp_parquet: Path) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter(tmp_parquet)
        assert isinstance(adapter, DataAdapter)

    def test_missing_parquet_raises_file_not_found(self) -> None:
        adapter = GdeltAdapter(
            config=GdeltConfig(),
            parquet_path=Path("/nonexistent/path/gdelt.parquet"),
        )
        with pytest.raises(FileNotFoundError):
            adapter.fetch(["sector_energy"], _date_range())
