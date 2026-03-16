"""Tests for smim/data/adapters/gdelt.py."""

from __future__ import annotations

from unittest.mock import MagicMock

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


def _make_timeline_response(dates_values: list[tuple[str, float]]) -> dict:
    """Build a mock GDELT timeline API response (nested series format)."""
    return {
        "timeline": [
            {
                "series": [
                    {"date": dt, "value": val} for dt, val in dates_values
                ]
            }
        ]
    }


def _make_flat_timeline_response(dates_values: list[tuple[str, float]]) -> dict:
    """Build a mock GDELT timeline API response (flat format)."""
    return {
        "timeline": [{"date": dt, "value": val} for dt, val in dates_values]
    }


def _make_mock_client(vol_data: dict, tone_data: dict) -> object:
    """Return a mock client that routes timelinevol vs timelinecovtone."""
    client = MagicMock()

    def _get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "timelinevol" in url:
            resp.json.return_value = vol_data
        else:
            resp.json.return_value = tone_data
        return resp

    client.get.side_effect = _get
    return client


# ── GdeltAdapter ──────────────────────────────────────────────────────────────

class TestGdeltAdapter:
    def _make_adapter(self, http_client=None) -> GdeltAdapter:
        config = GdeltConfig(themes=["ECON_ENERGY"])
        return GdeltAdapter(config=config, http_client=http_client or MagicMock())

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "gdelt"

    def test_returns_count_and_tone_columns(self) -> None:
        vol = _make_timeline_response([("20200101000000", 10.0), ("20200102000000", 12.0)])
        tone = _make_timeline_response([("20200101000000", -2.5), ("20200102000000", -3.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        assert "ECON_ENERGY_count" in df.columns
        assert "ECON_ENERGY_tone" in df.columns

    def test_index_is_event_date(self) -> None:
        vol = _make_timeline_response([("20200101000000", 10.0)])
        tone = _make_timeline_response([("20200101000000", -1.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        assert df.index.name == "event_date"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_volume_is_daily_sum(self) -> None:
        """Multiple intra-day rows for same date should be summed."""
        vol = _make_timeline_response([
            ("20200101000000", 5.0),
            ("20200101060000", 3.0),  # same day
        ])
        tone = _make_timeline_response([("20200101000000", -1.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        day_val = df.loc[pd.Timestamp("2020-01-01"), "ECON_ENERGY_count"]
        assert day_val == pytest.approx(8.0)

    def test_tone_is_daily_mean(self) -> None:
        """Multiple tone observations per day should be averaged."""
        vol = _make_timeline_response([("20200101000000", 10.0)])
        tone = _make_timeline_response([
            ("20200101000000", -2.0),
            ("20200101120000", -4.0),  # same day
        ])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        day_val = df.loc[pd.Timestamp("2020-01-01"), "ECON_ENERGY_tone"]
        assert day_val == pytest.approx(-3.0)

    def test_as_of_restricts_url_end_datetime(self) -> None:
        """as_of is applied as enddatetime in the GDELT API request URL."""
        vol = _make_timeline_response([("20200101000000", 5.0)])
        tone = _make_timeline_response([("20200101000000", -1.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        as_of = pd.Timestamp("2020-01-15")
        adapter.fetch(["ECON_ENERGY"], _date_range(), as_of=as_of)

        # GDELT enddatetime should reflect as_of (YYYYMMDDHHMMSS format)
        url = client.get.call_args_list[0][0][0]
        assert "20200115" in url

    def test_flat_timeline_format_parsed(self) -> None:
        """GDELT can return flat list (not nested series)."""
        vol = _make_flat_timeline_response([("20200103000000", 8.0)])
        tone = _make_flat_timeline_response([("20200103000000", -0.5)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        assert pd.Timestamp("2020-01-03") in df.index

    def test_empty_timeline_returns_empty_dataframe(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"timeline": []}
        client.get.return_value = resp

        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        assert df.empty

    def test_uses_config_themes_when_no_series_ids(self) -> None:
        vol = _make_timeline_response([("20200101000000", 1.0)])
        tone = _make_timeline_response([("20200101000000", -0.5)])
        client = _make_mock_client(vol, tone)

        config = GdeltConfig(themes=["ECON_ENERGY", "HEALTH"])
        adapter = GdeltAdapter(config=config, http_client=client)

        # fetch with empty list should use config themes
        df = adapter.fetch([], _date_range())
        assert "ECON_ENERGY_count" in df.columns
        assert "HEALTH_count" in df.columns

    def test_multiple_themes(self) -> None:
        vol = _make_timeline_response([("20200101000000", 5.0)])
        tone = _make_timeline_response([("20200101000000", -1.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY", "ECON_TRADE"], _date_range())
        assert "ECON_ENERGY_count" in df.columns
        assert "ECON_TRADE_count" in df.columns

    def test_output_dtype_is_float64(self) -> None:
        vol = _make_timeline_response([("20200101000000", 10.0)])
        tone = _make_timeline_response([("20200101000000", -2.0)])
        client = _make_mock_client(vol, tone)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["ECON_ENERGY"], _date_range())
        assert df["ECON_ENERGY_count"].dtype == "float64"
        assert df["ECON_ENERGY_tone"].dtype == "float64"

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)
