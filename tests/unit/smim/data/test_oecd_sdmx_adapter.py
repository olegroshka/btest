"""Tests for smim/data/adapters/oecd_sdmx.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import OecdConfig
from quantdsl_backtest.smim.data.adapters.oecd_sdmx import (
    OecdSdmxAdapter,
    _parse_sdmx_json_v1,
    _parse_sdmx_json_v2,
    _period_to_timestamp,
)
from quantdsl_backtest.smim.interfaces import DateRange


def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2022-12-31"),
    )


def _mock_sdmx_v2(period_values: list[tuple[str, float]]) -> dict:
    """Build a minimal SDMX-JSON 2.0 response."""
    time_values = [{"id": p} for p, _ in period_values]
    return {
        "meta": {
            "schema": {
                "dataStructure": {
                    "dimensions": {
                        "observation": [
                            {
                                "id": "TIME_PERIOD",
                                "values": time_values,
                            }
                        ]
                    }
                }
            }
        },
        "data": {
            "dataSets": [
                {
                    "observations": {
                        str(i): [val] for i, (_, val) in enumerate(period_values)
                    }
                }
            ]
        },
    }


def _mock_sdmx_v1(period_values: list[tuple[str, float]]) -> dict:
    """Build a minimal SDMX-JSON 1.0 response."""
    time_values = [{"id": p} for p, _ in period_values]
    return {
        "structure": {
            "dimensions": {
                "observation": [
                    {
                        "id": "TIME_PERIOD",
                        "values": time_values,
                    }
                ]
            }
        },
        "dataSets": [
            {
                "observations": {
                    str(i): [val] for i, (_, val) in enumerate(period_values)
                }
            }
        ],
    }


def _make_mock_client(response_data: dict, status: int = 200) -> object:
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_data
    client.get.return_value = resp
    return client


# ── _period_to_timestamp ──────────────────────────────────────────────────────

class TestPeriodToTimestamp:
    def test_quarterly(self) -> None:
        assert _period_to_timestamp("2020-Q1") == pd.Timestamp("2020-01-01")

    def test_quarterly_q3(self) -> None:
        assert _period_to_timestamp("2019-Q3") == pd.Timestamp("2019-07-01")

    def test_annual(self) -> None:
        assert _period_to_timestamp("2020") == pd.Timestamp("2020-01-01")

    def test_iso_date(self) -> None:
        ts = _period_to_timestamp("2020-01-15")
        assert ts == pd.Timestamp("2020-01-15")

    def test_invalid_returns_none(self) -> None:
        assert _period_to_timestamp("not-a-period") is None


# ── _parse_sdmx_json_v2 ───────────────────────────────────────────────────────

class TestParseSDMXJsonV2:
    def test_extracts_records(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 5.0), ("2020-Q2", 6.0)])
        records = _parse_sdmx_json_v2(data)
        assert len(records) == 2
        assert records[0]["TIME_PERIOD"] == "2020-Q1"
        assert records[0]["value"] == pytest.approx(5.0)

    def test_falls_back_to_v1_on_missing_meta(self) -> None:
        data = _mock_sdmx_v1([("2021-Q1", 3.0)])
        records = _parse_sdmx_json_v2(data)
        assert len(records) == 1

    def test_empty_datasets_returns_empty(self) -> None:
        data = {
            "meta": {"schema": {"dataStructure": {"dimensions": {"observation": []}}}},
            "data": {"dataSets": []},
        }
        records = _parse_sdmx_json_v2(data)
        assert records == []


# ── OecdSdmxAdapter ───────────────────────────────────────────────────────────

class TestOecdSdmxAdapter:
    def _make_adapter(self, http_client=None) -> OecdSdmxAdapter:
        return OecdSdmxAdapter(config=OecdConfig(), http_client=http_client or MagicMock())

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "oecd"

    def test_returns_wide_dataframe(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 5.0), ("2020-Q2", 5.5)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert "QNA.Q.USA.P51G.GPSA" in df.columns

    def test_index_is_event_date(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 5.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert df.index.name == "event_date"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_quarterly_period_parsed(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 100.0), ("2020-Q4", 105.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert pd.Timestamp("2020-01-01") in df.index
        assert pd.Timestamp("2020-10-01") in df.index

    def test_url_includes_dataflow_and_key(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 1.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        url = client.get.call_args[0][0]

        assert "OECD,QNA,1.0" in url
        assert "Q.USA.P51G.GPSA" in url

    def test_as_of_restricts_end_period(self) -> None:
        data = _mock_sdmx_v2([("2020-Q1", 1.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        as_of = pd.Timestamp("2020-06-01")
        adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range(), as_of=as_of)
        url = client.get.call_args[0][0]

        assert "2020-06-01" in url

    def test_series_id_without_dot_skipped(self) -> None:
        client = _make_mock_client(_mock_sdmx_v2([("2020-Q1", 1.0)]))
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA_NODOT"], _date_range())
        assert df.empty

    def test_non_200_status_skipped(self) -> None:
        client = _make_mock_client({}, status=404)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert df.empty

    def test_v1_fallback_parsed(self) -> None:
        """When v2 meta is missing, adapter falls back to SDMX-JSON v1."""
        data = _mock_sdmx_v1([("2019-Q2", 99.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert not df.empty

    def test_empty_response_returns_empty_df(self) -> None:
        data = _mock_sdmx_v2([])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert df.empty

    def test_output_sorted_by_date(self) -> None:
        data = _mock_sdmx_v2([("2021-Q2", 3.0), ("2019-Q1", 1.0), ("2020-Q3", 2.0)])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["QNA.Q.USA.P51G.GPSA"], _date_range())
        assert list(df.index) == sorted(df.index)

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)
