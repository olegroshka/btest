"""Tests for smim/data/adapters/imf_sdmx.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import ImfConfig
from quantdsl_backtest.smim.data.adapters.imf_sdmx import ImfSdmxAdapter, _parse_period
from quantdsl_backtest.smim.interfaces import DateRange


def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2022-12-31"),
    )


def _mock_compact_data(period_value_pairs: list[tuple[str, str]]) -> dict:
    """Build a mock IMF SDMX CompactData JSON response."""
    obs = [{"@TIME_PERIOD": p, "@OBS_VALUE": v} for p, v in period_value_pairs]
    return {
        "CompactData": {
            "DataSet": {
                "Series": {
                    "@FREQ": "Q",
                    "@REF_AREA": "US",
                    "@INDICATOR": "NGDP_R_K_IX",
                    "Obs": obs,
                }
            }
        }
    }


def _make_mock_client(response_data: dict, status: int = 200) -> object:
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_data
    client.get.return_value = resp
    return client


# ── _parse_period ─────────────────────────────────────────────────────────────

class TestParsePeriod:
    def test_quarterly(self) -> None:
        ts = _parse_period("2005-Q1")
        assert ts == pd.Timestamp("2005-01-01")

    def test_quarterly_q4(self) -> None:
        ts = _parse_period("2020-Q4")
        assert ts == pd.Timestamp("2020-10-01")

    def test_monthly(self) -> None:
        ts = _parse_period("2020-01")
        assert ts == pd.Timestamp("2020-01-01")

    def test_annual(self) -> None:
        ts = _parse_period("2020")
        assert ts == pd.Timestamp("2020-01-01")

    def test_invalid_returns_none(self) -> None:
        assert _parse_period("not-a-date") is None


# ── ImfSdmxAdapter ────────────────────────────────────────────────────────────

class TestImfSdmxAdapter:
    def _make_adapter(self, http_client=None) -> ImfSdmxAdapter:
        config = ImfConfig(datasets=["IFS"])
        return ImfSdmxAdapter(config=config, http_client=http_client or MagicMock())

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "imf"

    def test_returns_wide_dataframe(self) -> None:
        data = _mock_compact_data([("2020-Q1", "100.5"), ("2020-Q2", "101.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range())
        assert "Q.US.NGDP_R_K_IX" in df.columns

    def test_index_is_event_date(self) -> None:
        data = _mock_compact_data([("2020-Q1", "100.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range())
        assert df.index.name == "event_date"
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_quarterly_period_parsed(self) -> None:
        data = _mock_compact_data([("2020-Q1", "100.5"), ("2020-Q3", "102.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range())
        assert pd.Timestamp("2020-01-01") in df.index
        assert pd.Timestamp("2020-07-01") in df.index

    def test_values_are_float(self) -> None:
        data = _mock_compact_data([("2020-Q1", "100.5")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range())
        assert df["Q.US.NGDP_R_K_IX"].dtype == "float64"
        assert df["Q.US.NGDP_R_K_IX"].iloc[0] == pytest.approx(100.5)

    def test_as_of_limits_upper_bound(self) -> None:
        """as_of passed as endPeriod via URL construction."""
        data = _mock_compact_data([("2020-Q1", "100.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        as_of = pd.Timestamp("2020-06-01")
        adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range(), as_of=as_of)

        url = client.get.call_args[0][0]
        assert "2020-06-01" in url

    def test_series_id_too_short_skipped(self) -> None:
        """Series IDs with fewer than 3 dot-parts are silently skipped."""
        data = _mock_compact_data([("2020-Q1", "100.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["US.NGDP"], _date_range())  # only 2 parts
        assert df.empty or "US.NGDP" not in df.columns

    def test_non_200_response_skipped(self) -> None:
        client = _make_mock_client({}, status=404)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.NGDP_R_K_IX"], _date_range())
        assert df.empty

    def test_multiple_series(self) -> None:
        data = _mock_compact_data([("2020-Q1", "5.0")])
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.PCPI_IX", "Q.GB.NGDP_R_K_IX"], _date_range())
        assert "Q.US.PCPI_IX" in df.columns
        assert "Q.GB.NGDP_R_K_IX" in df.columns

    def test_series_list_response_handled(self) -> None:
        """DataSet.Series can be a list instead of a single dict."""
        data = {
            "CompactData": {
                "DataSet": {
                    "Series": [
                        {
                            "@FREQ": "Q",
                            "@REF_AREA": "US",
                            "Obs": [{"@TIME_PERIOD": "2020-Q1", "@OBS_VALUE": "99.0"}],
                        }
                    ]
                }
            }
        }
        client = _make_mock_client(data)
        adapter = self._make_adapter(http_client=client)

        df = adapter.fetch(["Q.US.PCPI_IX"], _date_range())
        assert not df.empty

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)
