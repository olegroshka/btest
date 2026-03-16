"""Tests for smim/data/adapters/bea_io.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import BeaConfig
from quantdsl_backtest.smim.data.adapters.bea_io import BeaIoAdapter, _PUB_LAG_DAYS
from quantdsl_backtest.smim.interfaces import DateRange


def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2018-01-01"),
        end=pd.Timestamp("2020-12-31"),
    )


def _mock_bea_response(year: int, row_code: str = "1", col_code: str = "T001", value: float = 0.5) -> dict:
    return {
        "BEAAPI": {
            "Results": {
                "Data": [
                    {
                        "Year": str(year),
                        "RowCode": row_code,
                        "RowDescription": f"Industry {row_code}",
                        "ColCode": col_code,
                        "ColDescription": f"Column {col_code}",
                        "DataValue": str(value),
                        "TableID": "259",
                    }
                ]
            }
        }
    }


def _make_mock_client(status: int = 200, response_data: dict | None = None) -> object:
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = response_data or _mock_bea_response(2019)
    resp.raise_for_status.return_value = None
    client.get.return_value = resp
    return client


# ── BeaIoAdapter ──────────────────────────────────────────────────────────────

class TestBeaIoAdapter:
    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEA_API_KEY", "dummy_bea_key")

    def _make_adapter(self, http_client=None) -> BeaIoAdapter:
        config = BeaConfig(
            api_key_env="BEA_API_KEY",
            table_ids=["259"],
            year_range=(1997, 2025),
        )
        return BeaIoAdapter(config=config, http_client=http_client or MagicMock())

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "bea"

    def test_returns_expected_columns(self) -> None:
        client = _make_mock_client()
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())

        expected = {
            "year", "source_industry", "source_desc",
            "target_industry", "target_desc", "coefficient", "table_id", "pub_date",
        }
        assert expected.issubset(set(df.columns))

    def test_pub_date_is_year_end_plus_lag(self) -> None:
        """pub_date must be year-end + 548 days (A1 buffer)."""
        client = _make_mock_client(response_data=_mock_bea_response(2019))
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())

        expected_pub = pd.Timestamp("2019-12-31") + pd.Timedelta(days=_PUB_LAG_DAYS)
        assert df.iloc[0]["pub_date"] == expected_pub

    def test_pub_date_not_period_end(self) -> None:
        """pub_date must NOT be 2019-12-31 (period end); it must be later."""
        client = _make_mock_client(response_data=_mock_bea_response(2019))
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())

        assert df.iloc[0]["pub_date"] > pd.Timestamp("2019-12-31")

    def test_as_of_filters_future_pub_dates(self) -> None:
        """Rows whose pub_date > as_of must be excluded."""
        response = {
            "BEAAPI": {
                "Results": {
                    "Data": [
                        # 2018 pub_date = 2018-12-31 + 548 days ≈ 2020-06-30
                        {
                            "Year": "2018",
                            "RowCode": "1",
                            "RowDescription": "Industry A",
                            "ColCode": "T001",
                            "ColDescription": "Col T001",
                            "DataValue": "0.3",
                            "TableID": "259",
                        },
                        # 2020 pub_date = 2020-12-31 + 548 days ≈ 2022-06-30
                        {
                            "Year": "2020",
                            "RowCode": "1",
                            "RowDescription": "Industry A",
                            "ColCode": "T001",
                            "ColDescription": "Col T001",
                            "DataValue": "0.4",
                            "TableID": "259",
                        },
                    ]
                }
            }
        }
        client = _make_mock_client(response_data=response)
        adapter = self._make_adapter(http_client=client)

        # as_of before 2020 pub_date → only 2018 row survives
        as_of = pd.Timestamp("2021-01-01")
        df = adapter.fetch(["259"], _date_range(), as_of=as_of)

        assert len(df) == 1
        assert df.iloc[0]["year"] == 2018

    def test_no_as_of_returns_all_rows(self) -> None:
        client = _make_mock_client()
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())
        assert len(df) == 1

    def test_empty_bea_response_returns_empty_dataframe(self) -> None:
        client = _make_mock_client(response_data={"BEAAPI": {"Results": {}}})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())
        assert df.empty
        assert "year" in df.columns

    def test_data_value_with_commas_parsed(self) -> None:
        """BEA returns large numbers with comma separators like '1,234,567'."""
        response = _mock_bea_response(2019, value=1_234_567)
        # Simulate BEA formatting
        response["BEAAPI"]["Results"]["Data"][0]["DataValue"] = "1,234,567"
        client = _make_mock_client(response_data=response)
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["259"], _date_range())

        assert df.iloc[0]["coefficient"] == pytest.approx(1_234_567.0)

    def test_uses_config_table_ids_when_series_ids_empty(self) -> None:
        client = _make_mock_client()
        adapter = self._make_adapter(http_client=client)
        adapter.fetch([], _date_range())

        url = client.get.call_args[0][0]
        assert "259" in url

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)
