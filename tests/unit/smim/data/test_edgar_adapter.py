"""Tests for smim/data/adapters/edgar.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import EdgarConfig
from quantdsl_backtest.smim.data.adapters.edgar import EdgarAdapter
from quantdsl_backtest.smim.interfaces import DateRange


def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2022-12-31"),
    )


def _mock_company_facts(cik: str) -> dict:
    """Build a mock SEC EDGAR company-facts JSON response."""
    return {
        "cik": int(cik.lstrip("0")),
        "entityName": f"Corp {cik}",
        "facts": {
            "us-gaap": {
                "CapitalExpenditures": {
                    "label": "Capital Expenditures",
                    "description": "Amount of cash outflow for capital expenditures.",
                    "units": {
                        "USD": [
                            {
                                "end": "2020-12-31",
                                "val": 1_000_000,
                                "accn": "0001234567-21-000001",
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-03-15",
                                "frame": "CY2020",
                            },
                            {
                                "end": "2021-12-31",
                                "val": 1_200_000,
                                "accn": "0001234567-22-000001",
                                "fy": 2021,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2022-03-10",
                                "frame": "CY2021",
                            },
                        ]
                    },
                }
            }
        },
    }


def _make_mock_client(cik_to_facts: dict) -> object:
    """Return a mock HTTP client that returns company_facts JSON."""
    client = MagicMock()

    def _get(url: str, **kwargs):
        resp = MagicMock()
        # Extract CIK from URL pattern: .../CIK{cik}/company-facts.json
        for cik, facts in cik_to_facts.items():
            if f"CIK{cik}" in url or cik.lstrip("0") in url:
                resp.status_code = 200
                resp.json.return_value = facts
                return resp
        resp.status_code = 404
        resp.json.return_value = {}
        return resp

    client.get.side_effect = _get
    return client


# ── EdgarAdapter ──────────────────────────────────────────────────────────────

class TestEdgarAdapter:
    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EDGAR_USER_AGENT", "test@example.com")

    def _make_adapter(self, http_client=None) -> EdgarAdapter:
        config = EdgarConfig(
            user_agent_env="EDGAR_USER_AGENT",
            xbrl_tags=["CapitalExpenditures"],
        )
        return EdgarAdapter(config=config, http_client=http_client or MagicMock())

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "edgar"

    def test_returns_dataframe_with_expected_columns(self) -> None:
        cik = "0001234567"
        client = _make_mock_client({cik: _mock_company_facts(cik)})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch([cik], _date_range())

        expected = {"actor_id", "tag", "event_date", "pub_date", "value"}
        assert expected.issubset(set(df.columns))

    def test_xbrl_tag_extraction_capital_expenditures(self) -> None:
        cik = "0001234567"
        client = _make_mock_client({cik: _mock_company_facts(cik)})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch([cik], _date_range())

        capex_rows = df[df["tag"] == "CapitalExpenditures"]
        assert len(capex_rows) > 0
        assert set(capex_rows["value"].tolist()) == {1_000_000.0, 1_200_000.0}

    def test_pub_date_is_filing_date_not_period_end(self) -> None:
        """A1 compliance: pub_date must be the EDGAR 'filed' date."""
        cik = "0001234567"
        client = _make_mock_client({cik: _mock_company_facts(cik)})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch([cik], _date_range())

        capex = df[df["tag"] == "CapitalExpenditures"].sort_values("pub_date")
        pub_dates = pd.to_datetime(capex["pub_date"].tolist())

        # Filed dates from mock: 2021-03-15 and 2022-03-10
        assert pd.Timestamp("2021-03-15") in pub_dates
        assert pd.Timestamp("2022-03-10") in pub_dates

        # NOT the period-end dates
        assert pd.Timestamp("2020-12-31") not in pub_dates
        assert pd.Timestamp("2021-12-31") not in pub_dates

    def test_as_of_filters_to_pub_date(self) -> None:
        """Rows with pub_date > as_of must be excluded."""
        cik = "0001234567"
        client = _make_mock_client({cik: _mock_company_facts(cik)})
        adapter = self._make_adapter(http_client=client)

        # Only the 2020 filing (filed 2021-03-15) should survive
        df = adapter.fetch([cik], _date_range(), as_of=pd.Timestamp("2021-06-01"))
        assert len(df) == 1
        assert df.iloc[0]["pub_date"] == pd.Timestamp("2021-03-15")

    def test_no_as_of_returns_all_rows_in_range(self) -> None:
        cik = "0001234567"
        client = _make_mock_client({cik: _mock_company_facts(cik)})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch([cik], _date_range())
        assert len(df) == 2

    def test_empty_response_returns_empty_dataframe(self) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        resp.json.return_value = {}
        client.get.return_value = resp

        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch(["9999999999"], _date_range())
        assert df.empty

    def test_user_agent_constant_is_set(self) -> None:
        """SEC requires a non-empty User-Agent; verify the module constant is set."""
        from quantdsl_backtest.smim.data.adapters.edgar import _USER_AGENT
        assert _USER_AGENT and "@" in _USER_AGENT

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)

    def test_multiple_ciks(self) -> None:
        cik1, cik2 = "0001234567", "0009876543"
        facts2 = {
            "cik": 9876543,
            "entityName": "Other Corp",
            "facts": {
                "us-gaap": {
                    "CapitalExpenditures": {
                        "label": "CapEx",
                        "description": "",
                        "units": {
                            "USD": [
                                {
                                    "end": "2021-06-30",
                                    "val": 500_000,
                                    "accn": "0009876543-21-000001",
                                    "fy": 2021,
                                    "fp": "Q2",
                                    "form": "10-Q",
                                    "filed": "2021-08-01",
                                    "frame": None,
                                }
                            ]
                        },
                    }
                }
            },
        }
        client = _make_mock_client({cik1: _mock_company_facts(cik1), cik2: facts2})
        adapter = self._make_adapter(http_client=client)
        df = adapter.fetch([cik1, cik2], _date_range())
        assert set(df["actor_id"].unique()) == {cik1, cik2}
