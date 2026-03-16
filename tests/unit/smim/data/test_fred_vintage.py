"""Tests for smim/data/adapters/fred_vintage.py."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import FredConfig
from quantdsl_backtest.smim.data.adapters.fred_vintage import (
    FredVintageAdapter,
    _load_api_key,
)
from quantdsl_backtest.smim.interfaces import DateRange


# ── helpers ──────────────────────────────────────────────────────────────────

def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-12-31"),
    )


def _mock_fred_series(values: dict[str, float]) -> object:
    """Build a pandas Series that mimics fredapi.Fred.get_series() output."""
    idx = pd.to_datetime(list(values.keys()))
    return pd.Series(list(values.values()), index=idx, dtype="float64")


# ── _load_api_key ─────────────────────────────────────────────────────────────

class TestLoadApiKey:
    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "testkey123")
        assert _load_api_key("FRED_API_KEY") == "testkey123"

    def test_raises_if_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FRED API key not found"):
            _load_api_key("FRED_API_KEY")


# ── FredVintageAdapter ────────────────────────────────────────────────────────

class TestFredVintageAdapter:
    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "dummy_test_key")

    def _make_adapter(self, enable_vintage: bool = True) -> FredVintageAdapter:
        return FredVintageAdapter(
            config=FredConfig(api_key_env="FRED_API_KEY", enable_vintage=enable_vintage)
        )

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "fred"

    def test_returns_dataframe_with_correct_columns(self) -> None:
        adapter = self._make_adapter()
        mock_series = _mock_fred_series({"2020-01-01": 1.5, "2020-04-01": 1.75})

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            MockFred.return_value.get_series.return_value = mock_series
            df = adapter.fetch(["FEDFUNDS"], _date_range())

        assert "FEDFUNDS" in df.columns
        assert df.index.name == "event_date"

    def test_as_of_passes_realtime_params(self) -> None:
        adapter = self._make_adapter(enable_vintage=True)
        mock_series = _mock_fred_series({"2020-01-01": 1.5})
        as_of = pd.Timestamp("2020-06-01")

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            mock_fred_instance = MockFred.return_value
            mock_fred_instance.get_series.return_value = mock_series
            adapter.fetch(["FEDFUNDS"], _date_range(), as_of=as_of)
            call_kwargs = mock_fred_instance.get_series.call_args[1]

        assert call_kwargs.get("realtime_start") == "2020-06-01"
        assert call_kwargs.get("realtime_end") == "2020-06-01"

    def test_no_as_of_omits_realtime_params(self) -> None:
        adapter = self._make_adapter()
        mock_series = _mock_fred_series({"2020-01-01": 1.5})

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            mock_fred_instance = MockFred.return_value
            mock_fred_instance.get_series.return_value = mock_series
            adapter.fetch(["FEDFUNDS"], _date_range())
            call_kwargs = mock_fred_instance.get_series.call_args[1]

        assert "realtime_start" not in call_kwargs
        assert "realtime_end" not in call_kwargs

    def test_enable_vintage_false_ignores_as_of(self) -> None:
        adapter = self._make_adapter(enable_vintage=False)
        mock_series = _mock_fred_series({"2020-01-01": 1.5})
        as_of = pd.Timestamp("2020-06-01")

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            mock_fred_instance = MockFred.return_value
            mock_fred_instance.get_series.return_value = mock_series
            adapter.fetch(["FEDFUNDS"], _date_range(), as_of=as_of)
            call_kwargs = mock_fred_instance.get_series.call_args[1]

        # vintage disabled — no realtime params even though as_of was given
        assert "realtime_start" not in call_kwargs

    def test_multiple_series(self) -> None:
        adapter = self._make_adapter()
        mock_series = _mock_fred_series({"2020-01-01": 1.0, "2020-04-01": 2.0})

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            MockFred.return_value.get_series.return_value = mock_series
            df = adapter.fetch(["FEDFUNDS", "DCOILBRENTEU"], _date_range())

        assert set(df.columns) == {"FEDFUNDS", "DCOILBRENTEU"}

    def test_empty_series_returns_nan_column(self) -> None:
        adapter = self._make_adapter()

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            MockFred.return_value.get_series.return_value = None
            df = adapter.fetch(["FEDFUNDS"], _date_range())

        # empty or NaN column but no exception
        assert "FEDFUNDS" in df.columns

    def test_output_index_is_datetime(self) -> None:
        adapter = self._make_adapter()
        mock_series = _mock_fred_series({"2020-01-01": 1.5})

        with patch(
            "fredapi.Fred"
        ) as MockFred:
            MockFred.return_value.get_series.return_value = mock_series
            df = adapter.fetch(["FEDFUNDS"], _date_range())

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None  # tz-naive

    def test_implements_data_adapter_protocol(self) -> None:
        from quantdsl_backtest.smim.interfaces import DataAdapter
        adapter = self._make_adapter()
        assert isinstance(adapter, DataAdapter)
