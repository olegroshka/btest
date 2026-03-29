"""Tests for smim/data/ adapters — FredVintageAdapter, EdgarAdapter, GdeltAdapter, PitStore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantdsl_backtest.smim.config import EdgarConfig, FredConfig, GdeltConfig
from quantdsl_backtest.smim.data.adapters.edgar import EdgarAdapter
from quantdsl_backtest.smim.data.adapters.fred_vintage import FredVintageAdapter
from quantdsl_backtest.smim.data.adapters.gdelt import GdeltAdapter
from quantdsl_backtest.smim.data.pit_store import PointInTimeStore
from quantdsl_backtest.smim.interfaces import DateRange


# ── helpers ───────────────────────────────────────────────────────────────────

def _date_range() -> DateRange:
    return DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-12-31"),
    )


def _pit_row(
    actor_id: str = "AAPL",
    signal_id: str = "CapEx",
    event_date: str = "2021-12-31",
    pub_date: str = "2022-03-15",
    value: float = 500.0,
    source: str = "edgar",
) -> dict:
    return dict(
        actor_id=actor_id,
        signal_id=signal_id,
        event_date=event_date,
        pub_date=pub_date,
        value=value,
        source=source,
        vintage_id=None,
    )


# ═══════════════════════════════════════════════════════════
# FredVintageAdapter
# ═══════════════════════════════════════════════════════════

class TestFredAdapter:
    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "dummy_key")

    def _make_adapter(self) -> FredVintageAdapter:
        return FredVintageAdapter(
            config=FredConfig(api_key_env="FRED_API_KEY", enable_vintage=True)
        )

    def _mock_series(self) -> pd.Series:
        idx = pd.to_datetime(["2020-01-01", "2020-04-01"])
        return pd.Series([1.5, 1.75], index=idx, dtype="float64")

    def test_source_name(self) -> None:
        assert self._make_adapter().source_name == "fred"

    def test_fetch_returns_dataframe(self) -> None:
        adapter = self._make_adapter()
        with patch("fredapi.Fred") as MockFred:
            MockFred.return_value.get_series.return_value = self._mock_series()
            result = adapter.fetch(["GDPC1"], _date_range())
        assert isinstance(result, pd.DataFrame)
        assert "GDPC1" in result.columns

    def test_vintage_retrieval_respects_as_of(self) -> None:
        adapter = self._make_adapter()
        as_of = pd.Timestamp("2020-06-01")
        with patch("fredapi.Fred") as MockFred:
            mock_instance = MockFred.return_value
            mock_instance.get_series.return_value = self._mock_series()
            adapter.fetch(["GDPC1"], _date_range(), as_of=as_of)
            kwargs = mock_instance.get_series.call_args[1]
        assert kwargs.get("realtime_start") == "2020-06-01"
        assert kwargs.get("realtime_end") == "2020-06-01"


# ═══════════════════════════════════════════════════════════
# EdgarAdapter
# ═══════════════════════════════════════════════════════════

class TestEdgarAdapter:
    def _make_adapter(self) -> EdgarAdapter:
        config = EdgarConfig(xbrl_tags=["CapitalExpenditures"], filing_types=["10-K"])
        mock_client = MagicMock()
        mock_client.get.return_value = self._mock_response()
        return EdgarAdapter(config=config, http_client=mock_client)

    def _mock_response(self) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "facts": {
                "us-gaap": {
                    "CapitalExpenditures": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "end": "2020-12-31",
                                    "filed": "2021-02-01",
                                    "val": 1_000_000,
                                }
                            ]
                        }
                    }
                }
            }
        }
        return resp

    def test_source_name(self) -> None:
        adapter = EdgarAdapter()
        assert adapter.source_name == "edgar"

    def test_fetch_xbrl_tags(self) -> None:
        adapter = self._make_adapter()
        result = adapter.fetch(["0000034088"], _date_range())
        assert isinstance(result, pd.DataFrame)
        assert "tag" in result.columns
        assert "value" in result.columns
        assert len(result) >= 1


# ═══════════════════════════════════════════════════════════
# GdeltAdapter
# ═══════════════════════════════════════════════════════════

class TestGdeltAdapter:
    """GdeltAdapter is a parquet-based file adapter; tests use a tmp fixture."""

    def _make_parquet(self, tmp_path) -> "Path":
        rows = [
            {"week_start": "2020-01-06", "theme_or_actor": "sector_energy",
             "article_count": 100.0, "avg_tone": -2.5, "intensity": 0.012},
        ]
        df = pd.DataFrame(rows)
        df["week_start"] = pd.to_datetime(df["week_start"])
        p = tmp_path / "gdelt.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_source_name(self) -> None:
        assert GdeltAdapter().source_name == "gdelt"

    def test_fetch_theme_intensity(self, tmp_path) -> None:
        p = self._make_parquet(tmp_path)
        adapter = GdeltAdapter(config=GdeltConfig(themes=["sector_energy"]), parquet_path=p)
        result = adapter.fetch(["sector_energy"], _date_range())
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "event_date"
        assert "sector_energy_count" in result.columns


# ═══════════════════════════════════════════════════════════
# PitStore
# ═══════════════════════════════════════════════════════════

class TestPitStore:
    def test_write_and_read_roundtrip(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        df = pd.DataFrame([_pit_row(value=42.0)])
        store.ingest(df)
        result = store.query(as_of=pd.Timestamp("2023-01-01"))
        assert len(result) == 1
        assert result.iloc[0]["value"] == pytest.approx(42.0)

    def test_pub_date_filtering_prevents_lookahead(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        # pub_date is in the future relative to as_of
        df = pd.DataFrame([_pit_row(pub_date="2099-01-01", value=99.0)])
        store.ingest(df)
        result = store.query(as_of=pd.Timestamp("2022-01-01"))
        assert result.empty
