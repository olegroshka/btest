"""Tests for smim/data/pit_store.py — M1.3-T1 and M1.3-T2."""

from __future__ import annotations

import pandas as pd
import pytest

from quantdsl_backtest.smim.data.pit_store import PointInTimeStore, _SCHEMA_COLS


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_row(
    actor_id: str = "AAPL",
    signal_id: str = "CapitalExpenditures",
    event_date: str = "2021-12-31",
    pub_date: str = "2022-03-15",
    value: float = 1_000_000.0,
    source: str = "edgar",
    vintage_id: str | None = None,
) -> dict:
    return dict(
        actor_id=actor_id,
        signal_id=signal_id,
        event_date=event_date,
        pub_date=pub_date,
        value=value,
        source=source,
        vintage_id=vintage_id,
    )


def _make_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ── M1.3-T1: schema + ingest + query ─────────────────────────────────────────

class TestSchemaValidation:
    def test_ingest_raises_on_missing_columns(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        bad = pd.DataFrame({"actor_id": ["AAPL"], "value": [1.0]})
        with pytest.raises(ValueError, match="missing required PIT columns"):
            store.ingest(bad)

    def test_ingest_accepts_valid_schema(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        df = _make_df(_make_row())
        store.ingest(df)  # should not raise


class TestIngest:
    def test_creates_parquet_file(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(_make_row(source="edgar")))
        assert (tmp_path / "edgar.parquet").exists()

    def test_creates_root_dir_if_missing(self, tmp_path) -> None:
        new_dir = tmp_path / "nested" / "store"
        store = PointInTimeStore(root_dir=new_dir)
        store.ingest(_make_df(_make_row()))
        assert new_dir.exists()

    def test_appends_on_second_ingest(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        row1 = _make_row(event_date="2021-12-31", value=1.0)
        row2 = _make_row(event_date="2022-12-31", value=2.0)
        store.ingest(_make_df(row1))
        store.ingest(_make_df(row2))
        result = pd.read_parquet(tmp_path / "edgar.parquet")
        assert len(result) == 2

    def test_deduplicates_on_natural_key(self, tmp_path) -> None:
        """Second ingest of same (actor, signal, event_date, source) keeps latest pub_date."""
        store = PointInTimeStore(root_dir=tmp_path)
        row_v1 = _make_row(pub_date="2022-03-01", value=1.0)
        row_v2 = _make_row(pub_date="2022-04-01", value=1.1)  # revised
        store.ingest(_make_df(row_v1))
        store.ingest(_make_df(row_v2))
        result = pd.read_parquet(tmp_path / "edgar.parquet")
        assert len(result) == 1
        assert result.iloc[0]["value"] == pytest.approx(1.1)

    def test_separate_files_per_source(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(_make_row(source="edgar"), _make_row(source="fred")))
        assert (tmp_path / "edgar.parquet").exists()
        assert (tmp_path / "fred.parquet").exists()

    def test_event_date_is_tz_naive(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(_make_row()))
        df = pd.read_parquet(tmp_path / "edgar.parquet")
        assert df["event_date"].dt.tz is None

    def test_pub_date_is_tz_naive(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(_make_row()))
        df = pd.read_parquet(tmp_path / "edgar.parquet")
        assert df["pub_date"].dt.tz is None

    def test_value_dtype_is_float64(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(_make_row()))
        df = pd.read_parquet(tmp_path / "edgar.parquet")
        assert df["value"].dtype == "float64"


class TestQuery:
    def test_a1_filter_excludes_future_pub_dates(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        # pub_date after as_of → must be excluded
        future = _make_row(pub_date="2023-01-01", value=99.0)
        past = _make_row(event_date="2020-12-31", pub_date="2022-01-01", value=1.0)
        store.ingest(_make_df(future, past))

        result = store.query(as_of=pd.Timestamp("2022-06-01"))
        assert len(result) == 1
        assert result.iloc[0]["value"] == pytest.approx(1.0)

    def test_a1_filter_includes_exact_as_of_date(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        row = _make_row(pub_date="2022-06-01")
        store.ingest(_make_df(row))

        result = store.query(as_of=pd.Timestamp("2022-06-01"))
        assert len(result) == 1

    def test_empty_store_returns_empty_dataframe(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        result = store.query(as_of=pd.Timestamp("2022-01-01"))
        assert result.empty
        assert set(_SCHEMA_COLS).issubset(set(result.columns))

    def test_filter_by_source(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(source="edgar", value=1.0),
            _make_row(source="fred", signal_id="FEDFUNDS", value=2.0),
        ))
        result = store.query(as_of=pd.Timestamp("2022-12-31"), sources=["edgar"])
        assert set(result["source"].unique()) == {"edgar"}

    def test_filter_by_actor_ids(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(actor_id="AAPL", value=1.0),
            _make_row(actor_id="MSFT", value=2.0),
        ))
        result = store.query(as_of=pd.Timestamp("2022-12-31"), actor_ids=["AAPL"])
        assert list(result["actor_id"].unique()) == ["AAPL"]

    def test_filter_by_signal_ids(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(signal_id="CapitalExpenditures", value=1.0),
            _make_row(signal_id="TotalAssets", value=2.0),
        ))
        result = store.query(
            as_of=pd.Timestamp("2022-12-31"),
            signal_ids=["CapitalExpenditures"],
        )
        assert list(result["signal_id"].unique()) == ["CapitalExpenditures"]

    def test_filter_by_date_range(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(event_date="2019-12-31", pub_date="2020-03-01", value=1.0),
            _make_row(event_date="2021-12-31", pub_date="2022-03-01", value=2.0),
        ))
        result = store.query(
            as_of=pd.Timestamp("2023-01-01"),
            date_range=(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")),
        )
        assert len(result) == 1
        assert result.iloc[0]["value"] == pytest.approx(2.0)


# ── M1.3-T2: bulk_ingest, query_signals, query_intensity, list_* ─────────────

class TestBulkIngest:
    def test_bulk_ingest_combines_frames(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        f1 = _make_df(_make_row(actor_id="AAPL", value=1.0))
        f2 = _make_df(_make_row(actor_id="MSFT", value=2.0))
        store.bulk_ingest([f1, f2])
        result = store.query(as_of=pd.Timestamp("2022-12-31"))
        assert len(result) == 2

    def test_bulk_ingest_empty_list_is_noop(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.bulk_ingest([])  # should not raise or write anything
        assert store.list_sources() == []


class TestQuerySignals:
    def test_returns_matching_signal_ids(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(signal_id="CapitalExpenditures", value=1.0),
            _make_row(signal_id="FEDFUNDS", source="fred", value=2.0),
        ))
        result = store.query_signals(
            ["CapitalExpenditures"], as_of=pd.Timestamp("2022-12-31")
        )
        assert set(result["signal_id"].unique()) == {"CapitalExpenditures"}

    def test_applies_a1_filter(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(signal_id="CapitalExpenditures", pub_date="2025-01-01", value=99.0)
        ))
        result = store.query_signals(
            ["CapitalExpenditures"], as_of=pd.Timestamp("2022-12-31")
        )
        assert result.empty


class TestQueryIntensity:
    def test_returns_matching_actor_ids(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(actor_id="AAPL", value=1.0),
            _make_row(actor_id="MSFT", value=2.0),
        ))
        result = store.query_intensity(["AAPL"], as_of=pd.Timestamp("2022-12-31"))
        assert list(result["actor_id"].unique()) == ["AAPL"]


class TestCatalogue:
    def test_list_actors(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(actor_id="AAPL"),
            _make_row(actor_id="MSFT"),
        ))
        assert store.list_actors() == ["AAPL", "MSFT"]

    def test_list_actors_empty_store(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        assert store.list_actors() == []

    def test_list_signals(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(signal_id="CapitalExpenditures"),
            _make_row(signal_id="TotalAssets"),
        ))
        assert store.list_signals() == ["CapitalExpenditures", "TotalAssets"]

    def test_list_sources(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(source="edgar"),
            _make_row(source="fred", signal_id="FEDFUNDS"),
        ))
        assert set(store.list_sources()) == {"edgar", "fred"}

    def test_list_actors_filtered_by_source(self, tmp_path) -> None:
        store = PointInTimeStore(root_dir=tmp_path)
        store.ingest(_make_df(
            _make_row(actor_id="AAPL", source="edgar"),
            _make_row(actor_id="US", source="fred", signal_id="FEDFUNDS"),
        ))
        assert store.list_actors(source="edgar") == ["AAPL"]
        assert store.list_actors(source="fred") == ["US"]
