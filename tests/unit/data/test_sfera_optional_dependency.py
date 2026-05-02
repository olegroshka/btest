from __future__ import annotations

import pytest

from quantdsl_backtest.data.orchestrator import default_registry
from quantdsl_backtest.data.requests import DataRequest, KIND_TIME_SERIES
from quantdsl_backtest.data.sources.sfera import SferaSource


def test_default_registry_can_be_created_without_sfera_db() -> None:
    reg = default_registry()
    provider_names = {getattr(provider, "name", provider.__class__.__name__) for provider in reg.providers}
    assert {
        "parquet_market_bars",
        "csv_market_bars",
        "yahoo_market_bars",
        "fred_market_bars",
        "fred_timeseries",
        "sfera",
    }.issubset(provider_names)


def test_sfera_source_defers_missing_dependency_until_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantdsl_backtest.data.sources.sfera as sfera_mod

    monkeypatch.setattr(sfera_mod, "_sfera_db", None)

    source = SferaSource()
    request = DataRequest(
        source="sfera://demo_schema/demo_table",
        kind=KIND_TIME_SERIES,
        start="2024-01-01",
        end="2024-01-02",
        frequency="1d",
    )

    with pytest.raises(ImportError, match="sfera-db package not found"):
        source.load(request, universe=None, cache=None)

