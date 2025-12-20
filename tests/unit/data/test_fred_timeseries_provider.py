from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.orchestrator import load_bundle
from quantdsl_backtest.data.requests import DataRequest, KIND_TIME_SERIES
from quantdsl_backtest.data.bundles import TimeSeriesBundle
from quantdsl_backtest.data.sources.cache import MemoryCacheStore
from quantdsl_backtest.data.sources.registry import DataSourceRegistry
from quantdsl_backtest.data.sources.fred import FredTimeSeriesSource


def test_fred_timeseries_provider_cache_flow(monkeypatch):
    cache = MemoryCacheStore()

    call_counter = {"n": 0}

    def fake_fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        call_counter["n"] += 1
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
        return pd.DataFrame({"date": idx, "value": range(len(idx))})

    import quantdsl_backtest.data.market as market_mod

    monkeypatch.setattr(market_mod, "fetch_fred_series", fake_fetch)

    reg = DataSourceRegistry()
    reg.register(FredTimeSeriesSource())

    req = DataRequest(
        source="fred://CPIAUCSL",
        kind=KIND_TIME_SERIES,
        start="2024-01-01",
        end="2024-01-05",
        frequency="1d",
        fields=["value"],
        dataset_id="macro",
    )

    b1 = load_bundle(req, universe=None, registry=reg, cache_factory=lambda _: cache)
    assert isinstance(b1, TimeSeriesBundle)
    assert b1.kind == KIND_TIME_SERIES
    assert b1.entities == ["CPIAUCSL"]
    assert call_counter["n"] == 1

    # Second load should be cached
    b2 = load_bundle(req, universe=None, registry=reg, cache_factory=lambda _: cache)
    assert isinstance(b2, TimeSeriesBundle)
    assert call_counter["n"] == 1

    # Extend range -> tail fetch
    req2 = DataRequest(
        source="fred://CPIAUCSL",
        kind=KIND_TIME_SERIES,
        start="2024-01-01",
        end="2024-01-07",
        frequency="1d",
        fields=["value"],
        dataset_id="macro",
    )
    b3 = load_bundle(req2, universe=None, registry=reg, cache_factory=lambda _: cache)
    assert isinstance(b3, TimeSeriesBundle)
    assert len(b3.frames["CPIAUCSL"]) == 7
    assert call_counter["n"] == 2

