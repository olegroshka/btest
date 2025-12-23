from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.platform_api.services.catalog_download import download_bundle


def test_download_bundle_includes_stats_by_entity_for_yahoo(monkeypatch):
    # Patch registry to return a Yahoo provider with a fake client and MemoryCache
    from quantdsl_backtest.data import orchestrator as orch
    from quantdsl_backtest.data.sources.cache import MemoryCacheStore
    from quantdsl_backtest.data.sources.registry import DataSourceRegistry
    from quantdsl_backtest.data.sources.yahoo import YahooMarketBarsSource

    class _Client:
        def download(self, symbols, *, start, end, interval):
            idx = pd.date_range("2024-01-01", "2024-01-03", freq="D")
            df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
            return {symbols[0]: df}

    prov = YahooMarketBarsSource()
    setattr(prov, "client", _Client())

    reg = DataSourceRegistry()
    reg.register(prov)

    monkeypatch.setattr(orch, "default_registry", lambda: reg)
    monkeypatch.setattr(orch, "default_cache_for_request", lambda req: MemoryCacheStore())

    req = DataRequest(source="yf://", kind="market_bars", start="2024-01-01", end="2024-01-04", frequency="1d")

    out = download_bundle(request=req, universe=None, dry_run=False, entities=["AAPL", "MSFT"])
    assert "stats_by_entity" in out
    assert set(out["stats_by_entity"].keys()) == {"AAPL", "MSFT"}

    assert "actions_by_entity" in out
    assert set(out["actions_by_entity"].keys()) == {"AAPL", "MSFT"}

    # With an empty cache (MemoryCacheStore) and deterministic fake downloader,
    # first run must be full_fetch for both entities.
    assert out["actions_by_entity"]["AAPL"] == "full_fetch"
    assert out["actions_by_entity"]["MSFT"] == "full_fetch"
