from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.data.sources.cache import MemoryCacheStore
from quantdsl_backtest.data.sources.yahoo import YahooMarketBarsSource
from quantdsl_backtest.dsl.universe import Universe


def test_yahoo_provider_records_stats_by_entity(monkeypatch):
    # Fake downloader returns deterministic frames
    class _Client:
        def download(self, symbols, *, start, end, interval):
            idx = pd.date_range("2024-01-01", "2024-01-03", freq="D")
            df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
            return {symbols[0]: df}

    src = YahooMarketBarsSource()
    setattr(src, "client", _Client())

    cache = MemoryCacheStore()

    req = DataRequest(source="yf://", kind="market_bars", start="2024-01-01", end="2024-01-04", frequency="1d")
    uni = Universe(name="u", static_instruments=["AAPL", "MSFT"])

    _ = src.load(req, uni, cache)

    by_ent = src.last_entity_cache_stats()
    assert set(by_ent.keys()) == {"AAPL", "MSFT"}
    # Should have written to cache for both on first run
    assert by_ent["AAPL"]["writes"] >= 1
    assert by_ent["MSFT"]["writes"] >= 1
