from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.sources.cache import MemoryCacheStore
from quantdsl_backtest.data.sources.yahoo import YahooMarketBarsSource
from quantdsl_backtest.data.requests import DataRequest, KIND_MARKET_BARS


def test_yahoo_cache_tail_fetch_only_missing(monkeypatch):
    cache = MemoryCacheStore()

    calls = []

    def fake_download(symbols, *, start, end, interval, missing_index):
        calls.append({"symbols": symbols, "start": start, "end": end, "interval": interval})

        idx = pd.date_range(
            pd.to_datetime(start),
            pd.to_datetime(end),
            freq="D",
            inclusive="left",
        )

        df = pd.DataFrame({
            "Open": pd.Series(range(len(idx)), index=idx),
            "High": pd.Series(range(len(idx)), index=idx),
            "Low": pd.Series(range(len(idx)), index=idx),
            "Close": pd.Series(range(len(idx)), index=idx),
            "Volume": pd.Series(range(len(idx)), index=idx),
        })

        class _FakeYF:
            def __init__(self, df):
                self._df = df

            def get(self, name: str):
                return self._df[name]

        return _FakeYF(df)

    import vectorbt as vbt

    monkeypatch.setattr(vbt.YFData, "download", fake_download)

    req = DataRequest(
        source="yf://AAPL",
        kind=KIND_MARKET_BARS,
        start="2024-01-01",
        end="2024-01-10",
        frequency="1d",
        fields=["open", "high", "low", "close", "volume"],
        dataset_id="prices",
    )

    src = YahooMarketBarsSource()

    # first load: full fetch
    b1 = src.load(req, universe=None, cache=cache)
    assert "AAPL" in b1.bars
    assert len(calls) == 1

    # second load: should be cached, no fetch
    _ = src.load(req, universe=None, cache=cache)
    assert len(calls) == 1

    # extend end -> should fetch tail only
    req2 = DataRequest(
        source="yf://AAPL",
        kind=KIND_MARKET_BARS,
        start="2024-01-01",
        end="2024-01-15",
        frequency="1d",
        fields=req.fields,
        dataset_id=req.dataset_id,
    )
    _ = src.load(req2, universe=None, cache=cache)
    assert len(calls) == 2

    assert pd.to_datetime(calls[1]["start"]) >= pd.to_datetime("2024-01-10")
