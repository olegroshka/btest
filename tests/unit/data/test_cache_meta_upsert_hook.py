from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.data.sources.cache import MemoryCacheStore, TailCachedFrameLoader, build_cache_key


def test_tail_cache_loader_calls_platform_meta_upsert_on_cache_write(monkeypatch):
    calls = []

    import quantdsl_backtest.data.sources.cache as cache_mod

    def fake_upsert(*, request, provider, entity, cache_key, df):
        calls.append(
            {
                "provider": provider,
                "entity": entity,
                "cache_key": cache_key,
                "rows": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
            }
        )

    monkeypatch.setattr(cache_mod, "_try_upsert_platform_meta", fake_upsert)

    cache = MemoryCacheStore()
    loader = TailCachedFrameLoader(provider="YF")

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out.sort_index()

    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
        return pd.DataFrame({"value": range(len(idx))}, index=idx)

    def last_needed(end_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
        return end_ts

    def next_start(last_dt: pd.Timestamp, frequency: str) -> pd.Timestamp:
        return (last_dt + pd.Timedelta(days=1)).normalize()

    req = DataRequest(
        source="yf://sp500",
        kind="market_bars",
        start="2024-01-01",
        end="2024-01-03",
        frequency="1d",
        dataset_id="sp500",
    )

    loader.load_one(
        req,
        entity="AAPL",
        cache=cache,
        normalize=normalize,
        fetch=fetch,
        last_needed_ts=last_needed,
        next_fetch_start=next_start,
    )

    assert len(calls) == 1
    expected_key = build_cache_key(provider="YF", request=req, entity="AAPL")
    assert calls[0]["cache_key"] == expected_key
    assert calls[0]["provider"] == "YF"
    assert calls[0]["entity"] == "AAPL"
    assert calls[0]["rows"] == 3

