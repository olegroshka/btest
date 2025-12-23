from __future__ import annotations

import pandas as pd

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.data.sources.cache import MemoryCacheStore, TailCachedFrameLoader


def test_tail_cache_loader_tail_fetch_and_merge(monkeypatch):
    # Disable platform meta side-effect for this unit test
    import quantdsl_backtest.data.sources.cache as cache_mod

    monkeypatch.setattr(cache_mod, "_try_upsert_platform_meta", lambda **_: None)

    cache = MemoryCacheStore()
    loader = TailCachedFrameLoader(provider="TEST")

    fetched_calls = []

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        return out

    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        fetched_calls.append((start, end))
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
        # include end day for this fake
        return pd.DataFrame({"value": range(len(idx))}, index=idx)

    def last_needed(end_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
        return end_ts

    def next_start(last_dt: pd.Timestamp, frequency: str) -> pd.Timestamp:
        return (last_dt + pd.Timedelta(days=1)).normalize()

    req1 = DataRequest(source="x://", start="2024-01-01", end="2024-01-05", frequency="1d")
    df1 = loader.load_one(
        req1,
        entity="A",
        cache=cache,
        normalize=normalize,
        fetch=fetch,
        last_needed_ts=last_needed,
        next_fetch_start=next_start,
    )
    assert len(fetched_calls) == 1

    # second: cache hit
    df2 = loader.load_one(
        req1,
        entity="A",
        cache=cache,
        normalize=normalize,
        fetch=fetch,
        last_needed_ts=last_needed,
        next_fetch_start=next_start,
    )
    assert len(fetched_calls) == 1
    pd.testing.assert_frame_equal(df1, df2)

    # extend end -> tail fetch
    req2 = DataRequest(source="x://", start="2024-01-01", end="2024-01-10", frequency="1d")
    _ = loader.load_one(
        req2,
        entity="A",
        cache=cache,
        normalize=normalize,
        fetch=fetch,
        last_needed_ts=last_needed,
        next_fetch_start=next_start,
    )
    assert len(fetched_calls) == 2
    # should start fetching from day after last cached day (2024-01-06)
    assert fetched_calls[1][0] == pd.Timestamp("2024-01-06")
