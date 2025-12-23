from __future__ import annotations


import pandas as pd
import pytest

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.data.adapters import load_market_data


@pytest.fixture
def temp_arctic_uri(tmp_path, monkeypatch):
    uri = f"lmdb://{tmp_path.as_posix()}"
    # Arctic expects path; ensure posix to avoid backslashes issues in URI
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", uri)
    # Provide a fake FRED api key to satisfy strict check, though we'll mock client
    monkeypatch.setenv("FRED_API_KEY", "DUMMY")
    return uri


def test_fred_adapter_cache_flow_with_tail_fetch(temp_arctic_uri, monkeypatch):
    import warnings

    # ArcticDB + pandas internals can emit a BlockManager-related DeprecationWarning.
    # We intentionally suppress this narrow warning in tests to keep the suite clean
    # without adding expensive deep-copy conversions in production code.
    warnings.filterwarnings(
        "ignore",
        message=r"Passing a BlockManagerUnconsolidated to DataFrame is deprecated.*",
        category=DeprecationWarning,
    )

    # Stub cache library with in-memory store to avoid arcticdb dependency during test
    class _MemLib:
        def __init__(self):
            self.store = {}

        def has_symbol(self, k):
            return k in self.store

        def read(self, k):
            class _Obj:
                def __init__(self, data):
                    self.data = data

            return _Obj(self.store[k])

        def write(self, k, df):
            self.store[k] = df.copy()

    memlib = _MemLib()

    from quantdsl_backtest.data import cache_arctic as ca

    # Patch cache entry point used by ArcticCacheStore
    monkeypatch.setattr(ca, "get_cache_lib", lambda provider, frequency: memlib)

    # Monkeypatch fetch_fred_series to a deterministic function and count calls
    call_counter = {"n": 0}

    def fake_fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        call_counter["n"] += 1
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
        # Return a normal time-indexed DataFrame (avoid pandas internals / BlockManager warnings)
        return pd.DataFrame({"value": range(len(idx))}, index=idx)

    import quantdsl_backtest.data.market as market_mod
    import quantdsl_backtest.data.sources.fred as fred_source_mod

    monkeypatch.setattr(market_mod, "fetch_fred_series", fake_fetch)
    # Provider imports fetch_fred_series lazily from market module, but we patch this too
    monkeypatch.setattr(fred_source_mod, "pd", pd)

    cfg = DataConfig(
        source="fred://CPIAUCSL",
        calendar="XNYS",
        frequency="1d",
        start="2024-01-01",
        end="2024-01-10",
        price_adjustment="none",
        fields=["close", "volume"],
    )
    uni = Universe(name="fred")

    # First load should fetch and write to cache
    md1 = load_market_data(cfg, uni)
    assert md1.instruments == ["CPIAUCSL"]
    bars1 = md1.get_bar_data("CPIAUCSL")
    assert "close" in bars1.columns
    assert "volume" in bars1.columns
    assert len(bars1) == 10
    assert call_counter["n"] == 1

    # Second load should hit cache and return identical data
    md2 = load_market_data(cfg, uni)
    bars2 = md2.get_bar_data("CPIAUCSL")
    pd.testing.assert_frame_equal(bars1, bars2)
    # No new fetch
    assert call_counter["n"] == 1

    # Third load: extend date range -> should fetch tail only and merge
    cfg_extended = DataConfig(
        source="fred://CPIAUCSL",
        calendar="XNYS",
        frequency="1d",
        start="2024-01-01",
        end="2024-01-15",
        price_adjustment="none",
        fields=["close", "volume"],
    )
    md3 = load_market_data(cfg_extended, uni)
    bars3 = md3.get_bar_data("CPIAUCSL")
    assert len(bars3) == 15
    # Should have fetched once more for the tail
    assert call_counter["n"] == 2

    # And now loading the original range again should not fetch
    md4 = load_market_data(cfg, uni)
    bars4 = md4.get_bar_data("CPIAUCSL")
    pd.testing.assert_frame_equal(bars4, bars1)
    assert call_counter["n"] == 2
