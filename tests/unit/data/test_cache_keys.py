from __future__ import annotations

from quantdsl_backtest.data.requests import DataRequest, KIND_MARKET_BARS, KIND_TIME_SERIES
from quantdsl_backtest.data.sources.cache import build_cache_key


def test_cache_key_includes_provider_kind_frequency_entity():
    req = DataRequest(
        source="fred://CPIAUCSL",
        kind=KIND_MARKET_BARS,
        start="2024-01-01",
        end="2024-01-02",
        frequency="1d",
        fields=["close"],
        dataset_id="macro",
    )
    k = build_cache_key(provider="FRED", request=req, entity="CPIAUCSL")
    assert k.startswith("v1/FRED/market_bars/1d/macro/CPIAUCSL")


def test_cache_key_separates_kinds_for_same_entity():
    req1 = DataRequest(
        source="fred://CPIAUCSL",
        kind=KIND_MARKET_BARS,
        start="2024-01-01",
        end="2024-01-02",
        frequency="1d",
        fields=["close"],
    )
    req2 = DataRequest(
        source="fred://CPIAUCSL",
        kind=KIND_TIME_SERIES,
        start="2024-01-01",
        end="2024-01-02",
        frequency="1d",
        fields=["value"],
    )
    k1 = build_cache_key(provider="FRED", request=req1, entity="CPIAUCSL")
    k2 = build_cache_key(provider="FRED", request=req2, entity="CPIAUCSL")
    assert k1 != k2

