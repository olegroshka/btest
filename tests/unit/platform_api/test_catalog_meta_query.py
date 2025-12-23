from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.services.catalog_meta_query import filter_meta_df


def test_filter_meta_df_filters_case_insensitive_provider_kind_and_exact_dataset_entity():
    df = pd.DataFrame(
        [
            {"provider": "YF", "frequency": "1d", "kind": "market_bars", "dataset": "sp500", "entity": "AAPL", "symbol": "s1"},
            {"provider": "FRED", "frequency": "1d", "kind": "time_series", "dataset": "macro", "entity": "CPI", "symbol": "s2"},
        ]
    )

    out = filter_meta_df(df, provider="yf")
    assert list(out["symbol"]) == ["s1"]

    out = filter_meta_df(df, kind="MARKET_BARS")
    assert list(out["symbol"]) == ["s1"]

    out = filter_meta_df(df, dataset="macro")
    assert list(out["symbol"]) == ["s2"]

    out = filter_meta_df(df, entity="AAPL")
    assert list(out["symbol"]) == ["s1"]


def test_filter_meta_df_limit():
    df = pd.DataFrame(
        [{"provider": "YF", "frequency": "1d", "kind": "market_bars", "dataset": "sp500", "entity": "AAPL", "symbol": "s1"}]
        * 5
    )
    out = filter_meta_df(df, limit=2)
    assert len(out) == 2

