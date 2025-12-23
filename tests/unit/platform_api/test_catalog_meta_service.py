from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.services.catalog_meta import (
    META_SYMBOL,
    CacheSymbolMeta,
    build_meta_row_from_df,
    read_catalog_index,
    upsert_catalog_index,
)


class _Obj:
    def __init__(self, data):
        self.data = data


class _MetaLib:
    def __init__(self):
        self._store = {}

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._store

    def read(self, symbol: str):
        if symbol not in self._store:
            raise KeyError(symbol)
        return _Obj(self._store[symbol])

    def write(self, symbol: str, data):
        self._store[symbol] = data


def test_read_catalog_index_missing_returns_empty():
    lib = _MetaLib()
    df = read_catalog_index(meta_lib=lib)
    assert df.empty
    assert "symbol" in df.columns


def test_upsert_catalog_index_inserts_and_updates_by_symbol():
    lib = _MetaLib()

    df1 = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    m1 = build_meta_row_from_df(
        provider="YF",
        frequency="1d",
        kind="market_bars",
        dataset="ds",
        entity="AAPL",
        symbol="v1/YF/market_bars/1d/ds/AAPL",
        df=df1,
    )
    upsert_catalog_index(meta_lib=lib, row=m1)

    out1 = lib._store[META_SYMBOL]
    assert isinstance(out1, pd.DataFrame)
    assert len(out1) == 1
    assert out1.iloc[0]["symbol"] == m1.symbol

    # Update same symbol, different coverage
    df2 = pd.DataFrame({"x": [1, 2, 3]}, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    m2 = build_meta_row_from_df(
        provider="YF",
        frequency="1d",
        kind="market_bars",
        dataset="ds",
        entity="AAPL",
        symbol=m1.symbol,
        df=df2,
    )
    upsert_catalog_index(meta_lib=lib, row=m2)

    out2 = lib._store[META_SYMBOL]
    assert len(out2) == 1
    assert int(out2.iloc[0]["rows"]) == 3

