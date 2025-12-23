from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.services.catalog_meta import META_SYMBOL
from quantdsl_backtest.platform_api.services.catalog_meta_refresh import refresh_catalog_meta_from_cache


class _Obj:
    def __init__(self, data):
        self.data = data


class _Lib:
    def __init__(self, symbols_to_df):
        self._symbols_to_df = symbols_to_df

    def list_symbols(self):
        return list(self._symbols_to_df.keys())

    def read(self, symbol: str):
        return _Obj(self._symbols_to_df[symbol])


class _MetaLib:
    def __init__(self):
        self.store = {}

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self.store

    def read(self, symbol: str):
        return _Obj(self.store[symbol])

    def write(self, symbol: str, data):
        self.store[symbol] = data


class _Arctic:
    def __init__(self, libs):
        self._libs = libs
        self._meta = _MetaLib()

    def list_libraries(self):
        return list(self._libs.keys()) + ["platform_meta/catalog"]

    def get_library(self, name: str, create_if_missing: bool = False):
        if name == "platform_meta/catalog":
            return self._meta
        return self._libs[name]


def test_refresh_catalog_meta_from_cache_scans_market_data_only():
    # One valid cache symbol and one non-v1 symbol
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))

    arctic = _Arctic(
        {
            "market_data/YF/1d": _Lib(
                {
                    "v1/YF/market_bars/1d/sp500/AAPL": bars,
                    "junk": bars,
                }
            ),
            "other_lib": _Lib({"v1/YF/market_bars/1d/sp500/MSFT": bars}),
        }
    )

    stats = refresh_catalog_meta_from_cache(arctic=arctic)
    assert stats["libraries_scanned"] == 1
    assert stats["symbols_scanned"] == 2
    assert stats["meta_upserts"] == 1

    # Verify meta index was written and contains expected fields
    assert arctic._meta.has_symbol(META_SYMBOL)
    meta_df = arctic._meta.read(META_SYMBOL).data
    assert isinstance(meta_df, pd.DataFrame)
    assert len(meta_df) == 1

    row = meta_df.iloc[0].to_dict()
    assert row["symbol"] == "v1/YF/market_bars/1d/sp500/AAPL"
    assert row["provider"] == "YF"
    assert row["kind"] == "market_bars"
    assert row["frequency"] == "1d"
    assert row["dataset"] == "sp500"
    assert row["entity"] == "AAPL"
    assert int(row["rows"]) == 2
    assert int(row["cols"]) == 1
