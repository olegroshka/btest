from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.services.catalog_meta import META_SYMBOL
from quantdsl_backtest.platform_api.services.catalog_symbol import get_symbol_meta, preview_frame


class _Obj:
    def __init__(self, data):
        self.data = data


class _MetaLib:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def has_symbol(self, symbol: str) -> bool:
        return symbol == META_SYMBOL

    def read(self, symbol: str):
        return _Obj(self._df)

    def write(self, symbol: str, data):
        raise AssertionError("not needed")


class _Lib:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def read(self, symbol: str):
        return _Obj(self._df)


class _Arctic:
    def __init__(self, meta_df: pd.DataFrame):
        self._meta = _MetaLib(meta_df)

    def get_library(self, name: str, create_if_missing: bool = False):
        assert name == "platform_meta/catalog"
        return self._meta


def test_get_symbol_meta_returns_last_row_for_symbol():
    df = pd.DataFrame(
        [
            {"symbol": "s", "provider": "YF", "rows": 1},
            {"symbol": "s", "provider": "YF", "rows": 2},
        ]
    )
    arctic = _Arctic(meta_df=df)

    rec = get_symbol_meta(arctic=arctic, symbol="s")
    assert rec is not None
    assert rec.row["rows"] == 2


def test_preview_frame_includes_head_tail_and_index_bounds():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

    out = preview_frame(lib=_Lib(df), symbol="v1", head=1, tail=1)
    assert out["rows"] == 3
    assert "ts" in out["columns"]
    assert "close" in out["columns"]
    assert out["index_start"].startswith("2020-01-01")
    assert out["index_end"].startswith("2020-01-03")
    assert len(out["head"]) == 1
    assert len(out["tail"]) == 1
    assert "ts" in out["head"][0]
    assert isinstance(out["head"][0]["ts"], str)
    assert out["head"][0]["ts"].startswith("2020-01-01")


def test_preview_frame_infers_ohlc_aliases_for_titlecase_columns():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    df = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Volume": [100, 200],
        },
        index=idx,
    )

    out = preview_frame(lib=_Lib(df), symbol="sym", head=2, tail=0)
    # API preview should surface canonical lowercase aliases for UI plotting.
    assert {"open", "high", "low", "close", "volume"}.issubset(set(out["columns"]))
    assert out["head"][0]["open"] == 10.0
    assert out["head"][0]["high"] == 12.0
    assert out["head"][0]["low"] == 9.0
    assert out["head"][0]["close"] == 11.0
    assert out["head"][0]["volume"] == 100
