from __future__ import annotations

from quantdsl_backtest.platform_api.services.catalog import list_arctic_cache_catalog


class _Lib:
    def __init__(self, symbols):
        self._symbols = list(symbols)

    def list_symbols(self):
        return list(self._symbols)


class _Arctic:
    def __init__(self, libs):
        # libs: dict[name, symbols]
        self._libs = dict(libs)

    def list_libraries(self):
        return list(self._libs.keys())

    def get_library(self, name: str):
        if name == "market_data/BROKEN/1d":
            raise RuntimeError("broken")
        if name not in self._libs:
            raise KeyError(name)
        return _Lib(self._libs[name])


def test_list_arctic_cache_catalog_filters_and_sorts():
    arctic = _Arctic(
        {
            "not_market_data/foo": ["x"],
            "market_data/YF/1d": ["v1/YF", "v0/YF"],
            "market_data/FRED/1d": ["s2", "s1"],
        }
    )

    libs = list_arctic_cache_catalog(arctic=arctic)

    assert [l.library for l in libs] == ["market_data/FRED/1d", "market_data/YF/1d"]
    assert libs[0].symbols == ["s1", "s2"]
    assert libs[1].symbols == ["v0/YF", "v1/YF"]


def test_list_arctic_cache_catalog_is_best_effort_on_bad_library():
    arctic = _Arctic(
        {
            "market_data/BROKEN/1d": ["ignored"],
            "market_data/YF/1d": ["ok"],
        }
    )

    libs = list_arctic_cache_catalog(arctic=arctic)
    assert [l.library for l in libs] == ["market_data/BROKEN/1d", "market_data/YF/1d"]
    assert libs[0].symbols == []
    assert libs[1].symbols == ["ok"]

