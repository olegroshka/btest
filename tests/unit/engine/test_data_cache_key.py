"""The in-process data cache must not hand one strategy another strategy's instruments."""
import pandas as pd

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.engine import data_loader


class _Strategy:
    def __init__(self, instruments):
        self.data = DataConfig(source="parquet://same.parquet", calendar="XLON", frequency="1d", start="2024-01-01", end="2024-01-31")
        self.universe = Universe(name="u", static_instruments=instruments)


def _fake_load(calls):
    index = pd.bdate_range("2024-01-02", periods=3)

    def load(_data, universe):
        calls.append(tuple(universe.static_instruments))
        bars = {n: pd.DataFrame({"close": 1.0, "volume": 1.0}, index=index) for n in universe.static_instruments}
        return MarketData(bars=bars, instruments=list(universe.static_instruments), fields=["close", "volume"], frequency="1d", calendar="XLON")

    return load


def test_a_different_universe_on_the_same_file_is_not_a_cache_hit(monkeypatch):
    calls = []
    monkeypatch.setattr(data_loader, "load_market_data", _fake_load(calls))
    monkeypatch.setattr(data_loader, "_DATA_CACHE", {})
    _, first, _ = data_loader.load_data_for_strategy(_Strategy(["A"]))
    _, second, _ = data_loader.load_data_for_strategy(_Strategy(["A", "B"]))
    assert list(first.columns) == ["A"]
    assert list(second.columns) == ["A", "B"]
    assert len(calls) == 2


def test_the_same_universe_is_still_a_cache_hit(monkeypatch):
    calls = []
    monkeypatch.setattr(data_loader, "load_market_data", _fake_load(calls))
    monkeypatch.setattr(data_loader, "_DATA_CACHE", {})
    data_loader.load_data_for_strategy(_Strategy(["A", "B"]))
    data_loader.load_data_for_strategy(_Strategy(["A", "B"]))
    assert len(calls) == 1
