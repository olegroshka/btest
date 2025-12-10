import dataclasses

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    MarginConfig,
    RiskChecks,
    Reporting,
)


def test_data_config_defaults_and_overrides():
    dc = DataConfig(
        source="dummy://source",
        calendar="XNYS",
        frequency="1d",
        start="2020-01-01",
        end="2020-12-31",
    )

    assert dc.price_adjustment == "split_dividend"
    assert isinstance(dc.fields, list)
    assert set(["open", "high", "low", "close", "volume"]) <= set(dc.fields)

    # Override fields
    dc2 = DataConfig(
        source="dummy://source",
        calendar="XNYS",
        frequency="1d",
        start="2020-01-01",
        end="2020-12-31",
        price_adjustment="none",
        fields=["close", "volume"],
    )
    assert dc2.price_adjustment == "none"
    assert dc2.fields == ["close", "volume"]


def test_universe_and_filters_are_carried():
    filters = [HasHistory(min_days=200), MinPrice(min_price=2.5), MinDollarADV(min_dollar_adv=2_000_000.0)]
    u = Universe(name="TEST", filters=filters, id_field="ticker", static_instruments=["A", "B"]) 

    assert u.name == "TEST"
    assert u.id_field == "ticker"
    assert u.static_instruments == ["A", "B"]
    assert isinstance(u.filters[0], HasHistory)
    assert u.filters[0].min_days == 200
    assert isinstance(u.filters[1], MinPrice)
    assert u.filters[1].min_price == 2.5
    assert isinstance(u.filters[2], MinDollarADV)
    assert u.filters[2].min_dollar_adv == 2_000_000.0


def test_backtest_config_defaults_and_factories():
    cfg1 = BacktestConfig()
    cfg2 = BacktestConfig()

    # Defaults
    assert cfg1.engine == "event_driven"
    assert cfg1.cash_initial == 1_000_000.0
    assert isinstance(cfg1.margin, MarginConfig)
    assert isinstance(cfg1.risk_checks, RiskChecks)
    assert isinstance(cfg1.reporting, Reporting)

    # Default factories should create new instances per BacktestConfig
    assert cfg1 is not cfg2
    assert cfg1.margin is not cfg2.margin
    assert cfg1.risk_checks is not cfg2.risk_checks
    assert cfg1.reporting is not cfg2.reporting
    assert cfg1.reporting.metrics is not cfg2.reporting.metrics

    # Reporting defaults
    assert cfg1.reporting.store_trades is True
    assert cfg1.reporting.store_positions is True
    assert isinstance(cfg1.reporting.metrics, list) and len(cfg1.reporting.metrics) == 0
