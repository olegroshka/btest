from __future__ import annotations

import pytest

from quantdsl_backtest.dsl.frequency import Frequency
from quantdsl_backtest.dsl.data_config import DataConfig


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1d", "1d"),
        ("5m", "5m"),
        ("15m", "15m"),
        ("1h", "1h"),
        ("2w", "2w"),
        (" 1D ", "1d"),
    ],
)
def test_frequency_parse_and_str(value: str, expected: str):
    assert str(Frequency.parse(value)) == expected


@pytest.mark.parametrize("bad", ["", "0d", "-1d", "d1", "1", "1x", "1mo"])
def test_frequency_parse_rejects_invalid(bad: str):
    with pytest.raises(ValueError):
        Frequency.parse(bad)


def test_data_config_validates_frequency():
    cfg = DataConfig(
        source="parquet://x",
        calendar="XNYS",
        frequency=" 1D ",
        start="2024-01-01",
        end="2024-01-02",
        price_adjustment="none",
        fields=["close"],
    )
    assert cfg.frequency == "1d"


def test_data_config_rejects_bad_frequency():
    with pytest.raises(ValueError):
        DataConfig(
            source="parquet://x",
            calendar="XNYS",
            frequency="0d",
            start="2024-01-01",
            end="2024-01-02",
            price_adjustment="none",
            fields=["close"],
        )

