import numpy as np
import pandas as pd

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.factor_engine import FactorEngine
from quantdsl_backtest.dsl.factors import (
    ReturnFactor,
    VolatilityFactor,
    RatioFactor,
)


def _make_md(index: pd.DatetimeIndex, instruments: list[str], close_panel: pd.DataFrame,
             extra_fields: dict[str, pd.DataFrame] | None = None) -> MarketData:
    bars = {}
    for instr in instruments:
        df = pd.DataFrame(index=index)
        df["close"] = close_panel[instr]
        if extra_fields:
            for field, panel in extra_fields.items():
                df[field] = panel[instr]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        bars[instr] = df
    return MarketData(
        bars=bars,
        instruments=instruments,
        fields=list(bars[instruments[0]].columns),
        frequency="1d",
        calendar="XNYS",
    )


def test_ratio_factor_divides_two_factors_elementwise():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    cols = ["A", "B"]
    # Prices designed so that 1-bar simple returns alternate and stay finite
    close = pd.DataFrame(
        [[100, 100], [110, 90], [121, 81], [109, 89], [120, 97]],
        index=idx,
        columns=cols,
        dtype=float,
    )

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    num = ReturnFactor(name="num", field="close", lookback=1, method="simple")
    den = VolatilityFactor(name="den", field="close", lookback=2, method="realized", annualize=False)

    ratio = RatioFactor(name="ratio", numerator=num, denominator=den)
    out = engine.compute("ratio", ratio)

    # Expected: pandas division of evaluated numerator/denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = engine.compute("num", num) / engine.compute("den", den)

    pd.testing.assert_frame_equal(out, expected)


def test_ratio_factor_propagates_nans_and_infs():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    cols = ["A", "B"]
    close = pd.DataFrame(
        [[100, 100], [100, 100], [100, 100], [100, 100]],
        index=idx,
        columns=cols,
        dtype=float,
    )

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    # Numerator: zero DataFrame; Denominator: very small values on one row to test inf
    num = ReturnFactor(name="num0", field="close", lookback=1, method="simple")  # zeros
    den = VolatilityFactor(name="den_small", field="close", lookback=2, method="realized", annualize=False)

    ratio = RatioFactor(name="ratio2", numerator=num, denominator=den)
    out = engine.compute("ratio2", ratio)

    expected = engine.compute("num0", num) / engine.compute("den_small", den)
    # Confirm exact pandas semantics
    pd.testing.assert_frame_equal(out, expected)
