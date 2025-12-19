import numpy as np
import pandas as pd

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.factor_engine import FactorEngine
from quantdsl_backtest.dsl.factors import VolatilityFactor


def _make_md(index: pd.DatetimeIndex, instruments: list[str], close_panel: pd.DataFrame) -> MarketData:
    bars = {}
    for instr in instruments:
        df = pd.DataFrame(index=index)
        df["close"] = close_panel[instr]
        # volume required by schema/tests
        df["volume"] = 0.0
        bars[instr] = df
    return MarketData(
        bars=bars,
        instruments=instruments,
        fields=list(bars[instruments[0]].columns),
        frequency="1d",
        calendar="XNYS",
    )


def test_volatility_min_periods_affects_nan_coverage():
    # 5 days, two instruments; B has sparse data to trigger NaNs in rolling std
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    cols = ["A", "B"]
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 99.0, 108.9, 98.01],
            # Insert NaNs to simulate missing sessions
            "B": [100.0, np.nan, np.nan, 110.0, 121.0],
        },
        index=idx,
        dtype=float,
    )

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    # Default: min_periods falls back to lookback (=3)
    vf_default = VolatilityFactor(name="vol_def", field="close", lookback=3, method="realized", annualize=False)
    out_default = engine.compute("vol_def", vf_default)

    # Looser: min_periods=1
    vf_loose = VolatilityFactor(name="vol_loose", field="close", lookback=3, method="realized", annualize=False, min_periods=1)
    out_loose = engine.compute("vol_loose", vf_loose)

    # Baseline via pandas for B: ensure our output matches pandas semantics
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close / close.shift(1))
    expected_default = r.rolling(3, min_periods=3).std()
    expected_loose = r.rolling(3, min_periods=1).std()

    pd.testing.assert_frame_equal(out_default, expected_default)
    pd.testing.assert_frame_equal(out_loose, expected_loose)

    # Coverage: looser should have fewer NaNs
    assert out_loose.isna().sum().sum() <= out_default.isna().sum().sum()