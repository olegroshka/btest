import numpy as np
import pandas as pd

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.factor_engine import FactorEngine
from quantdsl_backtest.dsl.factors import ReturnFactor, WinsorizedFactor


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


def test_winsorized_factor_clips_outliers_cross_sectionally():
    # Dates and instruments
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = ["A", "B", "C"]

    # Close prices designed so that simple 1-bar returns on t1 are [0.0, 0.0, 10.0]
    # and on t2 all equal (zero std) to test stability
    close = pd.DataFrame(
        [
            [100.0, 100.0, 100.0],  # t0
            [100.0, 100.0, 1100.0],  # t1 -> simple returns: [0.0, 0.0, 10.0]
            [100.0, 100.0, 1100.0],  # t2 -> simple returns: [0.0, 0.0, 0.0] (zero std)
        ],
        index=idx,
        columns=cols,
        dtype=float,
    )

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    base = ReturnFactor(name="ret1_raw", field="close", lookback=1, method="simple")
    wz = WinsorizedFactor(name="ret1_w", base=base, z=1.0)

    out = engine.compute("ret1_w", wz)

    # Compute expected clipping using pandas row-wise mean/std
    with np.errstate(divide="ignore", invalid="ignore"):
        base_rets = close / close.shift(1) - 1.0
    mean = base_rets.mean(axis=1, skipna=True)
    std = base_rets.std(axis=1, skipna=True)
    lower = mean - 1.0 * std
    upper = mean + 1.0 * std
    expected = base_rets.clip(lower=lower, upper=upper, axis=0)

    pd.testing.assert_frame_equal(out, expected)

    # Specific check: on t1, C should be clipped below 10.0
    t1 = idx[1]
    assert out.loc[t1, "C"] < base_rets.loc[t1, "C"]
    # On t2, std == 0 -> no change
    t2 = idx[2]
    assert (out.loc[t2] == base_rets.loc[t2]).all()
