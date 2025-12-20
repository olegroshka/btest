import numpy as np
import pandas as pd

from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.factor_engine import FactorEngine
from quantdsl_backtest.dsl.factors import ReturnFactor, VolatilityFactor, FiboRetraceFactor


def _make_md(index: pd.DatetimeIndex, instruments: list[str], close_panel: pd.DataFrame,
             extra_fields: dict[str, pd.DataFrame] | None = None) -> MarketData:
    bars = {}
    for instr in instruments:
        df = pd.DataFrame(index=index)
        df["close"] = close_panel[instr]
        if extra_fields:
            for field, panel in extra_fields.items():
                df[field] = panel[instr]
        # add volume as zeros if absent
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


def test_return_factor_simple_and_log():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = ["A", "B"]
    close = pd.DataFrame([[100.0, 100.0], [110.0, 90.0], [121.0, 81.0]], index=idx, columns=cols)

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    # Simple 1-step returns
    rf_simple = ReturnFactor(name="ret1", field="close", lookback=1, method="simple")
    out_simple = engine.compute("ret1", rf_simple)
    expected_simple = close / close.shift(1) - 1.0
    pd.testing.assert_frame_equal(out_simple, expected_simple)

    # Log 1-step returns over 2 steps (lookback=2)
    rf_log = ReturnFactor(name="ret2", field="close", lookback=2, method="log")
    out_log = engine.compute("ret2", rf_log)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_log = np.log(close / close.shift(2))
    pd.testing.assert_frame_equal(out_log, expected_log)


def test_volatility_factor_realized_annualized():
    # Price path with alternating 10% up and down to yield non-zero std
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    cols = ["A"]
    close = pd.DataFrame([100, 110, 99, 108.9, 98.01, 107.811], index=idx, columns=cols, dtype=float)

    md = _make_md(idx, cols, close)
    engine = FactorEngine(md, close)

    vf = VolatilityFactor(name="vol", field="close", lookback=3, method="realized", annualize=True)
    out = engine.compute("vol", vf)

    # Expected: rolling std of 1-bar log returns, window=3, annualized sqrt(252)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close / close.shift(1))
    expected = r.rolling(3, min_periods=3).std() * np.sqrt(252.0)
    pd.testing.assert_frame_equal(out, expected)


def test_fibo_retrace_factor_basic():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    cols = ["A", "B"]
    close = pd.DataFrame([[10, 20], [12, 18], [11, 22], [13, 19]], index=idx, columns=cols, dtype=float)
    high = pd.DataFrame([[11, 21], [13, 19], [12, 23], [14, 20]], index=idx, columns=cols, dtype=float)
    low = pd.DataFrame([[9, 19], [10, 17], [10, 20], [12, 18]], index=idx, columns=cols, dtype=float)

    md = _make_md(idx, cols, close, extra_fields={"high": high, "low": low})
    engine = FactorEngine(md, close)

    ff = FiboRetraceFactor(name="fibo", field_high="high", field_low="low", lookback=2, level=0.5)
    out = engine.compute("fibo", ff)

    # Expected: for each t, hi=max(high[-2:]), lo=min(low[-2:]); lo + (hi - lo)*0.5
    hi = high.rolling(2, min_periods=2).max()
    lo = low.rolling(2, min_periods=2).min()
    expected = lo + (hi - lo) * 0.5
    pd.testing.assert_frame_equal(out, expected)
