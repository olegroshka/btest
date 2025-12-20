import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.attribution import (
    asset_returns,
    contrib_return_panel,
    contrib_by_quantile,
    costs_by_instrument_day,
)


def test_asset_returns_pct_change_no_fill():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    cols = ["A", "B"]
    close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [11.0, 21.0]], index=idx, columns=cols
    )
    rets = asset_returns(close)
    # first row NaN
    assert rets.iloc[0].isna().all()
    # verify one step
    assert np.isclose(rets.loc[idx[1], "A"], 0.1)
    assert np.isclose(rets.loc[idx[2], "B"], 0.05)


def test_contrib_return_panel_alignment():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    cols = ["A", "B"]
    close = pd.DataFrame(
        [[100, 100], [110, 90], [121, 99], [133.1, 108.9]], index=idx, columns=cols
    )
    # weights at time t (we expect contrib uses w_{t-1})
    w = pd.DataFrame(
        [[0.0, 0.0], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], index=idx, columns=cols
    )
    contrib = contrib_return_panel(w, close)
    # For t=2 (third row), r_t = close[2]/close[1]-1 for A,B = [0.1, 0.1]; w_{t-1} = [0.5, 0.5]
    assert np.isclose(contrib.loc[idx[2]].sum(), 0.1 * (0.5 + 0.5))


def test_contrib_by_quantile_simple():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    cols = ["A", "B", "C", "D"]
    contrib = pd.DataFrame(
        [[0.1, 0.2, -0.1, -0.2], [0.0, 0.1, 0.2, -0.3]], index=idx, columns=cols
    )
    qdf = pd.DataFrame([[1, 1, 2, 2], [1, 2, 2, 1]], index=idx, columns=cols)
    out, ls = contrib_by_quantile(contrib, qdf, q=2)
    # day 1: Q1 total = 0.1+0.2=0.3, Q2 total = -0.1-0.2=-0.3, LS=-0.6
    assert np.isclose(out.loc[idx[0], 1], 0.3)
    assert np.isclose(out.loc[idx[0], 2], -0.3)
    assert np.isclose(ls.loc[idx[0]], -0.3 - 0.3)


def test_costs_by_instrument_day():
    # Two trades on same day different instruments; include commission, fees and slippage proxy
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2020-01-02 15:30"),
                pd.Timestamp("2020-01-02 15:35"),
            ],
            "instrument": ["A", "B"],
            "commission": [1.0, 2.0],
            "fees": [0.5, 1.5],
            "slippage_bps": [10.0, 20.0],
            "notional": [10000.0, 5000.0],
        }
    )
    piv = costs_by_instrument_day(df)
    # Expected costs: A = 1+0.5 + 10000*0.001 = 1.5 + 10 = 11.5; B = 2+1.5 + 5000*0.002 = 3.5 + 10 = 13.5
    d = pd.Timestamp("2020-01-02").normalize()
    assert np.isclose(float(piv.loc[d, "A"]), 11.5)
    assert np.isclose(float(piv.loc[d, "B"]), 13.5)
