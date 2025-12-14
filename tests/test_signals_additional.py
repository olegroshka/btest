import numpy as np
import pandas as pd

from quantdsl_backtest.engine.signal_engine import SignalEngine
from quantdsl_backtest.dsl.signals import (
    NotNull,
    Quantile,
    CrossSectionRank,
)


def _panel(idx=None, cols=None, values=None) -> pd.DataFrame:
    if idx is None:
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
    if cols is None:
        cols = ["A", "B", "C"]
    return pd.DataFrame(values, index=idx, columns=cols, dtype="float64")


def test_quantile_boundaries_q0_q1():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    x = _panel(idx, cols, [[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])

    nodes = {
        "q0": Quantile(factor_name="x", q=0.0),
        "q1": Quantile(factor_name="x", q=1.0),
    }
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # Expected row-wise min and max broadcast
    expected_min = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    expected_min.iloc[0] = x.iloc[0].min()
    expected_min.iloc[1] = x.iloc[1].min()

    expected_max = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    expected_max.iloc[0] = x.iloc[0].max()
    expected_max.iloc[1] = x.iloc[1].max()

    pd.testing.assert_frame_equal(out["q0"], expected_min)
    pd.testing.assert_frame_equal(out["q1"], expected_max)


def test_rank_percentile_with_ties_average():
    # Row with ties to check average ranking normalization
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    cols = ["A", "B", "C"]
    x = _panel(idx, cols, [[1.0, 1.0, 2.0]])

    nodes = {
        "r": CrossSectionRank(factor_name="x", method="percentile"),
    }
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()["r"]

    # pandas rank: [1.5, 1.5, 3.0]; engine normalizes as (rank-1)/(n-1) with n=3 -> [0.25,0.25,1.0]
    row = out.iloc[0]
    assert np.isclose(row["A"], 0.25)
    assert np.isclose(row["B"], 0.25)
    assert np.isclose(row["C"], 1.0)


def test_notnull_all_nan():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B"]
    x = _panel(idx, cols, [[np.nan, np.nan], [np.nan, np.nan]])

    nodes = {"nn": NotNull("x")}
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()["nn"]

    assert out.dtypes.eq(bool).all() or out.dtypes.eq(np.bool_).all()
    assert (~out).all().all()  # all False
