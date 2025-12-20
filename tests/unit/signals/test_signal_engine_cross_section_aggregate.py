import numpy as np
import pandas as pd

from quantdsl_backtest.engine.signal_engine import SignalEngine
from quantdsl_backtest.dsl.signals import (
    CrossSectionAggregate,
    MaskFromBoolean,
    GreaterEqual,
    CrossSectionRank,
)


def _make_panel():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])  # 3 rows
    cols = ["A", "B", "C"]

    # Base numeric factor with NaNs on later dates
    f = pd.DataFrame(
        [[1.0, 2.0, 3.0], [np.nan, 5.0, 7.0], [np.nan, np.nan, np.nan]],
        index=dates,
        columns=cols,
    )

    # Another factor for rank-based signal source tests
    x = pd.DataFrame(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        index=dates,
        columns=cols,
    )

    return {"f": f, "x": x}


def test_mean_no_mask_broadcast_and_nan_handling():
    factors = _make_panel()
    nodes = {
        "agg": CrossSectionAggregate(source="f", op="mean", name="agg"),
    }
    eng = SignalEngine(factors=factors, signal_nodes=nodes, use_polars=False)
    out = eng.compute_all()["agg"]

    # Row 1: mean(1,2,3) = 2.0
    row = out.loc[pd.Timestamp("2020-01-01")]
    assert np.allclose(row.dropna().values, 2.0)

    # Row 2: mean(5,7) = 6.0 (ignores NaN)
    row = out.loc[pd.Timestamp("2020-01-02")]
    assert np.allclose(row.dropna().values, 6.0)

    # Row 3: all NaN -> aggregate emits NaN across the row
    row = out.loc[pd.Timestamp("2020-01-03")]
    assert row.isna().all()


def test_aggregate_ops_min_max_sum_median():
    factors = _make_panel()
    nodes = {
        "min": CrossSectionAggregate(source="f", op="min", name="min"),
        "max": CrossSectionAggregate(source="f", op="max", name="max"),
        "sum": CrossSectionAggregate(source="f", op="sum", name="sum"),
        "median": CrossSectionAggregate(source="f", op="median", name="median"),
    }
    eng = SignalEngine(factors=factors, signal_nodes=nodes, use_polars=False)
    out = eng.compute_all()

    # Date 1 values: [1, 2, 3]
    ts1 = pd.Timestamp("2020-01-01")
    assert np.allclose(out["min"].loc[ts1].dropna().values, 1.0)
    assert np.allclose(out["max"].loc[ts1].dropna().values, 3.0)
    assert np.allclose(out["sum"].loc[ts1].dropna().values, 6.0)
    assert np.allclose(out["median"].loc[ts1].dropna().values, 2.0)

    # Date 2 values: [NaN, 5, 7] -> min=5, max=7, sum=12, median=6
    ts2 = pd.Timestamp("2020-01-02")
    assert np.allclose(out["min"].loc[ts2].dropna().values, 5.0)
    assert np.allclose(out["max"].loc[ts2].dropna().values, 7.0)
    assert np.allclose(out["sum"].loc[ts2].dropna().values, 12.0)
    assert np.allclose(out["median"].loc[ts2].dropna().values, 6.0)

    # Date 3 all-NaN -> aggregates emit NaN rows
    ts3 = pd.Timestamp("2020-01-03")
    assert out["min"].loc[ts3].isna().all()
    assert out["max"].loc[ts3].isna().all()
    assert out["sum"].loc[ts3].isna().all()
    assert out["median"].loc[ts3].isna().all()


def test_mask_restricts_universe():
    factors = _make_panel()
    nodes = {
        # mask m: use only entries where f >= 2
        "m": MaskFromBoolean(name="m", expr=GreaterEqual(left="f", right=2.0)),
        "agg_m": CrossSectionAggregate(source="f", op="mean", mask_name="m", name="agg_m"),
    }
    eng = SignalEngine(factors=factors, signal_nodes=nodes, use_polars=False)
    out = eng.compute_all()["agg_m"]

    # Date 1: values [1,2,3] with mask f>=2 -> [2,3] mean = 2.5
    ts1 = pd.Timestamp("2020-01-01")
    assert np.allclose(out.loc[ts1].dropna().values, 2.5)

    # Date 2: values [NaN,5,7] with mask f>=2 -> [5,7] mean = 6.0
    ts2 = pd.Timestamp("2020-01-02")
    assert np.allclose(out.loc[ts2].dropna().values, 6.0)

    # Date 3: all NaN -> aggregate emits NaN across the row
    ts3 = pd.Timestamp("2020-01-03")
    assert out.loc[ts3].isna().all()


def test_empty_mask_row_emits_nan():
    factors = _make_panel()
    nodes = {
        # mask all_false: 0 >= 1 -> False everywhere
        "all_false": MaskFromBoolean(name="all_false", expr=GreaterEqual(left=0.0, right=1.0)),
        "agg": CrossSectionAggregate(source="f", op="mean", mask_name="all_false", name="agg"),
    }
    eng = SignalEngine(factors=factors, signal_nodes=nodes, use_polars=False)
    out = eng.compute_all()["agg"]

    # All dates should be NaN across columns due to empty mask
    assert out.isna().all(axis=None)


def test_source_can_be_another_signal_cross_section_rank():
    factors = _make_panel()
    nodes = {
        "r": CrossSectionRank(factor_name="x", name="r"),
        "agg_r": CrossSectionAggregate(source="r", op="mean", name="agg_r"),
    }
    eng = SignalEngine(factors=factors, signal_nodes=nodes, use_polars=False)
    out = eng.compute_all()["agg_r"]

    # For rows with 3 valid entries, percentile rank average should be 0.5
    for ts in out.index:
        row = out.loc[ts]
        # All rows in our fixture have 3 valid entries for factor 'x'
        assert np.allclose(row.dropna().values, 0.5)
