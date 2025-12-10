import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.engine.signal_engine import SignalEngine
from quantdsl_backtest.dsl.signals import (
    NotNull,
    And,
    Or,
    Not,
    LessEqual,
    GreaterEqual,
    Quantile,
    CrossSectionRank,
    MaskFromBoolean,
)


def _simple_panel(index=None, columns=None, values=None) -> pd.DataFrame:
    if index is None:
        index = pd.date_range("2020-01-01", periods=3, freq="D")
    if columns is None:
        columns = ["A", "B", "C"]
    df = pd.DataFrame(values, index=index, columns=columns, dtype="float64")
    return df


def _engine_with_factors(factors: dict[str, pd.DataFrame]) -> SignalEngine:
    # Provide an empty nodes dict; we'll fill per-test
    return SignalEngine(factors=factors, signal_nodes={})


def test_notnull_and_basic_boolean_ops():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = ["A", "B", "C"]
    x = _simple_panel(
        index=idx,
        columns=cols,
        values=[
            [1.0, np.nan, 0.5],
            [np.nan, 2.0, np.nan],
            [3.0, 4.0, 5.0],
        ],
    )

    factors = {"x": x}

    nodes = {
        "notnull": NotNull("x"),
        "le1": LessEqual("x", 1.0),
        # Compose via signal names
        "and_sig": And("notnull", "le1"),
        "or_sig": Or("notnull", Not(LessEqual("x", 0.2))),
        "not_sig": Not("le1"),
    }

    eng = SignalEngine(factors=factors, signal_nodes=nodes)
    out = eng.compute_all()

    # NotNull should be True where x is not NaN
    expected_notnull = x.notna()
    pd.testing.assert_frame_equal(out["notnull"], expected_notnull)

    # le1 compares x <= 1.0
    expected_le1 = x <= 1.0
    pd.testing.assert_frame_equal(out["le1"], expected_le1)

    # and_sig: NotNull AND (x <= 1)
    pd.testing.assert_frame_equal(out["and_sig"], expected_notnull & expected_le1)

    # not_sig: NOT (x <= 1)
    pd.testing.assert_frame_equal(out["not_sig"], ~expected_le1)

    # or_sig: NotNull OR NOT (x <= 0.2)
    expected_or = expected_notnull | ~(x <= 0.2)
    pd.testing.assert_frame_equal(out["or_sig"], expected_or)


def test_comparisons_between_factors_and_scalars():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    a = _simple_panel(index=idx, columns=cols, values=[[1, 2, 3], [3, 2, 1]])
    b = _simple_panel(index=idx, columns=cols, values=[[2, 2, 2], [2, 2, 2]])

    factors = {"a": a, "b": b}
    nodes = {
        "le_scalar": LessEqual("a", 2),
        "ge_scalar": GreaterEqual("a", 2),
        "le_factor": LessEqual("a", "b"),
        "ge_factor": GreaterEqual("a", "b"),
    }

    eng = SignalEngine(factors=factors, signal_nodes=nodes)
    out = eng.compute_all()

    pd.testing.assert_frame_equal(out["le_scalar"], a <= 2)
    pd.testing.assert_frame_equal(out["ge_scalar"], a >= 2)
    pd.testing.assert_frame_equal(out["le_factor"], a <= b)
    pd.testing.assert_frame_equal(out["ge_factor"], a >= b)


def test_quantile_with_and_without_mask():
    # Factor with increasing values per row
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    x = _simple_panel(index=idx, columns=cols, values=[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    # Build mask where x <= 2.0
    nodes = {
        "mask": MaskFromBoolean(LessEqual("x", 2.0)),
        "q50_all": Quantile(factor_name="x", q=0.5),
        "q50_masked": Quantile(factor_name="x", q=0.5, within_mask="mask"),
    }

    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # q50_all should be median across columns for each row, broadcast to all columns
    med0 = float(x.iloc[0].quantile(0.5))
    med1 = float(x.iloc[1].quantile(0.5))
    # Broadcast row-wise medians across all columns
    expected_all = pd.DataFrame(index=x.index, columns=x.columns, dtype="float64")
    expected_all.iloc[0] = med0
    expected_all.iloc[1] = med1
    pd.testing.assert_frame_equal(out["q50_all"], expected_all)

    # q50_masked should compute median across masked subset per row
    m0 = x.iloc[0] <= 2.0  # True for A,B
    m1 = x.iloc[1] <= 2.0  # True for B,C
    med0_mask = float(x.iloc[0][m0].quantile(0.5))
    med1_mask = float(x.iloc[1][m1].quantile(0.5))
    expected_mask = pd.DataFrame(index=x.index, columns=x.columns, dtype="float64")
    expected_mask.iloc[0] = med0_mask
    expected_mask.iloc[1] = med1_mask
    pd.testing.assert_frame_equal(out["q50_masked"], expected_mask)


def test_cross_section_rank_percentile_and_zscore_with_mask():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    # Row 0: distinct values for clear percentile ranks 0, 0.5, 1.0
    # Row 1: constant values to test zscore -> zeros
    x = _simple_panel(index=idx, columns=cols, values=[[10.0, 20.0, 30.0], [5.0, 5.0, 5.0]])

    # Mask: only A,B on first row; only B,C on second row
    mask_vals = [[True, True, False], [False, True, True]]
    mask_df = _simple_panel(index=idx, columns=cols, values=mask_vals).astype(bool)

    nodes = {
        "mask": MaskFromBoolean(mask_df),  # pass DataFrame directly as expr
        "rank_pct": CrossSectionRank(factor_name="x", mask_name="mask", method="percentile"),
        "rank_z": CrossSectionRank(factor_name="x", mask_name="mask", method="zscore"),
    }

    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # Percentile ranks on row 0 among A,B: A is lower -> 0.0, B -> 1.0; C masked -> NaN
    r0 = out["rank_pct"].iloc[0]
    assert np.isclose(r0["A"], 0.0)
    assert np.isclose(r0["B"], 1.0)
    assert np.isnan(r0["C"])  # masked

    # Z-score on row 1 among B,C: both equal -> 0.0; A masked -> NaN
    r1 = out["rank_z"].iloc[1]
    assert np.isnan(r1["A"])  # masked
    assert np.isclose(r1["B"], 0.0)
    assert np.isclose(r1["C"], 0.0)


def test_mask_from_boolean_wraps_expression_to_bool():
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    cols = ["A", "B", "C"]
    x = _simple_panel(index=idx, columns=cols, values=[[0.1, 1.5, 2.5]])

    nodes = {
        "mask": MaskFromBoolean(LessEqual("x", 1.0)),
    }

    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    expected = (x <= 1.0).astype(bool)
    # Verify the mask DataFrame has boolean dtype for all columns
    assert out["mask"].dtypes.eq(bool).all()
    pd.testing.assert_frame_equal(out["mask"], expected)
