import numpy as np
import pandas as pd

from quantdsl_backtest.engine.signal_engine import SignalEngine
from quantdsl_backtest.dsl.signals import (
    LessEqual,
    And,
    MaskFromBoolean,
)


def _panel(idx=None, cols=None, values=None) -> pd.DataFrame:
    if idx is None:
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
    if cols is None:
        cols = ["A", "B", "C"]
    return pd.DataFrame(values, index=idx, columns=cols, dtype="float64")


def test_resolve_expr_series_with_datetimeindex_broadcasts_across_columns():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    cols = ["A", "B", "C"]
    x = _panel(idx, cols, [[1, 2, 3], [2, 2, 2], [3, 1, 0]])

    # Build a time-indexed threshold series to compare against
    thr = pd.Series([1.5, 1.5, 2.5], index=idx)

    nodes = {
        "le_time": LessEqual("x", thr),
    }
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # Expect x <= thr broadcast per-row across all columns
    thr_df = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    for c in cols:
        thr_df[c] = thr
    expected = x <= thr_df
    pd.testing.assert_frame_equal(out["le_time"], expected)


def test_resolve_expr_series_with_instrument_index_applies_per_column_constant():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    x = _panel(idx, cols, [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    # Per-instrument thresholds
    thr_by_name = pd.Series({"A": 2.0, "B": 2.5, "C": 1.0})

    nodes = {
        "le_name": LessEqual("x", thr_by_name),
    }
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # Build expected by expanding per-name thresholds into constant columns
    thr_df = pd.DataFrame(index=idx, columns=cols, dtype="float64")
    for c in cols:
        thr_df[c] = thr_by_name[c]
    expected = x <= thr_df
    pd.testing.assert_frame_equal(out["le_name"], expected)


def test_signal_string_references_use_cache_and_align():
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    cols = ["A", "B"]
    x = _panel(idx, cols, [[1.0, 2.0]])

    nodes = {
        "sig1": LessEqual("x", 1.5),
        "sig2": And("sig1", "sig1"),
        "mask": MaskFromBoolean("sig2"),
    }
    eng = SignalEngine(factors={"x": x}, signal_nodes=nodes)
    out = eng.compute_all()

    # sig2 should equal sig1 logically; mask wraps to boolean
    pd.testing.assert_frame_equal(out["sig2"], out["sig1"])
    pd.testing.assert_frame_equal(out["mask"], out["sig1"].astype(bool))

    # Ensure cache contains sig1 (evidence of named caching)
    assert "sig1" in eng.cache
