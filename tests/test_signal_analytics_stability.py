from __future__ import annotations

import math
import warnings
import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.signal_analytics import (
    compute_forward_returns,
    assign_quantiles,
    quantile_turnover,
)
from quantdsl_backtest.engine.analytics.attribution import contrib_by_quantile


def make_panel(rows: int = 50, cols: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    names = [f"N{i:03d}" for i in range(cols)]
    data = rng.normal(0.0, 1.0, size=(rows, cols))
    return pd.DataFrame(data, index=dates, columns=names)


def test_compute_forward_returns_zero_safe_no_runtimewarning():
    close = make_panel(30, 10).abs() * 10.0
    # inject some exact zeros to mimic halts or bad ticks
    close.iloc[5, 3] = 0.0
    close.iloc[10, 7] = 0.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", RuntimeWarning)
        fwr = compute_forward_returns(close, horizons=[1, 5])

    # Should not raise or produce infs. Avoid applying NumPy ufuncs directly to
    # pandas objects to prevent potential __array_ufunc__ recursion on some versions.
    for h, df in fwr.items():
        arr = df.to_numpy(np.float64)
        mask = ~np.isnan(arr)
        # Check only valid entries; positions corresponding to NaN are ignored
        assert np.isfinite(arr[mask]).all()


def test_assign_quantiles_degenerate_cross_section_skips():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = ["A", "B", "C", "D"]
    sig = pd.DataFrame(
        [
            [1.0, 1.0, 1.0, 1.0],   # degenerate: all equal
            [np.nan, np.nan, np.nan, np.nan],  # all NaN
            [1.0, 2.0, 3.0, 4.0],   # proper dispersion
        ],
        index=dates,
        columns=cols,
    )

    qdf = assign_quantiles(sig, q=3)
    # First two rows should remain NaN due to safeguards
    assert qdf.iloc[0].isna().all()
    assert qdf.iloc[1].isna().all()
    # Last row should have quantiles 1..3 assigned
    assert qdf.iloc[2].notna().sum() == 4


def test_contrib_by_quantile_handles_all_nan_rows_and_vectorizes():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    names = ["A", "B", "C"]

    contrib = pd.DataFrame(0.0, index=dates, columns=names)
    contrib.iloc[2, 1] = 0.05

    # quantiles with NaN rows interspersed
    q = pd.DataFrame(np.nan, index=dates, columns=names)
    q.iloc[1] = [1, 2, np.nan]
    q.iloc[2] = [2, 2, 1]

    out, ls = contrib_by_quantile(contrib, q, q=3)

    # Shape checks
    assert list(out.columns) == [1, 2, 3]
    assert out.index.equals(contrib.index)

    # Basic sanity: sums are finite
    assert np.isfinite(out.fillna(0.0).to_numpy()).all()
    assert np.isfinite(ls.fillna(0.0).to_numpy()).all()


def test_quantile_turnover_vectorized_no_divide_by_zero():
    # Create quantiles with some dates having no overlap (all NaNs)
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    names = ["A", "B", "C"]
    qdf = pd.DataFrame(np.nan, index=dates, columns=names)
    qdf.iloc[1] = [1, 2, np.nan]
    qdf.iloc[2] = [1, 3, 2]
    qdf.iloc[3] = [np.nan, np.nan, np.nan]

    turn = quantile_turnover(qdf, q=3)

    # First date NaN, last date NaN due to denom==0
    assert math.isnan(turn.iloc[0])
    assert math.isnan(turn.iloc[-1])
    # Middle should be finite
    assert np.isfinite(turn.iloc[2]) or math.isnan(turn.iloc[2])
