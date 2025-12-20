from __future__ import annotations

import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.signal_analytics import (
    assign_quantiles,
)
from quantdsl_backtest.engine.analytics.attribution import (
    contrib_return_panel,
    contrib_by_quantile,
)


def _make_dates_names(r: int = 10, c: int = 6):
    dates = pd.date_range("2020-01-01", periods=r, freq="B")
    names = [f"N{i:02d}" for i in range(c)]
    return dates, names


def test_contrib_by_quantile_matches_vectorized_reference():
    # Synthetic deterministic data
    dates, names = _make_dates_names(8, 5)
    # simple close series that yields stable returns pattern
    close = pd.DataFrame(
        [[100, 101, 102, 103, 104]] + [[100 + t] * 5 for t in range(1, 8)],
        index=dates,
        columns=names,
    ).astype(float)

    # weights: rotate one-hot vector to create non-trivial contributions
    eye = np.eye(len(names))
    w = pd.DataFrame([eye[i % len(names)] for i in range(len(dates))], index=dates, columns=names)

    contrib = contrib_return_panel(w, close)

    # Quantiles: fixed per row to avoid empties
    # Use a simple deterministic pattern with no ties
    base = pd.DataFrame([np.linspace(0, 1, len(names)) for _ in dates], index=dates, columns=names)
    qdf = assign_quantiles(base, q=3)

    out, ls = contrib_by_quantile(contrib, qdf, q=3)

    # Vectorized reference: per row, mask by quantile and sum
    cols = [1, 2, 3]
    ref = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for k in cols:
        mask = (qdf == float(k))
        ref[k] = contrib.where(mask).sum(axis=1).astype(float)
    ref_ls = ref[3] - ref[1]

    # Compare
    pd.testing.assert_frame_equal(out, ref)
    pd.testing.assert_series_equal(ls, ref_ls)


def test_ls_invariant_under_affine_monotonic_signal_transform():
    # If signal is strictly increasing across names, any positive affine transform
    # should not change bucket membership; L-S attribution must be identical.
    dates, names = _make_dates_names(12, 7)
    rng = np.random.default_rng(123)
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, size=(len(dates), len(names))), axis=0), index=dates, columns=names)

    # random dense weights
    w = pd.DataFrame(rng.normal(0, 1, size=(len(dates), len(names))), index=dates, columns=names)
    contrib = contrib_return_panel(w, close)

    base_sig = pd.DataFrame([np.linspace(0.0, 1.0, len(names)) for _ in dates], index=dates, columns=names)
    q1 = assign_quantiles(base_sig, q=5)

    # Apply positive affine transform
    a, b = 2.5, 7.3
    q2 = assign_quantiles(a * base_sig + b, q=5)

    out1, ls1 = contrib_by_quantile(contrib, q1, q=5)
    out2, ls2 = contrib_by_quantile(contrib, q2, q=5)

    pd.testing.assert_frame_equal(out1, out2)
    pd.testing.assert_series_equal(ls1, ls2)


def test_contrib_by_quantile_handles_empty_buckets_and_nan_rows():
    dates, names = _make_dates_names(5, 4)
    # contrived contributions
    contrib = pd.DataFrame(0.0, index=dates, columns=names)
    contrib.iloc[2, 1] = 0.1

    # Quantiles with an all-NaN row and an empty-top/bottom bucket scenario
    q = pd.DataFrame(np.nan, index=dates, columns=names)
    # only middle buckets populated
    q.iloc[1] = [2, 3, np.nan, 2]
    q.iloc[2] = [2, 2, 3, 3]

    out, ls = contrib_by_quantile(contrib, q, q=3)

    # Where no instruments are assigned to a bucket, the output stays NaN for that bucket
    assert np.isnan(out.loc[dates[0], 1]) and np.isnan(out.loc[dates[0], 3])
    # L-S reflects NaN if any side is NaN on that date
    assert np.isnan(ls.loc[dates[0]])
    # On populated date, sums are finite
    assert np.isfinite(out.loc[dates[2]].fillna(0.0).to_numpy()).all()


def test_end_to_end_quantiles_to_attribution_matches_direct_grouped_sums():
    # End-to-end: assign_quantiles -> contrib_return_panel -> contrib_by_quantile
    dates, names = _make_dates_names(15, 6)
    rng = np.random.default_rng(2024)
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, size=(len(dates), len(names))), axis=0), index=dates, columns=names)
    weights = pd.DataFrame(rng.normal(0, 1, size=(len(dates), len(names))), index=dates, columns=names)

    contrib = contrib_return_panel(weights, close)

    # signal with ties on some dates to exercise stability; ensure some dispersion
    sig = pd.DataFrame(rng.normal(0, 1, size=(len(dates), len(names))), index=dates, columns=names)
    sig.iloc[3] = 0.5  # no dispersion row -> becomes all NaN
    qdf = assign_quantiles(sig, q=4)

    out, ls = contrib_by_quantile(contrib, qdf, q=4)

    # Direct grouped sums per date
    cols = [1, 2, 3, 4]
    direct = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for k in cols:
        m = (qdf == float(k))
        sums = contrib.where(m).sum(axis=1)
        cnt = m.sum(axis=1)
        # Match contrib_by_quantile behavior: if no members in bucket, keep NaN
        sums[cnt == 0] = np.nan
        direct[k] = sums
    direct_ls = direct[4] - direct[1]

    pd.testing.assert_frame_equal(out, direct)
    pd.testing.assert_series_equal(ls, direct_ls)
