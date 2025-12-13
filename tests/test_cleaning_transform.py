from __future__ import annotations

import pandas as pd
import numpy as np

from quantdsl_backtest.dsl.transforms import CleaningTransform


def _df(data, index=None, cols=None):
    return pd.DataFrame(data, index=index, columns=cols)


def test_cleaning_transform_ffill_bfill_and_filtering():
    # Construct prices with NaNs such that only certain instruments meet coverage >= 0.8
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = _df(
        [
            [np.nan,  1.0,    5.0,   9.0],
            [1.0,     np.nan, 6.0,   10.0],
            [1.1,     1.2,    6.1,   np.nan],
            [1.2,     1.3,    np.nan, 10.3],
            [np.nan,  1.4,    6.3,   10.4],
            [1.4,     1.5,    6.4,   10.5],
            [1.5,     1.6,    6.5,   10.6],
            [1.6,     1.7,    6.6,   10.7],
            [1.7,     1.8,    6.7,   10.8],
            [1.8,     1.9,    6.8,   10.9],
        ],
        index=idx,
        cols=["A", "B", "C", "D"],
    )

    # Volumes with NaNs
    volumes = _df(
        [
            [np.nan, 100, np.nan, 400],
            [50,     110, 310,    np.nan],
            [55,     120, 320,    410],
            [60,     np.nan, 330, 420],
            [65,     140, 340,    430],
            [70,     150, 350,    440],
            [75,     160, 360,    450],
            [80,     170, 370,    460],
            [85,     180, 380,    470],
            [90,     190, 390,    480],
        ],
        index=idx,
        cols=["A", "B", "C", "D"],
    )

    t = CleaningTransform(min_coverage=0.8, min_keep=3)
    p2, v2 = t.apply(prices, volumes)

    # Check instruments kept count respects min_keep (at least 3)
    assert p2.shape[1] >= 3
    assert list(p2.columns) == list(v2.columns)

    # After ffill/bfill there should be no NaNs left in prices for kept instruments
    assert p2.isna().sum().sum() == 0

    # Volumes NaNs should be filled with 0.0 per transform
    assert (v2.isna().sum().sum() == 0) and (v2.values >= 0).all()


def test_cleaning_transform_top_coverage_when_too_few_pass_threshold():
    # Build data where only 1 instrument meets coverage threshold, but min_keep=2
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    prices = _df(
        [
            [1.0, np.nan, 5.0],
            [1.1, np.nan, np.nan],
            [1.2, np.nan, 5.2],
            [1.3, np.nan, 5.3],
            [1.4, np.nan, 5.4],
        ],
        index=idx,
        cols=["A", "B", "C"],
    )
    volumes = prices.copy() * 0.0

    t = CleaningTransform(min_coverage=0.9, min_keep=2)
    p2, v2 = t.apply(prices, volumes)

    # Ensure 2 instruments are kept (top-coverage fallback)
    assert p2.shape[1] == 2
    assert set(p2.columns).issubset({"A", "B", "C"})
    assert list(p2.columns) == list(v2.columns)
