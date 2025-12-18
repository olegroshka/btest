from __future__ import annotations

import math
import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.signal_analytics import (
    assign_quantiles as vec_assign_quantiles,
    compute_forward_returns as vec_compute_forward_returns,
    mean_forward_return_by_quantile as vec_mean_fwr_by_q,
)
from quantdsl_backtest.engine.analytics.attribution import (
    contrib_by_quantile as prod_contrib_by_quantile,
    contrib_return_panel,
)


# ---------------------------
# Reference (non-vectorized) implementations mirroring current semantics
# ---------------------------

def ref_assign_quantiles_loop(signal: pd.DataFrame, q: int) -> pd.DataFrame:
    # Semantics mirror current implementation:
    # - Rank across columns per row (method="average")
    # - If row has <=1 unique non-NaN values -> all NaN
    # - Map percentile to buckets 1..q using floor((rank-1)/(n-1)*q) + 1, then clamp [1, q]
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns, dtype="float32")
    for t in signal.index:
        row = signal.loc[t]
        if row.notna().sum() == 0:
            continue
        # degenerate dispersion check
        if row.dropna().nunique() <= 1:
            # leave NaNs
            continue
        ranks = row.rank(method="average")
        n = row.notna().sum()
        if n <= 1:
            continue
        denom = max(n - 1, 1)
        pct = (ranks - 1.0) / denom
        buckets = np.floor(pct * q) + 1.0
        buckets = buckets.clip(lower=1.0, upper=float(q))
        out.loc[t, ranks.notna()] = buckets[ranks.notna()].astype("float32").to_numpy()
    return out


def ref_compute_forward_returns(close: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        arr = np.full_like(close.to_numpy(dtype=float), np.nan, dtype=float)
        base = close.to_numpy(dtype=float)
        fwd = close.shift(-h).to_numpy(dtype=float)
        # elementwise safe divide
        mask = base != 0
        arr[mask] = (fwd[mask] / base[mask]) - 1.0
        out[h] = pd.DataFrame(arr, index=close.index, columns=close.columns)
    return out


def ref_mean_forward_return_by_quantile(
    quantile: pd.DataFrame, fwd_ret: pd.DataFrame, q: int
) -> tuple[pd.DataFrame, pd.Series]:
    cols = list(range(1, q + 1))
    out = pd.DataFrame(index=quantile.index, columns=cols, dtype="float64")
    r = fwd_ret.reindex_like(quantile)
    for t in quantile.index:
        qr = quantile.loc[t]
        rr = r.loc[t]
        for k in cols:
            idx = qr.index[qr == float(k)]
            if len(idx) == 0:
                # leave NaN
                continue
            ser = rr.reindex(idx)
            out.loc[t, k] = ser.sum(skipna=True) / max(len(idx), 1)
    ls = out[q] - out[1]
    return out, ls


def ref_contrib_by_quantile(contrib_ret: pd.DataFrame, quantile: pd.DataFrame, q: int):
    # Identical semantics to production but kept here for isolation in parity tests
    cols = list(range(1, q + 1))
    out = pd.DataFrame(index=contrib_ret.index, columns=cols, dtype="float64")
    for t in contrib_ret.index:
        cr = contrib_ret.loc[t]
        qr = quantile.loc[t]
        for k in cols:
            idx = qr.index[qr == float(k)]
            if len(idx) == 0:
                continue
            out.loc[t, k] = float(cr.reindex(idx).fillna(0.0).sum())
    ls = out[q] - out[1]
    return out, ls


# ---------------------------
# Helpers
# ---------------------------

def make_panel(rows: int = 60, cols: int = 25, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    names = [f"N{i:03d}" for i in range(cols)]
    data = rng.normal(0.0, 1.0, size=(rows, cols))
    # sprinkle NaNs and ties
    data[rng.random(data.shape) < 0.05] = np.nan
    df = pd.DataFrame(data, index=dates, columns=names)
    # make a few rows degenerate
    df.iloc[5] = 1.0
    df.iloc[10] = np.nan
    return df


# ---------------------------
# Tests
# ---------------------------

def test_assign_quantiles_parity_with_reference():
    sig = make_panel()
    q = 5
    got = vec_assign_quantiles(sig, q=q)
    ref = ref_assign_quantiles_loop(sig, q=q)

    # exact parity including NaN placement
    pd.testing.assert_frame_equal(got, ref)


def test_compute_forward_returns_parity():
    # build pseudo prices by cumulative product of small returns
    rng = np.random.default_rng(7)
    ret = 1.0 + rng.normal(0.0, 0.002, size=(80, 12))
    # inject exact zeros in base to test NaN on divide
    ret[20, 3] = 0.0
    prices = pd.DataFrame(ret, index=pd.date_range("2020-01-01", periods=80, freq="B")).cumprod()
    horizons = [1, 5, 10]

    got = vec_compute_forward_returns(prices, horizons)
    ref = ref_compute_forward_returns(prices, horizons)

    for h in horizons:
        pd.testing.assert_frame_equal(got[h], ref[h])


def test_mean_forward_return_by_quantile_parity():
    sig = make_panel(rows=50, cols=15, seed=99)
    q = 5
    qdf = vec_assign_quantiles(sig, q=q)

    # fabricate forward returns as shifted signal + noise to have structure
    r = sig.shift(-1).fillna(0.0) * 0.01 + 0.0005

    got_df, got_ls = vec_mean_fwr_by_q(qdf, r, q=q)
    ref_df, ref_ls = ref_mean_forward_return_by_quantile(qdf, r, q=q)

    pd.testing.assert_frame_equal(got_df, ref_df)
    pd.testing.assert_series_equal(got_ls, ref_ls)


def test_end_to_end_contrib_ls_stats_parity():
    # Simulate a small backtest-like panel
    rng = np.random.default_rng(2024)
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    names = [f"IDX{i:02d}" for i in range(18)]

    # Prices
    ret = 1.0 + rng.normal(0.0, 0.003, size=(len(dates), len(names)))
    close = pd.DataFrame(ret, index=dates, columns=names).cumprod()

    # Weights (t-1) will be derived from a simple signal quantiles-like rule
    raw_signal = pd.DataFrame(rng.normal(0.0, 1.0, size=close.shape), index=dates, columns=names)
    # add some cross-sectional structure akin to "day_20" by smoothing
    raw_signal = raw_signal.rolling(20, min_periods=1).mean()

    q = 5
    q_vec = vec_assign_quantiles(raw_signal, q=q)
    q_ref = ref_assign_quantiles_loop(raw_signal, q=q)
    pd.testing.assert_frame_equal(q_vec, q_ref)

    # Construct contribution panel from random weights derived from quantiles
    # Emulate equal-weight long Q5 / short Q1 with small scaling
    w = pd.DataFrame(0.0, index=dates, columns=names)
    w[q_vec == float(q)] = 1.0 / max(1, (q_vec == float(q)).sum(axis=1).max())
    w[q_vec == 1.0] = -1.0 / max(1, (q_vec == 1.0).sum(axis=1).max())

    contrib = contrib_return_panel(w, close)

    # Aggregate by quantile (vectorized path uses prod function; reference is local)
    out_prod, ls_prod = prod_contrib_by_quantile(contrib, q_vec, q=q)
    out_ref, ls_ref = ref_contrib_by_quantile(contrib, q_ref, q=q)

    pd.testing.assert_frame_equal(out_prod, out_ref)
    pd.testing.assert_series_equal(ls_prod, ls_ref)

    # Compare summary stats of L-S series
    ls = ls_prod.fillna(0.0)
    total = float(ls.sum())
    mean_daily = float(ls.mean())
    vol_daily = float(ls.std(ddof=0))

    ls_r = ls_ref.fillna(0.0)
    total_r = float(ls_r.sum())
    mean_daily_r = float(ls_r.mean())
    vol_daily_r = float(ls_r.std(ddof=0))

    # tight numeric parity
    assert math.isclose(total, total_r, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(mean_daily, mean_daily_r, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(vol_daily, vol_daily_r, rel_tol=0.0, abs_tol=0.0)
