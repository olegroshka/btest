# src/quantdsl_backtest/engine/analytics/signal_analytics.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

def compute_forward_returns(close: pd.DataFrame, horizons: list[int]) -> Dict[int, pd.DataFrame]:
    # fwd_ret_h[t] = close[t+h]/close[t] - 1
    out: Dict[int, pd.DataFrame] = {}
    base = close.to_numpy(dtype=float)
    for h in horizons:
        fwd = close.shift(-h).to_numpy(dtype=float)
        # Safe divide: where base==0 -> NaN (avoid inf), also preserves NaNs
        res = np.divide(fwd, base, out=np.full_like(fwd, np.nan, dtype=float), where=base != 0) - 1.0
        out[h] = pd.DataFrame(res, index=close.index, columns=close.columns)
    return out

def _spearman_rank_ic(signal_row: pd.Series, ret_row: pd.Series) -> float:
    # Spearman = Pearson corr of ranks
    s = signal_row.dropna()
    r = ret_row.reindex(s.index).dropna()
    idx = s.index.intersection(r.index)
    if len(idx) < 3:
        return np.nan
    sr = s.loc[idx].rank(method="average")
    rr = r.loc[idx].rank(method="average")
    return float(sr.corr(rr))

def compute_rank_ic(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    mask: Optional[pd.DataFrame] = None,
) -> pd.Series:
    # Vectorized Spearman rank IC computed row-wise
    # Apply mask by setting masked elements to NaN to exclude them
    s = signal.copy()
    r = fwd_ret.reindex_like(signal)
    if mask is not None:
        s = s.where(mask)
        r = r.where(mask)

    # Rank across columns for each row; NaNs remain NaN
    sr = s.rank(axis=1, method="average")
    rr = r.rank(axis=1, method="average")

    # Keep only positions where both ranks are present
    both = sr.notna() & rr.notna()
    sr = sr.where(both)
    rr = rr.where(both)

    # Count valid pairs per row; require at least 3
    n = both.sum(axis=1)
    valid = n >= 3

    # Row-wise Pearson correlation using nan-aware reductions
    # mean
    sr_mean = sr.mean(axis=1, skipna=True)
    rr_mean = rr.mean(axis=1, skipna=True)

    # center
    sr_c = sr.sub(sr_mean, axis=0)
    rr_c = rr.sub(rr_mean, axis=0)

    cov = (sr_c * rr_c).sum(axis=1, skipna=True) / (n - 1)
    sr_var = (sr_c.pow(2)).sum(axis=1, skipna=True) / (n - 1)
    rr_var = (rr_c.pow(2)).sum(axis=1, skipna=True) / (n - 1)

    denom = (sr_var.clip(lower=0).pow(0.5) * rr_var.clip(lower=0).pow(0.5))
    ic = cov / denom
    ic = ic.where(valid)
    ic = ic.astype("float64")
    return ic

def assign_quantiles(
    signal: pd.DataFrame,
    q: int,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # output values: 1..q, NaN where not assigned
    # Vectorized rank-based bucket assignment (matches previous fallback behavior and is stable with ties)
    x = signal.copy()
    if mask is not None:
        x = x.where(mask)

    # Ranks across columns per row; NaNs preserved
    ranks = x.rank(axis=1, method="average")
    # Count of non-NaN per row
    n = x.notna().sum(axis=1)
    # Normalized percentile in [0,1]; handle rows with n<=1 to avoid div by zero
    denom = (n - 1).replace(0, 1)
    pct = (ranks.sub(1.0, axis=0)).div(denom, axis=0)
    # Map to 1..q; keep NaNs
    buckets = np.floor(pct * q) + 1.0
    buckets = buckets.where(ranks.notna())
    # Rows with no dispersion (<=1 unique non-NaN) -> leave as NaN
    nunq = x.nunique(axis=1, dropna=True)
    no_disp = nunq <= 1
    if no_disp.any():
        buckets.loc[no_disp] = np.nan
    # Clamp to [1..q]
    buckets = buckets.clip(lower=1.0, upper=float(q))
    return buckets.astype("float32")

def mean_forward_return_by_quantile(
    quantile: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    # returns a [t x q] mean fwd return, and LS series (top - bottom)
    cols = list(range(1, q + 1))
    # Align fwd_ret to quantile index/columns
    r = fwd_ret.reindex_like(quantile)
    out = pd.DataFrame(index=quantile.index, columns=cols, dtype="float64")

    # Compute per-quantile mean via masks
    for k in cols:
        mask = (quantile == float(k))
        # sum of returns where mask true and count per row
        sum_k = (r.where(mask)).sum(axis=1, skipna=True)
        cnt_k = mask.sum(axis=1)
        out[k] = sum_k / cnt_k.replace(0, np.nan)

    ls = out[q] - out[1]
    return out, ls

def quantile_turnover(quantile: pd.DataFrame, q: int) -> pd.Series:
    # simple: fraction of names that changed quantile since prev day, among non-null
    prev = quantile.shift(1)
    valid = quantile.notna() & prev.notna()
    changed = (quantile != prev) & valid
    num = changed.sum(axis=1)
    den = valid.sum(axis=1)
    turn = num / den.replace(0, np.nan)
    turn.iloc[0] = np.nan
    return turn.astype("float64")
