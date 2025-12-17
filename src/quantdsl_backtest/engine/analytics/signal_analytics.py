# src/quantdsl_backtest/engine/analytics/signal_analytics.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

def compute_forward_returns(close: pd.DataFrame, horizons: list[int]) -> Dict[int, pd.DataFrame]:
    # fwd_ret_h[t] = close[t+h]/close[t] - 1
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
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
    ic = pd.Series(index=signal.index, dtype="float64")
    for t in signal.index:
        s = signal.loc[t]
        r = fwd_ret.loc[t]
        if mask is not None:
            m = mask.loc[t].astype(bool)
            s = s[m]
            r = r[m]
        ic.loc[t] = _spearman_rank_ic(s, r)
    return ic

def assign_quantiles(
    signal: pd.DataFrame,
    q: int,
    mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # output values: 1..q, NaN where not assigned
    out = pd.DataFrame(index=signal.index, columns=signal.columns, dtype="float32")
    for t in signal.index:
        row = signal.loc[t]
        if mask is not None:
            row = row[mask.loc[t].astype(bool)]
        row = row.dropna()
        if row.empty:
            continue
        # pandas qcut requires unique bins; handle ties
        try:
            bins = pd.qcut(row, q, labels=False, duplicates="drop")
            # map to 1..q (even if duplicates drop, keep max label+1)
            out.loc[t, bins.index] = (bins.astype("float32") + 1.0)
        except Exception:
            # fallback: rank-based bucket
            ranks = row.rank(method="average")
            pct = (ranks - 1.0) / max(len(ranks) - 1, 1)
            out.loc[t, row.index] = (np.floor(pct * q) + 1.0)
    return out

def mean_forward_return_by_quantile(
    quantile: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    # returns a [t x q] mean fwd return, and LS series (top - bottom)
    cols = list(range(1, q + 1))
    out = pd.DataFrame(index=quantile.index, columns=cols, dtype="float64")
    for t in quantile.index:
        qrow = quantile.loc[t]
        rrow = fwd_ret.loc[t]
        for k in cols:
            idx = qrow.index[qrow == k]
            if len(idx) == 0:
                continue
            out.loc[t, k] = float(rrow.reindex(idx).dropna().mean())
    ls = out[q] - out[1]
    return out, ls

def quantile_turnover(quantile: pd.DataFrame, q: int) -> pd.Series:
    # simple: fraction of names that changed quantile since prev day, among non-null
    turn = pd.Series(index=quantile.index, dtype="float64")
    prev = None
    for t in quantile.index:
        cur = quantile.loc[t]
        if prev is None:
            turn.loc[t] = np.nan
        else:
            idx = cur.index[cur.notna() & prev.notna()]
            if len(idx) == 0:
                turn.loc[t] = np.nan
            else:
                turn.loc[t] = float((cur.loc[idx] != prev.loc[idx]).mean())
        prev = cur
    return turn
