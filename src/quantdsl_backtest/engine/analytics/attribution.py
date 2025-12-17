# src/quantdsl_backtest/engine/analytics/attribution.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

def asset_returns(close: pd.DataFrame) -> pd.DataFrame:
    # Explicitly disable forward-fill to avoid deprecation warning
    rets = close.pct_change(fill_method=None)
    return rets.replace([np.inf, -np.inf], np.nan)

def contrib_return_panel(weights: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    # return-space contribution: w_{t-1} * r_t
    r = asset_returns(close).fillna(0.0)
    w_prev = weights.shift(1).fillna(0.0)
    return w_prev * r

def contrib_by_quantile(
    contrib_ret: pd.DataFrame,
    quantile: pd.DataFrame,
    q: int,
) -> tuple[pd.DataFrame, pd.Series]:
    # sum contributions for names in each quantile
    cols = list(range(1, q + 1))
    out = pd.DataFrame(index=contrib_ret.index, columns=cols, dtype="float64")
    for t in contrib_ret.index:
        cr = contrib_ret.loc[t]
        qr = quantile.loc[t]
        for k in cols:
            idx = qr.index[qr == k]
            if len(idx) == 0:
                continue
            out.loc[t, k] = float(cr.reindex(idx).fillna(0.0).sum())
    ls = out[q] - out[1]
    return out, ls

def costs_by_instrument_day(trades: pd.DataFrame) -> pd.DataFrame:
    # returns [date x instrument] cost_pnl (positive = cost)
    if trades is None or trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    else:
        df["date"] = pd.to_datetime(df.index).normalize()

    # conservative: commission + fees + (optional) slippage proxy if slippage_bps present and notional present
    cost = pd.Series(0.0, index=df.index, dtype="float64")
    if "commission" in df.columns:
        cost += df["commission"].astype(float).fillna(0.0)
    if "fees" in df.columns:
        cost += df["fees"].astype(float).fillna(0.0)
    if "slippage_bps" in df.columns and "notional" in df.columns:
        sbps = df["slippage_bps"].astype(float)
        notional = df["notional"].astype(float).abs()
        slip = notional * (sbps / 1e4)
        cost += slip.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["cost_pnl"] = cost
    piv = df.pivot_table(index="date", columns="instrument", values="cost_pnl", aggfunc="sum").fillna(0.0)
    return piv
