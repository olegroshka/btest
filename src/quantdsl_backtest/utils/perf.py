"""
quantdsl_backtest.utils.perf
=============================
Performance metric helpers for single-series backtests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_series_metrics(
    strat_ret: pd.Series,
    bh_ret: pd.Series,
    pos: pd.Series,
    label: str,
) -> dict:
    """
    Compute standard backtest metrics for a daily return series.

    Parameters
    ----------
    strat_ret : strategy daily returns
    bh_ret    : buy-and-hold daily returns (used for beta/alpha)
    pos       : daily position (0/1 integers)
    label     : row identifier in output dict

    Returns
    -------
    dict with keys: Label, CAGR, Sharpe, Sortino, MaxDD, Calmar,
                    Beta, Alpha, WinPct, InMktPct, TotalReturn
    """
    sr      = strat_ret.fillna(0)
    br      = bh_ret.reindex(sr.index).fillna(0)
    n_years = len(sr) / 252

    ann_ret = sr.mean() * 252
    ann_vol = sr.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    down    = sr[sr < 0]
    sortino = (ann_ret / (down.std() * np.sqrt(252))) if len(down) > 1 else 0.0

    equity = (1 + sr).cumprod()
    mdd    = float((equity / equity.cummax() - 1).min()) * 100
    total  = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr   = (total ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else float("nan")

    valid = br.notna() & sr.notna()
    if valid.sum() > 30:
        slope, intercept, *_ = scipy_stats.linregress(br[valid].values, sr[valid].values)
        beta  = float(slope)
        alpha = float(intercept) * 252 * 100
    else:
        beta = alpha = float("nan")

    p_aligned = pos.reindex(sr.index).fillna(0)
    active    = sr[p_aligned > 0]
    win_rate  = float((active > 0).mean() * 100) if len(active) > 0 else float("nan")

    return dict(
        Label       = label,
        CAGR        = round(cagr, 2),
        Sharpe      = round(sharpe, 3),
        Sortino     = round(sortino, 3),
        MaxDD       = round(mdd, 2),
        Calmar      = round(calmar, 3),
        Beta        = round(beta,  3) if beta  == beta  else float("nan"),
        Alpha       = round(alpha, 2) if alpha == alpha else float("nan"),
        WinPct      = round(win_rate, 1) if win_rate == win_rate else float("nan"),
        InMktPct    = round(float(p_aligned.mean() * 100), 1),
        TotalReturn = round((total - 1) * 100, 1),
    )
