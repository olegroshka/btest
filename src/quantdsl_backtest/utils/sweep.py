"""
quantdsl_backtest.utils.sweep
==============================
Threshold sensitivity sweep helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .perf import compute_series_metrics


def threshold_sweep(
    factor: pd.Series,
    daily_ret: pd.Series,
    bh_ret: pd.Series,
    gate: pd.Series | None = None,
    percentiles=range(5, 76, 5),
    signal_delay: int = 1,
) -> pd.DataFrame:
    """
    Sweep entry thresholds over ``factor`` and return a metrics DataFrame.

    Parameters
    ----------
    factor      : raw signal series (e.g. TKAN 5d prediction)
    daily_ret   : buy-and-hold daily log returns
    bh_ret      : same as daily_ret (used as benchmark in compute_metrics)
    gate        : optional boolean Series — additional filter ANDed with threshold mask
    percentiles : iterable of percentile cut-points (default 5th–75th in steps of 5)
    signal_delay: bars to shift position before applying (default 1)

    Returns
    -------
    pd.DataFrame indexed by label, columns include thr, gate_active, Sharpe, CAGR, etc.
    """
    t_range = np.percentile(factor.dropna(), list(percentiles))
    bh_pos  = pd.Series(1, index=daily_ret.index)
    rows    = []

    variants = [(False, "factor only")] + ([(True, "factor+gate")] if gate is not None else [])

    for thr in t_range:
        mask_base = factor >= thr
        for use_gate, suffix in variants:
            mask  = mask_base & gate if (use_gate and gate is not None) else mask_base
            label = f"{suffix}  thr={thr:.5f}"
            pos_s = mask.shift(signal_delay).fillna(False).infer_objects(copy=False).astype(int)
            ret_s = (pos_s * daily_ret).fillna(0)
            m = compute_series_metrics(ret_s, bh_ret, pos_s, label)
            m["thr"]        = round(float(thr), 6)
            m["gate_active"] = use_gate
            rows.append(m)

    return pd.DataFrame(rows).set_index("Label")
