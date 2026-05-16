from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


ExecutionMode = int | Literal["NO"]


def apply_execution_returns(
    signal: pd.Series,
    close_to_close_returns: pd.Series,
    execution: ExecutionMode = 1,
    open_to_close_returns: pd.Series | None = None,
    carry_in: bool = True,
    extra_delay_bars: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply execution timing to a signal and return strategy daily returns.

    Semantics mirror Signum-style execution handling:
    - int mode: signal at close[T], fill at close[T+execution], first earned bar is T+execution+1.
    - "NO" mode: signal at close[T], fill at open[T+1], entry bar earns open->close,
      subsequent bars earn close->close.

    Parameters
    ----------
    signal:
        Binary or continuous signal indexed like returns.
    close_to_close_returns:
        Daily close->close simple returns.
    execution:
        Integer lag or "NO" for next-open semantics.
    open_to_close_returns:
        Required for execution="NO".
    carry_in:
        If True, pre-window invested state is carried in from first signal value.
    extra_delay_bars:
        Additional bars added on top of execution mode lag.

    Returns
    -------
    strategy_returns, shifted_signal, entry_mask
    """
    idx = signal.index
    signal_arr = signal.reindex(idx).to_numpy(dtype=float, copy=False)
    cc_arr = close_to_close_returns.reindex(idx).to_numpy(dtype=float, copy=False)
    first_signal = float(signal_arr[0]) if (carry_in and len(signal_arr) > 0) else 0.0

    if isinstance(execution, str):
        mode = execution.upper()
        if mode != "NO":
            raise ValueError(f"Unknown execution mode: {execution!r}")
        if open_to_close_returns is None:
            raise ValueError("execution='NO' requires open_to_close_returns")

        oc_arr = open_to_close_returns.reindex(idx).to_numpy(dtype=float, copy=False)
        lag = 1 + int(extra_delay_bars)

        shifted_arr = np.full(len(signal_arr), first_signal, dtype=float)
        if lag < len(signal_arr):
            shifted_arr[lag:] = signal_arr[:-lag]

        prev_arr = np.empty_like(shifted_arr)
        prev_arr[0] = first_signal
        if len(shifted_arr) > 1:
            prev_arr[1:] = shifted_arr[:-1]

        entry_mask_arr = (shifted_arr > 0.0) & (prev_arr <= 0.0)
        blended_arr = np.where(entry_mask_arr, oc_arr, cc_arr)
        strategy_arr = shifted_arr * blended_arr

        strategy_returns = pd.Series(strategy_arr, index=idx, dtype="float64")
        shifted_signal = pd.Series(shifted_arr, index=idx, dtype="float64")
        entry_mask = pd.Series(entry_mask_arr, index=idx, dtype="bool")
        return strategy_returns, shifted_signal, entry_mask

    if not isinstance(execution, int):
        raise ValueError(f"execution must be int or 'NO', got {execution!r}")

    lag = int(execution) + 1 + int(extra_delay_bars)

    shifted_arr = np.full(len(signal_arr), first_signal, dtype=float)
    if lag < len(signal_arr):
        shifted_arr[lag:] = signal_arr[:-lag]

    prev_arr = np.empty_like(shifted_arr)
    prev_arr[0] = first_signal
    if len(shifted_arr) > 1:
        prev_arr[1:] = shifted_arr[:-1]

    entry_mask_arr = (shifted_arr > 0.0) & (prev_arr <= 0.0)
    strategy_arr = shifted_arr * cc_arr

    strategy_returns = pd.Series(strategy_arr, index=idx, dtype="float64")
    shifted_signal = pd.Series(shifted_arr, index=idx, dtype="float64")
    entry_mask = pd.Series(entry_mask_arr, index=idx, dtype="bool")
    return strategy_returns, shifted_signal, entry_mask


def apply_leveraged_returns(
    position: pd.Series,
    asset_execution_returns: pd.Series,
    leverage: float,
    borrow_annual: pd.Series | None = None,
    trading_days_per_year: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """Build leveraged strategy returns with financing on borrowed notional only.

    Formula:
        r_t = pos_t * (L * r_asset_t) - pos_t * max(L-1, 0) * (borrow_annual_t / N)

    where N is `trading_days_per_year`.

    Returns
    -------
    leveraged_returns, financing_drag
    """
    if leverage < 0:
        raise ValueError("leverage must be non-negative")

    idx = position.index
    pos = position.reindex(idx).fillna(0.0).astype(float)
    asset_ret = asset_execution_returns.reindex(idx).fillna(0.0).astype(float)

    if borrow_annual is None:
        borrow_annual_aligned = pd.Series(0.0, index=idx, dtype="float64")
    else:
        borrow_annual_aligned = borrow_annual.reindex(idx).ffill().fillna(0.0).astype(float)

    borrowed_notional = max(float(leverage) - 1.0, 0.0)
    borrow_daily = borrow_annual_aligned.to_numpy(dtype=float, copy=False) / float(trading_days_per_year)

    pos_arr = pos.to_numpy(dtype=float, copy=False)
    asset_arr = asset_ret.to_numpy(dtype=float, copy=False)
    financing_drag_arr = pos_arr * borrowed_notional * borrow_daily
    leveraged_arr = pos_arr * (float(leverage) * asset_arr) - financing_drag_arr

    financing_drag = pd.Series(financing_drag_arr, index=idx, dtype="float64")
    leveraged_returns = pd.Series(leveraged_arr, index=idx, dtype="float64")
    return leveraged_returns, financing_drag
