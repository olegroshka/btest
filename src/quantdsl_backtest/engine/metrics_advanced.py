# src/quantdsl_backtest/engine/metrics_advanced.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .results import BacktestResult


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _clean_returns_for_metrics(returns: pd.Series) -> pd.Series:
    """Return a sanitized daily returns series for higher-level metrics.

    Conventions:
      - NaNs → 0.0 (flat day)
      - inf → NaN → 0.0

    We intentionally do NOT clip here; clipping is already applied inside
    compute_basic_metrics for Sharpe/Sortino/DD. For these diagnostics we
    prefer to reflect the actual series shape while still being numerically safe.
    """

    if returns is None or len(returns) == 0:
        return pd.Series(dtype="float64")
    return returns.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_cagr_from_equity(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    eq0 = float(equity.iloc[0])
    eq1 = float(equity.iloc[-1])
    if not np.isfinite(eq0) or not np.isfinite(eq1) or eq0 <= 0:
        return float("nan")

    # Estimate years using index span; fall back to trading-days assumption.
    try:
        dt_days = (equity.index[-1] - equity.index[0]).days
        years = dt_days / 365.25 if dt_days > 0 else (len(equity) - 1) / 252.0
    except Exception:
        years = (len(equity) - 1) / 252.0

    if years <= 0:
        return float("nan")
    return (eq1 / eq0) ** (1.0 / years) - 1.0


def _ulcer_index_from_equity(equity: pd.Series) -> float:
    """Ulcer index as RMS of drawdown percentages (magnitude, not signed)."""
    if equity is None or len(equity) == 0:
        return 0.0

    eq = equity.astype("float64").replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if len(eq) == 0:
        return 0.0

    peak = eq.cummax()
    dd = (eq / peak) - 1.0  # negative in drawdown
    dd_mag = (-dd).clip(lower=0.0)

    try:
        return float(np.sqrt(np.mean(np.square(dd_mag.values))))
    except Exception:
        return float("nan")


def compute_advanced_metrics_from_result(result: BacktestResult) -> Dict[str, float]:
    """Compute additional PM-style metrics not covered by QuantStats.

    All metrics are daily-based (industry standard for daily backtests).

    Returns keys are intended to be stable user-facing names (as used in
    StrategyAnalyticsConfig.metrics):
      - calmar
      - win_rate
      - profit_factor
      - tail_ratio
      - ulcer_index
      - avg_leverage
      - max_leverage
      - pct_days_in_market

    Additionally, we compute engine-side cagr if missing.
    """

    metrics: Dict[str, float] = {}

    rets = _clean_returns_for_metrics(result.returns)

    # CAGR
    if "cagr" not in (result.metrics or {}):
        metrics["cagr"] = _safe_float(_compute_cagr_from_equity(result.equity))

    # Win rate (fraction of days with positive returns)
    if len(rets) > 0:
        metrics["win_rate"] = _safe_float(float((rets > 0).mean()))

        # Profit factor (daily): sum(pos) / abs(sum(neg))
        pos = rets[rets > 0].sum()
        neg = rets[rets < 0].sum()
        if neg == 0:
            metrics["profit_factor"] = float("nan") if pos != 0 else 0.0
        else:
            metrics["profit_factor"] = _safe_float(float(pos / abs(neg)))

        # Tail ratio: |q95| / |q05|
        try:
            q95 = float(rets.quantile(0.95))
            q05 = float(rets.quantile(0.05))
            if q05 == 0:
                metrics["tail_ratio"] = float("nan")
            else:
                metrics["tail_ratio"] = _safe_float(abs(q95) / abs(q05))
        except Exception:
            metrics["tail_ratio"] = float("nan")

    # Ulcer index
    metrics["ulcer_index"] = _safe_float(_ulcer_index_from_equity(result.equity))

    # Calmar = CAGR / |max_drawdown|
    try:
        cagr_val = float((result.metrics or {}).get("cagr", metrics.get("cagr", float("nan"))))
        max_dd = float((result.metrics or {}).get("max_drawdown", float("nan")))
        denom = abs(max_dd)
        if denom <= 0 or not np.isfinite(denom) or not np.isfinite(cagr_val):
            metrics["calmar"] = float("nan")
        else:
            metrics["calmar"] = _safe_float(cagr_val / denom)
    except Exception:
        metrics["calmar"] = float("nan")

    # Exposure/leverage summaries
    lev = getattr(result, "leverage", None)
    if isinstance(lev, pd.Series) and len(lev) > 0:
        lev = lev.astype("float64").replace([np.inf, -np.inf], np.nan)
        metrics["avg_leverage"] = _safe_float(float(lev.mean(skipna=True)))
        metrics["max_leverage"] = _safe_float(float(lev.max(skipna=True)))

    # % days in market: gross exposure > 0 (or leverage > 0)
    ge = getattr(result, "gross_exposure", None)
    if isinstance(ge, pd.Series) and len(ge) > 0:
        ge = ge.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        metrics["pct_days_in_market"] = _safe_float(float((ge.abs() > 1e-12).mean()))
    elif len(rets) > 0:
        metrics["pct_days_in_market"] = _safe_float(float((rets != 0).mean()))

    return metrics

