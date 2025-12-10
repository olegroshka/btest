# src/quantdsl_backtest/engine/accounting.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..dsl.costs import BorrowCost, FinancingCost
from ..utils.logging import get_logger


log = get_logger(__name__)


def mark_to_market(
    prev_positions: pd.Series,
    prev_prices: pd.Series,
    curr_prices: pd.Series,
    prev_cash: float,
) -> Tuple[float, float]:
    """
    Mark positions to current prices, returning:

        equity_before_trades, price_pnl

    Price PnL is computed as sum(positions * (curr - prev)).
    """
    prev_positions = prev_positions.fillna(0.0)
    prev_prices = prev_prices.reindex(prev_positions.index).ffill()
    curr_prices = curr_prices.reindex(prev_positions.index)

    price_pnl = float(((curr_prices - prev_prices) * prev_positions).sum())
    # Equity before trades
    equity_before = prev_cash + (curr_prices * prev_positions).sum()
    return equity_before, price_pnl


def apply_carry_costs(
    positions: pd.Series,
    prices: pd.Series,
    cash: float,
    borrow: BorrowCost,
    financing: FinancingCost,
    dt_years: float = 1.0 / 252.0,
) -> Tuple[float, float, float]:
    """
    Apply simple borrow and financing costs over a time step.

    Returns:
        new_cash, borrow_cost, financing_pnl
    """
    positions = positions.fillna(0.0)
    prices = prices.reindex(positions.index)

    long_notional = float((positions.clip(lower=0.0) * prices).sum())
    short_notional = float((-positions.clip(upper=0.0) * prices).sum())

    # Borrow cost on shorts
    borrow_rate = borrow.default_annual_rate
    borrow_cost = short_notional * borrow_rate * dt_years

    # Simple financing on cash (use spread_bps as full rate for now)
    fin_rate = financing.spread_bps / 1e4
    financing_pnl = cash * fin_rate * dt_years  # positive if earning interest

    new_cash = cash - borrow_cost + financing_pnl
    return new_cash, borrow_cost, financing_pnl


def compute_exposures(
    positions: pd.Series,
    prices: pd.Series,
) -> Dict[str, float]:
    """
    Compute long/short/gross/net exposures and leverage (assuming eq>0 provided elsewhere).
    """
    positions = positions.fillna(0.0)
    prices = prices.reindex(positions.index)

    notional = positions * prices
    long_exp = float(notional.clip(lower=0.0).sum())
    short_exp = float(notional.clip(upper=0.0).sum())  # negative
    gross = long_exp + abs(short_exp)
    net = long_exp + short_exp

    return {
        "long_exposure": long_exp,
        "short_exposure": short_exp,
        "gross_exposure": gross,
        "net_exposure": net,
    }


def compute_basic_metrics(
    returns: pd.Series,
    equity: pd.Series,
    weights: pd.DataFrame | None,
) -> Dict[str, float]:
    """
    Compute basic performance metrics using robust, comparable logic:
      - total_return
      - sharpe (annualized, 252 trading days)
      - sortino (annualized)
      - max_drawdown
      - turnover_annual (if weights provided)
    """
    metrics: Dict[str, float] = {}

    # Clean returns: replace infinities with NaN and treat NaNs as 0.0 (no move)
    rets = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Total return from equity if available
    try:
        if equity is not None and len(equity) >= 2:
            total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        else:
            total_ret = 0.0
    except Exception:
        total_ret = 0.0

    if len(rets) == 0:
        metrics["total_return"] = float(total_ret)
        metrics["sharpe"] = 0.0
        metrics["sortino"] = 0.0
        metrics["max_drawdown"] = 0.0
        metrics["turnover_annual"] = 0.0
        return metrics

    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    neg_ret = rets[rets < 0]

    ann_factor = float(np.sqrt(252.0))
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    if len(neg_ret) > 0:
        downside_std = float(neg_ret.std())
        sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown
    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Turnover: 0.5 * sum(|w_t - w_{t-1}|) per day, annualized
    turnover_daily = 0.0
    try:
        if weights is not None and len(weights) > 1:
            diff = weights.diff().abs().sum(axis=1) * 0.5
            turnover_daily = float(diff.mean())
    except Exception:
        turnover_daily = 0.0
    turnover_annual = float(turnover_daily * 252.0)

    metrics["total_return"] = float(total_ret)
    metrics["sharpe"] = float(sharpe)
    metrics["sortino"] = float(sortino)
    metrics["max_drawdown"] = float(max_dd)
    metrics["turnover_annual"] = float(turnover_annual)
    return metrics
