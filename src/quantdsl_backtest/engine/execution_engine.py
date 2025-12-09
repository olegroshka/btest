# src/quantdsl_backtest/engine/execution_engine.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..dsl.execution import Execution
from ..dsl.costs import Commission, StaticFees
from ..utils.logging import get_logger


log = get_logger(__name__)


def rebalance_to_target_weights(
    date: pd.Timestamp,
    execution: Execution,
    commission: Commission,
    fees: StaticFees,
    equity: float,
    prices: pd.Series,
    volumes: pd.Series,
    prev_positions: pd.Series,
    target_weights: pd.Series,
) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
    Given target weights and current positions/cash, construct and execute
    trades at the bar's price (with slippage and volume limits).

    Returns:
        new_positions: Series of positions (units) post-trade
        cash_delta: float (change in cash during this rebalance, negative if spent)
        trades_df: DataFrame with one row per execution
    """
    instruments = target_weights.index
    # Align everything
    prices = prices.reindex(instruments)
    volumes = volumes.reindex(instruments).fillna(0.0)
    prev_positions = prev_positions.reindex(instruments).fillna(0.0)

    # For simplicity, use 'close' as execution price before slippage
    base_prices = prices.copy()

    # Target notionals
    target_notional = target_weights * equity
    current_notional = prev_positions * base_prices

    delta_notional = target_notional - current_notional

    # Volume limits
    vp = execution.volume_limits
    max_participation = vp.max_participation
    min_notional = vp.min_fill_notional

    trades = []
    cash_delta = 0.0
    new_positions = prev_positions.copy()

    for instr in instruments:
        dn = float(delta_notional.get(instr, 0.0))
        price = float(base_prices.get(instr, np.nan))
        vol = float(volumes.get(instr, 0.0))

        if not np.isfinite(price) or price <= 0:
            continue

        if abs(dn) < 1e-8:
            continue

        # Apply volume participation limit -> cap notional change
        if vol > 0 and max_participation > 0:
            max_notional = max_participation * vol * price
            if max_notional <= 0:
                continue
            if abs(dn) > max_notional:
                if vp.mode == "proportional":
                    dn = np.sign(dn) * max_notional
                elif vp.mode == "clip":
                    dn = np.sign(dn) * max_notional

        if abs(dn) < min_notional:
            continue

        # Determine side and quantity
        side = "BUY" if dn > 0 else "SELL"
        qty = dn / price  # signed quantity

        # Slippage model: power-law in participation
        sl_model = execution.slippage
        participation = 0.0
        if vol > 0:
            participation = min(1.0, abs(qty) / vol)
        slippage_bps = sl_model.base_bps + sl_model.k * (participation ** sl_model.exponent)
        slippage_frac = slippage_bps / 1e4

        if side == "BUY":
            exec_price = price * (1.0 + slippage_frac)
        else:
            exec_price = price * (1.0 - slippage_frac)

        notional_exec = exec_price * qty  # signed
        # Commission
        if commission.type == "per_share":
            comm = commission.amount * abs(qty)
        elif commission.type == "bps_notional":
            comm = (commission.amount / 1e4) * abs(notional_exec)
        else:
            comm = 0.0

        trade_cash = -notional_exec - comm  # cash decreases on buy (qty>0)
        cash_delta += trade_cash

        new_positions[instr] = new_positions[instr] + qty

        trades.append(
            {
                "datetime": date,
                "instrument": instr,
                "side": side,
                "quantity": qty,
                "price": exec_price,
                "notional": notional_exec,
                "slippage_bps": slippage_bps,
                "commission": comm,
                "fees": 0.0,  # per-trade static fees not modeled in detail
                "realized_pnl": 0.0,
            }
        )

    trades_df = pd.DataFrame(trades)
    return new_positions, cash_delta, trades_df
