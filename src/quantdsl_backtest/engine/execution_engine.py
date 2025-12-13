# src/quantdsl_backtest/engine/execution_engine.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..dsl.execution import Execution
from ..models.slippage import build_slippage_model
from ..dsl.costs import Commission, StaticFees
from ..models.costs import build_cost_model
from ..models.volume_limits import build_volume_limit_model
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

    # Volume limits model
    vl_model = build_volume_limit_model(getattr(execution, "volume_limits", None))

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

        # Apply volume participation limit using model
        dn = vl_model.cap_notional_change(dn=dn, price=price, volume=vol)
        if not vl_model.passes_min_fill(abs(dn)):
            continue

        # Determine side and quantity
        side = "BUY" if dn > 0 else "SELL"
        qty = dn / price  # signed quantity

        # Slippage model (delegated to models.slippage)
        sl_model = build_slippage_model(execution.slippage)
        slippage_bps = sl_model.slippage_bps_from_order(qty=qty, volume=vol)
        slippage_frac = slippage_bps / 1e4

        if side == "BUY":
            exec_price = price * (1.0 + slippage_frac)
        else:
            exec_price = price * (1.0 - slippage_frac)

        notional_exec = exec_price * qty  # signed
        # Commission via model
        cost_model = build_cost_model(commission)
        comm = cost_model.commission_from_trade(qty=qty, exec_price=exec_price)

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
