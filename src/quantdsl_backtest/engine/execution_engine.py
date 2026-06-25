# src/quantdsl_backtest/engine/execution_engine.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..dsl.execution import Execution
from ..models.slippage import build_slippage_model
from ..dsl.costs import Commission, StaticFees, TransactionTax
from ..models.costs import build_cost_model
from ..models.volume_limits import build_volume_limit_model
from ..utils.logging import get_logger


log = get_logger(__name__)


def rebalance_to_target_weights(
    date: pd.Timestamp,
    execution: Execution,
    commission: Commission,
    fees: StaticFees,
    cash: float,
    prices: pd.Series,
    volumes: pd.Series,
    prev_positions: pd.Series,
    target_weights: pd.Series,
    tax: "TransactionTax | None" = None,
) -> Tuple[pd.Series, float, pd.DataFrame]:
    """
    Size and execute a rebalance ENTIRELY at ``prices`` — the prices at the execution instant.

    NO-LOOKAHEAD INVARIANT (structural, not by convention)
    ------------------------------------------------------
    There is exactly ONE price vector. Equity is derived here as ``cash + prev_positions·prices``, and
    every order is BOTH sized and filled at that same ``prices``. The function is never handed a
    pre-computed equity or a second price series, so an order can never be sized from one timestamp
    (e.g. today's close) and filled at another (e.g. today's open). Whoever calls this at the open
    passes the open; at the close, the close. Time flows one way, by construction — there is no
    argument through which a future (or stale) price could enter.

    Rebalance gates — vectorized boolean masks over the instrument axis; all default OFF:
      * ``OrderPolicy.no_trade_band``      — skip unless ``|target_w - current_w| >= band`` (relative,
                                             scale-free: |Δnotional|/equity is exactly the weight gap).
      * ``OrderPolicy.min_trade_notional`` — skip orders below this many dollars (absolute, scalar).
      * ``OrderPolicy.min_trade_units``    — skip orders below this many units; a per-instrument dict
                                             ``{symbol: min_units}`` or a scalar applied to all.

    Returns
    -------
    (new_positions, cash_delta, trades_df)
    """
    instruments = list(target_weights.index)
    px   = prices.reindex(instruments).to_numpy(dtype=float)
    prev = prev_positions.reindex(instruments).fillna(0.0).to_numpy(dtype=float)
    vol  = volumes.reindex(instruments).fillna(0.0).to_numpy(dtype=float)
    tw   = target_weights.to_numpy(dtype=float)

    valid = np.isfinite(px) & (px > 0.0)

    # Equity marked at THE SAME prices we fill at -> single price vector -> no lookahead.
    equity = float(cash) + float(np.sum(np.where(valid, prev * px, 0.0)))

    current_notional = np.where(valid, prev * px, 0.0)
    target_notional  = tw * equity
    dn  = target_notional - current_notional                       # signed $ to trade per instrument
    qty = np.divide(dn, px, out=np.zeros_like(dn), where=valid)     # signed units

    # ---------------------- vectorized rebalance gates ----------------------
    _op           = getattr(execution, "order_policy", None)
    band          = float(getattr(_op, "no_trade_band", 0.0) or 0.0)
    min_notional  = float(getattr(_op, "min_trade_notional", 0.0) or 0.0)
    min_units_cfg = getattr(_op, "min_trade_units", None)

    trade = valid & (np.abs(dn) > 1e-8)
    if band > 0.0 and equity > 0.0:
        trade &= (np.abs(dn) / equity >= band)                     # relative gate: |Δweight| >= band
    if min_notional > 0.0:
        trade &= (np.abs(dn) >= min_notional)                      # absolute $ gate
    if min_units_cfg is not None:
        if hasattr(min_units_cfg, "get"):
            mu = np.array([float(min_units_cfg.get(s, 0.0)) for s in instruments])
        else:
            mu = np.full(len(instruments), float(min_units_cfg))
        trade &= (np.abs(qty) >= mu)                               # absolute per-instrument unit gate

    # Models (built once)
    vl_model   = build_volume_limit_model(getattr(execution, "volume_limits", None))
    sl_model   = build_slippage_model(execution.slippage)
    cost_model = build_cost_model(commission)
    # tax may be a single TransactionTax or a list (a mixed-domicile book has different stamp rates per name).
    _taxes = list(tax) if isinstance(tax, (list, tuple)) else ([] if tax is None else [tax])

    trades = []
    cash_delta = 0.0
    new_positions = prev_positions.reindex(instruments).fillna(0.0).copy()

    # Only instruments that pass the gate trade — the loop is over survivors, not the whole universe.
    for k in np.nonzero(trade)[0]:
        instr = instruments[k]
        price = float(px[k])

        # Volume participation cap may shrink the order; re-check the absolute gates after it.
        dnk = vl_model.cap_notional_change(dn=float(dn[k]), price=price, volume=float(vol[k]))
        if min_notional > 0.0 and abs(dnk) < min_notional:
            continue
        if not vl_model.passes_min_fill(abs(dnk)):
            continue

        q = dnk / price
        side = "BUY" if dnk > 0 else "SELL"

        slippage_bps  = sl_model.slippage_bps_from_order(qty=q, volume=float(vol[k]))
        slippage_frac = slippage_bps / 1e4
        exec_price = price * (1.0 + slippage_frac) if side == "BUY" else price * (1.0 - slippage_frac)

        notional_exec = exec_price * q                     # signed
        comm = cost_model.commission_from_trade(qty=q, exec_price=exec_price)

        # Transaction tax (e.g. UK stamp duty) — on trade notional, only the taxed side, only the named instruments.
        # Sum over every configured tax so a mixed book charges each name its own domicile's rate.
        tax_cost = 0.0
        for _tx in _taxes:
            if _tx.rate > 0.0 \
                    and (_tx.on == "both" or (_tx.on == "buy" and side == "BUY") or (_tx.on == "sell" and side == "SELL")) \
                    and (_tx.instruments is None or instr in _tx.instruments):
                tax_cost += _tx.rate * abs(notional_exec)

        cash_delta += -notional_exec - comm - tax_cost     # cash decreases on a buy (q>0); tax charged on the taxed side
        new_positions[instr] = new_positions[instr] + q

        trades.append(
            {
                "datetime": date,
                "instrument": instr,
                "side": side,
                "quantity": q,
                "raw_price": price,                                  # market price BEFORE slippage (fill level)
                "price": exec_price,                                 # filled price, slippage included
                "notional": notional_exec,
                "slippage_bps": slippage_bps,
                "slippage_usd": abs(q) * abs(exec_price - price),    # $ slippage cost
                "commission": comm,
                "tax": tax_cost,                                     # transaction tax $ (UK stamp etc.)
                "fees": 0.0,  # per-trade static fees not modeled in detail
                "realized_pnl": 0.0,
            }
        )

    return new_positions, cash_delta, pd.DataFrame(trades)
