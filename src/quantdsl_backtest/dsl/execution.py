# src/quantdsl_backtest/dsl/execution.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# --- Order policy & latency -------------------------------------------------


@dataclass(slots=True)
class OrderPolicy:
    """
    High-level order defaults for the strategy.

    fill_on controls execution timing:
      "close"  — fill at today's close (same bar as signal, default)
      "open"   — fill at next bar's open (signal T close → fill T+1 open)
    """

    default_order_type: Literal["MKT", "MOC", "LIMIT"] = "MOC"
    time_in_force: Literal["DAY", "GTC"] = "DAY"
    fill_on: Literal["close", "open"] = "close"
    # --- Rebalance gates (vectorized, per instrument, checked every rebalance; all default off) ---
    no_trade_band: float = 0.0            # relative: skip unless |target_w - current_w| >= this
    min_trade_notional: float = 0.0       # absolute $: skip orders smaller than this (scalar)
    min_trade_units: Optional[dict] = None  # absolute units: {symbol: min_units} or a scalar; kills crumbs
    # --- Limit-on-open buys (fill_on="open" only; default off) ---
    # A buy (target weight above the weight held at the open) is a limit order at prev_close × (1 + pct):
    # it fills at the open only if open ≤ that limit, else it is cancelled for the day — the symbol keeps its
    # held weight, sells still execute, the unspent cash stays cash. The target is re-queued every bar, so an
    # unfilled buy is retried next morning at the new limit. Scalar = every symbol; dict = {symbol: pct}.
    limit_buy_pct: Optional[object] = None


@dataclass(slots=True)
class LatencyModel:
    """
    Simple latency model in bars/milliseconds.
    """

    signal_to_order_delay_bars: int = 0
    market_latency_ms: int = 0


# --- Slippage & volume limits ----------------------------------------------


@dataclass(slots=True)
class PowerLawSlippageModel:
    """
    Slippage model: slippage_bps = base_bps + k * (participation ** exponent)
    """

    base_bps: float = 1.0
    k: float = 20.0
    exponent: float = 0.5
    use_intraday_vol: bool = False
    # Per-instrument override of base_bps, e.g. {"TECL": 20.0, "TQQQ": 1.0, "BIL": 0.5}: one flat cost for every
    # instrument mis-prices a book that mixes a 1bp-spread ETF with a 40bp-at-the-open one. Instruments not
    # listed keep base_bps. Event-driven engine; the vectorized engine falls back to it when this is set.
    base_bps_by_symbol: Optional[dict] = None


@dataclass(slots=True)
class VolumeParticipation:
    """
    Volume-based participation limits. The engine will ensure that
    order sizes do not exceed a fraction of ADV or volume in the bar.
    """

    max_participation: float = 0.1      # e.g. <= 10% ADV
    mode: Literal["proportional", "clip"] = "proportional"
    min_fill_notional: float = 0.0


@dataclass(slots=True)
class LimitOrderBookModel:
    """
    Placeholder for a more detailed LOB simulation model.
    """

    levels: int = 1
    queue_priority: Literal["fifo", "pro_rata"] = "fifo"
    use_spread: bool = True


# --- Execution wrapper ------------------------------------------------------


@dataclass(slots=True)
class Execution:
    """
    Aggregate execution configuration: order policy, latency,
    slippage, volume limits, and optional LOB model.
    """

    order_policy: OrderPolicy
    latency: LatencyModel
    slippage: PowerLawSlippageModel
    volume_limits: VolumeParticipation
    book_model: Optional[LimitOrderBookModel] = None
