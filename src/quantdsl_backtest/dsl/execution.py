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

    exit_on controls how long a position is held:
      "hold"       — until a later rebalance changes it (default)
      "next_open"  — everything is sold at the NEXT bar's open, so the book is held overnight only and is
                     flat during the day: buy in the closing auction (MOC), sell in the next opening auction
                     (MOO). Needs fill_on="close". Event-driven engine only.
    """

    default_order_type: Literal["MKT", "MOC", "LIMIT"] = "MOC"
    time_in_force: Literal["DAY", "GTC"] = "DAY"
    fill_on: Literal["close", "open"] = "close"
    exit_on: Literal["hold", "next_open"] = "hold"
    # --- Rebalance gates (vectorized, per instrument, checked every rebalance; all default off) ---
    no_trade_band: float = 0.0            # relative: skip unless |target_w - current_w| >= this
    min_trade_notional: float = 0.0       # absolute $: skip orders smaller than this (scalar)
    min_trade_units: Optional[dict] = None  # absolute units: {symbol: min_units} or a scalar; kills crumbs
    # A cash account that cannot borrow (an ISA): sells execute first and buys are shrunk pro rata so cash,
    # after every cost, never goes below zero. Event-driven engine only. Long-only books.
    cash_only: bool = False

    def __post_init__(self) -> None:
        if self.exit_on not in ("hold", "next_open"):
            raise ValueError(f"exit_on must be 'hold' or 'next_open', got {self.exit_on!r}.")
        if self.exit_on == "next_open" and self.fill_on != "close":
            raise ValueError(
                "exit_on='next_open' sells at the open what was bought at the close: it needs fill_on='close'."
            )


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
