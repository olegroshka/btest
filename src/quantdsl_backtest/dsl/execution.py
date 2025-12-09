# src/quantdsl_backtest/dsl/execution.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# --- Order policy & latency -------------------------------------------------


@dataclass(slots=True)
class OrderPolicy:
    """
    High-level order defaults for the strategy.
    """

    default_order_type: Literal["MKT", "MOC", "LIMIT"] = "MOC"
    time_in_force: Literal["DAY", "GTC"] = "DAY"


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
