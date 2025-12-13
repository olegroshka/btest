from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(slots=True)
class LatencyRuntime:
    """
    Minimal runtime latency model built from DSL config.
    Only bar-based delay is applied in current implementation.
    """

    delay_bars: int = 0
    market_latency_ms: int = 0


def build_latency_model(cfg: Optional[object]) -> LatencyRuntime:
    if cfg is None:
        return LatencyRuntime(delay_bars=0, market_latency_ms=0)
    delay = int(getattr(cfg, "signal_to_order_delay_bars", 0) or 0)
    mkt_ms = int(getattr(cfg, "market_latency_ms", 0) or 0)
    return LatencyRuntime(delay_bars=delay, market_latency_ms=mkt_ms)


def apply_latency_to_weights(weights: pd.DataFrame, latency: LatencyRuntime) -> pd.DataFrame:
    """
    Apply bar delay by shifting target weights forward in time.
    NaNs introduced by the shift cause vectorbt to keep previous holdings.
    """
    if weights is None:
        return weights
    db = int(getattr(latency, "delay_bars", 0) or 0)
    if db <= 0:
        return weights
    return weights.shift(db)
