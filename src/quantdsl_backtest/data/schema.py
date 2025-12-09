# src/quantdsl_backtest/data/schema.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


InstrumentId = str


@dataclass(slots=True)
class MarketData:
    """
    In-memory representation of market data for a backtest.

    We deliberately keep it simple:

    - `bars`: mapping instrument_id -> DataFrame
      Each DataFrame is indexed by datetime, with columns like
      ["open", "high", "low", "close", "volume", ...].

    - `instruments`: sorted list of instrument IDs included.

    - `fields`: which data fields are present (e.g. OHLCV).

    - `frequency`: "1d", "1m", etc. (copied from DataConfig).

    - `calendar`: exchange calendar name, e.g. "XNYS".

    - `tz`: optional timezone of the timestamps.
    """

    bars: Dict[InstrumentId, pd.DataFrame]
    instruments: List[InstrumentId]
    fields: List[str]
    frequency: str
    calendar: str
    tz: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize instrument list
        self.instruments = sorted(set(self.instruments))

    def get_bar_data(self, instrument: InstrumentId) -> pd.DataFrame:
        """Convenience accessor for a single instrument."""
        return self.bars[instrument]
