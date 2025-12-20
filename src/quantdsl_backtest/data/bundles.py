from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Base type for loaded datasets.

    Bundles are typed by `kind` and may carry different payload shapes.
    """

    kind: str
    source: str
    start: str
    end: str
    frequency: str
    calendar: Optional[str] = None
    tz: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MarketBarsBundle(DataBundle):
    """Market OHLCV bars, stored per-instrument.

    Payload contract matches existing MarketData semantics:
      - bars[instrument] is a DataFrame indexed by datetime, with OHLCV-like columns.
    """

    bars: Dict[str, pd.DataFrame] = None  # type: ignore[assignment]
    instruments: List[str] = None  # type: ignore[assignment]
    fields: List[str] = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class TimeSeriesBundle(DataBundle):
    """Generic (alternative) time series bundle.

    - `frames` maps entity_id -> DataFrame indexed by datetime.
    - For single-series sources like FRED, entity_id is the series id.
    """

    frames: Dict[str, pd.DataFrame] = None  # type: ignore[assignment]
    entities: List[str] = None  # type: ignore[assignment]
    fields: List[str] = None  # type: ignore[assignment]

