from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


KIND_MARKET_BARS = "market_bars"
KIND_TIME_SERIES = "timeseries"


@dataclass(frozen=True, slots=True)
class DataRequest:
    """Provider-agnostic request for loading time series data.

    Notes:
      - `source` is a canonicalized, URI-like string (e.g. "parquet://...", "fred://CPIAUCSL").
      - `kind` specifies the expected dataset shape/semantics.
          - market_bars: OHLCV-like bars per instrument
          - timeseries: generic time series per entity (alt data)
      - `fields` meaning is kind-specific (OHLCV for bars, value columns for series, etc.).
      - `dataset_id` allows strategies to name multiple datasets even if they share source.
    """

    source: str
    kind: str = KIND_MARKET_BARS

    start: str = ""
    end: str = ""
    frequency: str = "1d"

    fields: List[str] = field(default_factory=list)

    calendar: Optional[str] = None
    tz: Optional[str] = None

    dataset_id: Optional[str] = None
