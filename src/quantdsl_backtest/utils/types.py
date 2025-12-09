# src/quantdsl_backtest/utils/types.py

from __future__ import annotations

from typing import Hashable, NewType

import pandas as pd


InstrumentId = NewType("InstrumentId", str)
DateLike = Hashable  # pandas Timestamp, datetime, string, etc.

EquitySeries = pd.Series
ReturnsSeries = pd.Series
