# src/quantdsl_backtest/utils/time.py

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd


def parse_iso_date(value: str) -> pd.Timestamp:
    """
    Parse an ISO date string like '2015-01-01' into a pandas Timestamp.
    """
    return pd.to_datetime(value).normalize()


def business_days_between(start: str, end: str) -> List[pd.Timestamp]:
    """
    Generate a list of business-day timestamps between [start, end].

    This is *not* an exchange-specific calendar; it just uses pandas'
    notion of business days (Mon–Fri). Good enough for a skeleton.
    """
    start_ts = parse_iso_date(start)
    end_ts = parse_iso_date(end)
    return list(pd.bdate_range(start=start_ts, end=end_ts))


def align_to_business_days(
    dates: Iterable[pd.Timestamp],
) -> List[pd.Timestamp]:
    """
    Ensure a list of timestamps are normalized to day-level (no intraday),
    sorted, and unique. Useful when reconciling with daily data.
    """
    normed = [pd.Timestamp(d).normalize() for d in dates]
    uniq_sorted = sorted(set(normed))
    return uniq_sorted


def now_utc() -> pd.Timestamp:
    """
    Return current UTC time as pandas Timestamp.
    """
    return pd.Timestamp(datetime.utcnow(), tz="UTC")


def to_timezone(ts: pd.Timestamp, tz: Optional[str]) -> pd.Timestamp:
    """
    Convert a timestamp to a target timezone (if given). If `tz` is None,
    returns ts unchanged.
    """
    if tz is None:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)
