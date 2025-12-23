from __future__ import annotations

import re

import pandas as pd


_FREQ_RE = re.compile(r"^(?P<n>\d+)?\s*(?P<unit>[a-zA-Z]+)$")


def parse_frequency(freq: str) -> tuple[int, str]:
    """Parse a frequency string like '1d', '5m', '1h'.

    Returns (n, unit) where unit is normalized to: 'd','h','m'.

    Unknown units default to daily.
    """

    f = (freq or "1d").strip().lower()
    m = _FREQ_RE.match(f)
    if not m:
        return 1, "d"

    n_s = m.group("n")
    unit = m.group("unit")
    n = int(n_s) if n_s else 1

    if unit in {"d", "day", "days", "1d"}:
        return n, "d"
    if unit in {"h", "hr", "hour", "hours", "1h"}:
        return n, "h"
    if unit in {"m", "min", "mins", "minute", "minutes", "1m"}:
        return n, "m"

    return n, "d"


def next_bar_start(last_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
    """Return the next bar timestamp after `last_ts` for a given frequency.

    This is used by the platform coverage planner and cache-tail fetch logic.

    Notes:
      - For daily bars we normalize to midnight.
      - For intraday bars we add a fixed timedelta.
    """

    n, unit = parse_frequency(frequency)

    ts = pd.Timestamp(last_ts)

    if unit == "d":
        return (ts + pd.Timedelta(days=n)).normalize()
    if unit == "h":
        return ts + pd.Timedelta(hours=n)
    if unit == "m":
        return ts + pd.Timedelta(minutes=n)

    return (ts + pd.Timedelta(days=n)).normalize()
