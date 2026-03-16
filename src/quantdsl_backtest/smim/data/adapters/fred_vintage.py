"""FRED / ALFRED vintage adapter for SMIM.

The standard btest FRED adapter (``data/sources/fred.py``) calls
``fredapi.Fred.get_series()`` without vintage parameters, so it always
returns the *current* revision of a series — violating A1 when used in
a historical backtest context.

This adapter adds ALFRED vintage retrieval: when ``as_of`` is supplied,
it passes ``realtime_start=as_of`` and ``realtime_end=as_of`` to the
FRED API, which returns the data exactly as it was published on that date.

Config: ``SmimConfig.data.fred``
  - ``api_key_env``: environment variable holding the FRED API key
  - ``enable_vintage``: if False, the ``as_of`` parameter is ignored and
    current data is returned (useful for exploration, never for backtesting)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from quantdsl_backtest.smim.config import FredConfig
from quantdsl_backtest.smim.interfaces import DateRange


def _load_api_key(env_var: str) -> str:
    """Load FRED API key from environment variable.

    Args:
        env_var: Name of the environment variable (default ``FRED_API_KEY``).

    Returns:
        API key string.

    Raises:
        RuntimeError: If the variable is not set or empty.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"FRED API key not found. Set the '{env_var}' environment variable "
            "or get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


@dataclass
class FredVintageAdapter:
    """FRED / ALFRED adapter with point-in-time vintage retrieval.

    Implements ``smim.interfaces.DataAdapter``.

    Args:
        config: ``FredConfig`` section from ``SmimConfig.data.fred``.
    """

    config: FredConfig = field(default_factory=FredConfig)

    @property
    def source_name(self) -> str:
        return "fred"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch one or more FRED series, optionally as a vintage snapshot.

        Args:
            series_ids: List of FRED series identifiers (e.g. ``["FEDFUNDS"]``).
            date_range: ``DateRange`` specifying ``start``, ``end``, ``frequency``.
            as_of: If provided and ``config.enable_vintage`` is True, retrieves
                the ALFRED vintage of each series as of this date.
                Both ``realtime_start`` and ``realtime_end`` are set to this value.

        Returns:
            DataFrame with ``pd.DatetimeIndex`` (tz-naive, name ``"event_date"``)
            and one float64 column per series_id. Missing values are NaN.
        """
        api_key = _load_api_key(self.config.api_key_env)

        try:
            from fredapi import Fred  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "fredapi is required for FredVintageAdapter. "
                "Install with: pip install fredapi"
            ) from exc

        fred = Fred(api_key=api_key)

        vintage_kwargs: dict[str, str] = {}
        if as_of is not None and self.config.enable_vintage:
            as_of_str = as_of.strftime("%Y-%m-%d")
            vintage_kwargs = {
                "realtime_start": as_of_str,
                "realtime_end": as_of_str,
            }

        frames: dict[str, pd.Series] = {}
        start_str = date_range.start.strftime("%Y-%m-%d")
        end_str = date_range.end.strftime("%Y-%m-%d")

        for sid in series_ids:
            series = fred.get_series(
                sid,
                observation_start=start_str,
                observation_end=end_str,
                **vintage_kwargs,
            )
            if series is None or len(series) == 0:
                frames[sid] = pd.Series(dtype="float64", name=sid)
            else:
                series.index = pd.to_datetime(series.index).tz_localize(None)
                series = series.astype("float64")
                series.name = sid
                frames[sid] = series

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "event_date"
        return df
