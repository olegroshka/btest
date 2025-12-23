from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ...dsl.universe import Universe
from ..bundles import MarketBarsBundle, TimeSeriesBundle
from ..requests import DataRequest, KIND_TIME_SERIES
from ..resolver import resolve_source
from .base import CacheStore
from .cache import TailCachedFrameLoader
from .stats import CacheStatsMixin


FRED_PREFIX = "fred://"


@dataclass(slots=True)
class FredClient:
    """Thin wrapper around the external FRED fetch function.

    Kept as an object to make dependency injection/testing straightforward.
    """

    def fetch(self, series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from ..market import fetch_fred_series

        return fetch_fred_series(series_id, start, end)


@dataclass(slots=True)
class FredNormalizer:
    """Normalize FRED frames into a single-column (value) Timeseries with tz-naive DatetimeIndex."""

    value_col: str

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[self.value_col], index=pd.DatetimeIndex([], name="date"))

        out = df.copy()

        if "date" in out.columns:
            # Vectorized + explicit tz-naive conversion
            dt = pd.to_datetime(out["date"], errors="coerce")
            # .dt is valid on datetime-like Series; guard for object dtype / NaT
            if hasattr(dt, "dt"):
                dt = dt.dt.tz_localize(None)
            out = out.drop(columns=["date"]).set_index(dt)
            out.index.name = "date"

        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")

        out.index = out.index.tz_localize(None)
        out = out.sort_index()

        if self.value_col not in out.columns:
            if "value" in out.columns:
                out = out.rename(columns={"value": self.value_col})
            elif "close" in out.columns:
                out = out.rename(columns={"close": self.value_col})
            else:
                first = out.columns[0]
                out = out.rename(columns={first: self.value_col})

        return out[[self.value_col]].astype("float64")


def _fred_last_needed_ts(end_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
    # FRED is effectively daily (calendar series). Treat end as inclusive.
    return end_ts


def _fred_next_fetch_start(last_dt: pd.Timestamp, frequency: str) -> pd.Timestamp:
    return (last_dt + pd.Timedelta(days=1)).normalize()


def _parse_fred_source(source: str) -> str:
    src = (source or "").strip()
    if not src.lower().startswith(FRED_PREFIX):
        raise ValueError(
            f"FRED source must use 'fred://<SERIES_ID>' scheme, got {source!r}"
        )
    return src[len(FRED_PREFIX):].strip()


@dataclass(slots=True)
class FredMarketBarsSource(CacheStatsMixin):
    """FRED provider that exposes a series as market-style bars."""

    name: str = "fred_market_bars"
    client: FredClient = field(default_factory=FredClient)
    cache_loader: TailCachedFrameLoader = field(
        default_factory=lambda: TailCachedFrameLoader(provider="FRED")
    )

    def can_load(self, request: DataRequest) -> bool:
        return request.kind == "market_bars" and request.source.lower().startswith(
            FRED_PREFIX
        )

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> MarketBarsBundle:
        resolved_source = resolve_source(request.source)
        series_id = _parse_fred_source(resolved_source)

        # snapshot stats for per-entity reporting
        before = self.cache_loader.stats.snapshot()

        norm = FredNormalizer("close")

        def _fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            df = self.client.fetch(series_id, start, end)
            # normalize expects the column to become "close"; rename here for clarity
            if "value" in df.columns:
                df = df.rename(columns={"value": "close"})
            return df

        df = self.cache_loader.load_one(
            request,
            entity=series_id,
            cache=cache,
            normalize=norm,
            fetch=_fetch,
            last_needed_ts=_fred_last_needed_ts,
            next_fetch_start=_fred_next_fetch_start,
        )

        after = self.cache_loader.stats.snapshot()
        try:
            object.__setattr__(self, "_last_entity_cache_stats", {series_id: after.delta(before).as_dict()})  # type: ignore[attr-defined]
        except Exception:
            pass

        cols: dict[str, pd.Series] = {}
        if "close" in request.fields:
            cols["close"] = df["close"]
        if "open" in request.fields:
            cols["open"] = df["close"]
        if "high" in request.fields:
            cols["high"] = df["close"]
        if "low" in request.fields:
            cols["low"] = df["close"]
        if "volume" in request.fields:
            cols["volume"] = pd.Series(index=df.index, data=float("nan"))

        df_bars = pd.DataFrame(cols).loc[request.start: request.end]

        return MarketBarsBundle(
            kind="market_bars",
            source=resolved_source,
            start=request.start,
            end=request.end,
            frequency=request.frequency,
            calendar=request.calendar,
            tz=request.tz,
            bars={series_id: df_bars},
            instruments=[series_id],
            fields=list(df_bars.columns),
        )

    def last_entity_cache_stats(self) -> dict[str, dict[str, int]]:
        return getattr(self, "_last_entity_cache_stats", {}) or {}


@dataclass(slots=True)
class FredTimeSeriesSource(CacheStatsMixin):
    """FRED provider that exposes a series as generic time series (alt data)."""

    name: str = "fred_timeseries"
    client: FredClient = field(default_factory=FredClient)
    cache_loader: TailCachedFrameLoader = field(
        default_factory=lambda: TailCachedFrameLoader(provider="FRED")
    )

    def can_load(self, request: DataRequest) -> bool:
        return request.kind == KIND_TIME_SERIES and request.source.lower().startswith(
            FRED_PREFIX
        )

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> TimeSeriesBundle:
        resolved_source = resolve_source(request.source)
        series_id = _parse_fred_source(resolved_source)

        before = self.cache_loader.stats.snapshot()

        norm = FredNormalizer("value")

        def _fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
            return self.client.fetch(series_id, start, end)

        df = self.cache_loader.load_one(
            request,
            entity=series_id,
            cache=cache,
            normalize=norm,
            fetch=_fetch,
            last_needed_ts=_fred_last_needed_ts,
            next_fetch_start=_fred_next_fetch_start,
        )

        after = self.cache_loader.stats.snapshot()
        try:
            object.__setattr__(self, "_last_entity_cache_stats", {series_id: after.delta(before).as_dict()})  # type: ignore[attr-defined]
        except Exception:
            pass

        return TimeSeriesBundle(
            kind=KIND_TIME_SERIES,
            source=resolved_source,
            start=request.start,
            end=request.end,
            frequency=request.frequency,
            calendar=request.calendar,
            tz=request.tz,
            frames={series_id: df},
            entities=[series_id],
            fields=list(df.columns),
        )

    def last_entity_cache_stats(self) -> dict[str, dict[str, int]]:
        return getattr(self, "_last_entity_cache_stats", {}) or {}

