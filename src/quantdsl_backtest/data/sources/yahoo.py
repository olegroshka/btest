from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ...dsl.universe import Universe
from ..bundles import MarketBarsBundle
from ..requests import DataRequest
from .base import CacheStore
from .cache import TailCachedFrameLoader
from .universe_filtering import UniverseFilterEngine
from .stats import CacheStatsMixin


YF_PREFIX = "yf://"


@dataclass(slots=True)
class YahooCoveragePolicy:
    """Define coverage semantics for Yahoo bars.

    vectorbt/yfinance has end semantics that are effectively exclusive for daily bars.
    """

    def last_needed_ts(self, end_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
        interval = YahooNormalizer.infer_interval(frequency)
        if interval == "1d":
            return (end_ts - pd.Timedelta(days=1)).normalize()
        return end_ts

    def next_fetch_start(self, last_dt: pd.Timestamp, frequency: str) -> pd.Timestamp:
        interval = YahooNormalizer.infer_interval(frequency)
        if interval == "1d":
            return (last_dt + pd.Timedelta(days=1)).normalize()
        # Conservative for intraday
        return last_dt


@dataclass(slots=True)
class YahooClient:
    """Data access client for Yahoo via vectorbt."""

    def download(self, symbols: List[str], *, start: str, end: str, interval: str) -> Dict[str, pd.DataFrame]:
        try:
            import vectorbt as vbt
        except Exception as e:
            raise RuntimeError(
                "vectorbt is required for Yahoo Finance access via YFData. Install with: pip install vectorbt"
            ) from e

        yf_data = vbt.YFData.download(
            symbols,
            start=start,
            end=end,
            interval=interval,
            missing_index="drop",
        )

        open_df = yf_data.get("Open")
        high_df = yf_data.get("High")
        low_df = yf_data.get("Low")
        close_df = yf_data.get("Close")
        vol_df = yf_data.get("Volume")

        def _ensure_df(obj, col_name: str) -> pd.DataFrame:
            if isinstance(obj, pd.Series):
                return obj.to_frame(col_name)
            return obj

        open_df = _ensure_df(open_df, symbols[0])
        high_df = _ensure_df(high_df, symbols[0])
        low_df = _ensure_df(low_df, symbols[0])
        close_df = _ensure_df(close_df, symbols[0])
        vol_df = _ensure_df(vol_df, symbols[0])

        open_df = open_df[symbols]
        high_df = high_df[symbols]
        low_df = low_df[symbols]
        close_df = close_df[symbols]
        vol_df = vol_df[symbols]

        frames: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            frames[sym] = pd.DataFrame(
                {
                    "open": open_df[sym],
                    "high": high_df[sym],
                    "low": low_df[sym],
                    "close": close_df[sym],
                    "volume": vol_df[sym],
                }
            ).dropna(how="all")

        return frames


@dataclass(slots=True)
class YahooNormalizer:
    """Normalize Yahoo OHLCV frames into sorted tz-naive DatetimeIndex."""

    @staticmethod
    def infer_interval(frequency: str) -> str:
        freq = (frequency or "1d").lower().strip()
        if freq in ("1d", "d", "daily"):
            return "1d"
        if freq in ("1h", "h", "60min", "60m"):
            return "1h"
        if freq in ("1m", "m", "1min"):
            return "1m"
        return "1d"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([]),
            )

        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()

        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
        return out[cols]


def _extract_symbols(source: str, universe: Optional[Universe]) -> List[str]:
    if universe is not None and universe.static_instruments:
        return list(universe.static_instruments)

    raw = source[len(YF_PREFIX):]
    if raw.strip():
        return [s.strip() for s in raw.split(",") if s.strip()]

    return []


@dataclass(slots=True)
class YahooMarketBarsSource(CacheStatsMixin):
    """Yahoo (YF) market bars provider with ArcticDB caching + tail fetch.

    Uses composition:
      - YahooClient for I/O
      - TailCachedFrameLoader for cache/tail-merge workflow
      - UniverseFilterEngine for universe filters
    """

    name: str = "yahoo_market_bars"

    client: YahooClient = field(default_factory=YahooClient)
    normalizer: YahooNormalizer = field(default_factory=YahooNormalizer)
    coverage: YahooCoveragePolicy = field(default_factory=YahooCoveragePolicy)
    cache_loader: TailCachedFrameLoader = field(
        default_factory=lambda: TailCachedFrameLoader(provider="YF")
    )
    universe_filters: UniverseFilterEngine = field(default_factory=UniverseFilterEngine)

    # last per-entity stats snapshot (platform/UI)
    _last_entity_cache_stats: Dict[str, Dict[str, int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def can_load(self, request: DataRequest) -> bool:
        return request.kind == "market_bars" and request.source.startswith(YF_PREFIX)

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> MarketBarsBundle:
        symbols = _extract_symbols(request.source, universe)
        if not symbols:
            raise ValueError(
                "No tickers specified for Yahoo Finance source. "
                "Provide Universe.static_instruments or use source='yf://AAPL,MSFT,...'."
            )

        interval = YahooNormalizer.infer_interval(request.frequency)

        raw_frames: Dict[str, pd.DataFrame] = {}
        # Per-entity cache stats deltas for platform/UI
        entity_stats: Dict[str, dict[str, int]] = {}

        for sym in symbols:
            before = self.cache_loader.stats.snapshot()

            def _fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
                # vectorbt expects strings / end semantics are provider-specific
                frames = self.client.download([sym], start=str(start.date()), end=str(end.date()), interval=interval)
                return frames[sym]

            df = self.cache_loader.load_one(
                request,
                entity=sym,
                cache=cache,
                normalize=self.normalizer,
                fetch=_fetch,
                last_needed_ts=self.coverage.last_needed_ts,
                next_fetch_start=self.coverage.next_fetch_start,
            )
            raw_frames[sym] = df

            after = self.cache_loader.stats.snapshot()
            entity_stats[sym] = after.delta(before).as_dict()

        # attach stats for platform consumers
        self._last_entity_cache_stats = entity_stats

        bars_by_instr: Dict[str, pd.DataFrame] = {}
        instruments: List[str] = []
        filters = universe.filters if universe is not None else []

        for sym in symbols:
            df = raw_frames[sym].loc[request.start: request.end]

            if filters and not self.universe_filters.passes(df, filters):
                continue

            cols = [c for c in (request.fields or []) if c in df.columns]
            extra_cols = [c for c in df.columns if c not in cols]
            df = df[cols + extra_cols]

            bars_by_instr[sym] = df.sort_index()
            instruments.append(sym)

        return MarketBarsBundle(
            kind="market_bars",
            source=request.source,
            start=request.start,
            end=request.end,
            frequency=request.frequency,
            calendar=request.calendar,
            tz=request.tz,
            bars=bars_by_instr,
            instruments=instruments,
            fields=list(request.fields or []),
        )

    def last_entity_cache_stats(self) -> Dict[str, dict[str, int]]:
        return self._last_entity_cache_stats or {}
