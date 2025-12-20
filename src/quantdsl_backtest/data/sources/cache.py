from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Optional

import pandas as pd

from ..requests import DataRequest


class MemoryCacheStore:
    """Test-friendly in-memory cache (raw frames only)."""

    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}

    def has(self, key: str) -> bool:
        return key in self._store

    def read_df(self, key: str) -> pd.DataFrame:
        return self._store[key].copy()

    def write_df(self, key: str, df: pd.DataFrame) -> None:
        self._store[key] = df.copy()


@dataclass(slots=True)
class ArcticCacheStore:
    """ArcticDB-backed raw frame store.

    This wraps the existing cache_arctic helper functions.
    """

    provider: str
    frequency: str

    def _lib(self):
        from ..cache_arctic import get_cache_lib

        return get_cache_lib(provider=self.provider, frequency=self.frequency)

    def has(self, key: str) -> bool:
        from ..cache_arctic import cache_has_symbol

        return cache_has_symbol(self._lib(), key)

    def read_df(self, key: str) -> pd.DataFrame:
        from ..cache_arctic import cache_read_symbol

        return cache_read_symbol(self._lib(), key)

    def write_df(self, key: str, df: pd.DataFrame) -> None:
        from ..cache_arctic import cache_write_symbol

        cache_write_symbol(self._lib(), key, df)


@dataclass(slots=True)
class SafeArcticCacheStore:
    """Fail-safe ArcticDB cache wrapper.

    If the local ArcticDB/LMDB store is missing/corrupted/incompatible, operations
    degrade to a no-op cache instead of raising. This keeps tests and local runs
    stable while still enabling ArcticDB as the preferred golden store.
    """

    provider: str
    frequency: str

    _disabled: bool = field(default=False, init=False, repr=False)
    _reason: str = field(default="", init=False, repr=False)

    def _inner(self) -> ArcticCacheStore:
        return ArcticCacheStore(provider=self.provider, frequency=self.frequency)

    def _disable(self, reason: str) -> None:
        object.__setattr__(self, "_disabled", True)
        object.__setattr__(self, "_reason", reason)

    def has(self, key: str) -> bool:
        if self._disabled:
            return False
        try:
            return self._inner().has(key)
        except Exception as e:
            self._disable(str(e))
            return False

    def read_df(self, key: str) -> pd.DataFrame:
        if self._disabled:
            raise KeyError(key)
        try:
            return self._inner().read_df(key)
        except Exception as e:
            self._disable(str(e))
            raise KeyError(key)

    def write_df(self, key: str, df: pd.DataFrame) -> None:
        if self._disabled:
            return
        try:
            self._inner().write_df(key, df)
        except Exception as e:
            self._disable(str(e))
            return


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def _hash_source(source: str) -> str:
    h = hashlib.sha1((source or "").encode("utf-8")).hexdigest()
    return h[:12]


def dataset_partition(request: DataRequest) -> str:
    if request.dataset_id:
        return request.dataset_id
    return _hash_source(request.source)


def build_cache_key(*, provider: str, request: DataRequest, entity: str) -> str:
    provider_norm = (provider or "").upper()
    # v1/<provider>/<kind>/<frequency>/<dataset>/<entity>
    return f"v1/{provider_norm}/{request.kind}/{request.frequency}/{dataset_partition(request)}/{entity}"


# ---------------------------------------------------------------------------
# Generic tail-fetch loader
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TailCacheStats:
    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    tail_fetches: int = 0


@dataclass(slots=True)
class TailCachedFrameLoader:
    """Generic raw-first cached frame loader with tail-fetch support.

    Provider injects:
      - normalize(df) -> df
      - fetch(start_ts, end_ts) -> df
      - last_needed_ts(end_ts, frequency) -> ts
      - next_fetch_start(last_dt, frequency) -> ts
    """

    provider: str
    stats: TailCacheStats = field(default_factory=TailCacheStats)

    def load_one(
        self,
        request: DataRequest,
        *,
        entity: str,
        cache,
        normalize: Callable[[pd.DataFrame], pd.DataFrame],
        fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
        last_needed_ts: Callable[[pd.Timestamp, str], pd.Timestamp],
        next_fetch_start: Callable[[pd.Timestamp, str], pd.Timestamp],
    ) -> pd.DataFrame:
        end_ts = pd.to_datetime(request.end)
        needed_last_ts = last_needed_ts(end_ts, request.frequency)

        cache_key = build_cache_key(provider=self.provider, request=request, entity=entity)

        cached: Optional[pd.DataFrame] = None
        if cache is not None and cache.has(cache_key):
            self.stats.reads += 1
            cached = normalize(cache.read_df(cache_key))
            last_dt = cached.index.max() if not cached.empty else None

            if last_dt is not None and last_dt >= needed_last_ts:
                self.stats.hits += 1
                return cached.loc[request.start: request.end]

        if cached is None:
            self.stats.misses += 1
            start_ts = pd.to_datetime(request.start)
            fetched = normalize(fetch(start_ts, end_ts))
            if cache is not None:
                cache.write_df(cache_key, fetched)
                self.stats.writes += 1
            return fetched.loc[request.start: request.end]

        self.stats.tail_fetches += 1
        last_dt = cached.index.max() if not cached.empty else None
        start_ts = pd.to_datetime(request.start) if last_dt is None else next_fetch_start(last_dt, request.frequency)

        fetched = normalize(fetch(start_ts, end_ts))
        merged = pd.concat([cached, fetched], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        if cache is not None:
            cache.write_df(cache_key, merged)
            self.stats.writes += 1

        return merged.loc[request.start: request.end]
