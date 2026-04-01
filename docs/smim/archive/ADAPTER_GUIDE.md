# SMIM Adapter Guide

How data adapters work in btest and how SMIM adapters fit in.

---

## 1  The btest Adapter Interface

All btest data providers implement the `DataSource` protocol
(`src/quantdsl_backtest/data/sources/base.py`):

```python
class DataSource(Protocol):
    name: str
    def can_load(self, request: DataRequest) -> bool: ...
    def load(self, request: DataRequest, universe, cache) -> DataBundle: ...
```

`DataRequest` is a provider-agnostic frozen dataclass:

```python
@dataclass(frozen=True)
class DataRequest:
    source: str        # canonicalized URI, e.g. "fred://FEDFUNDS"
    kind: str          # "market_bars" | "timeseries"
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: str     # "1d", "1W", "1ME", …
    fields: tuple[str, ...]
    calendar: str
    tz: str
    dataset_id: str | None  # optional user-controlled cache partition key
```

`load()` returns a `DataBundle` subclass (`MarketBarsBundle` or
`TimeSeriesBundle`). The engine then unpacks the bundle.

---

## 2  How URI Resolution Works

A strategy declares data as:

```python
DataConfig(source="fred://FEDFUNDS", kind="timeseries", ...)
```

The engine calls `DataConfigAdapter(cfg).to_request()` which internally
calls `resolve_source()` (`data/resolver.py`). `resolve_source` canonicalises
the URI (lowercases the scheme, expands FINRA shorthand aliases, etc.) and
returns a normalised string still in `scheme://id` form.

The `DataSourceRegistry` (`data/sources/registry.py`) iterates registered
providers in order and returns the first whose `can_load()` returns `True`:

```python
def resolve(self, request: DataRequest) -> DataSource:
    for p in self.providers:
        if p.can_load(request):
            return p
    raise ValueError(...)
```

The default registry (`data/orchestrator.py`) registers four providers:
`ParquetMarketBarsSource`, `YahooMarketBarsSource`, `FredMarketBarsSource`,
`FredTimeSeriesSource`.

---

## 3  ArcticDB Caching

The cache is **opt-in** and **fail-safe**.

### Three-layer design

| Layer | Class | File |
|---|---|---|
| Low-level | `get_cache_lib(provider, frequency)` | `data/cache_arctic.py` |
| Store protocol | `SafeArcticCacheStore` | `data/sources/cache.py` |
| Tail-fetch logic | `TailCachedFrameLoader` | `data/sources/cache.py` |

**`SafeArcticCacheStore`** wraps `ArcticCacheStore` in a try/except; if
ArcticDB is missing or the LMDB store is corrupted, all operations degrade
to no-ops rather than raising. This keeps tests and cold-start runs stable.

**`TailCachedFrameLoader`** provides the smart tail-fetch loop:

1. Check if `key` exists in the cache.
2. If yes, read cached data and check if it covers the full requested range.
3. If stale (last date < `last_needed_ts`), fetch only the missing tail and
   append it to the cache.
4. If no cache entry, fetch the full range and write it.

Cache keys follow the pattern `{kind}/{dataset}/{entity}` where `dataset`
is derived from `request.dataset_id` or the URI basename.

**Cache library naming**: `market_data/{PROVIDER}/{frequency}` in the local
LMDB store (`lmdb://local_cache` by default, overridable via
`QUANTDSL_ARCTIC_URI` env var).

### Using the cache in a new btest adapter

```python
from ..sources.cache import TailCachedFrameLoader
from ..cache_arctic import SafeArcticCacheStore   # not imported directly;
                                                  # passed in via orchestrator

cache_loader = TailCachedFrameLoader(provider="MY_PROVIDER")

df = cache_loader.load_one(
    request,
    entity=series_id,
    cache=cache,          # SafeArcticCacheStore instance from orchestrator
    normalize=normalizer, # callable: raw_df -> normalised_df
    fetch=fetch_fn,       # callable: (start, end) -> raw_df
    last_needed_ts=...,   # callable: (end_ts, freq) -> pd.Timestamp
    next_fetch_start=..., # callable: (last_dt, freq) -> pd.Timestamp
)
```

---

## 4  Adding a New btest Adapter (step by step)

This is for adapters that integrate with the backtest engine (market bars
or time series for strategies). See §5 for SMIM-specific adapters.

**Step 1** — Create `src/quantdsl_backtest/data/sources/my_source.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from ..requests import DataRequest
from ..bundles import TimeSeriesBundle
from ...dsl.universe import Universe
from .base import CacheStore
from .cache import TailCachedFrameLoader
from .stats import CacheStatsMixin

MY_PREFIX = "mysource://"

@dataclass(slots=True)
class MyClient:
    def fetch(self, series_id: str,
              start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Hit the real API here; return DataFrame with columns [date, value]
        ...

@dataclass(slots=True)
class MySource(CacheStatsMixin):
    name: str = "my_source"
    client: MyClient = field(default_factory=MyClient)
    cache_loader: TailCachedFrameLoader = field(
        default_factory=lambda: TailCachedFrameLoader(provider="MY")
    )

    def can_load(self, request: DataRequest) -> bool:
        return (request.kind == "timeseries"
                and request.source.lower().startswith(MY_PREFIX))

    def load(self, request: DataRequest,
             universe: Optional[Universe],
             cache: Optional[CacheStore]) -> TimeSeriesBundle:
        series_id = request.source[len(MY_PREFIX):]
        df = self.cache_loader.load_one(
            request, entity=series_id, cache=cache,
            normalize=lambda x: x,    # or a real normalizer
            fetch=lambda s, e: self.client.fetch(series_id, s, e),
            last_needed_ts=lambda end, _: end,
            next_fetch_start=lambda last, _: last + pd.Timedelta(days=1),
        )
        return TimeSeriesBundle(
            kind="timeseries", source=request.source,
            start=request.start, end=request.end,
            frequency=request.frequency,
            calendar=request.calendar, tz=request.tz,
            frames={series_id: df}, entities=[series_id],
            fields=list(df.columns),
        )
```

**Step 2** — Register in `data/orchestrator.py`:

```python
from .sources.my_source import MySource
def default_registry() -> DataSourceRegistry:
    ...
    reg.register(MySource())
    return reg
```

**Step 3** — Write unit tests in `tests/unit/data/test_my_source.py` using
`MemoryCacheStore` and a mocked client.

---

## 5  SMIM Adapters — How They Differ

SMIM adapters implement a **different protocol** (`smim/interfaces.py`):

```python
class DataAdapter(Protocol):
    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch data. If as_of is given, return vintage as of that date."""
        ...

    @property
    def source_name(self) -> str: ...
```

Key differences from btest `DataSource`:

| Dimension | btest `DataSource` | SMIM `DataAdapter` |
|---|---|---|
| Return type | `DataBundle` (typed) | `pd.DataFrame` |
| Caching | `TailCachedFrameLoader` + ArcticDB | PIT store (Parquet) |
| URI scheme | `scheme://id` dispatched by registry | Direct instantiation |
| Point-in-time | Not a first-class concern | `as_of` parameter enforces A1 |
| Integration | Engine pulls via `DataSourceRegistry` | SMIM pipeline pulls directly |

SMIM adapters **may reuse** `SafeArcticCacheStore` for raw-frame caching,
but primary persistence is the `PointInTimeStore` (Parquet, dual-timestamp).

### SMIM adapter skeleton

```python
# src/quantdsl_backtest/smim/data/adapters/my_smim_adapter.py
from __future__ import annotations

import pandas as pd

from quantdsl_backtest.smim.config import SmimConfig
from quantdsl_backtest.smim.interfaces import DateRange


class MySmimAdapter:
    """Adapter for <source> — implements smim.interfaces.DataAdapter."""

    def __init__(self, config: SmimConfig) -> None:
        self._cfg = config.data.my_source  # relevant config section

    @property
    def source_name(self) -> str:
        return "my_source"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame indexed by event_date, columns = series_ids.

        If as_of is provided, return data as it was known on that date
        (point-in-time discipline, Assumption A1).
        """
        # 1. Build HTTP request(s) — use httpx or urllib, never requests-library
        # 2. Apply as_of filtering (e.g., filter filed_date <= as_of)
        # 3. Return tidy DataFrame:
        #      Index: pd.DatetimeIndex (event_date, tz-naive)
        #      Columns: series_ids
        #      Values: float64 (NaN for missing)
        raise NotImplementedError
```

### Output DataFrame contract

All SMIM `DataAdapter.fetch()` implementations return:

| Property | Requirement |
|---|---|
| Index | `pd.DatetimeIndex`, tz-naive, name `"event_date"` |
| Columns | One column per requested `series_id` (or subset if unavailable) |
| Dtype | `float64` throughout |
| Missing | `NaN` (never drop rows) |
| Frequency | Aligned to `date_range.frequency` before returning |

### Registering SMIM adapters

SMIM adapters are not registered in `DataSourceRegistry`. They are
instantiated directly by the SMIM pipeline:

```python
from quantdsl_backtest.smim.data.adapters.fred_vintage import FredVintageAdapter
adapter = FredVintageAdapter(config)
df = adapter.fetch(["FEDFUNDS", "DCOILBRENTEU"], date_range, as_of=t)
```

### Reusing btest ArcticDB cache

If you want ArcticDB caching in a SMIM adapter, import and use
`SafeArcticCacheStore` directly — but note that the PIT store is the
primary caching layer and this would create a redundant second cache:

```python
from quantdsl_backtest.data.sources.cache import SafeArcticCacheStore

cache = SafeArcticCacheStore(provider="SMIM_FRED", frequency="1Q")
if cache.has(key):
    return cache.read_df(key)
df = _fetch_live(...)
cache.write_df(key, df)
return df
```
