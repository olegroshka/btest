from __future__ import annotations

from typing import Optional, Callable

from ..dsl.universe import Universe
from .bundles import DataBundle
from .requests import DataRequest
from .sources.cache import SafeArcticCacheStore
from .sources.registry import DataSourceRegistry
from .sources.parquet import ParquetMarketBarsSource
from .sources.csv_source import CsvMarketBarsSource
from .sources.yahoo import YahooMarketBarsSource
from .sources.fred import FredMarketBarsSource, FredTimeSeriesSource
from .sources.sfera import SferaSource


def default_registry() -> DataSourceRegistry:
    reg = DataSourceRegistry()
    reg.register(ParquetMarketBarsSource())
    reg.register(CsvMarketBarsSource())
    reg.register(YahooMarketBarsSource())
    reg.register(FredMarketBarsSource())
    reg.register(FredTimeSeriesSource())
    # Sfera PostgreSQL — single adapter handles sfera:// for both OHLCV and
    # time-series data; return type is driven by DataConfig kind="market_bars"
    # or kind="timeseries".  sfera-bars:// still accepted as a backward-compat alias.
    reg.register(SferaSource())
    return reg


def default_cache_for_request(request: DataRequest):
    """Default cache selection.

    Golden-store behavior:
    - We *always* try to use ArcticDB as the write-through cache, including parquet sources,
      so that running any strategy ingests its requested raw data into the cache catalog.
    - We wrap with SafeArcticCacheStore so a corrupted/missing LMDB store degrades
      gracefully (strategy still runs), while the platform API can show a clear 503.
    """

    provider = "GLOBAL"
    if request.source.lower().startswith("fred://"):
        provider = "FRED"
    if request.source.lower().startswith("yf://"):
        provider = "YF"
    if request.source.lower().startswith("parquet://"):
        provider = "PARQUET"
    if request.source.lower().startswith("csv://"):
        provider = "CSV"
    if request.source.lower().startswith("sfera://") or request.source.lower().startswith("sfera-bars://"):
        provider = "SFERA"

    return SafeArcticCacheStore(provider=provider, frequency=request.frequency)


def load_bundle(
    request: DataRequest,
    universe: Optional[Universe],
    *,
    registry: Optional[DataSourceRegistry] = None,
    cache_factory: Optional[Callable[[DataRequest], object]] = None,
) -> DataBundle:
    """Load a typed DataBundle using the provider registry.

    Parameters are injectable so tests/platform can supply:
      - a custom registry (e.g., plugin providers)
      - a custom cache implementation
    """

    reg = registry or default_registry()
    cache = (cache_factory or default_cache_for_request)(request)
    return reg.load(request, universe=universe, cache=cache)
