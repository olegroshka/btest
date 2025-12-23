from __future__ import annotations

from typing import Optional, Callable

from ..dsl.universe import Universe
from .bundles import DataBundle
from .requests import DataRequest
from .sources.cache import SafeArcticCacheStore
from .sources.registry import DataSourceRegistry
from .sources.parquet import ParquetMarketBarsSource
from .sources.yahoo import YahooMarketBarsSource
from .sources.fred import FredMarketBarsSource, FredTimeSeriesSource


def default_registry() -> DataSourceRegistry:
    reg = DataSourceRegistry()
    reg.register(ParquetMarketBarsSource())
    reg.register(YahooMarketBarsSource())
    reg.register(FredMarketBarsSource())
    reg.register(FredTimeSeriesSource())
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
