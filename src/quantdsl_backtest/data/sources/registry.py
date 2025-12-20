from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ...dsl.universe import Universe
from ..requests import DataRequest
from ..bundles import DataBundle
from .base import DataSource, CacheStore


@dataclass(slots=True)
class DataSourceRegistry:
    providers: List[DataSource] = field(default_factory=list)

    def register(self, provider: DataSource) -> None:
        self.providers.append(provider)

    def resolve(self, request: DataRequest) -> DataSource:
        for p in self.providers:
            if p.can_load(request):
                return p
        supported = ", ".join(getattr(p, "name", p.__class__.__name__) for p in self.providers)
        raise ValueError(
            f"No data source provider can load request source={request.source!r} kind={request.kind!r}. "
            f"Registered providers: [{supported}]"
        )

    def load(self, request: DataRequest, universe: Optional[Universe], cache: Optional[CacheStore]) -> DataBundle:
        provider = self.resolve(request)
        return provider.load(request, universe, cache)

