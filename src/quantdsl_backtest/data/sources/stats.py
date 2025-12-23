from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


class HasTailCacheStats(Protocol):
    def tail_cache_stats(self) -> Dict[str, int]: ...


@dataclass(slots=True)
class CacheStatsMixin:
    """Opt-in mixin to expose cache stats for platform/UI tooling."""

    cache_loader: object

    def tail_cache_stats(self) -> Dict[str, int]:
        stats = getattr(self.cache_loader, "stats", None)
        if stats is None:
            return {}

        out = {}
        for k in ["reads", "writes", "hits", "misses", "tail_fetches"]:
            v = getattr(stats, k, None)
            if isinstance(v, int):
                out[k] = v
        return out

