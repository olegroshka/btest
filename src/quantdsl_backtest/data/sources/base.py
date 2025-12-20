from __future__ import annotations

from typing import Optional, Protocol

from ...dsl.universe import Universe
from ..requests import DataRequest
from ..bundles import DataBundle


class CacheStore(Protocol):
    """Minimal cache interface for raw (provider-native) frames."""

    def has(self, key: str) -> bool: ...

    def read_df(self, key: str): ...

    def write_df(self, key: str, df) -> None: ...


class DataSource(Protocol):
    """A pluggable data provider that returns a bundle subtype."""

    name: str

    def can_load(self, request: DataRequest) -> bool: ...

    def load(self, request: DataRequest, universe: Optional[Universe], cache: Optional[CacheStore]) -> DataBundle: ...

