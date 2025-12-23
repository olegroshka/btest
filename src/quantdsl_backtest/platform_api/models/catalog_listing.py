from __future__ import annotations

from typing import Any, Optional


try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class CatalogSymbolEntry(BaseModel):
    symbol: str
    meta: Optional[dict[str, Any]] = None


class CatalogLibraryEntry(BaseModel):
    library: str
    symbols: list[CatalogSymbolEntry] = Field(default_factory=list)
    count: int


class CatalogListResponse(BaseModel):
    libraries: list[CatalogLibraryEntry] = Field(default_factory=list)

