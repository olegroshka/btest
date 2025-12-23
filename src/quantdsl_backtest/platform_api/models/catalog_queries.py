from __future__ import annotations

from typing import Any, Optional


try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class CatalogMetaQuery(BaseModel):
    provider: Optional[str] = None
    frequency: Optional[str] = None
    dataset: Optional[str] = None
    kind: Optional[str] = None
    entity: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1, le=10000)


class CatalogRefreshResponse(BaseModel):
    status: str
    stats: Optional[dict[str, int]] = None


class CatalogSymbolMetaResponse(BaseModel):
    symbol: str
    meta: Optional[dict[str, Any]] = None


class CatalogPreviewResponse(BaseModel):
    library: Optional[str] = None
    symbol: str
    columns: list[str] = Field(default_factory=list)
    rows: int = 0
    index_start: Optional[str] = None
    index_end: Optional[str] = None
    head: list[dict[str, Any]] = Field(default_factory=list)
    tail: list[dict[str, Any]] = Field(default_factory=list)

