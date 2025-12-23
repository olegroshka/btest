from __future__ import annotations

from typing import Any, Optional

try:
    from pydantic import BaseModel, Field
    from pydantic import field_validator
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


_ALLOWED_KINDS = {"market_bars", "time_series"}


def _normalize_kind(kind: str) -> str:
    return (kind or "").strip().lower()


def _normalize_frequency(freq: str) -> str:
    return (freq or "").strip().lower()


def _validate_frequency(freq: str) -> str:
    # Keep this tight but extendable.
    # Accept common aliases; reject anything unknown for now.
    f = _normalize_frequency(freq)
    aliases = {
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "h": "1h",
        "hour": "1h",
        "m": "1m",
        "min": "1m",
    }
    f = aliases.get(f, f)

    allowed = {"1d", "1h", "1m"}
    if f not in allowed:
        raise ValueError(f"Unsupported frequency {freq!r}. Allowed: {sorted(allowed)}")
    return f


class CatalogDownloadRequest(BaseModel):
    source: str
    kind: str = "market_bars"
    start: Optional[str] = ""
    end: Optional[str] = ""
    frequency: str = "1d"

    dataset_id: Optional[str] = None
    calendar: Optional[str] = None
    tz: Optional[str] = None

    # explicit entity list (preferred over comma-separated)
    entities: list[str] = Field(default_factory=list)

    dry_run: bool = False

    @field_validator("kind")
    @classmethod
    def _kind_allowed(cls, v: str) -> str:
        k = _normalize_kind(v)
        if k not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported kind {v!r}. Allowed: {sorted(_ALLOWED_KINDS)}")
        return k

    @field_validator("frequency")
    @classmethod
    def _frequency_allowed(cls, v: str) -> str:
        return _validate_frequency(v)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _none_to_empty_str(cls, v):
        return "" if v is None else v


class CatalogDownloadResponse(BaseModel):
    dry_run: bool

    # dry-run response
    request: Optional[dict[str, Any]] = None
    plan: Optional[list[dict[str, Any]]] = None

    # execute response
    kind: Optional[str] = None
    source: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    frequency: Optional[str] = None

    entities: Optional[list[str]] = None

    cache_stats: Optional[dict[str, int]] = None
    stats_by_entity: Optional[dict[str, dict[str, int]]] = None
    actions_by_entity: Optional[dict[str, str]] = None


class CatalogPlanRequest(BaseModel):
    source: str
    kind: str = "market_bars"
    start: str = ""
    end: str = ""
    frequency: str = "1d"
    dataset_id: Optional[str] = None
    entities: list[str] = Field(default_factory=list)

    @field_validator("kind")
    @classmethod
    def _kind_allowed(cls, v: str) -> str:
        k = _normalize_kind(v)
        if k not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported kind {v!r}. Allowed: {sorted(_ALLOWED_KINDS)}")
        return k

    @field_validator("frequency")
    @classmethod
    def _frequency_allowed(cls, v: str) -> str:
        return _validate_frequency(v)


class CatalogPlanResponse(BaseModel):
    request: dict[str, Any]
    entities: list[str]
    plan: Optional[list[dict[str, Any]]] = None


class CatalogMetaResponse(BaseModel):
    rows: list[dict[str, Any]]
    count: int

