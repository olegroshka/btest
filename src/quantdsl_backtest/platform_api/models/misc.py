from __future__ import annotations


try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class HealthResponse(BaseModel):
    status: str


class ProvidersResponse(BaseModel):
    providers: list[dict[str, str]]

