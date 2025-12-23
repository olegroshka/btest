from __future__ import annotations

from typing import Any, Optional


try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class ApiError(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error")
    details: Optional[dict[str, Any]] = Field(default=None)

