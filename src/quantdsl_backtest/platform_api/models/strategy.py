from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class StrategyInfo(BaseModel):
    id: str = Field(..., description="Strategy id (file stem)")
    path: str = Field(..., description="Absolute file path")
    strategy_hash: str = Field(..., description="SHA-256 hex digest of current file content")
    name: Optional[str] = Field(default=None, description="Human name (best-effort from module/docstring)")
    description: Optional[str] = Field(default=None, description="Description (best-effort)")


class StrategyDetail(StrategyInfo):
    source: str = Field(..., description="Full Python source")


class StrategySaveRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="Strategy id; derived from filename stem if omitted")
    source: str = Field(..., description="Full Python source")

