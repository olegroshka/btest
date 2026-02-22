from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

try:
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


RunStatus = Literal["pending", "running", "succeeded", "failed"]


class RunSubmitRequest(BaseModel):
    strategy_id: str = Field(..., description="Strategy id (file stem)")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional run params")

    # Temporary back-compat (will be removed once UI always uses strategy filesystem)
    source: Optional[str] = Field(default=None, description="Optional source snapshot override")
    strategy_hash: Optional[str] = Field(default=None, description="Optional hash override")


class RunSubmitResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunRecord(BaseModel):
    """Persisted run record (source of truth for Runs UI)."""

    run_id: str = Field(..., description="Unique id for this run (uuid hex)")
    strategy_id: str = Field(..., description="Stable strategy identifier (file stem)")
    strategy_hash: str = Field(..., description="SHA-256 of strategy source at submission time")

    # Reproducibility
    source_snapshot: Optional[str] = Field(default=None, description="Full Python source text executed")
    params: dict[str, Any] = Field(default_factory=dict, description="Resolved run parameters")

    # Lifecycle
    status: RunStatus = Field(default="pending")
    submitted_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_s: Optional[float] = None

    # Outputs
    metrics: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    artifacts_dir: Optional[str] = None
    reports_url: Optional[str] = None


class StrategySummary(BaseModel):
    """Derived view for a strategy row (last-run snapshot)."""

    strategy_id: str
    last_run_id: Optional[str] = None
    last_status: Optional[RunStatus] = None
    last_run_at: Optional[datetime] = None
    last_metrics: Optional[dict[str, Any]] = None

