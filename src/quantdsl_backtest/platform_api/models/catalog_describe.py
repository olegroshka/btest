from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
    ) from e


class CatalogDescribeResponse(BaseModel):
    library: Optional[str] = None
    symbol: str

    rows: int
    columns: List[str]

    index_start: Optional[str] = None
    index_end: Optional[str] = None

    dtypes: Dict[str, str]
    missing: Dict[str, int]
    non_null_pct: Dict[str, float]
    unique: Dict[str, int]
    numeric: Dict[str, Dict[str, Any]]
    gaps: Dict[str, Any]
