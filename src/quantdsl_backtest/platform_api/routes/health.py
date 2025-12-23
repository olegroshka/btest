from __future__ import annotations

from quantdsl_backtest.platform_api.models.misc import HealthResponse


def _router():
    try:
        from fastapi import APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return APIRouter(tags=["health"])


router = _router()


@router.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}
