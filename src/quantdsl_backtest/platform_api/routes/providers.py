from __future__ import annotations

from quantdsl_backtest.platform_api.models.misc import ProvidersResponse

from ...data.orchestrator import default_registry


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["providers"])


router = _router()


@router.get("/providers", response_model=ProvidersResponse)
def list_providers() -> dict:
    """List available data sources registered in the local runtime."""
    reg = default_registry()

    providers = []
    for src in reg.providers:
        providers.append(
            {
                "name": getattr(src, "name", src.__class__.__name__),
                "class": src.__class__.__name__,
            }
        )

    return {"providers": providers}
