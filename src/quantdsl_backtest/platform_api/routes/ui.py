from __future__ import annotations

import pathlib

from starlette.responses import HTMLResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["ui"])


router = _router()


@router.get("/", response_class=HTMLResponse)
def ui_index():
    from quantdsl_backtest.platform_ui.site import html_index

    return HTMLResponse(html_index())


@router.get("/static/{path:path}")
def ui_static(path: str):
    """Serve Platform UI static assets (local-first)."""

    from starlette.responses import FileResponse, Response

    # Special case for plotly.min.js - serve from plotly package if available.
    # This avoids committing a ~5MB binary to the repo.
    if path == "plotly.min.js":
        try:
            import importlib.resources
            # In Python 3.9+, .files() is the modern way to access package data.
            plotly_js = importlib.resources.files("plotly").joinpath("package_data/plotly.min.js")
            if plotly_js.exists():
                return FileResponse(str(plotly_js))
        except (ImportError, AttributeError, TypeError):
            pass

    base = pathlib.Path(__file__).resolve().parents[2] / "platform_ui" / "assets"
    file_path = (base / path).resolve()

    # Prevent path traversal
    if base not in file_path.parents and file_path != base:
        return Response(status_code=404)

    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)

    return FileResponse(str(file_path))
