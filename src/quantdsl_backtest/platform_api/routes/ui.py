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


@router.get("/favicon.ico")
def favicon_ico():
    """Serve a minimal favicon to avoid noisy 404s in browser consoles."""

    from starlette.responses import Response

    # Empty 204 is fine; browsers stop retrying and console stays clean.
    return Response(status_code=204)


@router.get("/apple-touch-icon.png")
def apple_touch_icon():
    """Serve apple-touch-icon to avoid 404 noise."""

    from starlette.responses import Response

    return Response(status_code=204)


@router.get("/robots.txt")
def robots_txt():
    """Serve minimal robots.txt to avoid 404 noise."""

    from starlette.responses import PlainTextResponse

    return PlainTextResponse("User-agent: *\nDisallow:\n")


@router.get("/manifest.json")
def manifest_json():
    """Serve minimal web manifest to avoid 404 noise."""

    from starlette.responses import JSONResponse

    return JSONResponse({"name": "Platform UI", "short_name": "Platform UI"})


@router.get("/", response_class=HTMLResponse)
def ui_index():
    """Serve the Platform UI shell.

    Current UI: serve the committed static index.html from platform_ui/assets_dist.

    There is no legacy inline HTML fallback anymore; the React/Vite bundle is the
    single source of truth.
    """

    base = pathlib.Path(__file__).resolve().parents[2] / "platform_ui" / "assets_dist"
    index_path = (base / "index.html").resolve()

    if not (index_path.exists() and index_path.is_file()):
        # Keep the failure mode explicit so missing UI assets is obvious in dev/CI.
        raise FileNotFoundError(f"Missing Platform UI build: {index_path}")

    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@router.get("/static/{path:path}")
def ui_static(path: str):
    """Serve Platform UI static assets from the committed SPA build."""

    from starlette.responses import FileResponse, Response

    # Special case for plotly.min.js - serve from plotly package if available.
    # This avoids committing a ~5MB binary to the repo.
    if path == "plotly.min.js":
        try:
            import importlib.resources

            plotly_js = importlib.resources.files("plotly").joinpath("package_data/plotly.min.js")
            try:
                opened = plotly_js.open("rb")  # type: ignore[attr-defined]
            except Exception:
                opened = None
            if opened is not None:
                opened.close()
                return FileResponse(str(plotly_js))
        except (ImportError, AttributeError, TypeError):
            pass

    base_dist = pathlib.Path(__file__).resolve().parents[2] / "platform_ui" / "assets_dist"
    file_path = (base_dist / path).resolve()

    # Prevent path traversal
    if base_dist not in file_path.parents and file_path != base_dist:
        return Response(status_code=404)

    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)

    return FileResponse(str(file_path))


@router.get("/apple-touch-icon-precomposed.png")
def apple_touch_icon_precomposed():
    from starlette.responses import Response

    return Response(status_code=204)


@router.get("/favicon-16x16.png")
def favicon_16():
    from starlette.responses import Response

    return Response(status_code=204)


@router.get("/favicon-32x32.png")
def favicon_32():
    from starlette.responses import Response

    return Response(status_code=204)


@router.get("/site.webmanifest")
def site_webmanifest():
    from starlette.responses import JSONResponse

    return JSONResponse({"name": "Platform UI", "short_name": "Platform UI"})


@router.get("/browserconfig.xml")
def browserconfig_xml():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/sitemap.xml")
def sitemap_xml():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/humans.txt")
def humans_txt():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/favicon.svg")
def favicon_svg():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/favicon.png")
def favicon_png():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/static/favicon.ico")
def static_favicon_ico():
    from starlette.responses import Response
    return Response(status_code=204)


@router.get("/static/manifest.json")
def static_manifest_json():
    from starlette.responses import JSONResponse
    return JSONResponse({"name": "Platform UI", "short_name": "Platform UI"})


@router.get("/static/site.webmanifest")
def static_site_webmanifest():
    from starlette.responses import JSONResponse
    return JSONResponse({"name": "Platform UI", "short_name": "Platform UI"})


@router.get("/static/apple-touch-icon.png")
def static_apple_touch_icon():
    from starlette.responses import Response
    return Response(status_code=204)

