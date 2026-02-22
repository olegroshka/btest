from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import FileResponse, Response

from ..services.run_paths import default_runs_root_dir


def _router() -> APIRouter:
    return APIRouter(tags=["reports"])


router = _router()


def _safe_file_response(*, run_id: str, path: str) -> Response:
    """Return a FileResponse for a report asset, preventing path traversal."""

    base = (default_runs_root_dir() / str(run_id)).resolve()
    file_path = (base / path).resolve()

    if base not in file_path.parents and file_path != base:
        return Response(status_code=404)

    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)

    return FileResponse(str(file_path))


@router.get("/reports/runs/{run_id}")
@router.get("/reports/runs/{run_id}/")
def run_reports_root(run_id: str, request: Request):
    """Serve the report index for a run.

    This makes the route more robust for users and prevents broken tabs when links omit
    the file name.
    """

    # Prefer index.html; fall back to index.htm if present.
    r = _safe_file_response(run_id=run_id, path="index.html")
    if r.status_code != 404:
        return r
    return _safe_file_response(run_id=run_id, path="index.htm")


@router.get("/reports/runs/{run_id}/{path:path}")
def run_reports(run_id: str, path: str, request: Request):
    """Serve run-scoped report artifacts.

    Security contract:
      - Prevent path traversal: requested file must stay inside the run dir.

    UX contract:
      - If a legacy link points to index.htm, redirect to index.html when possible.
    """

    # Normalize a few common legacy/broken paths.
    norm = str(path or "")
    if norm in {"", "/"}:
        return run_reports_root(run_id=run_id, request=request)

    # Some users/browsers will try /index.htm. We don't generate that; redirect if possible.
    if norm.lower() == "index.htm":
        r = _safe_file_response(run_id=run_id, path="index.html")
        if r.status_code != 404:
            return r

    return _safe_file_response(run_id=run_id, path=norm)
