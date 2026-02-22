from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from starlette.responses import FileResponse, Response

from ..services.run_paths import default_runs_root_dir


def _router() -> APIRouter:
    return APIRouter(tags=["reports"])


router = _router()


@router.get("/reports/runs/{run_id}/{path:path}")
def run_reports(run_id: str, path: str, request: Request):
    """Serve run-scoped report artifacts.

    Security contract:
      - Prevent path traversal: requested file must stay inside the run dir.
    """

    base = (default_runs_root_dir() / str(run_id)).resolve()
    file_path = (base / path).resolve()

    if base not in file_path.parents and file_path != base:
        return Response(status_code=404)

    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)

    return FileResponse(str(file_path))

