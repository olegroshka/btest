from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..errors import to_api_error
from ..services.run_store import RunStore


def _router() -> APIRouter:
    return APIRouter(tags=["run_artifacts"])


router = _router()


def _rid(request: Request) -> str | None:
    return getattr(getattr(request, "state", None), "request_id", None)


def _get_store(request: Request) -> RunStore:
    store = getattr(getattr(request, "app", None), "state", None)
    store = getattr(store, "run_store", None)
    if store is None:
        raise RuntimeError("RunStore is not configured")
    return store


def _load_run_dir(run_id: str, request: Request) -> Path:
    store = _get_store(request)
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=to_api_error(
                code="RUN_NOT_FOUND",
                message=f"Run not found: {run_id}",
                status=404,
                request_id=_rid(request),
            ),
        )
    if not run.artifacts_dir:
        raise HTTPException(
            status_code=404,
            detail=to_api_error(
                code="ARTIFACTS_NOT_FOUND",
                message=f"Artifacts not available for run: {run_id}",
                status=404,
                request_id=_rid(request),
            ),
        )
    return Path(run.artifacts_dir).resolve()


def _safe_file_under(base: Path, rel_path: str) -> Path | None:
    try:
        p = (base / rel_path).resolve()
    except Exception:
        return None
    if base not in p.parents and p != base:
        return None
    return p


@router.get("/runs/{run_id}/artifacts", response_model=None)
def list_run_artifacts(run_id: str, request: Request):
    """List artifact relative paths for a run.

    Prefers `summary.json` if present (stable ordering), otherwise falls back to filesystem scan.
    """

    try:
        run_dir = _load_run_dir(run_id, request)

        # Load from summary.json when available
        summary_p = _safe_file_under(run_dir, "summary.json")
        if summary_p and summary_p.exists() and summary_p.is_file():
            try:
                payload = json.loads(summary_p.read_text(encoding="utf-8"))
                artifacts = payload.get("artifacts")
                if isinstance(artifacts, list) and all(isinstance(x, str) for x in artifacts):
                    return {"artifacts": artifacts}
            except Exception:
                pass

        # Fallback: walk filesystem
        artifacts: list[str] = []
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                artifacts.append(str(p.relative_to(run_dir)).replace("\\", "/"))
        return {"artifacts": artifacts}

    except HTTPException as exc:
        # preserve error shape
        detail = exc.detail if isinstance(exc.detail, dict) else to_api_error(code=f"HTTP_{exc.status_code}", message=str(exc.detail), status=int(exc.status_code), request_id=_rid(request))
        return JSONResponse(status_code=int(exc.status_code), content=detail, headers={"X-Request-Id": _rid(request) or ""})

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="RUNS_UNAVAILABLE",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )


@router.get("/runs/{run_id}/artifact/{path:path}", response_model=None)
def get_run_artifact(run_id: str, path: str, request: Request):
    """Download/serve a single artifact file from a run directory."""

    try:
        run_dir = _load_run_dir(run_id, request)
        file_path = _safe_file_under(run_dir, path)
        if file_path is None or not file_path.exists() or not file_path.is_file():
            return JSONResponse(
                status_code=404,
                content=to_api_error(
                    code="ARTIFACT_NOT_FOUND",
                    message=f"Artifact not found: {path}",
                    status=404,
                    request_id=_rid(request),
                ),
                headers={"X-Request-Id": _rid(request) or ""},
            )
        return FileResponse(str(file_path))
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else to_api_error(code=f"HTTP_{exc.status_code}", message=str(exc.detail), status=int(exc.status_code), request_id=_rid(request))
        return JSONResponse(status_code=int(exc.status_code), content=detail, headers={"X-Request-Id": _rid(request) or ""})
