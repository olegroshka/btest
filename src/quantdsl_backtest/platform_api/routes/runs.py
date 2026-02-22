from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.exceptions import HTTPException

from ..errors import to_api_error
from ..models.run import RunStatus, RunSubmitRequest, RunSubmitResponse
from ..services.run_store import RunStore


def _router() -> APIRouter:
    return APIRouter(tags=["runs"])


router = _router()


def _rid(request: Request) -> str | None:
    return getattr(getattr(request, "state", None), "request_id", None)


def _get_store(request: Request) -> RunStore:
    store = getattr(getattr(request, "app", None), "state", None)
    store = getattr(store, "run_store", None)
    if store is None:
        raise RuntimeError("RunStore is not configured")
    return store


def _get_task_runner(request: Request):
    runner = getattr(getattr(request, "app", None), "state", None)
    runner = getattr(runner, "task_runner", None)
    if runner is None:
        raise RuntimeError("TaskRunner is not configured")
    return runner


@router.get("/runs")
def list_runs(
    request: Request,
    strategy_id: str | None = Query(default=None),
    status: RunStatus | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List runs from the run index (SQLite)."""

    try:
        store = _get_store(request)
        res = store.list_runs(strategy_id=strategy_id, status=status, limit=limit, offset=offset)
        return {"runs": [r.model_dump() for r in res.runs], "total": res.total}
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


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    """Get a single run by id."""

    try:
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
        return {"run": run.model_dump()}
    except HTTPException:
        raise
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


@router.post("/runs")
def submit_run(request: Request, body: RunSubmitRequest) -> RunSubmitResponse:
    """Submit a run.

    Preferred contract:
      - strategy_id
      - params

    Backward compatibility (temporary):
      - source / strategy_hash override (must provide BOTH)
    """

    try:
        from ..services import strategy_discovery as sd

        strategy_id = str(body.strategy_id).strip()
        if not strategy_id:
            raise HTTPException(status_code=422, detail="strategy_id is required")

        # Strict override: either provide both or neither.
        has_source = bool(body.source and str(body.source).strip())
        has_hash = bool(body.strategy_hash and str(body.strategy_hash).strip())
        if has_source != has_hash:
            raise HTTPException(status_code=422, detail="source and strategy_hash must be provided together")

        if has_source and has_hash:
            source_snapshot = str(body.source)
            hash_snapshot = str(body.strategy_hash)
        else:
            strategies_dir = sd.default_strategies_dir()
            source_snapshot = sd.read_strategy_source(strategies_dir=strategies_dir, strategy_id=strategy_id)
            hash_snapshot = sd.sha256_text(source_snapshot)

        runner = _get_task_runner(request)
        res = runner.submit(
            strategy_id=strategy_id,
            strategy_hash=hash_snapshot,
            source_snapshot=source_snapshot,
            params=body.params,
        )
        return RunSubmitResponse(run_id=res.run_id, status=res.status)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=to_api_error(
                code="STRATEGY_NOT_FOUND",
                message=f"Strategy not found: {body.strategy_id}",
                status=404,
                request_id=_rid(request),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="RUN_SUBMIT_FAILED",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )
