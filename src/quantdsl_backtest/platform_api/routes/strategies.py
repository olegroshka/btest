from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException

from ..errors import to_api_error
from ..services import strategy_discovery as sd


def _router() -> APIRouter:
    return APIRouter(tags=["strategies"])


router = _router()


def _rid(request: Request) -> str | None:
    return getattr(getattr(request, "state", None), "request_id", None)


@router.get("/strategies")
def list_strategies(request: Request) -> dict[str, Any]:
    """List strategies from the local filesystem.

    Note: this is local-first and reads from the gitignored `strategies/` dir.
    If empty on first server start, we bootstrap it from committed examples.
    """

    try:
        strategies_dir = sd.default_strategies_dir()
        # Best-effort bootstrap to make first-run UX smooth.
        sd.bootstrap_strategies(target_dir=strategies_dir)
        infos = sd.discover_strategies(strategies_dir=strategies_dir)
        return {"strategies": [i.model_dump() for i in infos]}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="STRATEGIES_UNAVAILABLE",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )


@router.get("/strategies/{strategy_id}")
def get_strategy(strategy_id: str, request: Request) -> dict[str, Any]:
    """Return strategy detail including full source code."""

    try:
        strategies_dir = sd.default_strategies_dir()
        source = sd.read_strategy_source(strategies_dir=strategies_dir, strategy_id=strategy_id)
        # Build StrategyInfo from discovery results.
        infos = sd.discover_strategies(strategies_dir=strategies_dir)
        info = next((i for i in infos if i.id == strategy_id), None)
        if info is None:
            raise FileNotFoundError(f"Strategy not found: {strategy_id}")

        d = info.model_dump()
        d["source"] = source
        return {"strategy": d}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=to_api_error(
                code="STRATEGY_NOT_FOUND",
                message=f"Strategy not found: {strategy_id}",
                status=404,
                request_id=_rid(request),
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="STRATEGIES_UNAVAILABLE",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )


@router.post("/strategies")
def create_strategy(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Create a new strategy file in the local strategies directory."""

    try:
        strategy_id = body.get("id")
        source = body.get("source")

        if strategy_id is None or not isinstance(strategy_id, str) or not strategy_id.strip():
            raise HTTPException(status_code=422, detail="id is required")
        if not isinstance(source, str) or not source.strip():
            raise HTTPException(status_code=422, detail="source is required")

        strategies_dir = sd.default_strategies_dir()
        p = strategies_dir / f"{sd.slugify(strategy_id)}.py"
        if p.exists():
            raise HTTPException(status_code=409, detail="strategy already exists")

        out = sd.write_strategy_source(strategies_dir=strategies_dir, strategy_id=strategy_id, source=source)
        return {"id": sd.slugify(strategy_id), "path": str(out)}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="STRATEGY_SAVE_FAILED",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )


@router.put("/strategies/{strategy_id}")
def update_strategy(strategy_id: str, request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Overwrite an existing strategy file."""

    try:
        source = body.get("source")
        if not isinstance(source, str) or not source.strip():
            raise HTTPException(status_code=422, detail="source is required")

        strategies_dir = sd.default_strategies_dir()
        p = strategies_dir / f"{strategy_id}.py"
        if not p.exists():
            raise HTTPException(status_code=404, detail="strategy not found")

        out = sd.write_strategy_source(strategies_dir=strategies_dir, strategy_id=strategy_id, source=source)
        # Return new hash so UI can show it / use for runs.
        h = sd.sha256_file(out)
        return {"id": strategy_id, "path": str(out), "strategy_hash": h}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=to_api_error(
                code="STRATEGY_SAVE_FAILED",
                message=str(exc),
                status=503,
                request_id=_rid(request),
            ),
        )
