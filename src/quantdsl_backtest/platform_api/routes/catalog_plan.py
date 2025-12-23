from __future__ import annotations

from quantdsl_backtest.platform_api.models.catalog import CatalogPlanRequest, CatalogPlanResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.post("/catalog/plan_download")
def plan_download(
    source: str = "",
    kind: str = "market_bars",
    start: str = "",
    end: str = "",
    frequency: str = "1d",
    dataset_id: str | None = None,
    entities: str | None = None,
    body: CatalogPlanRequest | None = None,
) -> dict:
    """Compute a download plan (no side-effects).

    Single supported endpoint:
      - Preferred: JSON body (validated; produces 422 on invalid frequency, etc.)
      - Back-compat: query params
    """

    from fastapi import HTTPException

    from quantdsl_backtest.data.requests import DataRequest
    from quantdsl_backtest.platform_api.services.catalog_download_plan import plan_download_for_request
    from quantdsl_backtest.platform_api.services.catalog_request_validation import normalize_and_validate_request

    if body is not None:
        source = body.source
        kind = body.kind
        start = body.start
        end = body.end
        frequency = body.frequency
        dataset_id = body.dataset_id
        entity_list = list(body.entities or [])
    else:
        entity_list: list[str] = []
        if entities:
            entity_list = [e.strip() for e in entities.split(",") if e.strip()]

    req = DataRequest(
        source=source,
        kind=kind,
        start=start,
        end=end,
        frequency=frequency,
        fields=[],
        dataset_id=dataset_id,
    )

    try:
        req = normalize_and_validate_request(req)
        plan = plan_download_for_request(request=req, entities=entity_list)
        return {
            "request": {
                "source": source,
                "kind": req.kind,
                "start": start,
                "end": end,
                "frequency": req.frequency,
                "dataset_id": dataset_id,
            },
            "entities": entity_list,
            "plan": plan,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
