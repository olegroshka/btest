from __future__ import annotations

from quantdsl_backtest.platform_api.models.catalog import CatalogDownloadRequest, CatalogDownloadResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.post("/catalog/download", response_model=CatalogDownloadResponse)
def download_data(
    source: str = "",
    kind: str = "market_bars",
    start: str = "",
    end: str = "",
    frequency: str = "1d",
    dataset_id: str | None = None,
    calendar: str | None = None,
    tz: str | None = None,
    entities: str | None = None,
    dry_run: bool = False,
    body: CatalogDownloadRequest | None = None,
) -> CatalogDownloadResponse:
    """Download/fill missing data for a request.

    Single supported endpoint:
      - Preferred: JSON body (validated)
      - Back-compat: query params

    This returns a stable response model (CatalogDownloadResponse).
    """

    from fastapi import HTTPException

    from quantdsl_backtest.data.requests import DataRequest
    from quantdsl_backtest.platform_api.services.catalog_download import download_bundle
    from quantdsl_backtest.platform_api.services.catalog_request_validation import normalize_and_validate_request

    # Body wins when provided
    if body is not None:
        source = body.source
        kind = body.kind
        start = body.start
        end = body.end
        frequency = body.frequency
        dataset_id = body.dataset_id
        calendar = body.calendar
        tz = body.tz
        dry_run = bool(body.dry_run)
        entity_list = list(body.entities or [])
    else:
        entity_list = None
        if entities:
            entity_list = [e.strip() for e in entities.split(",") if e.strip()]

    req = DataRequest(
        source=source,
        kind=kind,
        start=start,
        end=end,
        frequency=frequency,
        fields=[],
        calendar=calendar,
        tz=tz,
        dataset_id=dataset_id,
    )

    try:
        req = normalize_and_validate_request(req)
        out = download_bundle(request=req, universe=None, entities=entity_list, dry_run=bool(dry_run))
        return CatalogDownloadResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
