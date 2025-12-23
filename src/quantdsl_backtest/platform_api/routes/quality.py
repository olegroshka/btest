from __future__ import annotations

from fastapi import HTTPException


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["quality"])


router = _router()


@router.post("/quality/scan")
def quality_scan(
    provider: str | None = None,
    frequency: str | None = None,
    dataset: str | None = None,
    kind: str | None = None,
    entity: str | None = None,
    limit: int = 200,
) -> dict:
    """Run an in-process quality scan and persist issues.

    Query params mirror /api/catalog/meta filters.
    """

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.quality_registry import run_quality_scan

        arctic = default_arctic_client()
        return run_quality_scan(
            arctic=arctic,
            provider=provider,
            frequency=frequency,
            dataset=dataset,
            kind=kind,
            entity=entity,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/quality/issues")
def list_issues(
    provider: str | None = None,
    frequency: str | None = None,
    dataset: str | None = None,
    kind: str | None = None,
    entity: str | None = None,
    issue_type: str | None = None,
    severity: str | None = None,
    limit: int = 500,
) -> dict:
    """List recorded quality issues (filterable)."""

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.quality_registry import list_quality_issues

        arctic = default_arctic_client()
        df = list_quality_issues(
            arctic=arctic,
            provider=provider,
            frequency=frequency,
            dataset=dataset,
            kind=kind,
            entity=entity,
            issue_type=issue_type,
            severity=severity,
        )
        rows = df.head(int(limit)).where(df.notna(), None).to_dict(orient="records") if not df.empty else []
        return {"count": int(len(df)), "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/quality/issues/{issue_id}")
def get_issue(issue_id: str) -> dict:
    """Get details for a single quality issue."""

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.quality_registry import get_quality_issue

        arctic = default_arctic_client()
        row = get_quality_issue(arctic=arctic, issue_id=issue_id)
        return {"issue": row}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

