from __future__ import annotations

from typing import List

from ...data.requests import DataRequest


def _provider_from_source(source: str) -> str:
    src = (source or "").lower()
    if src.startswith("fred://"):
        return "FRED"
    if src.startswith("yf://"):
        return "YF"
    if src.startswith("parquet://"):
        return "PARQUET"
    return "GLOBAL"


def plan_download_for_request(*, request: DataRequest, entities: List[str]) -> List[dict] | None:
    """Compute a best-effort download plan from metadata index.

    Returns list of plan rows or None if planning unavailable.
    """

    if not entities:
        return []

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_meta import get_meta_library, read_catalog_index
        from quantdsl_backtest.platform_api.services.catalog_coverage import plan_download
        from quantdsl_backtest.data.sources.cache import dataset_partition

        arctic = default_arctic_client()
        meta_lib = get_meta_library(arctic=arctic)
        meta_df = read_catalog_index(meta_lib=meta_lib)

        provider = _provider_from_source(request.source)
        dataset = str(dataset_partition(request))

        return plan_download(
            request_start=request.start,
            request_end=request.end,
            entities=list(entities),
            provider=provider,
            frequency=request.frequency,
            kind=request.kind,
            dataset=dataset,
            meta_df=meta_df,
        )
    except Exception:
        return None
