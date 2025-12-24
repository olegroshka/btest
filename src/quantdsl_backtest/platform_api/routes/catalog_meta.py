from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.models.catalog import CatalogMetaResponse
from quantdsl_backtest.platform_api.models.catalog_queries import CatalogRefreshResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.get("/catalog/meta", response_model=CatalogMetaResponse)
def get_catalog_meta(
    provider: str | None = None,
    frequency: str | None = None,
    dataset: str | None = None,
    kind: str | None = None,
    entity: str | None = None,
    library: str | None = None,
    symbol: str | None = None,
    limit: int | None = None,
) -> dict:
    """Return the raw catalog metadata index (catalog_index)."""

    from fastapi import HTTPException

    if limit is not None and (limit < 1 or limit > 10000):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 10000")

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_meta import (
            get_meta_library,
            read_catalog_index,
        )
        from quantdsl_backtest.platform_api.services.catalog_meta_query import filter_meta_df

        arctic = default_arctic_client()
        meta_lib = get_meta_library(arctic=arctic)
        df = read_catalog_index(meta_lib=meta_lib)

        # Back-compat: older local caches can have library=null in the meta index.
        # Derive it from provider/frequency so UI queries using (library, symbol) work.
        try:
            if "library" in df.columns and "provider" in df.columns and "frequency" in df.columns:
                missing = df["library"].isna() | (df["library"].astype(str).str.strip() == "")
                if missing.any():
                    df.loc[missing, "library"] = (
                        "market_data/" + df.loc[missing, "provider"].astype(str).str.upper() + "/" + df.loc[missing, "frequency"].astype(str).str.lower()
                    )
        except Exception:
            pass

        df = filter_meta_df(
            df,
            provider=provider,
            frequency=frequency,
            dataset=dataset,
            kind=kind,
            entity=entity,
            library=library,
            symbol=symbol,
            limit=limit,
        )

        # JSON friendly (replace NaN/NaT with None)
        records = df.mask(pd.isna(df), None).to_dict(orient="records")
        return {"rows": records, "count": int(len(records))}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/catalog/refresh", response_model=CatalogRefreshResponse)
def refresh_catalog_meta() -> dict:
    """Rebuild catalog metadata by scanning cache libraries and updating meta index."""

    from fastapi import HTTPException

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_meta_refresh import (
            refresh_catalog_meta_from_cache,
        )

        arctic = default_arctic_client()
        stats = refresh_catalog_meta_from_cache(arctic=arctic)
        return {"status": "ok", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
