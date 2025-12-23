from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.models.catalog_listing import CatalogListResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.get("/catalog", response_model=CatalogListResponse)
def list_cached_datasets() -> dict:
    """List cached dataset symbols currently stored in ArcticDB.

    Best-effort endpoint:
    - If ArcticDB isn't installed or the LMDB store is corrupted, returns an empty list.
    - If metadata library is missing/unreadable, metadata is omitted.
    """

    from fastapi import HTTPException

    try:
        from ..services.catalog import default_arctic_client, list_arctic_cache_catalog
        from ..services.catalog_meta import get_meta_library, read_catalog_index

        arctic = default_arctic_client()
        libs = list_arctic_cache_catalog(arctic=arctic)

        # Metadata is optional.
        meta_rows = None
        try:
            meta_lib = get_meta_library(arctic=arctic)
            meta_rows = read_catalog_index(meta_lib=meta_lib)
        except Exception:
            meta_rows = None

        meta_by_symbol = {}
        if meta_rows is not None and not meta_rows.empty and "symbol" in meta_rows.columns:
            # Last write wins if duplicates
            for _, r in meta_rows.iterrows():
                sym = r.get("symbol")
                if not isinstance(sym, str) or not sym:
                    continue
                meta_by_symbol[sym] = {
                    "provider": r.get("provider"),
                    "frequency": r.get("frequency"),
                    "kind": r.get("kind"),
                    "dataset": r.get("dataset"),
                    "entity": r.get("entity"),
                    "start": None if pd.isna(r.get("start")) else str(r.get("start")),
                    "end": None if pd.isna(r.get("end")) else str(r.get("end")),
                    "updated_at": None if pd.isna(r.get("updated_at")) else str(r.get("updated_at")),
                    "rows": None if pd.isna(r.get("rows")) else int(r.get("rows")),
                    "cols": None if pd.isna(r.get("cols")) else int(r.get("cols")),
                }

        return {
            "libraries": [
                {
                    "library": l.library,
                    "symbols": [
                        {"symbol": s, "meta": meta_by_symbol.get(s)} for s in l.symbols
                    ],
                    "count": l.count,
                }
                for l in libs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
