from __future__ import annotations

from fastapi import HTTPException
from quantdsl_backtest.platform_api.models.catalog_queries import CatalogPreviewResponse, CatalogSymbolMetaResponse
from quantdsl_backtest.platform_api.models.catalog_describe import CatalogDescribeResponse


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.get("/catalog/meta/{symbol:path}", response_model=CatalogSymbolMetaResponse)
def get_symbol_meta(symbol: str) -> dict:
    """Return metadata row for a single cache symbol."""

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_symbol import get_symbol_meta as _get

        arctic = default_arctic_client()
        rec = _get(arctic=arctic, symbol=symbol)
        if rec is None:
            return {"symbol": symbol, "meta": None}
        return {"symbol": symbol, "meta": rec.row}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/catalog/preview/{library:path}", response_model=CatalogPreviewResponse)
def preview_cached_symbol_v2(
    library: str,
    symbol: str,
    head: int = 5,
    tail: int = 5,
) -> dict:
    """Preview cached raw frame by library + symbol (symbol passed as query param).

    This avoids ambiguous routing when `symbol` itself contains slashes (our cache keys
    are `<kind>/<dataset>/<entity>`).

    Example:
      GET /api/catalog/preview/market_data/PARQUET/1d?symbol=market_bars/equities_indicies.parquet/SPY&head=3&tail=2
    """

    if head < 0 or head > 1000:
        raise HTTPException(status_code=400, detail="head must be between 0 and 1000")
    if tail < 0 or tail > 1000:
        raise HTTPException(status_code=400, detail="tail must be between 0 and 1000")

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_symbol import preview_frame

        arctic = default_arctic_client()
        lib = arctic.get_library(library)
        out = preview_frame(lib=lib, symbol=symbol, head=head, tail=tail)
        # route layer knows the canonical library name
        out["library"] = str(library)
        return out
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/catalog/preview/{library:path}/{symbol:path}", response_model=CatalogPreviewResponse)
def preview_cached_symbol(
    library: str,
    symbol: str,
    head: int = 5,
    tail: int = 5,
) -> dict:
    """Preview cached raw frame by library+symbol.

    Note: this is intentionally separated from the metadata store. It allows inspecting
    raw cached data in `market_data/...` libraries.
    """

    if head < 0 or head > 1000:
        raise HTTPException(status_code=400, detail="head must be between 0 and 1000")
    if tail < 0 or tail > 1000:
        raise HTTPException(status_code=400, detail="tail must be between 0 and 1000")

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_symbol import preview_frame

        arctic = default_arctic_client()
        lib = arctic.get_library(library)
        out = preview_frame(lib=lib, symbol=symbol, head=head, tail=tail)
        out["library"] = str(library)
        return out
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/catalog/describe/{library:path}", response_model=CatalogDescribeResponse)
def describe_cached_symbol(
    library: str,
    symbol: str,
) -> dict:
    """Describe cached raw frame by library + symbol.

    Example:
      GET /api/catalog/describe/market_data/PARQUET/1d?symbol=market_bars/equities_indicies.parquet/SPY

    Returns accurate stats (not sample-based): row count, index range, dtypes, missing counts,
    and numeric aggregates.
    """

    try:
        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
        from quantdsl_backtest.platform_api.services.catalog_symbol import describe_frame

        arctic = default_arctic_client()
        lib = arctic.get_library(library)
        out = describe_frame(lib=lib, symbol=symbol)
        out["library"] = str(library)
        return out
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/catalog/chart/{library:path}")
def chart_cached_symbol(
    library: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = 1500,
) -> dict:
    """Return chart-oriented data for a cached symbol.

    Behavior:
      - If the user specifies a bounded range (start and/or end), we return the FULL resolution
        for that slice, up to a safety cap.
      - If the range is unbounded and the series is large, we downsample evenly to `limit` points.

    This supports a "research" workflow: full resolution when the user zooms into a time window,
    but fast default loading for multi-year history.
    """

    if limit < 10 or limit > 200000:
        raise HTTPException(status_code=400, detail="limit must be between 10 and 200000")

    # Safety cap for full-resolution responses (browser + JSON payload size)
    FULL_MAX_ROWS = 20000

    try:
        import pandas as pd

        from quantdsl_backtest.platform_api.services.catalog import default_arctic_client

        arctic = default_arctic_client()
        lib = arctic.get_library(library)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*BlockManagerUnconsolidated.*",
                category=DeprecationWarning,
            )
            obj = lib.read(symbol)

        df = getattr(obj, "data", obj)

        if not isinstance(df, pd.DataFrame):
            if isinstance(df, pd.Series):
                df = df.to_frame(name=df.name or "value")
            else:
                df = pd.DataFrame(df)

        if df is None or df.empty:
            return {
                "library": str(library),
                "symbol": str(symbol),
                "rows": 0,
                "index_start": None,
                "index_end": None,
                "columns": [],
                "data": [],
            }

        # normalize index to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[df.index.notna()]

        if start:
            ts_start = pd.to_datetime(start, utc=True)
            if df.index.tz is None:
                ts_start = ts_start.tz_localize(None)
            df = df.loc[ts_start:]
        if end:
            ts_end = pd.to_datetime(end, utc=True)
            if df.index.tz is None:
                ts_end = ts_end.tz_localize(None)
            df = df.loc[:ts_end]

        if df.empty:
            return {
                "library": str(library),
                "symbol": str(symbol),
                "rows": 0,
                "index_start": None,
                "index_end": None,
                "columns": [],
                "data": [],
            }

        # Decide whether to return full-resolution
        is_bounded = bool(start or end)
        if is_bounded:
            if len(df) > FULL_MAX_ROWS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested range is too large for full resolution ({len(df)} rows). Narrow the range or rely on downsampling.",
                )
            df_out = df
        else:
            # Unbounded: downsample if needed
            if len(df) > limit:
                step = max(1, int(len(df) / limit))
                df_out = df.iloc[::step]
                # Ensure last row is included
                if df_out.index[-1] != df.index[-1]:
                    df_out = pd.concat([df_out, df.iloc[-1:]])
            else:
                df_out = df

        out = df_out.reset_index(names="ts")
        out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cols = [c for c in out.columns if c != "ts"]
        data = out.to_dict(orient="records")

        return {
            "library": str(library),
            "symbol": str(symbol),
            "rows": int(len(df_out)),
            "index_start": df_out.index.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "index_end": df_out.index.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": cols,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
