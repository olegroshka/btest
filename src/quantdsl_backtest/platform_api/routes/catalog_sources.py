from __future__ import annotations

from typing import Any


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform] (or uv sync --extra platform)"
        ) from e

    return _APIRouter(tags=["catalog"])


router = _router()


@router.get("/catalog/sources")
def list_catalog_sources() -> dict[str, Any]:
    """List supported source types for Download UI.

    This is intentionally a small, UI-oriented capability endpoint.

    Fields:
      - id: stable identifier used by UI
      - scheme: URI prefix the API expects in CatalogDownloadRequest.source
      - label: human readable
      - file_based: if True, UI should offer a file picker and construct source as <scheme> + <path>
      - supports_frequency: if True, frequency must be provided/selected
      - examples: example source strings
    """

    return {
        "sources": [
            {
                "id": "parquet",
                "scheme": "parquet://",
                "label": "Parquet (local file)",
                "file_based": True,
                "supports_frequency": True,
                "examples": ["parquet://equities/sp500_daily"],
            },
            {
                "id": "yf",
                "scheme": "yf://",
                "label": "Yahoo Finance (YF)",
                "file_based": False,
                "supports_frequency": True,
                "examples": ["yf://AAPL", "yf://SPY"],
            },
            {
                "id": "fred",
                "scheme": "fred://",
                "label": "FRED",
                "file_based": False,
                "supports_frequency": True,
                "examples": ["fred://CPIAUCSL"],
            },
        ]
    }

