from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd


def _get_arctic_uri() -> str:
    # Allow override via env; default local LMDB folder
    uri = os.environ.get("QUANTDSL_ARCTIC_URI", "lmdb://local_cache")

    # For lmdb:// URIs, ensure the directory exists.
    # ArcticDB/LMDB can throw if the directory doesn't exist (common under parallel tests).
    if uri.startswith("lmdb://"):
        path = uri[len("lmdb://") :]
        if path:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                # If we can't create it, leave as-is; SafeArcticCacheStore will disable on failure.
                pass

    return uri


_ARCTIC_CLIENTS: dict[str, Any] = {}


def get_arctic_client() -> Any:
    """Return a cached ArcticDB client for the current URI.

    LMDB (local_cache) does not support multiple opens in the same process,
    so we must use a singleton per URI.
    """
    try:
        from arcticdb import Arctic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "arcticdb is required for caching. Install with: pip install arcticdb"
        ) from e

    uri = _get_arctic_uri()
    if uri not in _ARCTIC_CLIENTS:
        _ARCTIC_CLIENTS[uri] = Arctic(uri)
    return _ARCTIC_CLIENTS[uri]


def get_cache_lib(provider: str, frequency: str):
    """
    Open (and create if needed) an ArcticDB library for given provider/frequency.
    Library naming: market_data/<PROVIDER>/<FREQ>
    """
    ac = get_arctic_client()
    lib_name = f"market_data/{provider.upper()}/{frequency.lower()}"
    lib = ac.get_library(lib_name, create_if_missing=True)
    return lib


def cache_has_symbol(lib: Any, symbol: str) -> bool:
    try:
        return lib.has_symbol(symbol)
    except Exception:
        return False


def cache_read_symbol(lib: Any, symbol: str) -> pd.DataFrame:
    # ArcticDB can return DataFrames backed by internal pandas managers that trigger
    # a pandas DeprecationWarning. Copying the whole frame is expensive; instead,
    # we narrowly suppress that specific warning for this read path.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*BlockManagerUnconsolidated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Passing a BlockManagerUnconsolidated to DataFrame is deprecated.*",
            category=DeprecationWarning,
        )
        obj = lib.read(symbol)

    data = obj.data
    # Ensure DataFrame
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "value")

    if isinstance(data, pd.DataFrame):
        return data

    return pd.DataFrame(data)


def cache_write_symbol(lib: Any, symbol: str, df: pd.DataFrame) -> None:
    # ArcticDB write normalization may also touch internal pandas managers.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*BlockManagerUnconsolidated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Passing a BlockManagerUnconsolidated to DataFrame is deprecated.*",
            category=DeprecationWarning,
        )
        lib.write(symbol, df)
