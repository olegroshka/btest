from __future__ import annotations

import os
from typing import Any

import pandas as pd


def _get_arctic_uri() -> str:
    # Allow override via env; default local LMDB folder
    return os.environ.get("QUANTDSL_ARCTIC_URI", "lmdb://local_cache")


def get_cache_lib(provider: str, frequency: str):
    """
    Open (and create if needed) an ArcticDB library for given provider/frequency.
    Library naming: market_data/<PROVIDER>/<FREQ>
    """
    try:
        from arcticdb import Arctic  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "arcticdb is required for caching. Install with: pip install arcticdb"
        ) from e

    uri = _get_arctic_uri()
    ac = Arctic(uri)
    lib_name = f"market_data/{provider.upper()}/{frequency.lower()}"
    lib = ac.get_library(lib_name, create_if_missing=True)
    return lib


def cache_has_symbol(lib: Any, symbol: str) -> bool:
    try:
        return lib.has_symbol(symbol)
    except Exception:
        return False


def cache_read_symbol(lib: Any, symbol: str) -> pd.DataFrame:
    obj = lib.read(symbol)
    data = obj.data
    # Ensure DataFrame
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "value")
    return data


def cache_write_symbol(lib: Any, symbol: str, df: pd.DataFrame) -> None:
    lib.write(symbol, df)
