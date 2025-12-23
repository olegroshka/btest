from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol, Tuple

import pandas as pd

from .catalog_meta import get_meta_library, upsert_catalog_index, build_meta_row_from_df


class _ArcticLibLike(Protocol):
    def list_symbols(self) -> Iterable[str]: ...

    def read(self, symbol: str): ...


class _ArcticLike(Protocol):
    def list_libraries(self) -> Iterable[str]: ...

    def get_library(self, name: str, create_if_missing: bool = ...): ...


@dataclass(slots=True)
class RefreshStats:
    libraries_scanned: int = 0
    symbols_scanned: int = 0
    meta_upserts: int = 0
    errors: int = 0


def _parse_library_name(lib_name: str) -> tuple[str, str] | None:
    # market_data/<PROVIDER>/<FREQ>
    parts = str(lib_name).split("/")
    if len(parts) < 3:
        return None
    if parts[0] != "market_data":
        return None
    return str(parts[1]), str(parts[2])


def _parse_symbol_new(symbol: str) -> tuple[str, str, str] | None:
    # <kind>/<dataset>/<entity>
    parts = str(symbol).split("/")
    if len(parts) < 3:
        return None
    # Guard: avoid mis-parsing legacy v1 keys as new keys
    if parts[0] == "v1":
        return None
    kind = parts[0]
    dataset = parts[1]
    entity = "/".join(parts[2:])
    return str(kind), str(dataset), str(entity)


def _parse_symbol_legacy_v1(symbol: str) -> Tuple[str, str, str, str, str] | None:
    """Parse legacy v1 cache key: v1/<provider>/<kind>/<frequency>/<dataset>/<entity>"""
    try:
        parts = str(symbol).split("/")
        if len(parts) < 6:
            return None
        if parts[0] != "v1":
            return None
        provider, kind, frequency, dataset = parts[1], parts[2], parts[3], parts[4]
        entity = "/".join(parts[5:])
        return str(provider), str(kind), str(frequency), str(dataset), str(entity)
    except Exception:
        return None


def refresh_catalog_meta_from_cache(*, arctic: _ArcticLike) -> Dict[str, int]:
    """Scan cache libraries and upsert metadata index.

    Supports both:
      - New keying: symbol=<kind>/<dataset>/<entity> (preferred)
      - Legacy v1 keying for migration

    The stored `symbol` in the meta index is always the per-library symbol string.
    """

    stats = RefreshStats()

    try:
        libs = sorted(list(arctic.list_libraries()))
    except Exception:
        return {"libraries_scanned": 0, "symbols_scanned": 0, "meta_upserts": 0, "errors": 1}

    meta_lib = get_meta_library(arctic=arctic)

    for lib_name in libs:
        if not str(lib_name).startswith("market_data/"):
            continue

        parsed_lib = _parse_library_name(str(lib_name))
        if parsed_lib is None:
            continue
        provider, frequency = parsed_lib

        stats.libraries_scanned += 1

        try:
            lib = arctic.get_library(lib_name)
            symbols = list(lib.list_symbols())  # type: ignore[call-arg]
        except Exception:
            stats.errors += 1
            continue

        for sym in symbols:
            stats.symbols_scanned += 1
            try:
                obj = lib.read(sym)  # type: ignore[misc]
                data = getattr(obj, "data", obj)
                if isinstance(data, pd.Series):
                    df = data.to_frame(name=data.name or "value")
                elif isinstance(data, pd.DataFrame):
                    df = data
                else:
                    df = pd.DataFrame(data)

                symbol_key = str(sym)

                # Prefer parsing the new key format.
                parsed_new = _parse_symbol_new(symbol_key)
                if parsed_new is not None:
                    kind, dataset, entity = parsed_new
                else:
                    # Legacy fallback
                    parsed_old = _parse_symbol_legacy_v1(symbol_key)
                    if parsed_old is None:
                        continue
                    _provider, kind, _frequency, dataset, entity = parsed_old
                    # If legacy values disagree with library naming, we still trust the library.

                row = build_meta_row_from_df(
                    provider=str(provider),
                    frequency=str(frequency),
                    kind=str(kind),
                    dataset=str(dataset),
                    entity=str(entity),
                    symbol=str(symbol_key),
                    df=df,
                )

                upsert_catalog_index(meta_lib=meta_lib, row=row)
                stats.meta_upserts += 1
            except Exception:
                stats.errors += 1
                continue

    return {
        "libraries_scanned": int(stats.libraries_scanned),
        "symbols_scanned": int(stats.symbols_scanned),
        "meta_upserts": int(stats.meta_upserts),
        "errors": int(stats.errors),
    }
