from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class _ArcticLike(Protocol):
    def list_libraries(self) -> Iterable[str]: ...

    def get_library(self, name: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class CatalogLibrary:
    library: str
    symbols: list[str]

    @property
    def count(self) -> int:
        return len(self.symbols)


def list_arctic_cache_catalog(*, arctic: _ArcticLike) -> list[CatalogLibrary]:
    """Return a best-effort list of cached dataset symbols in ArcticDB.

    Contract:
      - Only considers libraries under the repo naming convention: `market_data/...`.
      - Never raises for per-library failures (e.g., corrupted library). Those become empty.

    This function is intentionally pure (caller injects an arctic-like object), making it
    easy to unit test without requiring the real `arcticdb` package.
    """

    libs: Iterable[str]
    try:
        libs = arctic.list_libraries()
    except Exception:
        return []

    out: list[CatalogLibrary] = []
    for lib_name in sorted(libs):
        if not str(lib_name).startswith("market_data/"):
            continue

        symbols: list[str]
        try:
            lib = arctic.get_library(lib_name)
            symbols = sorted(list(lib.list_symbols()))
        except Exception:
            symbols = []

        out.append(CatalogLibrary(library=str(lib_name), symbols=symbols))

    return out


def _is_lmdb_corruption_error(exc: Exception) -> bool:
    msg = str(exc)
    return ("MDB_INVALID" in msg) or ("File is not an LMDB file" in msg)


def default_arctic_client() -> Any:
    """Create a real ArcticDB client.

    Kept in a separate helper so the pure catalog logic above can be tested without
    importing `arcticdb`.
    """

    from ...data.cache_arctic import get_arctic_client

    return get_arctic_client()
