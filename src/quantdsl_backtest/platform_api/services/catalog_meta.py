from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import pandas as pd


class _MetaLibLike(Protocol):
    def has_symbol(self, symbol: str) -> bool: ...

    def read(self, symbol: str) -> Any: ...

    def write(self, symbol: str, data: Any) -> Any: ...


class _ArcticLike(Protocol):
    def get_library(self, name: str, create_if_missing: bool = ...): ...


META_LIBRARY_NAME = "platform_meta/catalog"
META_SYMBOL = "catalog_index"
LEGACY_META_SYMBOL = "catalog_index_v1"


@dataclass(frozen=True, slots=True)
class CacheSymbolMeta:
    """Minimal metadata about one cached symbol."""

    provider: str
    frequency: str
    kind: str
    dataset: str
    entity: str
    symbol: str
    library: str

    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]

    updated_at: pd.Timestamp
    rows: int
    cols: int


def _utcnow_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=timezone.utc))


def build_meta_row_from_df(
    *,
    provider: str,
    frequency: str,
    kind: str,
    dataset: str,
    entity: str,
    symbol: str,
    df: pd.DataFrame,
) -> CacheSymbolMeta:
    idx = getattr(df, "index", None)
    if idx is None or len(df) == 0:
        start_ts = None
        end_ts = None
        rows = 0
    else:
        # Assumption: time series indexed frames
        try:
            start_ts = pd.Timestamp(idx.min())
            end_ts = pd.Timestamp(idx.max())
        except Exception:
            start_ts = None
            end_ts = None
        rows = int(len(df))

    cols = int(getattr(df, "shape", (rows, 0))[1])

    library = f"market_data/{str(provider).upper()}/{str(frequency).lower()}"

    return CacheSymbolMeta(
        provider=provider,
        frequency=frequency,
        kind=kind,
        dataset=dataset,
        entity=entity,
        symbol=symbol,
        library=library,
        start=start_ts,
        end=end_ts,
        updated_at=_utcnow_ts(),
        rows=rows,
        cols=cols,
    )


def _meta_to_record(m: CacheSymbolMeta) -> dict[str, Any]:
    return {
        "provider": m.provider,
        "frequency": m.frequency,
        "kind": m.kind,
        "dataset": m.dataset,
        "entity": m.entity,
        "symbol": m.symbol,
        "library": m.library,
        "start": m.start,
        "end": m.end,
        "updated_at": m.updated_at,
        "rows": m.rows,
        "cols": m.cols,
    }


def _records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "provider",
                "frequency",
                "kind",
                "dataset",
                "entity",
                "symbol",
                "library",
                "start",
                "end",
                "updated_at",
                "rows",
                "cols",
            ]
        )

    df = pd.DataFrame.from_records(records)
    for c in ["start", "end", "updated_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def read_catalog_index(*, meta_lib: _MetaLibLike) -> pd.DataFrame:
    """Read the catalog metadata index as a DataFrame.

    Best-effort: returns empty on errors/missing.
    """

    sym = get_meta_symbol_name(meta_lib)

    try:
        if not meta_lib.has_symbol(sym):
            return _records_to_df([])
        obj = meta_lib.read(sym)
        data = getattr(obj, "data", obj)
        if isinstance(data, pd.Series):
            return data.to_frame(name=data.name or "value")
        if isinstance(data, pd.DataFrame):
            # Ensure expected dtypes
            for c in ["start", "end", "updated_at"]:
                if c in data.columns:
                    data[c] = pd.to_datetime(data[c], utc=True, errors="coerce")
            return data
        return _records_to_df([])
    except Exception:
        return _records_to_df([])


def upsert_catalog_index(*, meta_lib: _MetaLibLike, row: CacheSymbolMeta) -> None:
    """Insert/update one row in the catalog metadata index."""

    df = read_catalog_index(meta_lib=meta_lib)
    rec = _meta_to_record(row)

    if df.empty:
        out = _records_to_df([rec])
        meta_lib.write(META_SYMBOL, out)
        return

    # Key by symbol (globally unique cache key)
    if "symbol" not in df.columns:
        df = _records_to_df([])

    existing = df["symbol"] == rec["symbol"]
    if existing.any():
        # Replace row
        df = df.loc[~existing].copy()

    out = pd.concat([df, _records_to_df([rec])], axis=0, ignore_index=True)
    # Stable sort for readability
    sort_cols = [c for c in ["provider", "frequency", "kind", "dataset", "entity", "symbol"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    meta_lib.write(META_SYMBOL, out)


def get_meta_symbol_name(meta_lib: _MetaLibLike) -> str:
    """Return the canonical meta symbol name.

    We write to the unversioned symbol. For reads, if the unversioned symbol isn't
    present but the legacy one is, we transparently read legacy (to support existing
    local caches).
    """

    try:
        if meta_lib.has_symbol(META_SYMBOL):
            return META_SYMBOL
    except Exception:
        pass

    try:
        if meta_lib.has_symbol(LEGACY_META_SYMBOL):
            return LEGACY_META_SYMBOL
    except Exception:
        pass

    return META_SYMBOL


def get_meta_library(*, arctic: _ArcticLike) -> Any:
    return arctic.get_library(META_LIBRARY_NAME, create_if_missing=True)
