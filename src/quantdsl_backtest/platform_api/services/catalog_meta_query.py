from __future__ import annotations

from typing import Optional

import pandas as pd


def _norm_opt(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s or None


def filter_meta_df(
    df: pd.DataFrame,
    *,
    provider: Optional[str] = None,
    frequency: Optional[str] = None,
    dataset: Optional[str] = None,
    kind: Optional[str] = None,
    entity: Optional[str] = None,
    library: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Filter the metadata index.

    - All filters are exact-match (case-insensitive for provider/kind, case-sensitive for dataset/entity).
    - Returns a copy suitable for serialization.
    """

    if df is None or df.empty:
        return df

    provider = _norm_opt(provider)
    frequency = _norm_opt(frequency)
    dataset = _norm_opt(dataset)
    kind = _norm_opt(kind)
    entity = _norm_opt(entity)
    library = _norm_opt(library)
    symbol = _norm_opt(symbol)

    out = df

    if provider is not None and "provider" in out.columns:
        out = out[out["provider"].astype(str).str.upper() == provider.upper()]

    if frequency is not None and "frequency" in out.columns:
        out = out[out["frequency"].astype(str) == frequency]

    if kind is not None and "kind" in out.columns:
        out = out[out["kind"].astype(str).str.lower() == kind.lower()]

    if dataset is not None and "dataset" in out.columns:
        out = out[out["dataset"].astype(str) == dataset]

    if entity is not None and "entity" in out.columns:
        out = out[out["entity"].astype(str) == entity]

    if library is not None and "library" in out.columns:
        out = out[out["library"].astype(str) == library]

    if symbol is not None and "symbol" in out.columns:
        out = out[out["symbol"].astype(str) == symbol]

    out = out.copy()

    if limit is not None:
        try:
            lim = int(limit)
            if lim >= 0:
                out = out.head(lim)
        except Exception:
            pass

    return out

