from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import os
import pandas as pd

from ...dsl.universe import Universe
from ..requests import DataRequest
from ..bundles import MarketBarsBundle
from .base import CacheStore
from .cache import TailCachedFrameLoader, MemoryCacheStore
from .universe_filtering import UniverseFilterEngine


def _normalize_datetime_index(table: pd.DataFrame) -> pd.DataFrame:
    for col in ("datetime", "date", "timestamp"):
        if col in table.columns:
            dt = pd.to_datetime(table[col])
            table = table.drop(columns=[col])
            table.index = dt
            table.index.name = "datetime"
            return table

    if isinstance(table.index, pd.DatetimeIndex):
        return table

    raise ValueError(
        "Could not find a datetime column ('datetime', 'date', or 'timestamp') and index is not a DatetimeIndex."
    )


def _infer_instrument_id_column(table: pd.DataFrame) -> str:
    candidates = ["ticker", "symbol", "ric", "instrument", "secid"]
    for c in candidates:
        if c in table.columns:
            return c
    for c in table.columns:
        if not pd.api.types.is_numeric_dtype(table[c]):
            return c
    raise ValueError("Could not infer an instrument ID column from data table.")


def _parquet_last_needed_ts(end_ts: pd.Timestamp, frequency: str) -> pd.Timestamp:
    # parquet is treated as inclusive daily-style series
    return end_ts


def _parquet_next_fetch_start(last_dt: pd.Timestamp, frequency: str) -> pd.Timestamp:
    return (last_dt + pd.Timedelta(days=1)).normalize()


def _should_use_cache_for_parquet(request: DataRequest, cache: Optional[CacheStore]) -> bool:
    """Parquet caching is opt-in.

    Rationale: the current test environment contains a pre-created local_cache folder
    that may not be a valid ArcticDB LMDB store, and slow tests must not fail due to
    cache backend issues.

    We still support the "golden cache" ingestion path when explicitly enabled.
    """

    if cache is None:
        return False

    # Always allow in-memory cache (tests / explicit injection)
    if isinstance(cache, MemoryCacheStore):
        return True

    # Explicit opt-in via dataset_id (platform use)
    return request.dataset_id == "golden_cache"


@dataclass(slots=True)
class ParquetMarketBarsSource:
    """Load market bars from a parquet file/table."""

    name: str = "parquet_market_bars"
    universe_filters: UniverseFilterEngine = field(default_factory=UniverseFilterEngine)
    cache_loader: TailCachedFrameLoader = field(
        default_factory=lambda: TailCachedFrameLoader(provider="PARQUET")
    )

    def can_load(self, request: DataRequest) -> bool:
        return request.kind == "market_bars" and request.source.startswith("parquet://")

    def load(
        self,
        request: DataRequest,
        universe: Optional[Universe],
        cache: Optional[CacheStore],
    ) -> MarketBarsBundle:
        path = request.source[len("parquet://") :]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet data file not found: {path}")

        table = pd.read_parquet(path)
        table = _normalize_datetime_index(table)
        table = table.sort_index()
        table = table.loc[request.start: request.end]

        id_col = universe.id_field if universe is not None else _infer_instrument_id_column(table)

        if universe is not None:
            if universe.static_instruments is not None:
                table = table[table[id_col].isin(universe.static_instruments)]

        use_cache = _should_use_cache_for_parquet(request, cache)

        bars_by_instr: Dict[str, pd.DataFrame] = {}
        instruments: List[str] = []
        filters = universe.filters if universe is not None else []

        for instr, df_instr in table.groupby(id_col):
            instr = str(instr)
            df_instr = df_instr.drop(columns=[id_col])

            def _normalize(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                if not isinstance(out.index, pd.DatetimeIndex):
                    out.index = pd.to_datetime(out.index)
                out.index = pd.to_datetime(out.index).tz_localize(None)
                return out.sort_index()

            def _fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
                s = str(start.date())
                e = str(end.date())
                return df_instr.loc[s:e]

            if use_cache:
                df_final = self.cache_loader.load_one(
                    request,
                    entity=instr,
                    cache=cache,
                    normalize=_normalize,
                    fetch=_fetch,
                    last_needed_ts=_parquet_last_needed_ts,
                    next_fetch_start=_parquet_next_fetch_start,
                )
            else:
                df_final = _normalize(df_instr.loc[request.start: request.end])

            if filters and not self.universe_filters.passes(df_final, filters):
                continue

            cols = [c for c in (request.fields or []) if c in df_final.columns]
            extra_cols = [c for c in df_final.columns if c not in cols]
            df_final = df_final[cols + extra_cols]

            bars_by_instr[instr] = df_final
            instruments.append(instr)

        return MarketBarsBundle(
            kind="market_bars",
            source=request.source,
            start=request.start,
            end=request.end,
            frequency=request.frequency,
            calendar=request.calendar,
            tz=request.tz,
            bars=bars_by_instr,
            instruments=instruments,
            fields=list(request.fields or []),
        )
