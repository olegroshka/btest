from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quantdsl_backtest.data.frequency import next_bar_start


@dataclass(frozen=True, slots=True)
class CoverageRow:
    symbol: str
    provider: str
    frequency: str
    kind: str
    dataset: str
    entity: str
    cached_start: Optional[pd.Timestamp]
    cached_end: Optional[pd.Timestamp]


def _to_ts(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        ts = pd.to_datetime(x, utc=False, errors="coerce")
        if pd.isna(ts):
            return None
        # normalize to tz-naive for comparisons with request start/end
        if hasattr(ts, "tz_localize"):
            try:
                ts = ts.tz_localize(None)
            except Exception:
                pass
        return pd.Timestamp(ts)
    except Exception:
        return None


def build_coverage_rows(meta_df: pd.DataFrame) -> List[CoverageRow]:
    if meta_df is None or meta_df.empty:
        return []

    required = {"symbol", "provider", "frequency", "kind", "dataset", "entity"}
    if not required.issubset(set(meta_df.columns)):
        return []

    out: List[CoverageRow] = []
    for _, r in meta_df.iterrows():
        out.append(
            CoverageRow(
                symbol=str(r.get("symbol")),
                provider=str(r.get("provider")),
                frequency=str(r.get("frequency")),
                kind=str(r.get("kind")),
                dataset=str(r.get("dataset")),
                entity=str(r.get("entity")),
                cached_start=_to_ts(r.get("start")),
                cached_end=_to_ts(r.get("end")),
            )
        )

    # Last write wins per symbol/entity
    seen: Dict[str, CoverageRow] = {}
    for row in out:
        seen[row.symbol] = row

    return list(seen.values())


def plan_download(
    *,
    request_start: str,
    request_end: str,
    entities: List[str],
    provider: str,
    frequency: str,
    kind: str,
    dataset: str,
    meta_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Return a per-entity cache plan based on metadata coverage.

    action:
      - miss -> full_fetch
      - cached covers end -> cache_hit
      - cached partial -> tail_fetch

    Notes:
      - This is an approximation: provider end semantics can differ. The real load path
        is still authoritative.
    """

    start_ts = _to_ts(request_start)
    end_ts = _to_ts(request_end)

    if start_ts is None or end_ts is None:
        return []

    # Build coverage keyed by (entity, provider/freq/kind/dataset)
    cov = build_coverage_rows(meta_df)

    by_entity: Dict[str, CoverageRow] = {}
    for row in cov:
        if row.provider.upper() != provider.upper():
            continue
        if row.frequency != frequency:
            continue
        if row.kind != kind:
            continue
        if row.dataset != dataset:
            continue
        by_entity[row.entity] = row

    plans: List[Dict[str, Any]] = []
    for e in entities:
        row = by_entity.get(e)
        if row is None or row.cached_end is None:
            plans.append(
                {
                    "entity": e,
                    "action": "full_fetch",
                    "symbol": f"{kind}/{dataset}/{e}",
                    "cached_start": None,
                    "cached_end": None,
                }
            )
            continue

        cached_start = row.cached_start
        cached_end = row.cached_end

        if cached_end >= end_ts:
            plans.append(
                {
                    "entity": e,
                    "action": "cache_hit",
                    "symbol": row.symbol,
                    "cached_start": None if cached_start is None else str(cached_start),
                    "cached_end": str(cached_end),
                }
            )
        else:
            fetch_start = next_bar_start(cached_end, frequency)
            plans.append(
                {
                    "entity": e,
                    "action": "tail_fetch",
                    "symbol": row.symbol,
                    "cached_start": None if cached_start is None else str(cached_start),
                    "cached_end": str(cached_end),
                    "fetch_start": str(fetch_start),
                    "fetch_end": str(end_ts),
                }
            )

    return plans
