"""
Data transformation interface for market data post-processing.

A transform takes wide `prices` and `volumes` DataFrames [datetime x instrument]
and returns transformed `(prices, volumes)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import pandas as pd


class DataTransform(Protocol):
    """Protocol for data transforms applied after loading raw market data."""

    def apply(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...


@dataclass(slots=True)
class CleaningTransform:
    """
    Generic cleaning transform suitable for sparse daily index datasets.

    Operations performed (configurable):
      - Forward/backward fill `prices` to bridge short gaps
      - Compute per-instrument coverage ratio on `prices` and keep instruments
        with coverage >= `min_coverage`. If fewer than `min_keep` satisfy the
        threshold, keep the top-coverage instruments up to `min_keep`.
      - Restrict `volumes` to the kept instruments and fill NaNs with 0.0.

    This mirrors the previous hardcoded logic that was applied for
    `equities/indicies.parquet` dataset, but is now opt-in via DSL.
    """

    min_coverage: float = 0.80
    min_keep: int = 4
    fill_prices: bool = True
    fill_volumes: bool = True

    # Optional friendly name
    name: str = "CleaningTransform"

    def apply(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:  # type: ignore[override]
        # Ensure numeric dtype; coerce any non-numeric (e.g., None/strings) to NaN
        p = prices.apply(pd.to_numeric, errors="coerce")
        v = volumes.apply(pd.to_numeric, errors="coerce")

        if self.fill_prices:
            p = p.ffill().bfill()

        coverage = p.notna().mean(axis=0)
        keep = coverage[coverage >= self.min_coverage].index.tolist()
        if len(keep) < self.min_keep:
            # Fallback: match original behavior by taking top-coverage names
            # even if coverage may be low; cap at min_keep.
            eligible = coverage.sort_values(ascending=False)
            k = min(self.min_keep, len(eligible))
            keep = eligible.head(k).index.tolist()

        p = p[keep]
        v = v[keep]
        if self.fill_volumes:
            v = v.fillna(0.0)

        return p, v
