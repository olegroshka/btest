# src/quantdsl_backtest/dsl/universe.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# --- Universe filters -------------------------------------------------------


@dataclass(slots=True)
class UniverseFilter:
    """Base class for universe filters (marker only)."""
    pass


@dataclass(slots=True)
class HasHistory(UniverseFilter):
    min_days: int = 252


@dataclass(slots=True)
class MinPrice(UniverseFilter):
    min_price: float = 1.0


@dataclass(slots=True)
class MinDollarADV(UniverseFilter):
    min_dollar_adv: float = 1_000_000.0


# --- Universe ---------------------------------------------------------------


@dataclass(slots=True)
class Universe:
    """
    Definition of the tradable universe for a strategy.

    The engine will:
      - start from some base universe implied by the data source
      - then apply these filters to produce the effective tradable set.
    """

    name: str                            # e.g. "SP500"
    filters: List[UniverseFilter] = field(default_factory=list)
    id_field: str = "ticker"             # column that identifies instruments
    # Optional: subset (e.g. list of tickers) if you want to hard-code.
    static_instruments: Optional[List[str]] = None
