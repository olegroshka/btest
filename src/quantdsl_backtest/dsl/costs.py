# src/quantdsl_backtest/dsl/costs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(slots=True)
class Commission:
    """
    Commission scheme. You can extend this later with more modes.
    """

    type: Literal["per_share", "bps_notional"] = "per_share"
    amount: float = 0.0                 # per-share, or bps depending on type


@dataclass(slots=True)
class BorrowCost:
    """
    Short borrow cost specification.
    """

    default_annual_rate: float = 0.0    # e.g. 0.02 = 2% p.a.
    curve_name: Optional[str] = None    # name of curve in your data store


@dataclass(slots=True)
class FinancingCost:
    """
    Cash financing rate specification.
    """

    base_rate_curve: str = "SOFR"       # name of rate curve
    spread_bps: float = 0.0             # spread in basis points


@dataclass(slots=True)
class StaticFees:
    """
    Misc static fees like management/performance fees.
    """

    nav_fee_annual: float = 0.0         # mgmt fee in fraction of NAV p.a.
    perf_fee_fraction: float = 0.0      # e.g. 0.2 for 20% performance fee


@dataclass(slots=True)
class Costs:
    """
    All cost / fee / financing related parameters.
    """

    commission: Commission
    borrow: BorrowCost
    financing: FinancingCost
    fees: StaticFees
