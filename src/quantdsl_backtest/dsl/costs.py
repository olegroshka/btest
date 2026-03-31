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

    Parameters
    ----------
    base_rate_curve : str
        Named rate curve (e.g. ``"SOFR"``, ``"ESTR"``).  Used as a fallback
        key when no explicit ``rate_csv_path`` is provided.
    spread_bps : float
        Lender spread over the base rate, in basis points (e.g. 150 for IBKR
        EUR Pro tier = €STR + 1.50 %).
    rate_csv_path : str or None
        Optional path to a CSV file with columns ``date`` and ``rate_pct``
        (rate expressed as percentage, e.g. 2.50 for 2.50 %).  When provided,
        ``SingleAssetRunner`` loads this file directly instead of looking up
        ``base_rate_curve`` in its registry.  Leave as ``None`` and register
        well-known curve paths via ``SingleAssetRunner(rate_registry={...})``
        for portable strategies.
    """

    base_rate_curve: str = "SOFR"       # name of rate curve
    spread_bps: float = 0.0             # spread over base rate in bps
    rate_csv_path: Optional[str] = None # direct path to rate CSV (overrides registry)


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
