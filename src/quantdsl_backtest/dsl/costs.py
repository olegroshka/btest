# src/quantdsl_backtest/dsl/costs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional


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

    ``default_annual_rate`` is charged on every short notional unless the
    instrument has its own rate: ``rates`` maps instrument to an annual rate
    (a fraction, 0.03 = 3% p.a.), or ``rates_path`` names a parquet/CSV file
    with one row per instrument (``instrument_column``, ``rate_column``),
    read once per run. Instruments absent from both pay the default. The
    file is a snapshot (one level of cost), not a history.

    Examples:
        BorrowCost(default_annual_rate=0.01)
        BorrowCost(default_annual_rate=0.01, rates={"GME": 0.48})
        BorrowCost(default_annual_rate=0.01, rates_path="exports/borrow_rates.parquet")
    """

    default_annual_rate: float = 0.0    # e.g. 0.02 = 2% p.a.
    curve_name: Optional[str] = None    # name of curve in your data store
    rates: Optional[Mapping[str, float]] = None   # per-instrument annual rates, a fraction
    rates_path: Optional[str] = None              # parquet or CSV with one row per instrument
    instrument_column: str = "ticker"
    rate_column: str = "fee_rate"


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
class TransactionTax:
    """A transaction tax on trade notional — generic across markets (UK SDRT = 0.5% on buys; French / Italian FTT;
    etc.). Charged ON TOP of commission/slippage, scoped to a trade side and (optionally) an instrument set.

    Examples:
        TransactionTax(rate=0.005, on="buy")                       # UK stamp duty, whole book
        TransactionTax(rate=0.005, on="buy", instruments=uk_set)   # UK stamp, only the UK names in a mixed book
    """

    rate: float = 0.0                                  # fraction of trade notional, e.g. 0.005 = 0.5%
    on: Literal["buy", "sell", "both"] = "buy"         # which side is taxed (UK stamp = buy only)
    instruments: Optional[frozenset] = None            # None = every instrument; else only these tickers


@dataclass(slots=True)
class Costs:
    """
    All cost / fee / financing related parameters.
    """

    commission: Commission
    borrow: BorrowCost
    financing: FinancingCost
    fees: StaticFees
    tax: Optional[TransactionTax] = None               # transaction tax (e.g. UK stamp duty); None = no tax
