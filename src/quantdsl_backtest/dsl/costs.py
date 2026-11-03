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
    # Smallest charge per order, in account currency (bps_notional only). A broker's "3 or 0.05%,
    # whichever is more" is Commission(type="bps_notional", amount=5.0, min_per_order=3.0).
    min_per_order: float = 0.0


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
    # A broker's minimum monthly activity fee, in account currency: at each month end the account pays
    # this less the commission it paid that month, never below zero. Event-driven engine only.
    min_monthly_fee: float = 0.0


@dataclass(slots=True)
class TransactionTax:
    """A transaction tax on trade notional — generic across markets (UK SDRT = 0.5% on buys; French / Italian FTT;
    etc.). Charged ON TOP of commission/slippage, scoped to a trade side and (optionally) an instrument set.

    Examples:
        TransactionTax(rate=0.005, on="buy")                       # UK stamp duty, whole book
        TransactionTax(rate=0.005, on="buy", instruments=uk_set)   # UK stamp, only the UK names in a mixed book
        TransactionTax(rate=0.0003, on="both", min_per_trade=1.5, instruments=usd_set)   # FX conversion on foreign names
    """

    rate: float = 0.0                                  # fraction of trade notional, e.g. 0.005 = 0.5%
    on: Literal["buy", "sell", "both"] = "buy"         # which side is taxed (UK stamp = buy only)
    instruments: Optional[frozenset] = None            # None = every instrument; else only these tickers
    min_per_trade: float = 0.0                         # smallest charge per taxed trade, in account currency


@dataclass(slots=True)
class FundingCost:
    """Perpetual-futures funding. Per event: a position of q units pays q * price * funding_rate
    (rate > 0 → longs pay shorts; rate < 0 → shorts pay longs). Each bar is charged the SUM of the
    rates of the events that settle inside the bar's holding window (bar open, bar open + 1 bar],
    on the positions held over that bar, at that bar's mark price.

    source:
        "sfera://bncrpt/funding_rates"  (default; columns symbol, funding_ts, funding_rate, venue)
        "parquet://path.parquet"        long format with columns (ts, ticker, funding_rate)
    venue: filters the sfera table (ignored for parquet).
    """

    source: str = "sfera://bncrpt/funding_rates"
    venue: str = "binance_um"


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
    funding: Optional[FundingCost] = None              # perp funding (crypto); None = no funding
