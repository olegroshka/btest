"""Disclosed positioning: rank-weighted books of the funds' short profit on the public registers.

Inputs are written by datacli (``scripts/export_register_inputs.py``): the
daily price panel of the euro registers' issuers (Germany, France, the
Netherlands, Ireland; one ticker per listing, ``SAP.XETRA``) and
``register_profit.parquet``, the per-date demeaned rank in [-0.5, 0.5] of
the issuer panel's funds' short profit residualised on reversal and momentum
(the ``factors`` cleaning), among the names with a visible holder,
shown from the day after it is marked (point in time); ``_neg`` is its
negation and ``register_profit_raw`` the unresidualised rank.

Orientation: where the disclosed shorts are in the
money the stock keeps falling, so the short book is the top of the profit
ranks and the long book the bottom (the shorts that are losing). As in the
second iteration: books of the top and bottom 30 percent weighted by the
demeaned rank, a turnover limit, commission, square-root impact, a volume
cap, and the borrow charged per name at the broker's rate from the country
files (``BorrowCost(rates_path=...)``, ``register_borrow_rates.parquet``),
the flat default for the rest; datacli's overlay
(``scripts/btest_borrow_overlay.py``) remains the check. The effect lives
at one to four weeks, so the rebalance is weekly. The books are small (the
pooled euro registers carry 40 to 170 ranked names a day): 25 a side, a 4
percent cap, and ``min_names_per_book`` so the engine trades as soon as ten
names are ranked on each side instead of waiting for full books.

Run from the btest root:  uv run python strategies/dpos/register_profit.py

The UK sizing check (the frozen FCA history) runs the same strategy
on datacli's ``exports/btest_uk`` (``DATACLI_BTEST_EXPORTS``), in pounds,
with ``DPOS_TAG=uk`` so its outputs sit next to the euro ones, and
``DPOS_START``/``DPOS_END`` on the halves split at 2019-09.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    DrawdownPolicy,
    Reporting,
    RiskChecks,
)
from quantdsl_backtest.dsl.costs import (
    BorrowCost,
    Commission,
    Costs,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.factors import ExternalFactor
from quantdsl_backtest.dsl.portfolio import (
    Book,
    LongShortPortfolio,
    SignalWeight,
    TopN,
    TurnoverLimit,
)
from quantdsl_backtest.dsl.signals import (
    And,
    GreaterEqual,
    Identity,
    MaskFromBoolean,
    NotNull,
)
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import HasHistory, MinDollarADV, MinPrice, Universe
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

_REPO = (
    Path(__file__).resolve().parents[2]
)  # the btest root; the datacli exports live under data/raw/ (a local copy, not committed)
EXPORTS = Path(
    os.environ.get(
        "DATACLI_BTEST_EXPORTS",
        str(_REPO / "data" / "raw" / "positioning" / "exports" / "btest"),
    )
)
SIGNAL = os.environ.get("DPOS_SIGNAL", "register_profit")
BOOK_EDGE = float(
    os.environ.get("DPOS_EDGE", "0.2")
)  # demeaned rank at the edge of each book (0.2 = the top 30 percent)
NAMES_PER_SIDE = int(os.environ.get("DPOS_NAMES", "25"))
MIN_NAMES_PER_BOOK = int(
    os.environ.get("DPOS_MIN_NAMES", "10")
)  # trade as soon as each book has this many ranked names
BORROW_RATES = os.environ.get(
    "DPOS_BORROW_RATES", "register_borrow_rates"
)  # per-name rates from the broker's country files; "" for the flat rate only
TURNOVER = float(
    os.environ.get("DPOS_TURNOVER", "0.5")
)  # fraction of gross per rebalance
REBALANCE = os.environ.get("DPOS_REBALANCE", "1w")
MIN_ADV = float(
    os.environ.get("DPOS_MIN_ADV", "1000000")
)  # in the panel's currency a day (euro, or pounds for the UK)
MIN_PRICE = float(os.environ.get("DPOS_MIN_PRICE", "1.0"))
TAG = os.environ.get(
    "DPOS_TAG", ""
)  # names the panel in the output folder ("" for the euro registers)
START = os.environ.get("DPOS_START", "2015-01-01")
END = os.environ.get("DPOS_END", "2026-10-02")
NO_COSTS = (
    os.environ.get("DPOS_NO_COSTS") == "1"
)  # the gross reference: no commission, impact or borrow


def build() -> Strategy:
    data = DataConfig(
        source=f"parquet://{(EXPORTS / 'register_prices_daily.parquet').as_posix()}",
        calendar="XETR",
        frequency="1d",
        start=START,
        end=END,
    )
    universe = Universe(
        name=f"{TAG or 'euro'}_register_issuers",
        filters=[
            HasHistory(min_days=126),
            MinPrice(min_price=MIN_PRICE),
            MinDollarADV(min_dollar_adv=MIN_ADV),
        ],
    )
    factors = {
        "s_bear": ExternalFactor(
            name="s_bear",
            path=(EXPORTS / f"{SIGNAL}.parquet").as_posix(),
            per_instrument=True,
        ),
        "s_bull": ExternalFactor(
            name="s_bull",
            path=(EXPORTS / f"{SIGNAL}_neg.parquet").as_posix(),
            per_instrument=True,
        ),
    }
    signals = {
        "bear": Identity(
            base="s_bear", name="bear"
        ),  # the exported rank is the score itself
        "bull": Identity(base="s_bull", name="bull"),
        "valid": MaskFromBoolean(name="valid", expr=NotNull("s_bear")),
        "short_candidates": MaskFromBoolean(
            name="short_candidates",
            expr=And(left="valid", right=GreaterEqual(left="bear", right=BOOK_EDGE)),
        ),
        "long_candidates": MaskFromBoolean(
            name="long_candidates",
            expr=And(left="valid", right=GreaterEqual(left="bull", right=BOOK_EDGE)),
        ),
    }
    portfolio = LongShortPortfolio(
        long_book=Book(
            name="long",
            selector=TopN(
                factor_name="bull",
                n=NAMES_PER_SIDE,
                mask_name="long_candidates",
                fill_from_unmasked=False,
            ),
            weighting=SignalWeight(signal_name="bull"),
        ),
        short_book=Book(
            name="short",
            selector=TopN(
                factor_name="bear",
                n=NAMES_PER_SIDE,
                mask_name="short_candidates",
                fill_from_unmasked=False,
            ),
            weighting=SignalWeight(signal_name="bear"),
        ),
        rebalance_frequency=REBALANCE,
        rebalance_at="market_close",
        signal_delay_bars=0,  # the external signal is already shown from the day after it is marked
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.04,
        turnover_limit=TurnoverLimit(window_bars=1, max_fraction=TURNOVER),
        min_names_per_book=MIN_NAMES_PER_BOOK,
    )
    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(
            base_bps=0.0 if NO_COSTS else 2.0, k=0.0 if NO_COSTS else 20.0, exponent=0.5
        ),
        volume_limits=VolumeParticipation(max_participation=0.1, mode="proportional"),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=0.0 if NO_COSTS else 1.0),
        borrow=BorrowCost(
            default_annual_rate=0.0 if NO_COSTS else 0.01,
            rates_path=(
                None
                if (NO_COSTS or not BORROW_RATES)
                else (EXPORTS / f"{BORROW_RATES}.parquet").as_posix()
            ),
        ),  # each short at the broker's rate for the name, the default for the rest
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    name = (
        f"dpos_register_{TAG + '_' if TAG else ''}{SIGNAL}_n{NAMES_PER_SIDE}_{REBALANCE}_{START[:4]}_{END[:4]}"
        + ("_nocost" if NO_COSTS else ("_rates" if BORROW_RATES else ""))
    )
    bt = BacktestConfig(
        engine="event_driven",
        cash_initial=10_000_000.0,
        risk_checks=RiskChecks(
            max_gross_leverage=2.5, drawdown=DrawdownPolicy(mode="none")
        ),
        reporting=Reporting(
            output_dir=f"outputs/{name}",
            store_trades=True,
            store_positions=True,
            strategyAnalytics=StrategyAnalyticsConfig(
                title=f"Disclosed positioning: register profit ({SIGNAL})"
            ),
        ),
        extra={"hold_when_no_targets": True},
    )
    return Strategy(
        name=name,
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=bt,
    )


if __name__ == "__main__":
    result = run_backtest(build())
    print(result.summary())
    sys.exit(0)
