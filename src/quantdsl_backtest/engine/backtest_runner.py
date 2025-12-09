# src/quantdsl_backtest/engine/backtest_runner.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..dsl.strategy import Strategy
from ..engine.results import BacktestResult
from ..utils.logging import get_logger
from .data_loader import load_data_for_strategy
from .factor_engine import FactorEngine
from .signal_engine import SignalEngine
from .portfolio_engine import compute_target_weights_for_date
from .execution_engine import rebalance_to_target_weights
from .accounting import (
    mark_to_market,
    apply_carry_costs,
    compute_exposures,
    compute_basic_metrics,
)


log = get_logger(__name__)


def run_backtest(strategy: Strategy) -> BacktestResult:
    """
    Run an event-driven daily backtest for the given Strategy.
    """

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    md, prices, volumes = load_data_for_strategy(strategy)
    dates = prices.index
    instruments = prices.columns

    # ------------------------------------------------------------------ #
    # 2. Compute factors & signals
    # ------------------------------------------------------------------ #
    factor_engine = FactorEngine(md, prices)
    factor_panels = factor_engine.compute_all(strategy.factors)

    signal_engine = SignalEngine(factor_panels, strategy.signals)
    signal_panels = signal_engine.compute_all()

    # ------------------------------------------------------------------ #
    # 3. Initialize state containers
    # ------------------------------------------------------------------ #
    n = len(dates)

    equity_series = pd.Series(index=dates, dtype="float64")
    return_series = pd.Series(index=dates, dtype="float64")
    cash_series = pd.Series(index=dates, dtype="float64")
    gross_series = pd.Series(index=dates, dtype="float64")
    net_series = pd.Series(index=dates, dtype="float64")
    long_series = pd.Series(index=dates, dtype="float64")
    short_series = pd.Series(index=dates, dtype="float64")
    lev_series = pd.Series(index=dates, dtype="float64")

    positions_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)
    weights_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)

    all_trades = []

    init_cash = strategy.backtest.cash_initial
    cash = float(init_cash)
    prev_positions = pd.Series(0.0, index=instruments, dtype="float64")
    prev_prices = prices.iloc[0].ffill()
    prev_equity = cash

    # ------------------------------------------------------------------ #
    # 4. Main daily loop
    # ------------------------------------------------------------------ #
    for i, dt in enumerate(dates):
        price_t = prices.loc[dt]
        volume_t = volumes.loc[dt]

        if i == 0:
            # Day 0: no prior PnL, just initialize equity
            equity_before = cash + (prev_positions * price_t).sum()
            price_pnl = 0.0
        else:
            equity_before, price_pnl = mark_to_market(
                prev_positions=prev_positions,
                prev_prices=prev_prices,
                curr_prices=price_t,
                prev_cash=cash,
            )

        # Apply carry costs (borrow + financing)
        cash, borrow_cost, fin_pnl = apply_carry_costs(
            positions=prev_positions,
            prices=price_t,
            cash=cash,
            borrow=strategy.costs.borrow,
            financing=strategy.costs.financing,
        )

        # Optionally management fees (we'll apply continuously on equity_before)
        nav_fee = strategy.costs.fees.nav_fee_annual
        if nav_fee > 0:
            nav_fee_daily = nav_fee / 252.0
            fee_amt = equity_before * nav_fee_daily
            cash -= fee_amt

        # Equity before trades at today's prices
        equity_before = cash + (prev_positions * price_t).sum()

        # Decide if we rebalance today
        do_rebalance = _is_rebalance_date(i, dates, strategy.portfolio)

        trades_today = pd.DataFrame()
        if do_rebalance:
            target_weights = compute_target_weights_for_date(
                date=dt,
                portfolio=strategy.portfolio,
                signals=signal_panels,
                prev_weights=weights_df.iloc[i - 1] if i > 0 else pd.Series(0.0, index=instruments),
            )

            new_positions, cash_delta, trades_today = rebalance_to_target_weights(
                date=dt,
                execution=strategy.execution,
                commission=strategy.costs.commission,
                fees=strategy.costs.fees,
                equity=equity_before,
                prices=price_t,
                volumes=volume_t,
                prev_positions=prev_positions,
                target_weights=target_weights,
            )

            cash += cash_delta
            cur_positions = new_positions
        else:
            cur_positions = prev_positions

        # Final equity at end of day t
        equity = cash + (cur_positions * price_t).sum()

        if i == 0:
            ret = 0.0
        else:
            ret = (equity / prev_equity) - 1.0 if prev_equity != 0 else 0.0

        # Exposures
        exps = compute_exposures(cur_positions, price_t)
        gross = exps["gross_exposure"]
        net = exps["net_exposure"]
        long_exp = exps["long_exposure"]
        short_exp = exps["short_exposure"]
        lev = gross / equity if equity > 0 else 0.0

        # Store
        equity_series.iloc[i] = equity
        return_series.iloc[i] = ret
        cash_series.iloc[i] = cash
        gross_series.iloc[i] = gross
        net_series.iloc[i] = net
        long_series.iloc[i] = long_exp
        short_series.iloc[i] = short_exp
        lev_series.iloc[i] = lev

        positions_df.iloc[i] = cur_positions
        if equity != 0:
            weights_df.iloc[i] = (cur_positions * price_t) / equity
        else:
            weights_df.iloc[i] = 0.0

        if not trades_today.empty:
            all_trades.append(trades_today)

        prev_prices = price_t
        prev_positions = cur_positions
        prev_equity = equity

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(
        columns=[
            "datetime",
            "instrument",
            "side",
            "quantity",
            "price",
            "notional",
            "slippage_bps",
            "commission",
            "fees",
            "realized_pnl",
        ]
    )

    # ------------------------------------------------------------------ #
    # 5. Metrics & BacktestResult
    # ------------------------------------------------------------------ #
    metrics = compute_basic_metrics(return_series, equity_series, weights_df)

    result = BacktestResult(
        equity=equity_series,
        returns=return_series,
        cash=cash_series,
        gross_exposure=gross_series,
        net_exposure=net_series,
        long_exposure=long_series,
        short_exposure=short_series,
        leverage=lev_series,
        positions=positions_df,
        weights=weights_df,
        trades=trades_df,
        metrics=metrics,
        start_date=dates[0],
        end_date=dates[-1],
        benchmark=None,
        metadata={"strategy_name": strategy.name, "data_source": strategy.data.source},
    )

    log.info(
        "Backtest complete: total return %.2f%%, Sharpe %.2f, max DD %.2f%%",
        result.total_return * 100.0,
        metrics.get("sharpe", 0.0),
        metrics.get("max_drawdown", 0.0) * 100.0,
    )

    return result


def _is_rebalance_date(
    idx: int,
    dates: pd.DatetimeIndex,
    portfolio,
) -> bool:
    """
    Simple daily rebalance logic for now. You can extend to weekly/monthly.
    """
    freq = portfolio.rebalance_frequency
    if freq == "1d":
        return True
    # For now just do daily; you can extend later.
    return True
