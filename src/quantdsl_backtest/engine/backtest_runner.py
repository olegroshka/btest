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


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run_backtest(strategy: Strategy) -> BacktestResult:
    """
    Dispatch to the appropriate engine implementation based on
    `strategy.backtest.engine`.

    Supported values:
      - "event_driven" : custom daily event loop (current implementation)
      - "vectorized"   : vectorbt-backed engine in `vectorized_engine.py`
    """
    engine_name = getattr(strategy.backtest, "engine", "event_driven")

    if engine_name == "event_driven":
        return _run_backtest_event_driven(strategy)

    if engine_name == "vectorized":
        from .vectorized_engine import run_backtest_vectorized

        log.info("Running strategy %s with vectorized engine", strategy.name)
        return run_backtest_vectorized(strategy)

    raise ValueError(f"Unknown backtest engine: {engine_name!r}")


# --------------------------------------------------------------------------- #
# Existing event-driven engine (unchanged logic)
# --------------------------------------------------------------------------- #


def _run_backtest_event_driven(strategy: Strategy) -> BacktestResult:
    """
    Run an event-driven daily backtest for the given Strategy.

    This is the original implementation which we keep intact for:
      - fine-grained execution control
      - custom slippage / volume / latency models
      - easier debugging
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

    # --- Risk checks context ---
    risk = strategy.backtest.risk_checks
    peak_equity = float(strategy.backtest.cash_initial)
    cooldown_days = 0  # simple "no-risk" cooldown counter
    trading_halted = False  # terminal kill-switch state

    # ------------------------------------------------------------------ #
    # 4. Main daily loop
    # ------------------------------------------------------------------ #
    for i, dt in enumerate(dates):
        price_t = prices.loc[dt]
        volume_t = volumes.loc[dt]

        # Use effective prices for valuation/P&L: if current price is missing (holiday),
        # carry forward the previous price to avoid artificial PnL spikes.
        price_t_eff = price_t.reindex(prev_positions.index)
        price_t_eff = price_t_eff.where(~price_t_eff.isna(), prev_prices)

        if i == 0:
            # Day 0: no prior PnL, just initialize equity
            equity_before = cash + (prev_positions * price_t_eff).sum()
            price_pnl = 0.0
        else:
            equity_before, price_pnl = mark_to_market(
                prev_positions=prev_positions,
                prev_prices=prev_prices,
                curr_prices=price_t_eff,
                prev_cash=cash,
            )

        # Apply carry costs (borrow + financing)
        cash, borrow_cost, fin_pnl = apply_carry_costs(
            positions=prev_positions,
            prices=price_t_eff,
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
        equity_before = cash + (prev_positions * price_t_eff).sum()

        # Decide if we rebalance today
        do_rebalance = _is_rebalance_date(i, dates, strategy.portfolio)

        trades_today = pd.DataFrame()
        # --- Pre-trade drawdown vs running peak (use equity_before) ---
        dd_pretrade = 0.0 if peak_equity == 0 else (equity_before / peak_equity - 1.0)
        dd_mag = -dd_pretrade if dd_pretrade < 0 else 0.0

        # Determine drawdown policy (backward-compatible):
        # If new policy present, use it; else, if max_drawdown set, treat as hard_kill threshold.
        policy = getattr(risk, "drawdown", None)
        policy_mode = getattr(policy, "mode", None) if policy is not None else None
        # Back-compat mapping
        if policy is None and getattr(risk, "max_drawdown", None) is not None:
            policy_mode = "hard_kill"
            policy_threshold = float(getattr(risk, "max_drawdown"))
        else:
            policy_threshold = float(getattr(policy, "threshold", 0.0) or 0.0)

        # Compute scaling according to policy
        scale = 1.0
        if policy_mode == "hard_kill":
            if dd_mag >= (policy_threshold or 0.0):
                trading_halted = True
                scale = 0.0
        elif policy_mode == "soft_scale":
            start = float(getattr(policy, "start", 0.1) or 0.1)
            full = float(getattr(policy, "full", 0.35) or 0.35)
            curve = getattr(policy, "curve", "linear") or "linear"
            if full <= start:
                # guard: if misconfigured, treat as immediate flat beyond start
                full = start
            if dd_mag <= start:
                scale = 1.0
            elif dd_mag >= full:
                scale = 0.0
            else:
                x = (dd_mag - start) / (full - start)  # in (0,1)
                if curve == "quadratic":
                    scale = 1.0 - x * x
                elif curve == "sqrt":
                    scale = 1.0 - np.sqrt(x)
                else:  # linear
                    scale = 1.0 - x
        else:
            # "none" or not set: keep scale 1.0
            pass

        if do_rebalance:
            target_weights = compute_target_weights_for_date(
                date=dt,
                portfolio=strategy.portfolio,
                signals=signal_panels,
                prev_weights=weights_df.iloc[i - 1] if i > 0 else pd.Series(0.0, index=instruments),
            )

            # Apply drawdown policy scaling / halting
            if trading_halted:
                target_weights = pd.Series(0.0, index=target_weights.index)
            elif scale < 1.0:
                target_weights = target_weights * scale

            new_positions, cash_delta, trades_today = rebalance_to_target_weights(
                    date=dt,
                    execution=strategy.execution,
                    commission=strategy.costs.commission,
                    fees=strategy.costs.fees,
                    equity=equity_before,
                    prices=price_t_eff,
                    volumes=volume_t,
                    prev_positions=prev_positions,
                    target_weights=target_weights,
                )

            cash += cash_delta
            cur_positions = new_positions
        else:
            cur_positions = prev_positions

        # Final equity at end of day t
        equity = cash + (cur_positions * price_t_eff).sum()

        if i == 0:
            ret = 0.0
        else:
            ret = (equity / prev_equity) - 1.0 if prev_equity != 0 else 0.0

        # --- Update peak equity & drawdown (before exposures/storage) ---
        if equity > peak_equity:
            peak_equity = equity
        drawdown = 0.0 if peak_equity == 0 else (equity / peak_equity - 1.0)

        # --- Risk checks: drawdown policy informational logging ---
        if policy_mode == "hard_kill" and trading_halted:
            print(
                f"[RISK] Kill-switch: DD {dd_mag: .2%} >= {policy_threshold: .2%} on {dt.date()}, halting trading and staying in cash."
            )

        # --- Risk checks: max_daily_loss + cooldown ---
        if getattr(risk, "max_daily_loss", None) is not None:
            if ret <= -risk.max_daily_loss:
                cooldown_days = max(cooldown_days, 5)  # stay flat for 5 days

        if cooldown_days > 0:
            cooldown_days -= 1
            # Override: ensure we end the day flat & no new positions next loop
            cur_positions = pd.Series(0.0, index=instruments)
            cash = equity

        # Exposures
        exps = compute_exposures(cur_positions, price_t_eff)
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
            weights_df.iloc[i] = (cur_positions * price_t_eff) / equity
        else:
            weights_df.iloc[i] = 0.0

        if not trades_today.empty:
            all_trades.append(trades_today)

        prev_prices = price_t_eff
        prev_positions = cur_positions
        prev_equity = equity

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame(
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
        metadata={
            "strategy_name": strategy.name,
            "data_source": strategy.data.source,
            "engine": "event_driven",
        },
    )

    log.info(
        "Backtest complete (event_driven): total return %.2f%%, Sharpe %.2f, max DD %.2f%%",
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
    Simple daily rebalance logic for now. Need to extend to weekly/monthly.
    """
    freq = portfolio.rebalance_frequency
    if freq == "1d":
        return True
    # For now just do daily; you can extend later.
    return True
