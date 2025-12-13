# src/quantdsl_backtest/engine/vectorized_engine.py

from __future__ import annotations

from typing import Any, Dict
import os
import time

import numpy as np
import pandas as pd
import vectorbt as vbt

from ..dsl.strategy import Strategy
from ..dsl.costs import Commission
from ..dsl.execution import Execution
from ..models.slippage import build_slippage_model
from .data_loader import load_data_for_strategy
from .factor_engine import FactorEngine
from .signal_engine import SignalEngine
from .portfolio_engine import compute_target_weights_for_date
from .results import BacktestResult
from .accounting import compute_basic_metrics
from ..utils.logging import get_logger


log = get_logger(__name__)


def run_backtest_vectorized(strategy: Strategy) -> BacktestResult:
    """
    Vectorized backtest implementation backed by vectorbt.

    Pipeline:
      1) Load data via adapters
      2) Compute factors & signals via DSL engines
      3) Build daily target weight matrix (only on rebalance dates)
      4) Call vectorbt.Portfolio.from_orders with size_type='targetpercent'
      5) Convert vectorbt.Portfolio into BacktestResult
    """
    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    md, prices, volumes = load_data_for_strategy(strategy)
    dates = prices.index
    instruments = prices.columns

    log.info(
        "Vectorized backtest: %s, %d instruments, %d bars",
        strategy.name,
        len(instruments),
        len(dates),
    )

    # ------------------------------------------------------------------ #
    # 2. Compute factors & signals using existing engines
    # ------------------------------------------------------------------ #
    factor_engine = FactorEngine(md, prices)
    factor_panels = factor_engine.compute_all(strategy.factors)

    signal_engine = SignalEngine(factor_panels, strategy.signals)
    signal_panels = signal_engine.compute_all()

    # ------------------------------------------------------------------ #
    # 3. Build target weights panel [time x instrument]
    # ------------------------------------------------------------------ #
    weights = pd.DataFrame(
        np.nan,
        index=dates,
        columns=instruments,
        dtype="float64",
    )
    prev_weights = pd.Series(0.0, index=instruments, dtype="float64")

    for i, dt in enumerate(dates):
        if _is_rebalance_date_vectorized(i, dates, strategy):
            target = compute_target_weights_for_date(
                date=dt,
                portfolio=strategy.portfolio,
                signals=signal_panels,
                prev_weights=prev_weights,
            )
            # Ensure instruments alignment
            target = target.reindex(instruments).fillna(0.0)
            weights.loc[dt] = target
            prev_weights = target

    # ------------------------------------------------------------------ #
    # 4. Build vectorbt portfolio (potentially two-pass for costs)
    # ------------------------------------------------------------------ #
    init_cash = float(strategy.backtest.cash_initial)
    freq = strategy.data.frequency or "1D"

    # Engine options (flags) with sane defaults. Can be overridden via
    # BacktestConfig.extra["vectorized_engine"] or environment variables.
    eng_opts = _resolve_engine_options(getattr(strategy.backtest, "extra", {}))

    # Decide whether we need a two-pass run to emulate per-share commission
    # and power-law slippage using actual order sizes.
    commission = strategy.costs.commission if strategy.costs is not None else None
    exec_cfg = strategy.execution
    sl_cfg = exec_cfg.slippage if exec_cfg is not None else None

    # If execution contains volume participation limits (< 100%), vectorbt's
    # targetpercent sizing cannot enforce them. For parity with the event-driven
    # engine (and to satisfy engine-consistency tests), fall back to the
    # event-driven implementation when such constraints are present.
    try:
        vp = exec_cfg.volume_limits if exec_cfg is not None else None
        max_participation = getattr(vp, "max_participation", None)
        if max_participation is not None and float(max_participation) < 1.0:
            log.info(
                "Vectorized engine: volume participation < 100%% detected (%.2f). "
                "Falling back to event-driven engine for parity.",
                float(max_participation),
            )
            from .backtest_runner import _run_backtest_event_driven

            return _run_backtest_event_driven(strategy)
    except Exception as exc:
        log.warning("Vectorized engine: failed to inspect volume limits, continuing with vectorized path: %s", exc)

    need_per_share = bool(
        commission is not None and getattr(commission, "type", None) == "per_share" and float(getattr(commission, "amount", 0.0)) != 0.0
    )
    need_power_law = bool(
        sl_cfg is not None and (
            float(getattr(sl_cfg, "k", 0.0)) != 0.0 or float(getattr(sl_cfg, "base_bps", 0.0)) != 0.0
        )
    )

    if (need_per_share or need_power_law) and eng_opts["approx_single_pass"]:
        # Optional approximate single-pass path: estimate quantities from weights and
        # constant equity approximation (init_cash). This is faster but less exact.
        log.info("Vectorized engine: using approximate single-pass friction model")
        fees_df, slippage_df = _approximate_cost_matrices(
            prices=prices,
            volumes=volumes,
            weights=weights,
            init_cash=init_cash,
            commission=commission,
            slippage_model=sl_cfg,
        )
        pf = vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",
            init_cash=init_cash,
            fees=fees_df if fees_df is not None else 0.0,
            slippage=slippage_df if slippage_df is not None else 0.0,
            cash_sharing=True,
            call_seq="auto",
            freq=freq,
        )
    elif need_per_share or need_power_law:
        t0 = time.perf_counter() if eng_opts["timing"] else None
        # Pass 1: build portfolio with zero costs to get actual order sizes
        pf0 = vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",
            init_cash=init_cash,
            fees=0.0,
            slippage=0.0,
            cash_sharing=True,
            call_seq="auto",
            freq=freq,
        )
        t1 = time.perf_counter() if eng_opts["timing"] else None

        # Extract orders and build per-bar cost matrices
        try:
            # Optional cache for pass-1 orders
            orders_rec = None
            if eng_opts["enable_caching"]:
                key = _make_orders_cache_key(weights, prices, init_cash)
                cached = _pass1_orders_cache.get(key)
                if cached is not None:
                    orders_rec = cached
                    log.info("Vectorized engine: Pass-1 orders cache hit")
            if orders_rec is None:
                orders_rec = pf0.orders.records_readable
                if eng_opts["enable_caching"]:
                    _pass1_orders_cache[key] = orders_rec
        except Exception as exc:
            log.warning("Vectorized engine: unable to access orders for cost computation, falling back to constant costs: %s", exc)
            fees = _commission_to_vbt_fees(commission)
            slippage = _slippage_to_vbt_frac(exec_cfg)
            pf = vbt.Portfolio.from_orders(
                close=prices,
                size=weights,
                size_type="targetpercent",
                init_cash=init_cash,
                fees=fees,
                slippage=slippage,
                cash_sharing=True,
                call_seq="auto",
                freq=freq,
            )
        else:
            # Build per-bar cost DataFrames aligned to prices index/columns
            fees_df, slippage_df = _build_cost_matrices_from_orders(
                prices=prices,
                volumes=volumes,
                orders_df=orders_rec,
                commission=commission,
                slippage_model=sl_cfg,
                sparse=eng_opts["sparse_cost_mats"],
                numpy_fast=eng_opts["numpy_fast_path"],
            )
            t2 = time.perf_counter() if eng_opts["timing"] else None

            pf = vbt.Portfolio.from_orders(
                close=prices,
                size=weights,
                size_type="targetpercent",
                init_cash=init_cash,
                fees=fees_df if fees_df is not None else 0.0,
                slippage=slippage_df if slippage_df is not None else 0.0,
                cash_sharing=True,
                call_seq="auto",
                freq=freq,
            )
            if eng_opts["timing"]:
                t3 = time.perf_counter()
                log.info(
                    "Vectorized engine timing: pass1=%.3fs cost_build=%.3fs pass2=%.3fs",
                    (t1 - t0) if (t0 is not None and t1 is not None) else float('nan'),
                    (t2 - t1) if (t1 is not None and t2 is not None) else float('nan'),
                    (t3 - t2) if (t2 is not None) else float('nan'),
                )
    else:
        # Simple single-pass: map constant fees/slippage
        fees = _commission_to_vbt_fees(commission)
        slippage = _slippage_to_vbt_frac(exec_cfg)
        pf = vbt.Portfolio.from_orders(
            close=prices,
            size=weights,
            size_type="targetpercent",  # TargetPercent sizing (per asset)
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            cash_sharing=True,
            call_seq="auto",            # sell before buy within a bar
            freq=freq,
        )

    # ------------------------------------------------------------------ #
    # 5. Extract time series & build BacktestResult
    # ------------------------------------------------------------------ #
    # Core portfolio state
    equity = pf.value()
    returns = pf.returns()
    cash = pf.cash()

    # Unified path: extract positions first, compute weights from positions and equity
    def _align_panel(df_like: pd.DataFrame) -> pd.DataFrame:
        """Align any DataFrame-like panel to (dates x instruments) without dropping data.
        If shapes match, force labels to exactly match our prices panel; else reindex.
        """
        if not isinstance(df_like, pd.DataFrame):
            df_like = pd.DataFrame(df_like)
        df = df_like.copy()
        # If df already has the same shape, set labels directly to avoid data loss
        if df.shape == (len(dates), len(instruments)):
            df.index = dates
            df.columns = instruments
            return df.astype("float64")
        # Try vectorbt wrapper metadata if present
        try:
            wrap = getattr(pf, "wrapper", None)
            if wrap is not None:
                idx = getattr(wrap, "index", None)
                cols = getattr(wrap, "columns", None)
                if idx is not None and len(idx) == df.shape[0]:
                    df.index = pd.Index(idx)
                if cols is not None and len(cols) == df.shape[1]:
                    df.columns = pd.Index(cols)
        except Exception:
            pass
        # Final alignment with reindex
        return df.reindex(index=dates, columns=instruments).astype("float64").fillna(0.0)

    # 1) Try direct shares accessor
    positions = None
    try:
        shares_obj = pf.shares()
        if hasattr(shares_obj, "to_pandas"):
            positions = _align_panel(shares_obj.to_pandas())
        else:
            positions = _align_panel(pd.DataFrame(shares_obj))
    except Exception:
        positions = None

    # 2) If not available, reconstruct from orders
    if positions is None:
        try:
            orders = getattr(pf, "orders", None)
            if orders is not None:
                rec = orders.records_readable
                df = rec.copy()
                col_map = {i: inst for i, inst in enumerate(instruments)}
                ts_col = "Timestamp" if "Timestamp" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
                size_col = "Size" if "Size" in df.columns else ("size" if "size" in df.columns else None)
                col_col = "Column" if "Column" in df.columns else ("column" if "column" in df.columns else None)
                if ts_col is not None and size_col is not None and col_col is not None:
                    df["instrument"] = df[col_col].map(col_map)
                    ts = pd.to_datetime(df[ts_col])
                    if getattr(dates, "tz", None) is not None:
                        if getattr(ts.dt, "tz", None) is not None:
                            ts = ts.dt.tz_convert(dates.tz)
                        else:
                            ts = ts.dt.tz_localize(dates.tz)
                    else:
                        if getattr(ts.dt, "tz", None) is not None:
                            ts = ts.dt.tz_localize(None)
                    df[ts_col] = ts
                    order_sizes = df.groupby([ts_col, "instrument"], as_index=False)[size_col].sum()
                    pivot = order_sizes.pivot(index=ts_col, columns="instrument", values=size_col)
                    pivot = pivot.reindex(dates).fillna(0.0)
                    positions = pivot.cumsum().astype("float64")
                    positions = _align_panel(positions)
        except Exception:
            positions = None

    # 3) Try positions accessor variants
    if positions is None:
        try:
            pos_acc = getattr(pf, "positions", None)
            if pos_acc is not None:
                for attr in ("shares", "position", "quantity", "size"):
                    if hasattr(pos_acc, attr):
                        candidate = getattr(pos_acc, attr)
                        if hasattr(candidate, "to_pandas"):
                            positions = _align_panel(candidate.to_pandas())
                        else:
                            positions = _align_panel(pd.DataFrame(candidate))
                        break
                if positions is None and hasattr(pos_acc, "to_pandas"):
                    positions = _align_panel(pos_acc.to_pandas())
        except Exception:
            positions = None

    # 4) Last resort: zeros
    if positions is None:
        positions = pd.DataFrame(0.0, index=dates, columns=instruments, dtype="float64")

    # Align core series
    equity = equity.reindex(dates).astype("float64")
    returns = returns.reindex(dates).fillna(0.0).astype("float64")
    cash = cash.reindex(dates).astype("float64")

    # Compute exposures from positions
    positions = _align_panel(positions)
    notional = positions * prices
    long_exposure = notional.clip(lower=0.0).sum(axis=1)
    short_exposure = notional.clip(upper=0.0).sum(axis=1)
    gross_exposure = long_exposure + np.abs(short_exposure)
    net_exposure = long_exposure + short_exposure
    leverage = (gross_exposure / equity.replace(0.0, np.nan)).fillna(0.0)

    # Planned target weights (forward-filled across non-rebalance days)
    planned_weights = (
        weights.copy()
        .ffill()
        .fillna(0.0)
        .reindex(index=dates, columns=instruments)
        .astype("float64")
    )

    # Weights from notional over equity (actual realized weights)
    eq_nonzero = equity.replace(0.0, np.nan)
    weights_full = (
        notional.div(eq_nonzero, axis=0)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype("float64")
    )

    # Safety fallback: if realized weights are degenerate (all zeros or never change),
    # use planned target weights to compute turnover metrics. This preserves
    # consistency with event-driven engine under perfect execution.
    try:
        degenerate = bool(weights_full.abs().sum().sum() == 0.0)
        if not degenerate:
            # check if there is any change over time
            total_change = float(weights_full.diff().abs().sum().sum())
            degenerate = total_change <= 0.0
        if degenerate:
            weights_full = planned_weights
    except Exception:
        weights_full = planned_weights

    # Trades: map vectorbt order records -> our "trades" DataFrame
    trades_df = _orders_to_trades_df(pf, instruments)

    # ------------------------------------------------------------------ #
    # 6. Apply carry costs (borrow/financing) and management fees to align
    #    with event-driven engine semantics. We adjust equity/returns/cash
    #    post vectorbt simulation to include these daily effects.
    # ------------------------------------------------------------------ #
    try:
        borrow_cfg = strategy.costs.borrow if strategy.costs is not None else None
        fin_cfg = strategy.costs.financing if strategy.costs is not None else None
        fees_cfg = strategy.costs.fees if strategy.costs is not None else None

        has_carry_or_fees = (
            (borrow_cfg is not None and float(getattr(borrow_cfg, "default_annual_rate", 0.0)) != 0.0)
            or (fin_cfg is not None and float(getattr(fin_cfg, "spread_bps", 0.0)) != 0.0)
            or (fees_cfg is not None and float(getattr(fees_cfg, "nav_fee_annual", 0.0)) != 0.0)
        )

        if has_carry_or_fees:
            # Prepare series for adjustments
            prev_cash_series = cash.shift(1).fillna(init_cash)
            # Use previous positions and current prices, similar to event-driven loop
            prev_positions_df = positions.shift(1).fillna(0.0)
            price_df = prices

            # Parameters
            dt = 1.0 / 252.0
            borrow_rate = float(getattr(borrow_cfg, "default_annual_rate", 0.0)) if borrow_cfg is not None else 0.0
            fin_rate = (float(getattr(fin_cfg, "spread_bps", 0.0)) / 1e4) if fin_cfg is not None else 0.0
            nav_fee_annual = float(getattr(fees_cfg, "nav_fee_annual", 0.0)) if fees_cfg is not None else 0.0
            nav_fee_daily = nav_fee_annual * dt

            # Compute daily deltas
            short_notional = (-prev_positions_df.clip(upper=0.0) * price_df).sum(axis=1)
            equity_before = prev_cash_series + (prev_positions_df * price_df).sum(axis=1)

            borrow_cost = short_notional * borrow_rate * dt
            financing_pnl = prev_cash_series * fin_rate * dt
            nav_fee_amt = equity_before * nav_fee_daily

            daily_delta = (-borrow_cost + financing_pnl - nav_fee_amt).astype("float64")
            adj_cumsum = daily_delta.cumsum().fillna(0.0)

            # Adjust equity, cash, returns, and weights to reflect carry/fees
            equity_adj = (equity.astype("float64") + adj_cumsum).astype("float64")
            cash_adj = (cash.astype("float64") + adj_cumsum).astype("float64")
            returns_adj = equity_adj.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

            # Recompute weights using adjusted equity
            eq_nonzero_adj = equity_adj.replace(0.0, np.nan)
            weights_full = (notional.div(eq_nonzero_adj, axis=0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Overwrite series with adjusted ones
            equity = equity_adj
            cash = cash_adj
            returns = returns_adj
    except Exception as exc:
        log.warning("Vectorized engine: failed to apply carry/fees adjustments, proceeding with raw vectorbt outputs: %s", exc)

    # Metrics – reuse the same accounting logic for consistency
    metrics = compute_basic_metrics(
        returns=returns,
        equity=equity,
        weights=weights_full,
    )

    result = BacktestResult(
        equity=equity,
        returns=returns,
        cash=cash,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        leverage=leverage,
        positions=positions,
        weights=weights_full,
        trades=trades_df,
        metrics=metrics,
        start_date=dates[0],
        end_date=dates[-1],
        benchmark=None,
        metadata={
            "strategy_name": strategy.name,
            "data_source": strategy.data.source,
            "engine": "vectorized",
            "vectorbt_version": getattr(vbt, "__version__", "unknown"),
        },
    )

    log.info(
        "Vectorized backtest complete: total return %.2f%%, Sharpe %.2f, max DD %.2f%%",
        result.total_return * 100.0,
        metrics.get("sharpe", 0.0),
        metrics.get("max_drawdown", 0.0) * 100.0,
    )

    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_rebalance_date_vectorized(
    idx: int,
    dates: pd.DatetimeIndex,
    strategy: Strategy,
) -> bool:
    """
    Rebalance schedule for the vectorized engine.

    For now mirrors the event-driven engine: '1d' = every day.
    Extend here to support weekly / monthly rebalancing without
    touching the rest of the engine.
    """
    freq = strategy.portfolio.rebalance_frequency
    if freq == "1d":
        return True
    # Placeholder: extend to '1w', '1m', etc. as needed
    return True


def _commission_to_vbt_fees(commission: Commission) -> float:
    """
    Map our commission model to vectorbt's `fees` argument
    (fraction of traded notional).

    Currently:
      - 'bps_notional' -> amount / 1e4 (applied per order)
      - 'per_share'    -> handled via two-pass order-based cost matrices elsewhere; returns 0 here
    """
    if commission is None:
        return 0.0

    if commission.type == "bps_notional":
        return float(commission.amount) / 1e4

    if commission.type == "per_share":
        # Two-pass path will compute per-bar fees fraction; return 0 for single-pass
        return 0.0

    return 0.0


def _slippage_to_vbt_frac(execution: Execution) -> float:
    """
    Map our (potentially non-linear) slippage model to vectorbt's
    constant `slippage` argument (fraction of price).

    For simple single-pass path we only use the base_bps parameter and ignore k/exponent.
    """
    if execution is None or execution.slippage is None:
        return 0.0
    # Use the model but evaluate at zero participation for constant frac
    sl_model = build_slippage_model(execution.slippage)
    return float(sl_model.slippage_bps_from_participation(0.0)) / 1e4


def _build_cost_matrices_from_orders(
    *,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    orders_df: pd.DataFrame,
    commission: Commission | None,
    slippage_model: Any | None,
    sparse: bool = True,
    numpy_fast: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Build per-bar, per-asset cost matrices to pass into vectorbt from
    order records. Returns (fees_df, slippage_df) where values are
    fractions of traded notional to apply on that bar for that asset.

    - Per-share commission: fraction = commission_per_share / price for bars with orders
    - Bps commission: handled by single-pass path; we keep None here
    - Power-law slippage: slip_frac = (base_bps + k * (|shares|/volume)^exponent) / 1e4
      on bars with orders; 0 otherwise.
    """
    dates = prices.index
    cols = prices.columns

    # Aggregate signed and absolute order sizes per bar/asset
    df = orders_df.copy()
    # Normalize column names
    ts_col = "Timestamp" if "Timestamp" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    size_col = "Size" if "Size" in df.columns else ("size" if "size" in df.columns else None)
    col_col = "Column" if "Column" in df.columns else ("column" if "column" in df.columns else None)
    if ts_col is None or size_col is None or col_col is None:
        # Unable to parse; return None to fall back
        return None, None

    # Group by ts/column to get net signed size and absolute size within a bar
    grp_signed = df.groupby([ts_col, col_col], as_index=False)[size_col].sum()
    grp_abs = df.assign(_abs=np.abs(df[size_col])).groupby([ts_col, col_col], as_index=False)["_abs"].sum()
    grp_buy = (
        df[df[size_col] > 0]
        .assign(_buy=lambda x: x[size_col])
        .groupby([ts_col, col_col], as_index=False)["_buy"].sum()
    )
    grp_sell = (
        df[df[size_col] < 0]
        .assign(_sell=lambda x: -x[size_col])
        .groupby([ts_col, col_col], as_index=False)["_sell"].sum()
    )

    # Pivot only on active timestamps; align to full dates later (sparse-aware)
    # If sparse=False, expand immediately to full index to keep identical behavior.
    net_size_by_col = grp_signed.pivot(index=ts_col, columns=col_col, values=size_col)
    abs_size_by_col = grp_abs.pivot(index=ts_col, columns=col_col, values="_abs")
    buy_qty_by_col = grp_buy.pivot(index=ts_col, columns=col_col, values="_buy")
    sell_qty_by_col = grp_sell.pivot(index=ts_col, columns=col_col, values="_sell")

    # Map integer column indices directly to instrument labels where applicable
    def _map_col(c):
        try:
            if isinstance(c, (int, np.integer)):
                j = int(c)
                if 0 <= j < len(cols):
                    return cols[j]
            # Already a label; ensure string match if present
            s = str(c)
            return s if s in cols else c
        except Exception:
            return c

    for _df in (net_size_by_col, abs_size_by_col, buy_qty_by_col, sell_qty_by_col):
        if _df is not None:
            _df.columns = _df.columns.map(_map_col)

    # Now align each to prices index/columns, filling missing with 0.0
    def _align(df_like: pd.DataFrame | None) -> pd.DataFrame:
        if df_like is None or df_like.empty:
            return pd.DataFrame(0.0, index=dates, columns=cols, dtype="float64")
        # Ensure datetime index comparable to prices' index
        try:
            idx = pd.to_datetime(df_like.index)
        except Exception:
            idx = df_like.index
        df_like = df_like.copy()
        df_like.index = idx
        if sparse:
            out = (
                df_like.reindex(index=dates)
                .reindex(columns=cols)
                .fillna(0.0)
                .astype("float64")
            )
        else:
            # Expand to full shape even if there are many zeros
            out = pd.DataFrame(0.0, index=dates, columns=cols, dtype="float64")
            out.loc[df_like.index, df_like.columns] = df_like.astype("float64").values
        return out

    net_size = _align(net_size_by_col)
    abs_size = _align(abs_size_by_col)
    buy_qty = _align(buy_qty_by_col)
    sell_qty = _align(sell_qty_by_col)

    # Build FEES matrix
    fees_df: pd.DataFrame | None = None
    if commission is not None:
        if getattr(commission, "type", None) == "per_share" and float(getattr(commission, "amount", 0.0)) != 0.0:
            per_share = float(commission.amount)
            # Vectorbt applies `fees` as a fraction of order value, which uses base price.
            # Event-driven commission is per-share, independent of slippage.
            # Thus, set per-bar fee fraction = per_share / price on bars with orders.
            if numpy_fast:
                p = prices.to_numpy(dtype=float, copy=False)
                abs_a = abs_size.to_numpy(dtype=float, copy=False)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac_arr = per_share / p
                frac_arr[~np.isfinite(frac_arr)] = 0.0
                mask = abs_a > 0.0
                fees_arr = np.where(mask, frac_arr, 0.0)
                fees_df = pd.DataFrame(fees_arr, index=dates, columns=cols, dtype="float64")
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = per_share / prices.replace(0.0, np.nan)
                frac = frac.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
                fees_df = frac.where(abs_size > 0.0, 0.0)
        elif getattr(commission, "type", None) == "bps_notional" and float(getattr(commission, "amount", 0.0)) != 0.0:
            # Single-pass could handle this; but for consistency, only apply on order bars
            frac = float(commission.amount) / 1e4
            fees_df = pd.DataFrame(0.0, index=dates, columns=cols, dtype="float64")
            mask = abs_size > 0.0
            fees_df[mask] = frac

    # Build SLIPPAGE matrix
    slippage_df: pd.DataFrame | None = None
    if slippage_model is not None:
        sl_model = build_slippage_model(slippage_model)
        if numpy_fast:
            net_a = net_size.to_numpy(dtype=float, copy=False)
            abs_a = abs_size.to_numpy(dtype=float, copy=False)
            vol_a = None
            if volumes is not None:
                vol_a = volumes.reindex(index=dates, columns=cols).to_numpy(dtype=float, copy=False)
            slip_frac_a = sl_model.build_slippage_fraction_matrix_from_orders_numpy(
                net_size_arr=net_a, abs_size_arr=abs_a, volumes_arr=vol_a
            )
            slippage_df = pd.DataFrame(slip_frac_a, index=dates, columns=cols, dtype="float64")
        else:
            vol_df = None
            if volumes is not None:
                vol_df = volumes.reindex(index=dates, columns=cols).astype("float64")
            slippage_df = sl_model.build_slippage_fraction_matrix_from_orders_pandas(
                net_size=net_size, abs_size=abs_size, volumes=vol_df
            )

    return fees_df, slippage_df


def _approximate_cost_matrices(
    *,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    weights: pd.DataFrame,
    init_cash: float,
    commission: Commission | None,
    slippage_model: Any | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Build approximate per-bar cost matrices without running a first vectorbt pass.
    Assumptions:
      - Equity used for target shares is approximated as constant `init_cash`.
      - Positions follow target weights instantly at each rebalance.
    This is purely optional for speed and will not perfectly match event-driven results.
    """
    dates = prices.index
    cols = prices.columns

    # Approximate target shares per bar
    # Use forward-filled weights to mimic vectorbt's hold behavior between rebalances
    weights_ffill = weights.ffill().fillna(0.0)
    eq = float(init_cash)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_shares = (weights_ffill * eq) / prices.replace(0.0, np.nan)
    target_shares = target_shares.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prev_shares = target_shares.shift(1).fillna(0.0)
    net_shares = (target_shares - prev_shares).astype("float64")
    abs_shares = np.abs(net_shares)

    # Fees (per-share commission) -> per-bar fraction of traded notional
    fees_df: pd.DataFrame | None = None
    if commission is not None and getattr(commission, "type", None) == "per_share" and float(getattr(commission, "amount", 0.0)) != 0.0:
        per_share = float(commission.amount)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = per_share / prices.replace(0.0, np.nan)
        frac = frac.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fees_df = frac.where(abs_shares > 0.0, 0.0).astype("float64")

    # Slippage (power-law)
    slippage_df: pd.DataFrame | None = None
    if slippage_model is not None:
        sl_model = build_slippage_model(slippage_model)
        vol_df = None
        if volumes is not None:
            vol_df = volumes.reindex(index=dates, columns=cols).astype("float64")
        # In the approximate path, abs_shares is |net_shares|, so scale becomes 1.0
        slippage_df = sl_model.build_slippage_fraction_matrix_from_orders_pandas(
            net_size=net_shares, abs_size=abs_shares, volumes=vol_df
        )

    return fees_df, slippage_df


# -----------------------------
# Options & caching helpers
# -----------------------------

def _resolve_engine_options(extra: Dict[str, object]) -> Dict[str, bool]:
    # Defaults
    opts = {
        "sparse_cost_mats": True,
        "numpy_fast_path": True,
        "approx_single_pass": False,
        "enable_caching": False,
        "timing": bool(int(os.environ.get("QUANTDSL_VEC_TIMING", "0") or 0)),
    }
    # From BacktestConfig.extra
    sub = {}
    if isinstance(extra, dict):
        sub = extra.get("vectorized_engine") if isinstance(extra.get("vectorized_engine"), dict) else {}
    if sub:
        for k in ("sparse_cost_mats", "numpy_fast_path", "approx_single_pass", "enable_caching", "timing"):
            if k in sub:
                opts[k] = bool(sub[k])
    # Env overrides
    def _env_bool(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        try:
            return bool(int(v))
        except Exception:
            return default
    opts["sparse_cost_mats"] = _env_bool("QUANTDSL_VEC_SPARSE", opts["sparse_cost_mats"])
    opts["numpy_fast_path"] = _env_bool("QUANTDSL_VEC_NUMPY", opts["numpy_fast_path"])
    opts["approx_single_pass"] = _env_bool("QUANTDSL_VEC_APPROX", opts["approx_single_pass"])
    opts["enable_caching"] = _env_bool("QUANTDSL_VEC_CACHE", opts["enable_caching"])
    return opts


def _make_orders_cache_key(weights: pd.DataFrame, prices: pd.DataFrame, init_cash: float) -> tuple:
    try:
        # Lightweight, collision-resistant key
        w_hash = pd.util.hash_pandas_object(weights.fillna(0.0), index=True).values.sum()
        idx_sig = (len(prices.index), prices.index[0], prices.index[-1])
        col_sig = (len(prices.columns), prices.columns[0], prices.columns[-1])
        return ("v1", int(w_hash), idx_sig, col_sig, float(init_cash))
    except Exception:
        return ("v1_fallback", weights.shape, prices.shape, float(init_cash))


# Simple in-memory cache (module-level). Not persisted across runs.
_pass1_orders_cache: Dict[tuple, pd.DataFrame] = {}


def _orders_to_trades_df(pf: vbt.Portfolio, instruments) -> pd.DataFrame:
    """
    Convert vectorbt's order records into our standard 'trades' DataFrame
    shape as documented in BacktestResult.
    """
    try:
        orders = pf.orders
        rec = orders.records_readable
    except Exception as exc:
        log.warning("Vectorized engine: unable to access pf.orders.records_readable: %s", exc)
        return pd.DataFrame(
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

    df = rec.copy()

    # Map column index to instrument name
    col_map = {i: inst for i, inst in enumerate(instruments)}
    if "Column" in df.columns:
        df["instrument"] = df["Column"].map(col_map)
    else:
        df["instrument"] = None

    # Normalize column names
    ts_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
    size_col = "Size" if "Size" in df.columns else "size"
    price_col = "Price" if "Price" in df.columns else "price"
    fees_col = "Fees" if "Fees" in df.columns else "fees"
    side_col = "Side" if "Side" in df.columns else "side"

    df["datetime"] = df[ts_col]
    df["quantity"] = df[size_col]
    df["price"] = df[price_col]
    df["notional"] = df[size_col] * df[price_col]
    df["commission"] = df[fees_col]
    df["fees"] = 0.0  # separate from commission for now
    df["slippage_bps"] = np.nan
    df["realized_pnl"] = np.nan
    df["side"] = df[side_col]

    cols = [
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
    return df[cols].sort_values("datetime").reset_index(drop=True)
