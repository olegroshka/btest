# src/quantdsl_backtest/engine/vectorized_engine.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import vectorbt as vbt

from ..dsl.strategy import Strategy
from ..dsl.costs import Commission
from ..dsl.execution import Execution
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

    # Decide whether we need a two-pass run to emulate per-share commission
    # and power-law slippage using actual order sizes.
    commission = strategy.costs.commission if strategy.costs is not None else None
    exec_cfg = strategy.execution
    sl_cfg = exec_cfg.slippage if exec_cfg is not None else None

    need_per_share = bool(
        commission is not None and getattr(commission, "type", None) == "per_share" and float(getattr(commission, "amount", 0.0)) != 0.0
    )
    need_power_law = bool(
        sl_cfg is not None and (
            float(getattr(sl_cfg, "k", 0.0)) != 0.0 or float(getattr(sl_cfg, "base_bps", 0.0)) != 0.0
        )
    )

    if need_per_share or need_power_law:
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

        # Extract orders and build per-bar cost matrices
        try:
            orders_rec = pf0.orders.records_readable
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

    # Positions (shares) over time – build robustly across vectorbt versions
    positions = None
    # Prefer reconstructing from orders to avoid wrapper objects
    try:
        orders = getattr(pf, "orders", None)
        if orders is not None:
            rec = orders.records_readable
            df = rec.copy()
            # Normalize column names and map column index to instrument
            col_map = {i: inst for i, inst in enumerate(instruments)}
            ts_col = "Timestamp" if "Timestamp" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
            size_col = "Size" if "Size" in df.columns else ("size" if "size" in df.columns else None)
            col_col = "Column" if "Column" in df.columns else ("column" if "column" in df.columns else None)
            if ts_col is not None and size_col is not None and col_col is not None:
                df["instrument"] = df[col_col].map(col_map)
                # Harmonize timezone with price index
                ts = pd.to_datetime(df[ts_col])
                if hasattr(dates, "tz") and dates.tz is not None:
                    # Make ts tz-aware in the same tz
                    if getattr(ts.dt, "tz", None) is not None:
                        ts = ts.dt.tz_convert(dates.tz)
                    else:
                        ts = ts.dt.tz_localize(dates.tz)
                else:
                    # Make ts naive
                    if getattr(ts.dt, "tz", None) is not None:
                        ts = ts.dt.tz_localize(None)
                df[ts_col] = ts
                # Sum order sizes per date/instrument
                order_sizes = (
                    df.groupby([ts_col, "instrument"], as_index=False)[size_col]
                    .sum()
                )
                # Pivot to daily net size per instrument, reindex to trading calendar
                pivot = (
                    order_sizes.pivot(index=ts_col, columns="instrument", values=size_col)
                    .reindex(dates)
                    .fillna(0.0)
                )
                positions = pivot.cumsum().astype("float64")
    except Exception:
        positions = None

    # If still not available, try vectorbt helpers
    if positions is None:
        # 1) Newer vectorbt often exposes `pf.shares()`
        try:
            cand = pf.shares()  # may raise or return wrapper
            if hasattr(cand, "to_pandas"):
                positions = cand.to_pandas()
            else:
                # Try pandas conversion
                positions = pd.DataFrame(cand)
        except Exception:
            positions = None

    if positions is None:
        # 2) Accessor with various attribute names
        try:
            pos_acc = getattr(pf, "positions", None)
            if pos_acc is not None:
                for attr in ("shares", "position", "quantity", "size"):
                    if hasattr(pos_acc, attr):
                        candidate = getattr(pos_acc, attr)
                        if hasattr(candidate, "to_pandas"):
                            positions = candidate.to_pandas()
                        else:
                            positions = pd.DataFrame(candidate)
                        break
                if positions is None and hasattr(pos_acc, "to_pandas"):
                    positions = pos_acc.to_pandas()
        except Exception:
            positions = None

    if positions is None:
        # Last resort: zero positions with correct shape
        positions = pd.DataFrame(0.0, index=dates, columns=instruments, dtype="float64")

    # Align everything to same index/columns
    equity = equity.reindex(dates).astype("float64")
    returns = returns.reindex(dates).fillna(0.0).astype("float64")
    cash = cash.reindex(dates).astype("float64")
    # Align to our price panel and ensure DataFrame dtype
    if not isinstance(positions, pd.DataFrame):
        positions = pd.DataFrame(positions)
    positions = positions.reindex(index=dates, columns=instruments).fillna(0.0).astype("float64")

    # Notional exposure per asset
    notional = positions * prices
    long_exposure = notional.clip(lower=0.0).sum(axis=1)
    short_exposure = notional.clip(upper=0.0).sum(axis=1)  # negative
    gross_exposure = long_exposure + np.abs(short_exposure)
    net_exposure = long_exposure + short_exposure
    leverage = gross_exposure / equity.replace(0.0, np.nan)
    leverage = leverage.fillna(0.0)

    # Weights = exposure / equity
    weights_full = pd.DataFrame(
        0.0, index=dates, columns=instruments, dtype="float64"
    )
    eq_nonzero = equity.replace(0.0, np.nan)
    weights_full.loc[:, :] = notional.div(eq_nonzero, axis=0).fillna(0.0)

    # Trades: map vectorbt order records -> our "trades" DataFrame
    trades_df = _orders_to_trades_df(pf, instruments)

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
    sl = execution.slippage
    return float(sl.base_bps) / 1e4


def _build_cost_matrices_from_orders(
    *,
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None,
    orders_df: pd.DataFrame,
    commission: Commission | None,
    slippage_model: Any | None,
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
        return (
            df_like.reindex(index=dates)
            .reindex(columns=cols)
            .fillna(0.0)
            .astype("float64")
        )

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
        base_bps = float(getattr(slippage_model, "base_bps", 0.0) or 0.0)
        k = float(getattr(slippage_model, "k", 0.0) or 0.0)
        exponent = float(getattr(slippage_model, "exponent", 1.0) or 1.0)

        if base_bps != 0.0 or k != 0.0:
            if volumes is None:
                part_net = pd.DataFrame(0.0, index=dates, columns=cols, dtype="float64")
            else:
                vol = volumes.reindex(index=dates, columns=cols).astype("float64")
                with np.errstate(divide="ignore", invalid="ignore"):
                    part_net = np.abs(net_size) / vol.replace(0.0, np.nan)
                part_net = part_net.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                part_net = part_net.clip(lower=0.0, upper=1.0)
            # Compute slippage in bps and convert to fraction
            slip_bps_ev = base_bps + k * np.power(part_net, exponent)
            # Scale slippage by net/legs ratio so that total slippage dollars across legs
            # approximates event-driven slippage on the net trade
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = (np.abs(net_size)) / abs_size.replace(0.0, np.nan)
            scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
            slippage_df = ((slip_bps_ev / 1e4) * scale).astype("float64").where(abs_size > 0.0, 0.0)

    return fees_df, slippage_df


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
