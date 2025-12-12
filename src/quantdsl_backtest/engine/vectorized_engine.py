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
    # 4. Build vectorbt portfolio
    # ------------------------------------------------------------------ #
    fees = _commission_to_vbt_fees(strategy.costs.commission)
    slippage = _slippage_to_vbt_frac(strategy.execution)
    init_cash = float(strategy.backtest.cash_initial)
    freq = strategy.data.frequency or "1D"

    # vectorbt expects NaNs for "no order" rows; that's exactly what we built
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",  # TargetPercent sizing (per asset) :contentReference[oaicite:2]{index=2}
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
      - 'bps_notional' -> amount / 1e4
      - 'per_share'    -> not supported in vectorized mode (falls back to 0)
    """
    if commission is None:
        return 0.0

    if commission.type == "bps_notional":
        return float(commission.amount) / 1e4

    if commission.type == "per_share":
        log.warning(
            "Vectorized engine: commission.type='per_share' not supported "
            "by constant-fee model; treating as zero. "
            "Use event_driven engine for exact per-share costs."
        )
        return 0.0

    return 0.0


def _slippage_to_vbt_frac(execution: Execution) -> float:
    """
    Map our (potentially non-linear) slippage model to vectorbt's
    constant `slippage` argument (fraction of price).

    For now we only use the base_bps parameter and ignore k/exponent.
    """
    if execution is None or execution.slippage is None:
        return 0.0
    sl = execution.slippage
    return float(sl.base_bps) / 1e4


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
