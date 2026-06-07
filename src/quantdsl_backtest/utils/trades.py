"""
quantdsl_backtest.utils.trades
==============================
Post-processing of backtest outputs. Fully vectorised — no Python loops.

Integrates with the engine's existing outputs:
  - positions.parquet  -> wide (dates x tickers) shares, engine-computed
  - trades.parquet     -> per-execution trade records

Usage:
    from quantdsl_backtest.utils.trades import build_roundtrips, build_daily_positions

    rt   = build_roundtrips(trades)                   # trades.parquet
    dpos = build_daily_positions(positions, prices)   # positions.parquet x prices
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# build_roundtrips
# ---------------------------------------------------------------------------

def build_roundtrips(trades_df: pd.DataFrame) -> pd.DataFrame:
    """FIFO round-trip table via vectorised cumulative-quantity interval matching.

    Each buy lot is assigned an interval [cum_lo, cum_hi) in the running
    buy-quantity space per instrument.  Each sell similarly gets an interval.
    A lot-pair matches when the intervals overlap AND the sell occurred after
    the buy -- no per-ticker Python loops.

    Parameters
    ----------
    trades_df:
        trades.parquet DataFrame.  Required columns: datetime, instrument,
        side, quantity, price.  Optional: commission, slippage_bps.

    Returns
    -------
    pd.DataFrame with columns:
        instrument, entry_date, exit_date, holding_days,
        quantity, entry_price, exit_price,
        cost_basis, proceeds, gross_pnl, gross_pnl_pct,
        total_commission, net_pnl, net_pnl_pct,
        entry_slippage_bps, exit_slippage_bps
    """
    tr = trades_df.copy()
    tr["datetime"] = pd.to_datetime(tr["datetime"])
    tr["qty_abs"]  = tr["quantity"].abs()
    if "commission"   not in tr.columns: tr["commission"]   = 0.0
    if "slippage_bps" not in tr.columns: tr["slippage_bps"] = 0.0

    tr = tr.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    side_u = tr["side"].str.upper().str.strip()

    buys  = tr[side_u == "BUY"].copy()
    sells = tr[side_u == "SELL"].copy()

    _EMPTY_COLS = [
        "instrument", "entry_date", "exit_date", "holding_days",
        "quantity", "entry_price", "exit_price",
        "cost_basis", "proceeds", "gross_pnl", "gross_pnl_pct",
        "total_commission", "net_pnl", "net_pnl_pct",
        "entry_slippage_bps", "exit_slippage_bps",
    ]
    if buys.empty or sells.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)

    # Cumulative quantity intervals within each instrument (already sorted by datetime)
    buys["cum_hi_b"]  = buys.groupby("instrument")["qty_abs"].cumsum()
    buys["cum_lo_b"]  = buys["cum_hi_b"] - buys["qty_abs"]

    sells["cum_hi_s"] = sells.groupby("instrument")["qty_abs"].cumsum()
    sells["cum_lo_s"] = sells["cum_hi_s"] - sells["qty_abs"]

    # Cross-join within instrument; filter to overlapping intervals + temporal order
    merged = buys.merge(sells, on="instrument", suffixes=("_b", "_s"))

    lo    = merged[["cum_lo_b", "cum_lo_s"]].max(axis=1)
    hi    = merged[["cum_hi_b", "cum_hi_s"]].min(axis=1)
    valid = (hi > lo + 1e-8) & (merged["datetime_s"] >= merged["datetime_b"])
    m     = merged[valid].copy()
    m["quantity"] = (hi - lo)[valid].values

    # Commission pro-rated by matched fraction of original lot size
    m["entry_commission"] = m["commission_b"].abs() * (m["quantity"] / m["qty_abs_b"])
    m["exit_commission"]  = m["commission_s"].abs() * (m["quantity"] / m["qty_abs_s"])
    m["total_commission"] = m["entry_commission"] + m["exit_commission"]

    # P&L (vectorised)
    m["cost_basis"]    = m["quantity"] * m["price_b"]
    m["proceeds"]      = m["quantity"] * m["price_s"]
    m["gross_pnl"]     = m["proceeds"] - m["cost_basis"]
    m["net_pnl"]       = m["gross_pnl"] - m["total_commission"]
    m["gross_pnl_pct"] = np.where(m["cost_basis"] > 0, (m["proceeds"] / m["cost_basis"] - 1) * 100, 0.0)
    m["net_pnl_pct"]   = np.where(m["cost_basis"] > 0,  m["net_pnl"]  / m["cost_basis"]       * 100, 0.0)
    m["holding_days"]  = (m["datetime_s"] - m["datetime_b"]).dt.days

    out = (
        m[[
            "instrument",
            "datetime_b", "datetime_s", "holding_days",
            "quantity", "price_b", "price_s",
            "cost_basis", "proceeds", "gross_pnl", "gross_pnl_pct",
            "total_commission", "net_pnl", "net_pnl_pct",
            "slippage_bps_b", "slippage_bps_s",
        ]]
        .rename(columns={
            "datetime_b":     "entry_date",
            "datetime_s":     "exit_date",
            "price_b":        "entry_price",
            "price_s":        "exit_price",
            "slippage_bps_b": "entry_slippage_bps",
            "slippage_bps_s": "exit_slippage_bps",
        })
        .copy()
    )

    for c in ("cost_basis", "proceeds", "gross_pnl", "net_pnl", "total_commission"):
        out[c] = out[c].round(2)
    for c in ("entry_price", "exit_price"):
        out[c] = out[c].round(6)
    for c in ("gross_pnl_pct", "net_pnl_pct"):
        out[c] = out[c].round(3)
    out["quantity"] = out["quantity"].round(4)

    return out.sort_values(["entry_date", "instrument"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# roundtrip_summary
# ---------------------------------------------------------------------------

def roundtrip_summary(rt: pd.DataFrame) -> dict:
    """Aggregate statistics from a build_roundtrips() result."""
    if rt.empty:
        return {}
    winners = rt[rt["net_pnl"] > 0]
    losers  = rt[rt["net_pnl"] < 0]
    return {
        "n_trips"                : len(rt),
        "win_rate"               : len(winners) / len(rt),
        "profit_factor"          : (winners["net_pnl"].sum() / abs(losers["net_pnl"].sum())
                                    if len(losers) > 0 else float("inf")),
        "avg_hold_days"          : rt["holding_days"].mean(),
        "median_hold_days"       : rt["holding_days"].median(),
        "avg_win"                : winners["net_pnl"].mean() if len(winners) else 0.0,
        "avg_loss"               : losers["net_pnl"].mean()  if len(losers)  else 0.0,
        "total_gross_pnl"        : rt["gross_pnl"].sum(),
        "total_net_pnl"          : rt["net_pnl"].sum(),
        "total_commission"       : rt["total_commission"].sum(),
        "avg_commission_per_trip": rt["total_commission"].mean(),
    }


# ---------------------------------------------------------------------------
# build_daily_positions
# ---------------------------------------------------------------------------

def build_daily_positions(
    positions: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    *,
    min_qty: float = 1e-6,
) -> pd.DataFrame:
    """Daily per-ticker position snapshot -- pure matrix operations, no loops.

    Uses the engine's pre-computed positions.parquet directly (wide format:
    dates x tickers = fractional shares held).  Market value is a single
    element-wise matrix multiply.

    Parameters
    ----------
    positions:
        Wide DataFrame from positions.parquet (dates x tickers = shares).
        Already computed by the engine -- do NOT pass trades_df here.
    prices:
        Optional wide closing-price DataFrame (dates x tickers).
        Index timezone / time-of-day is normalised automatically.
        When provided adds a ``market_value`` column.
    min_qty:
        Positions <= this threshold are filtered out (float-rounding guard).

    Returns
    -------
    pd.DataFrame (long format): date, ticker, qty  [, market_value]

    Recipes:
        # Wide market-value matrix for Signum / chart:
        dpos.pivot(index="date", columns="ticker", values="market_value").fillna(0)

        # Daily portfolio total:
        dpos.groupby("date")["market_value"].sum()
    """
    pos = positions.copy()
    pos.index = pd.to_datetime(pos.index).normalize()

    if prices is not None:
        px = prices.copy()
        px.index = pd.to_datetime(px.index).normalize()

        common_dates   = pos.index.intersection(px.index)
        common_tickers = pos.columns.intersection(px.columns)

        pos = pos.loc[common_dates, common_tickers]
        px  = px.loc[common_dates, common_tickers]

        mv_wide = pos * px                           # element-wise matrix multiply -- O(T*N)

        qty_long = pos.stack(future_stack=True).rename("qty")
        mv_long  = mv_wide.stack(future_stack=True).rename("market_value")

        result = pd.concat([qty_long, mv_long], axis=1).reset_index()
        result.columns = ["date", "ticker", "qty", "market_value"]
    else:
        qty_long = pos.stack(future_stack=True).rename("qty")
        result   = qty_long.reset_index()
        result.columns = ["date", "ticker", "qty"]

    result = result[result["qty"] > min_qty].copy()
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)
