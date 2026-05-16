"""
LSE PEAD — S004 — build_signals.py  (consistent universe)
==========================================================
Universe defined in: research/generated/Dividend Growth/universe.py
Prices come from LSEUniverse — no separate price pull.
PEAD tickers = intersection of canonical universe + EPS-qualified tickers.

SUE = eps_difference / rolling_std(eps_actual, 8Q), clipped ±5σ
- Rolling std uses shift(1) → no current-quarter lookahead
- Forward-fill period = HOLD_DAYS trading days → signal expires naturally
- Signal is NaN after HOLD_DAYS → position exits on next daily rebalance

Outputs (written to signals/004_lse_pead/data/)
────────────────────────────────────────────────────────────────────────
  sue_signal.parquet  — wide date×ticker, SUE active for HOLD_DAYS after report
  lse_prices.parquet  — OHLCV for the PEAD universe (subset of canonical)
  events.parquet      — cleaned events table (for check_data + drilldown)

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/004_lse_pead/build_signals.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

# ── Universe import ───────────────────────────────────────────────────────────
STRATEGY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRATEGY_ROOT))
from universe import LSEUniverse, DB, PULL_START, BACKTEST_START, SHARED_DATA_DIR, SHARED_PRICES

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Signal-specific constants ─────────────────────────────────────────────────
START               = BACKTEST_START
END                 = "2026-01-01"
HOLD_DAYS           = 21             # trading days SUE remains active after report_date
MIN_EVENTS          = 4              # min events with non-null estimate per ticker
SUE_CLIP            = 5.0            # clip at ±5σ — handles unit/feed errors
SUE_VOL_WINDOW      = 8              # rolling quarters for EPS volatility (denominator)
SUE_VOL_MIN_PERIODS = 4              # min periods to compute volatility
MIN_PRICE           = 100.0          # GBX — penny stock filter
MIN_DAYS            = 252            # min trading days of price history

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _conn():
    return psycopg2.connect(**DB)


def build_sue(universe_tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    conn = _conn()
    print("Pulling earnings_history for LSE...")
    earnings = pd.read_sql("""
        SELECT ticker, date, report_date, eps_actual, eps_estimate,
               eps_difference, before_after_market
        FROM eodhd.earnings_history
        WHERE exchange = 'LSE'
          AND deprecated_at IS NULL
          AND report_date IS NOT NULL
        ORDER BY ticker, date
    """, conn)
    conn.close()

    print(f"  Raw: {len(earnings)} rows, {earnings['ticker'].nunique()} tickers")
    earnings["date"] = pd.to_datetime(earnings["date"])
    earnings["report_date"] = pd.to_datetime(earnings["report_date"])
    earnings = earnings.sort_values(["ticker", "date"]).reset_index(drop=True)

    # SUE denominator: rolling std of eps_actual (shifted to avoid lookahead)
    def _sue(grp: pd.DataFrame) -> pd.Series:
        eps_vol = grp["eps_actual"].shift(1).rolling(
            SUE_VOL_WINDOW, min_periods=SUE_VOL_MIN_PERIODS
        ).std()
        return (grp["eps_difference"] / eps_vol).clip(-SUE_CLIP, SUE_CLIP)

    earnings["sue"] = earnings.groupby("ticker", group_keys=False).apply(_sue, include_groups=False)

    # Filter: must have analyst estimate, valid SUE, report_date in range
    valid = (
        earnings["eps_estimate"].notna()
        & earnings["sue"].notna()
        & (earnings["report_date"] >= pd.Timestamp(START))
        & (earnings["report_date"] < pd.Timestamp(END))
    )
    events = earnings.loc[valid].copy()

    # Filter to canonical universe — intersection ensures consistent universe
    in_universe = events["ticker"].isin(universe_tickers)
    events = events[in_universe].copy()

    counts = events.groupby("ticker").size()
    qualified = counts[counts >= MIN_EVENTS].index.tolist()
    events = events[events["ticker"].isin(qualified)].reset_index(drop=True)
    print(f"  Qualified (>= {MIN_EVENTS} events, in universe): {len(events)} events, {len(qualified)} tickers")
    return events, qualified
    print(f"Pulling prices for {len(tickers)} tickers...")
    tlist = ", ".join(f"'{t}'" for t in tickers)
    prices = pd.read_sql(f"""
        SELECT ticker, trade_date AS date,
               open_price AS open, high_price AS high,
               low_price AS low,
               adj_close_price AS close,
               volume
        FROM eodhd.prices
        WHERE exchange = 'LSE'
          AND ticker IN ({tlist})
          AND trade_date >= '{PULL_START}'
          AND trade_date < '{END}'
          AND adj_close_price > 0
        ORDER BY ticker, trade_date
    """, conn)
    conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"  Pulled {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    return prices


def build_sue_parquet(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill each event's SUE for HOLD_DAYS trading days from report_date."""
    trading_days = pd.DatetimeIndex(sorted(prices["date"].unique()))
    tickers = sorted(events["ticker"].unique())

    sue_wide = pd.DataFrame(np.nan, index=trading_days, columns=tickers, dtype=float)

    for row in events.sort_values("report_date").itertuples():
        if row.ticker not in sue_wide.columns:
            continue
        start_idx = sue_wide.index.searchsorted(row.report_date)
        if start_idx >= len(sue_wide.index):
            continue
        end_idx = min(start_idx + HOLD_DAYS, len(sue_wide.index))
        sue_wide.iloc[start_idx:end_idx, sue_wide.columns.get_loc(row.ticker)] = row.sue

    active = sue_wide.notna().sum(axis=1)
    print(f"SUE parquet: {sue_wide.shape[1]} tickers | active/day: "
          f"median={active.median():.0f}  max={active.max():.0f}")
    return sue_wide


def main() -> None:
    # ── Step 0: schema introspection + canonical universe ─────────────────────
    print("Step 0 — Schema introspection + canonical universe …")
    conn = psycopg2.connect(**DB)
    u = LSEUniverse(conn)
    u.introspect()
    universe_tickers = u.tickers()

    # ── Step 1: SUE events filtered to canonical universe ─────────────────────
    events, tickers = build_sue(universe_tickers)

    # ── Step 2: prices from universe module (consistent, bad-tick cleaned) ────
    prices_long = u.prices()
    # Filter to PEAD tickers only (subset of full universe)
    prices = prices_long[prices_long["ticker"].isin(tickers)].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"Final PEAD universe: {len(tickers)} tickers  (of {len(universe_tickers)} canonical)")

    sue_wide = build_sue_parquet(events, prices)
    sue_wide.to_parquet(DATA_DIR / "sue_signal.parquet")
    # prices are in shared_data/lse_prices.parquet — no per-signal copy needed
    events.to_parquet(DATA_DIR / "events.parquet", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
