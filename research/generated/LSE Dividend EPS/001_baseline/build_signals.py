#!/usr/bin/env python
"""
Build price + signal parquets for the LSE Dividend EPS strategy.

Outputs (all written to 001_baseline/data/)
-------------------------------------------
lse_prices.parquet
    Long format — date(UTC 05:00:00), ticker, open, high, low, close, close_unadj, volume
    close     = adj_close_price  (split+div adjusted — engine uses this for P&L)
    close_unadj = close_price    (unadjusted — used only for return attribution)

composite.parquet
    Wide ExternalFactor — tz-naive 05:00:00 index × ticker columns
    Values = 0.4 × trailing-12M yield rank + 0.6 × YoY EPS growth rank
    NaN    = ticker fails quality filter (no dividends, no EPS, or negative history)

div_income.parquet
    Wide — trailing 12M dividends per share per ticker per trading day
    Used in notebook for return attribution (dividend income component)

eps_yoy_raw.parquet
    Wide — point-in-time YoY EPS growth per ticker per trading day
    Used in notebook for diagnostics / signal attribution

Run from btest/ root:
    uv run python "research/LSE Dividend EPS/001_baseline/build_signals.py"
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

DB = dict(host="localhost", dbname="sfera", user="postgres", password="lokomotiv")

RESEARCH_ROOT = Path(__file__).resolve().parent
DATA_DIR      = RESEARCH_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dates — warmup starts 2 years before backtest so EPS lag-4 has data
PULL_START     = "2013-01-01"
PULL_END       = "2026-06-01"
BACKTEST_START = "2015-01-01"   # consistent with engine start


# ══════════════════════════════════════════════════════════════════════════════
# 1. LSE PRICES
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("Step 1 — Loading LSE prices from sfera …")
conn = psycopg2.connect(**DB)
prices_raw = pd.read_sql(
    """
    SELECT
        ticker,
        trade_date::date           AS trade_date,
        open_price                 AS open,
        high_price                 AS high,
        low_price                  AS low,
        adj_close_price            AS close,        -- adjusted = engine P&L
        close_price                AS close_unadj,  -- unadjusted = attribution
        volume
    FROM eodhd.prices
    WHERE exchange = 'LSE'
      AND trade_date >= %(start)s
      AND trade_date <= %(end)s
      AND close_price    > 0
      AND adj_close_price > 0
      AND deprecated_at IS NULL
    ORDER BY ticker, trade_date
    """,
    conn,
    params={"start": PULL_START, "end": PULL_END},
    parse_dates=["trade_date"],
)
print(f"  Raw rows: {len(prices_raw):,}  |  tickers: {prices_raw['ticker'].nunique():,}")

# Build canonical UTC 05:00:00 timestamps (matching sp500_daily convention)
prices_raw["date"] = (
    pd.to_datetime(prices_raw["trade_date"])
    .dt.tz_localize("UTC")
    .dt.normalize()
    + pd.Timedelta(hours=5)
)
prices_raw = prices_raw.drop(columns=["trade_date"])

# Drop rows where adj_close <= 0 or close_unadj <= 0 (data quality)
prices_raw = prices_raw[
    (prices_raw["close"] > 0) & (prices_raw["close_unadj"] > 0)
].copy()

# Save long-format prices parquet (same schema as sp500_daily + close_unadj)
out_prices = DATA_DIR / "lse_prices.parquet"
prices_raw[["date", "ticker", "open", "high", "low", "close", "close_unadj", "volume"]].to_parquet(
    out_prices, index=False
)
print(f"  Saved  → {out_prices}  ({len(prices_raw):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRADING DATES + PRICE WIDE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2 — Building trading-date index …")

# Keep only tickers active in the backtest window for signal alignment
prices_bt = prices_raw[prices_raw["date"] >= pd.Timestamp(BACKTEST_START, tz="UTC")].copy()
tickers_bt = sorted(prices_bt["ticker"].unique())
print(f"  Tickers in backtest window: {len(tickers_bt):,}")

# Wide close matrix (adj) — UTC tz-aware timestamps as row index
prices_wide = (
    prices_bt[["date", "ticker", "close"]]
    .set_index(["date", "ticker"])["close"]
    .unstack("ticker")
)
trading_dates     = prices_wide.index                   # UTC tz-aware e.g. 2015-01-05 05:00:00+00:00
trading_dates_naive = trading_dates.tz_localize(None).normalize()  # tz-naive midnight for calendar align
print(f"  Trading dates: {len(trading_dates):,}  |  {trading_dates[0].date()} → {trading_dates[-1].date()}")

# Unadjusted close wide matrix (for attribution)
prices_wide_unadj = (
    prices_bt[["date", "ticker", "close_unadj"]]
    .set_index(["date", "ticker"])["close_unadj"]
    .unstack("ticker")
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LSE DIVIDENDS → TRAILING 12M YIELD
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3 — Computing trailing 12M dividend yield …")
tickers_sql_list = tickers_bt  # use backtest-window tickers only

divs = pd.read_sql(
    """
    SELECT ticker, ex_date, dividend
    FROM eodhd.dividends
    WHERE exchange = 'LSE'
      AND ticker = ANY(%(tickers)s)
      AND ex_date IS NOT NULL
      AND ex_date >= %(start)s
      AND dividend > 0
    ORDER BY ex_date
    """,
    conn,
    params={"tickers": tickers_sql_list, "start": "2012-01-01"},
    parse_dates=["ex_date"],
)
print(f"  Dividend rows: {len(divs):,}  |  tickers: {divs['ticker'].nunique():,}")

# ── Currency normalisation: LSE prices are in GBX (pence), but some tickers
# (e.g. SHEL, formerly Anglo-Dutch) have dividends stored in GBP (pounds).
# Multiply those dividends by 100 to convert GBP → GBX so yield = div/price
# stays in the same units across the whole universe.
instr_currency = pd.read_sql(
    """
    SELECT ticker, currency
    FROM eodhd.instruments
    WHERE exchange_code = 'LSE'
      AND ticker = ANY(%(tickers)s)
    """,
    conn,
    params={"tickers": tickers_sql_list},
).set_index("ticker")["currency"]

gbp_tickers = instr_currency[instr_currency == "GBP"].index.tolist()
if gbp_tickers:
    divs.loc[divs["ticker"].isin(gbp_tickers), "dividend"] *= 100
    print(f"  GBP→GBX ×100 applied to {len(gbp_tickers)} tickers: {gbp_tickers[:10]}")
else:
    print("  No GBP-currency tickers found — all dividends assumed already in GBX.")

divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.normalize()

# Pivot: sum same-day dividends per ticker
div_pivot = (
    divs.groupby(["ex_date", "ticker"])["dividend"]
    .sum()
    .unstack("ticker")
    .reindex(columns=tickers_bt)
    .fillna(0.0)
)

# Extended daily calendar for rolling window (400-day buffer before backtest)
cal_start = trading_dates_naive.min() - pd.Timedelta(days=400)
all_cal = pd.date_range(start=cal_start, end=trading_dates_naive.max(), freq="D")
div_cal = div_pivot.reindex(all_cal, fill_value=0.0).fillna(0.0)

# Rolling 12M trailing sum
t12m_divs = div_cal.rolling("365D").sum()

# Align to trading dates (tz-naive)
t12m_td = t12m_divs.reindex(trading_dates_naive)

# Trailing yield = 12M dividends / unadjusted close price
prices_aligned = prices_wide_unadj.copy()
prices_aligned.index = trading_dates_naive  # strip tz for alignment
is_payer = t12m_td > 0.001
yield_raw = t12m_td.div(prices_aligned).where(is_payer)
yield_raw = yield_raw.where(yield_raw <= 0.40)   # cap at 40% (data errors)

# Save div_income for attribution
out_div_income = DATA_DIR / "div_income.parquet"
t12m_td.index = trading_dates.tz_localize(None)  # restore 05:00:00 tz-naive
t12m_td.to_parquet(out_div_income)
print(f"  Avg payers per day: {is_payer.sum(axis=1).mean():.0f}")
print(f"  Saved  → {out_div_income}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LSE EARNINGS HISTORY → YOY EPS GROWTH (POINT-IN-TIME)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4 — Computing point-in-time YoY EPS growth …")

eps_raw = pd.read_sql(
    """
    SELECT ticker, date AS period_date, report_date, eps_actual
    FROM eodhd.earnings_history
    WHERE exchange = 'LSE'
      AND ticker = ANY(%(tickers)s)
      AND date >= %(start)s
      AND date <= %(end)s
      AND eps_actual IS NOT NULL
    ORDER BY ticker, date, report_date
    """,
    conn,
    params={"tickers": tickers_sql_list, "start": "2012-01-01", "end": PULL_END},
    parse_dates=["period_date", "report_date"],
)
conn.close()
print(f"  EPS rows: {len(eps_raw):,}  |  tickers: {eps_raw['ticker'].nunique():,}")

# Drop rows with no report_date (can't establish point-in-time)
eps_raw = eps_raw.dropna(subset=["report_date"]).copy()

# Sort by ticker, period_date — then lag by 4 periods (≈ 1 year, handles quarterly + annual)
eps_raw = eps_raw.sort_values(["ticker", "period_date"]).reset_index(drop=True)
eps_raw["eps_lag4"] = eps_raw.groupby("ticker")["eps_actual"].shift(4)

# YoY growth — guard against zero denominator and sign changes
# Only compute when both current and lagged EPS have same sign (meaningful YoY)
eps_raw["yoy_growth"] = np.where(
    eps_raw["eps_lag4"].notna()
    & (eps_raw["eps_lag4"].abs() > 0.001)
    & (np.sign(eps_raw["eps_actual"]) == np.sign(eps_raw["eps_lag4"])),  # same sign
    eps_raw["eps_actual"] / eps_raw["eps_lag4"] - 1.0,
    np.nan,
)
eps_raw["yoy_growth"] = eps_raw["yoy_growth"].clip(-3.0, 5.0)

# Deduplicate: if multiple reports on same (ticker, report_date), keep latest period
eps_clean = (
    eps_raw.dropna(subset=["yoy_growth"])
    .sort_values(["ticker", "report_date", "period_date"])
    .groupby(["ticker", "report_date"])
    .last()
    .reset_index()
)
eps_clean["report_date"] = pd.to_datetime(eps_clean["report_date"]).dt.normalize()

# Build sparse wide matrix: index=report_date, columns=ticker
eps_wide_sparse = (
    eps_clean.set_index(["report_date", "ticker"])["yoy_growth"]
    .unstack("ticker")
    .reindex(columns=tickers_bt)
)

# Forward-fill on daily calendar (carry last known value — point-in-time OK since
# we gate on report_date, not period_date)
all_cal2 = pd.date_range(start="2012-01-01", end=PULL_END, freq="D")
eps_daily = eps_wide_sparse.reindex(all_cal2).ffill()  # carry indefinitely until new report

# Align to trading dates
eps_td = eps_daily.reindex(trading_dates_naive)
eps_valid = eps_td.notna()

print(f"  Avg tickers with EPS signal per day: {eps_valid.sum(axis=1).mean():.0f}")

# Save raw EPS YoY for attribution
out_eps = DATA_DIR / "eps_yoy_raw.parquet"
eps_td.index = trading_dates.tz_localize(None)
eps_td.to_parquet(out_eps)
print(f"  Saved  → {out_eps}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPOSITE SIGNAL: 0.4 × yield rank + 0.6 × EPS YoY rank
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5 — Building composite signal …")

# Re-align yield to same index as eps_td
yield_td = yield_raw.copy()
yield_td.index = trading_dates.tz_localize(None)  # 05:00:00 tz-naive

# Both signals must be non-null for composite; otherwise fall back to yield-only
both_valid = yield_td.notna() & eps_td.notna()
yield_only = yield_td.notna() & ~eps_td.notna()


def cs_winsorize(df: pd.DataFrame, z: float = 2.5) -> pd.DataFrame:
    """Cross-sectional winsorize at ±z·std per row."""
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    lo   = (mean - z * std).values[:, None]
    hi   = (mean + z * std).values[:, None]
    return df.clip(lo, hi)


yield_w  = cs_winsorize(yield_td)
eps_w    = cs_winsorize(eps_td)

yield_pct = yield_w.rank(axis=1, pct=True)
eps_pct   = eps_w.rank(axis=1, pct=True)

# Full composite (both signals)
composite = yield_pct.mul(0.4).add(eps_pct.mul(0.6))

# Fallback: new payer with no EPS history → yield rank only
composite = composite.where(both_valid, yield_pct.where(yield_only, np.nan))

# Diagnostics
n_full  = both_valid.sum(axis=1).mean()
n_yfb   = yield_only.sum(axis=1).mean()
n_total = composite.notna().sum(axis=1).mean()
print(f"  Avg tickers in composite (full):      {n_full:.0f}")
print(f"  Avg tickers (yield-only fallback):     {n_yfb:.0f}")
print(f"  Avg total tickers with signal:         {n_total:.0f}")
print(f"  Composite range: [{composite.stack().min():.3f}, {composite.stack().max():.3f}]")

# ── 6. Save composite with tz-naive 05:00:00 index (must match engine's self.index)
out_composite = DATA_DIR / "composite.parquet"
composite.to_parquet(out_composite)
print(f"\nStep 6 — Saved composite → {out_composite}  shape={composite.shape}")
print(f"  Index tz:    {composite.index.tz}")
print(f"  Index sample: {composite.index[0]}")
print("\n✓  build_signals.py complete.")
