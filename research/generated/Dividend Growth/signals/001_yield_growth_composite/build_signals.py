"""
LSE Yield + Dividend Growth Composite — S001 (LSE rewrite)
===========================================================
Signal = 40% trailing-12M yield rank  +  60% YoY dividend growth rank
         Both computed from eodhd.dividends (LSE).

Distinction vs S003:
- S003 uses EPS growth (eodhd.earnings_history) for the 60% growth leg
- S001 uses dividend growth (does the company grow its actual payout?)

Universe: shared LSEUniverse from universe.py (~1,347 tickers)

Outputs (written to signals/001_yield_growth_composite/data/)
-------------------------------------------------------------
  lse_prices.parquet    -- OHLCV for canonical LSE universe (from universe.py)
  div_composite.parquet -- wide date x ticker composite signal

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/001_yield_growth_composite/build_signals.py"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2

# -- Universe import ----------------------------------------------------------
STRATEGY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRATEGY_ROOT))
from universe import LSEUniverse, DB, PULL_START, PULL_END, BACKTEST_START, BACKTEST_END, SHARED_DATA_DIR, SHARED_PRICES

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# -- CLI args -----------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--yield-cap", type=float, default=0.15,
                     help="Max yield before nullifying (data quality guard). Default 0.15")
_args, _ = _parser.parse_known_args()

# -- Signal-specific constants ------------------------------------------------
YIELD_CAP       = _args.yield_cap   # e.g. 0.15, 0.20, 0.25, 0.30
GROWTH_CLIP_LO  = -1.0
GROWTH_CLIP_HI  = 5.0
WINSORIZE_Z     = 2.5
YIELD_WEIGHT    = 0.40
GROWTH_WEIGHT   = 0.60

# =============================================================================
print("-" * 60)
print("Step 0 - Schema introspection + canonical universe ...")
conn = psycopg2.connect(**DB)
u = LSEUniverse(conn)
u.introspect()
tickers_bt  = u.tickers()
prices_long = u.prices()
conn.close()
print(f"  Universe: {len(tickers_bt):,} tickers")

# =============================================================================
# Step 1 -- prices wide + trading calendar
# =============================================================================
SHARED_DATA_DIR.mkdir(exist_ok=True)
if not SHARED_PRICES.exists():
    prices_long.to_parquet(SHARED_PRICES, index=False)
    print(f"  Shared prices written to {SHARED_PRICES}")
else:
    print(f"  Shared prices already exist, skipping write ({SHARED_PRICES})")

prices_wide = (
    prices_long[["date", "ticker", "close"]]
    .set_index(["date", "ticker"])["close"]
    .unstack("ticker")
    .reindex(columns=tickers_bt)
)
trading_dates       = prices_wide.index               # tz-naive 05:00:00
trading_dates_naive = trading_dates.normalize()       # midnight for calendar alignment
quality_mask = prices_wide.notna()

print(f"\nStep 1 - Prices: {len(prices_long):,} rows, {len(tickers_bt):,} tickers")
print(f"  Trading dates: {len(trading_dates):,}  |  {trading_dates_naive.min().date()} to {trading_dates_naive.max().date()}")

# =============================================================================
# Step 2 -- LSE dividends -> trailing 12M yield + YoY growth
# =============================================================================
print("\nStep 2 - Computing trailing 12M dividend yield + YoY growth ...")

conn2 = psycopg2.connect(**DB)
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
    conn2,
    params={"tickers": tickers_bt, "start": "2012-01-01"},
    parse_dates=["ex_date"],
)
conn2.close()
print(f"  Dividend rows: {len(divs):,}  |  tickers: {divs['ticker'].nunique():,}")

# eodhd stores LSE dividends in GBP (pounds), prices are in GBX (pence)
divs["dividend"] *= 100
print(f"  GBP->GBX x100 applied (eodhd stores in GBP, prices in pence)")

divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.normalize()

div_pivot = (
    divs.groupby(["ex_date", "ticker"])["dividend"].sum()
    .unstack("ticker")
    .reindex(columns=tickers_bt)
    .fillna(0.0)
)

# Build daily calendar with 400-day lookback buffer
cal_start = trading_dates_naive.min() - pd.Timedelta(days=400)
all_cal   = pd.date_range(start=cal_start, end=trading_dates_naive.max(), freq="D")
div_cal   = div_pivot.reindex(all_cal, fill_value=0.0).fillna(0.0)

# Rolling 12M sums
t12m      = div_cal.rolling("365D").sum()
t12m_prev = t12m.shift(365, freq="D")   # prior-year 12M -- for YoY growth

# Align to trading dates
t12m_td      = t12m.reindex(trading_dates_naive)
t12m_prev_td = t12m_prev.reindex(trading_dates_naive)

is_payer = t12m_td > 0.001

# Trailing yield: GBX dividend / GBX price
prices_aligned = prices_wide.copy()
prices_aligned.index = trading_dates_naive
yield_raw = t12m_td.div(prices_aligned).where(is_payer)
yield_raw = yield_raw.where(yield_raw <= YIELD_CAP)

# YoY dividend growth: (TTM_now / TTM_prev) - 1
growth_raw = (
    t12m_td.div(t12m_prev_td.where(t12m_prev_td > 0.001)) - 1.0
).where(is_payer)
growth_raw = growth_raw.clip(GROWTH_CLIP_LO, GROWTH_CLIP_HI)

# Apply quality mask
yield_raw  = yield_raw.where(quality_mask.values)
growth_raw = growth_raw.where(quality_mask.values)

print(f"  Avg payers/day: {is_payer.sum(axis=1).mean():.0f}")
print(f"  Avg with growth signal/day: {growth_raw.notna().sum(axis=1).mean():.0f}")

# =============================================================================
# Step 2.5 -- EWMA smoothing (90-day yield, 60-day growth)
# =============================================================================
print("\nStep 2.5 - Applying EWMA smoothing (yield span=90, growth span=60) ...")
yield_smooth  = yield_raw.ewm(span=90,  min_periods=30).mean()
growth_smooth = growth_raw.ewm(span=60, min_periods=20).mean()

# Re-apply is_payer mask after smoothing (EWMA can bleed into non-payer periods)
yield_smooth  = yield_smooth.where(is_payer & quality_mask.values)
growth_smooth = growth_smooth.where(is_payer & quality_mask.values)

print(f"  Yield smooth  non-null/day: {yield_smooth.notna().sum(axis=1).mean():.0f}")
print(f"  Growth smooth non-null/day: {growth_smooth.notna().sum(axis=1).mean():.0f}")

# =============================================================================
# Step 3 -- Composite = 40% yield rank + 60% growth rank  (smoothed signals)
# =============================================================================
print("\nStep 3 - Building composite (using EWMA-smoothed signals) ...")


def cs_winsorize(df, z=WINSORIZE_Z):
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, float("nan"))
    lo   = (mean - z * std).values[:, None]
    hi   = (mean + z * std).values[:, None]
    return df.clip(lo, hi)


yield_w  = cs_winsorize(yield_smooth)
growth_w = cs_winsorize(growth_smooth)

yield_pct  = yield_w.rank(axis=1, pct=True)
growth_pct = growth_w.rank(axis=1, pct=True)

both_valid = yield_pct.notna() & growth_pct.notna()
yield_only = yield_pct.notna() & growth_pct.isna()

composite = yield_pct.mul(YIELD_WEIGHT).add(growth_pct.mul(GROWTH_WEIGHT))
composite = composite.where(both_valid, yield_pct.where(yield_only))
composite = composite.where(is_payer & quality_mask.values)

print(f"  Avg tickers in composite (both signals): {both_valid.sum(axis=1).mean():.0f}")
print(f"  Avg tickers yield-only (new payers):     {yield_only.sum(axis=1).mean():.0f}")
print(f"  Composite range: [{composite.stack().min():.3f}, {composite.stack().max():.3f}]")

# =============================================================================
# Step 4 -- Save
# =============================================================================
composite.index   = trading_dates.tz_localize(None)
yield_pct.index   = trading_dates.tz_localize(None)
growth_pct.index  = trading_dates.tz_localize(None)

out_comp   = DATA_DIR / "div_composite.parquet"
out_yield  = DATA_DIR / "yield_rank.parquet"
out_growth = DATA_DIR / "div_growth_rank.parquet"

composite.to_parquet(out_comp)
yield_pct.to_parquet(out_yield)
growth_pct.to_parquet(out_growth)

print(f"\nStep 4 - Saved:")
print(f"  composite    -> {out_comp}   shape={composite.shape}")
print(f"  yield_rank   -> {out_yield}  shape={yield_pct.shape}")
print(f"  div_growth   -> {out_growth} shape={growth_pct.shape}")
print(f"  Index tz: {composite.index.tz}  sample: {composite.index[0]}")
print("\nDone: build_signals.py (001 LSE) complete.")