#!/usr/bin/env python
"""
Build price + signal parquets — S003 LSE Div+EPS (consistent universe)
=======================================================================
Universe defined in: research/generated/Dividend Growth/universe.py
All price loading, bad-tick filtering, and universe filters come from
LSEUniverse — this script only computes the signal-specific data layers.

Outputs (written to signals/003_lse_div_eps/data/)
────────────────────────────────────────────────────────────────────────
  lse_prices.parquet   — cleaned prices for DSL engine
  composite.parquet    — 0.4×yield rank + 0.6×EPS_YoY rank
  div_income.parquet   — trailing 12M divs per share (for attribution)
  eps_yoy_raw.parquet  — point-in-time YoY EPS growth (for drilldown)
  yield_clean.parquet  — capped trailing yield wide matrix (for drilldown)

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/003_lse_div_eps/build_signals.py"
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ── Universe import ───────────────────────────────────────────────────────────
STRATEGY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRATEGY_ROOT))
from universe import LSEUniverse, DB, PULL_START, PULL_END, BACKTEST_START, MIN_PRICE_GBX, MIN_HISTORY_DAYS, SHARED_DATA_DIR, SHARED_PRICES

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal-specific parameters (importable by drilldown notebooks) ────────────
YIELD_CAP = 0.15    # max plausible trailing yield for LSE equities
EPS_LAG   = 4       # periods to lag for YoY comparison (4 = ~1 year)
YIELD_W   = 0.40    # weight of yield rank in composite
EPS_W     = 0.60    # weight of EPS rank in composite


# ══════════════════════════════════════════════════════════════════════════════
# 0. SCHEMA INTROSPECTION + UNIVERSE LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("-" * 60)
print("Step 0 - Schema introspection + canonical universe ...")
conn = psycopg2.connect(**DB)
u = LSEUniverse(conn)
u.introspect()          # prints all source table columns — catches renames immediately

prices_long = u.prices()                          # cleaned OHLCV, bad ticks filtered
tickers_bt  = u.tickers()                         # canonical universe tickers

# ── Save prices parquet for DSL engine (shared, written once) ───────────────
SHARED_DATA_DIR.mkdir(exist_ok=True)
if not SHARED_PRICES.exists():
    prices_long[["date", "ticker", "open", "high", "low", "close", "close_unadj", "volume"]].to_parquet(
        SHARED_PRICES, index=False
    )
    print(f"\nStep 1 - Prices written to {SHARED_PRICES}  ({len(prices_long):,} rows, {len(tickers_bt):,} tickers)")
else:
    print(f"\nStep 1 - Shared prices already exist, skipping write ({SHARED_PRICES})")

# ── Build wide matrices for signal computation ────────────────────────────────
prices_bt = prices_long[prices_long["date"] >= pd.Timestamp(BACKTEST_START)].copy()

prices_wide = (
    prices_bt[["date", "ticker", "close"]]
    .set_index(["date", "ticker"])["close"]
    .unstack("ticker")
)
trading_dates       = prices_wide.index
trading_dates_naive = trading_dates.normalize()  # always normalize to midnight for calendar alignment

prices_wide_unadj = (
    prices_bt[["date", "ticker", "close_unadj"]]
    .set_index(["date", "ticker"])["close_unadj"]
    .unstack("ticker")
)
print(f"  Trading dates: {len(trading_dates):,}  |  {prices_wide.index[0].date()} to {prices_wide.index[-1].date()}")

# Quality mask: price >= MIN_PRICE_GBX AND cumulative history >= MIN_HISTORY_DAYS
mask_price   = prices_wide_unadj >= MIN_PRICE_GBX
mask_history = prices_wide.notna().cumsum() >= MIN_HISTORY_DAYS
quality_mask = mask_price & mask_history
print(f"  Avg eligible tickers per day: {quality_mask.sum(axis=1).mean():.0f}")

tickers_sql_list = tickers_bt


# ══════════════════════════════════════════════════════════════════════════════
# 2. LSE DIVIDENDS → TRAILING 12M YIELD
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2 - Computing trailing 12M dividend yield ...")
tickers_sql_list = tickers_bt

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

# eodhd stores ALL LSE dividends in GBP (pounds) regardless of instrument currency
# classification. Prices are in GBX (pence). Multiply every dividend by 100.
# Verified: BP raw dividend ≈ 0.083 GBP, price ≈ 572 GBX → correct yield ~5.8%
# only after ×100 (0.083 × 100 × 4 payments ÷ 572 = 5.8%).
divs["dividend"] *= 100
print(f"  GBP->GBX x100 applied to all {len(divs):,} dividend rows (eodhd stores in GBP, prices in pence)")

divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.normalize()

div_pivot = (
    divs.groupby(["ex_date", "ticker"])["dividend"].sum()
    .unstack("ticker")
    .reindex(columns=tickers_bt)
    .fillna(0.0)
)

cal_start = trading_dates_naive.min() - pd.Timedelta(days=400)
all_cal   = pd.date_range(start=cal_start, end=trading_dates_naive.max(), freq="D")
div_cal   = div_pivot.reindex(all_cal, fill_value=0.0).fillna(0.0)
t12m_divs = div_cal.rolling("365D").sum()
t12m_td   = t12m_divs.reindex(trading_dates_naive)

prices_aligned = prices_wide_unadj.copy()
prices_aligned.index = trading_dates_naive

is_payer  = t12m_td > 0.001
yield_raw = t12m_td.div(prices_aligned).where(is_payer)
yield_raw = yield_raw.where(yield_raw <= YIELD_CAP)

# Apply quality mask — no signal for tickers failing price/history filter
yield_clean = yield_raw.where(quality_mask.values)

# Save for attribution and drilldown
out_div_income = DATA_DIR / "div_income.parquet"
t12m_td.index = trading_dates.tz_localize(None)
t12m_td.to_parquet(out_div_income)

out_yield = DATA_DIR / "yield_clean.parquet"
yield_clean.index = trading_dates.tz_localize(None)
yield_clean.to_parquet(out_yield)

print(f"  Avg eligible payers per day: {(is_payer & quality_mask.values).sum(axis=1).mean():.0f}")
print(f"  Saved to {out_div_income}")
print(f"  Saved to {out_yield}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LSE EARNINGS HISTORY → YOY EPS GROWTH (POINT-IN-TIME)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3 - Computing point-in-time YoY EPS growth ...")

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

eps_raw = eps_raw.dropna(subset=["report_date"]).copy()
eps_raw = eps_raw.sort_values(["ticker", "period_date"]).reset_index(drop=True)
eps_raw["eps_lag"] = eps_raw.groupby("ticker")["eps_actual"].shift(EPS_LAG)

eps_raw["yoy_growth"] = np.where(
    eps_raw["eps_lag"].notna()
    & (eps_raw["eps_lag"].abs() > 0.001)
    & (np.sign(eps_raw["eps_actual"]) == np.sign(eps_raw["eps_lag"])),
    eps_raw["eps_actual"] / eps_raw["eps_lag"] - 1.0,
    np.nan,
)
eps_raw["yoy_growth"] = eps_raw["yoy_growth"].clip(-3.0, 5.0)

eps_clean = (
    eps_raw.dropna(subset=["yoy_growth"])
    .sort_values(["ticker", "report_date", "period_date"])
    .groupby(["ticker", "report_date"])
    .last()
    .reset_index()
)
eps_clean["report_date"] = pd.to_datetime(eps_clean["report_date"]).dt.normalize()

eps_wide_sparse = (
    eps_clean.set_index(["report_date", "ticker"])["yoy_growth"]
    .unstack("ticker")
    .reindex(columns=tickers_bt)
)
all_cal2  = pd.date_range(start="2012-01-01", end=PULL_END, freq="D")
eps_daily = eps_wide_sparse.reindex(all_cal2).ffill()
eps_td    = eps_daily.reindex(trading_dates_naive)

# Apply quality mask
eps_td_masked = eps_td.where(quality_mask.values)

out_eps = DATA_DIR / "eps_yoy_raw.parquet"
eps_td_masked.index = trading_dates.tz_localize(None)
eps_td_masked.to_parquet(out_eps)
print(f"  Avg tickers with EPS signal per day: {eps_td_masked.notna().sum(axis=1).mean():.0f}")
print(f"  Saved to {out_eps}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPOSITE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4 - Building composite signal ...")


def cs_winsorize(df: pd.DataFrame, z: float = 2.5) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    lo   = (mean - z * std).values[:, None]
    hi   = (mean + z * std).values[:, None]
    return df.clip(lo, hi)


yield_td   = yield_clean.copy()
yield_td.index = trading_dates.tz_localize(None)
eps_td_use = eps_td_masked.copy()

both_valid = yield_td.notna() & eps_td_use.notna()
yield_only = yield_td.notna() & ~eps_td_use.notna()

yield_w_df  = cs_winsorize(yield_td)
eps_w_df    = cs_winsorize(eps_td_use)
yield_pct   = yield_w_df.rank(axis=1, pct=True)
eps_pct     = eps_w_df.rank(axis=1, pct=True)

composite  = yield_pct.mul(YIELD_W).add(eps_pct.mul(EPS_W))
# Require BOTH signals — yield-only tickers are excluded.
# Allowing yield-only falls back to a pure yield percentile (0–1) which
# systematically outranks tickers that have both signals (composite ≤ YIELD_W+EPS_W).
composite  = composite.where(both_valid)

n_full  = both_valid.sum(axis=1).mean()
n_yfb   = yield_only.sum(axis=1).mean()
n_total = composite.notna().sum(axis=1).mean()
print(f"  Avg tickers in composite (both signals):  {n_full:.0f}")
print(f"  Avg tickers excluded (yield-only, no EPS):{n_yfb:.0f}")
print(f"  Avg total tickers with signal:            {n_total:.0f}")
print(f"  Composite range: [{composite.stack().min():.3f}, {composite.stack().max():.3f}]")

out_comp = DATA_DIR / "composite.parquet"
composite.to_parquet(out_comp)
print(f"\nStep 5 - Saved composite to {out_comp}  shape={composite.shape}")
print(f"  Index tz:    {composite.index.tz}")
print(f"  Index sample: {composite.index[0]}")

print("\nDone: build_signals.py (003_lse_div_eps) complete.")
