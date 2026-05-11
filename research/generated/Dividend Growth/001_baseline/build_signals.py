#!/usr/bin/env python
"""
Build dividend signals for the dividend_growth_ls strategy.

Outputs
-------
btest/data/dividends/div_composite.parquet
    Wide DataFrame: index=trading_date (tz-naive), columns=ticker
    Values : composite score (0.4 * yield_pct_rank + 0.6 * growth_pct_rank)
    NaN    : ticker paid no dividends in the trailing 12 months
"""
import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

DB = dict(host="localhost", dbname="sfera", user="postgres", password="lokomotiv")
PARQUET_PATH = Path("equities/sp500_daily")
OUT_PATH = Path("research/Dividend Growth/001_baseline/data/div_composite.parquet")


# ── 1. Load SP500 prices ───────────────────────────────────────────────────────
print("Loading sp500_daily...")
prices_long = pd.read_parquet(PARQUET_PATH)
# Keep UTC timestamps EXACTLY as they are (e.g. 2015-01-02 05:00:00+00:00)
# These must match self.index in the engine exactly.
prices_long["date"] = pd.to_datetime(prices_long["date"])
tickers = sorted(prices_long["ticker"].unique())
prices_wide = (
    prices_long[["date", "ticker", "close"]]
    .set_index(["date", "ticker"])["close"]
    .unstack("ticker")
)
trading_dates = prices_wide.index  # UTC e.g. 2015-01-02 05:00:00+00:00
# Tz-naive normalized version for dividend calendar alignment
trading_dates_naive = trading_dates.tz_localize(None).normalize()
print(f"  {len(tickers)} tickers, {len(trading_dates_naive)} trading dates")
print(f"  Date range: {trading_dates_naive.min().date()} → {trading_dates_naive.max().date()}")

# ── 2. Load dividends from Postgres ───────────────────────────────────────────
print("Loading dividends from Postgres...")
tickers_sql = "','".join(tickers)
query = f"""
    SELECT ticker, ex_date, dividend
    FROM eodhd.dividends
    WHERE exchange = 'US'
      AND ticker IN ('{tickers_sql}')
      AND ex_date IS NOT NULL
      AND dividend > 0
    ORDER BY ex_date
"""
conn = psycopg2.connect(**DB)
divs = pd.read_sql(query, conn, parse_dates=["ex_date"])
conn.close()
divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.normalize()
print(f"  {len(divs)} records for {divs['ticker'].nunique()} tickers")

# ── 3. Build daily dividend cash-flow matrix ───────────────────────────────────
# sum multiple dividends on same ex_date (e.g. special + regular)
div_pivot = (
    divs.groupby(["ex_date", "ticker"])["dividend"]
    .sum()
    .unstack("ticker")
    .reindex(columns=tickers)
    .fillna(0.0)
)

# Full calendar range with 400-day lookback buffer so 2015 has a full year
all_cal_dates = pd.date_range(
    start=trading_dates_naive.min() - pd.Timedelta(days=400),
    end=trading_dates_naive.max(),
    freq="D",
)
div_cal = div_pivot.reindex(all_cal_dates, fill_value=0.0).fillna(0.0)

# ── 4. Rolling 12-month sum → trailing yield ───────────────────────────────────
print("Computing rolling 12M sums...")
t12m = div_cal.rolling("365D").sum()
# Shift by 365 calendar days to get the PREVIOUS 12M sum (for YoY growth)
t12m_prev = t12m.shift(365, freq="D")

# Align to SP500 trading dates (tz-naive calendar alignment)
t12m_td      = t12m.reindex(trading_dates_naive)
t12m_prev_td = t12m_prev.reindex(trading_dates_naive)

# ── 5. Yield and growth factors ────────────────────────────────────────────────
is_payer = t12m_td > 0.001   # True for tickers that paid something in last 12M

# Trailing yield: annualised dividend sum / price
# prices_wide has UTC index; strip tz for alignment with t12m_td
prices_aligned = prices_wide.copy()
prices_aligned.index = trading_dates_naive
yield_raw = t12m_td.div(prices_aligned).where(is_payer)
yield_raw = yield_raw.where(yield_raw <= 0.30)   # drop data errors (> 30% yield)

# YoY dividend growth rate (clipped at [-100%, +500%])
growth_raw = (
    t12m_td.div(t12m_prev_td.where(t12m_prev_td > 0.001)) - 1.0
).where(is_payer)
growth_raw = growth_raw.clip(-1.0, 5.0)

# ── 6. Cross-sectional winsorize (z=2.5) before ranking ───────────────────────
def cs_winsorize(df: pd.DataFrame, z: float = 2.5) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1)
    lo   = (mean - z * std).values[:, None]
    hi   = (mean + z * std).values[:, None]
    return df.clip(lo, hi)

yield_w  = cs_winsorize(yield_raw)
growth_w = cs_winsorize(growth_raw)

# ── 7. Percentile ranks then composite ────────────────────────────────────────
yield_pct  = yield_w.rank(axis=1, pct=True)
growth_pct = growth_w.rank(axis=1, pct=True)

# Composite: 40% yield + 60% growth
composite = yield_pct.mul(0.4).add(growth_pct.mul(0.6))

# Fallback: new payer (< 2 years history) with no prior-year data → yield rank only
mask_no_growth = growth_pct.isna() & yield_pct.notna()
composite = composite.where(~mask_no_growth, yield_pct)

# Enforce: non-payers → NaN (engine will see no signal for them)
composite = composite.where(is_payer, float("nan"))

# ── 8. Diagnostics ────────────────────────────────────────────────────────────
avg_payers = is_payer.sum(axis=1).mean()
sample_2020 = composite.loc["2020"].notna().sum(axis=1).mean()
print(f"  Avg payers/trading-day (full period): {avg_payers:.1f}")
print(f"  Avg composite non-null tickers (2020): {sample_2020:.1f}")
print(f"  Composite range: [{composite.stack().min():.3f}, {composite.stack().max():.3f}]")

# ── 9. Restore tz-naive 05:00:00 index to match engine's prices.index ─────────
# DataLoader strips timezone from the parquet timestamps, so self.index in the
# FactorEngine is tz-naive with times at 05:00:00 (e.g. 2015-01-02 05:00:00).
# composite was computed on midnight tz-naive dates; reassign to the exact
# tz-stripped UTC timestamps so ExternalFactor.reindex() gets exact matches.
composite.index = trading_dates.tz_localize(None)

# ── 10. Save ──────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
composite.to_parquet(OUT_PATH)
print(f"\nSaved → {OUT_PATH}   shape={composite.shape}")
print(f"Index tz: {composite.index.tz}  sample: {composite.index[0]}")
