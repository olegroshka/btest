"""
GTA 6 / TTWO Pre-Release Drift Research
=========================================
Hypothesis: Take-Two Interactive (TTWO) rallies in the N weeks before a major
Rockstar release and sells off after. If true, and GTA 6 is Nov 2026, we want
to be long TTWO from ~60 days before release — contingent on postponement risk
staying low (Polymarket signal).

Historical events studied:
  - GTA 5       :  17 Sep 2013
  - GTA 5 PC    :  14 Apr 2015
  - Red Dead 2  :  26 Oct 2018
  - GTA Online (Cayo Perico): 15 Dec 2020  (DLC — smaller effect)
  - GTA 6       : ~Nov 2026  (target trade)

Polymarket gate:
  "GTA 6 launch postponed again?" — current price 0.32 (32% probability)
  We only enter if postponement prob < POSTPONE_THRESHOLD (default 0.20)
  i.e. market must be >80% confident release is on schedule.

Outputs:
  - Pre/post return windows for each historical release
  - Cumulative return chart (pylight via matplotlib or signum)
  - Forward trade setup for GTA 6 Nov 2026

Run from btest venv (no tensorflow needed):
  cd btest
  .venv\Scripts\python.exe "research/GTA6/ttwo_prerelease.py"
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
TICKER = "TTWO"

RELEASES = {
    "GTA 5":         pd.Timestamp("2013-09-17"),
    "GTA 5 PC":      pd.Timestamp("2015-04-14"),
    "Red Dead 2":    pd.Timestamp("2018-10-26"),
    "Cayo Perico":   pd.Timestamp("2020-12-15"),
    "GTA 6 (fwd)":   pd.Timestamp("2026-11-01"),   # update when confirmed
}

WINDOWS = [-60, -45, -30, -20, -10, -5, 0, +5, +10, +20, +30]  # days relative to release

# Polymarket postponement gate (live snapshot — update daily from pipeline)
POSTPONE_PROB       = 0.32   # current market price of "GTA 6 postponed again?"
POSTPONE_THRESHOLD  = 0.20   # only trade if postponement prob < this

# ── Download TTWO ─────────────────────────────────────────────────────────────
print(f"Downloading {TICKER}...")
ttwo = yf.download(TICKER, start="2010-01-01", progress=False)["Close"].squeeze()
ttwo.index = pd.DatetimeIndex(ttwo.index).normalize()
print(f"  {ttwo.index[0].date()} → {ttwo.index[-1].date()}  ({len(ttwo):,} trading days)")

log_ret = np.log(ttwo / ttwo.shift(1)).dropna()

# ── Pre/post return windows per release ──────────────────────────────────────
def nearest_trading_day(ts, index):
    """Return nearest date in index at or after ts."""
    loc = index.searchsorted(ts)
    return index[min(loc, len(index) - 1)]

print("\n=== PRE/POST RELEASE RETURNS ===")
rows = []
for name, release_date in RELEASES.items():
    if name == "GTA 6 (fwd)":
        continue
    rd = nearest_trading_day(release_date, ttwo.index)
    rd_idx = ttwo.index.get_loc(rd)
    row = {"Event": name, "Release": rd.date()}
    for w in WINDOWS:
        target_idx = rd_idx + w
        if 0 <= target_idx < len(ttwo):
            r = ttwo.iloc[target_idx] / ttwo.iloc[rd_idx] - 1
            row[f"d{w:+d}"] = round(r * 100, 1)
        else:
            row[f"d{w:+d}"] = None
    rows.append(row)

summary = pd.DataFrame(rows).set_index("Event")
print(summary.to_string())

# ── Average pre-release drift ─────────────────────────────────────────────────
pre_cols = [f"d{w:+d}" for w in WINDOWS if w <= 0]
post_cols = [f"d{w:+d}" for w in WINDOWS if w >= 0]

print("\n=== AVERAGE DRIFT ACROSS HISTORICAL RELEASES ===")
avg = summary[pre_cols + post_cols].mean()
for col, val in avg.items():
    bar = "█" * int(abs(val) / 1) if not np.isnan(val) else ""
    sign = "+" if val >= 0 else ""
    print(f"  {col:8s}: {sign}{val:.1f}%  {bar}")

# ── Best entry/exit window ────────────────────────────────────────────────────
best_pre  = avg[pre_cols].idxmin()   # most negative (cheapest entry)
best_exit = avg[post_cols].idxmax()  # most positive (best exit)
best_pre_val  = avg[best_pre]
best_exit_val = avg[best_exit]

print(f"\n  Best avg entry : {best_pre}  ({best_pre_val:+.1f}% vs release day)")
print(f"  Best avg exit  : {best_exit}  ({best_exit_val:+.1f}% vs release day)")

# ── Volatility-adjusted Sharpe of the strategy ───────────────────────────────
# Simple: buy N_ENTRY days before, sell N_EXIT days after, per release
N_ENTRY = -60
N_EXIT  = +10

trade_returns = []
for name, release_date in RELEASES.items():
    if name == "GTA 6 (fwd)":
        continue
    rd = nearest_trading_day(release_date, ttwo.index)
    rd_idx = ttwo.index.get_loc(rd)
    entry_idx = rd_idx + N_ENTRY
    exit_idx  = rd_idx + N_EXIT
    if entry_idx >= 0 and exit_idx < len(ttwo):
        r = ttwo.iloc[exit_idx] / ttwo.iloc[entry_idx] - 1
        trade_returns.append({"Event": name, "Return": round(r * 100, 1),
                              "Entry": ttwo.index[entry_idx].date(),
                              "Exit":  ttwo.index[exit_idx].date()})

print(f"\n=== INDIVIDUAL TRADES (buy d{N_ENTRY:+d}, sell d{N_EXIT:+d}) ===")
trades_df = pd.DataFrame(trade_returns)
print(trades_df.to_string(index=False))
print(f"\n  Mean return : {trades_df['Return'].mean():+.1f}%")
print(f"  Win rate    : {(trades_df['Return'] > 0).mean():.0%}")
print(f"  Best        : {trades_df['Return'].max():+.1f}%")
print(f"  Worst       : {trades_df['Return'].min():+.1f}%")

# ── Forward trade: GTA 6 Nov 2026 ─────────────────────────────────────────────
gta6_release = RELEASES["GTA 6 (fwd)"]
gta6_entry   = gta6_release + pd.tseries.offsets.BDay(N_ENTRY)
gta6_exit    = gta6_release + pd.tseries.offsets.BDay(N_EXIT)
today        = pd.Timestamp.today().normalize()
days_to_entry = np.busday_count(today.date(), gta6_entry.date())
days_to_release = np.busday_count(today.date(), gta6_release.date())

print(f"\n=== FORWARD TRADE: GTA 6 (expected {gta6_release.date()}) ===")
print(f"  Target entry : {gta6_entry.date()}  ({days_to_entry:+d} trading days from today)")
print(f"  Target exit  : {gta6_exit.date()}")
print(f"  Days to release : {days_to_release} trading days")
print(f"  TTWO current : ${ttwo.iloc[-1]:.2f}  (as of {ttwo.index[-1].date()})")
print()
print(f"  Polymarket gate:")
print(f"    'GTA 6 postponed again?' = {POSTPONE_PROB:.0%}")
print(f"    Entry threshold           = {POSTPONE_THRESHOLD:.0%}")
if POSTPONE_PROB < POSTPONE_THRESHOLD:
    print(f"    ✅ GATE OPEN  — proceed with entry plan")
else:
    diff = POSTPONE_PROB - POSTPONE_THRESHOLD
    print(f"    ❌ GATE CLOSED — postponement risk too high ({diff:.0%} above threshold)")
    print(f"       Wait for market to price below {POSTPONE_THRESHOLD:.0%} before entering")
print()
print(f"  Expected return (hist avg): {trades_df['Return'].mean():+.1f}%")
print(f"  TTWO target price at exit : ${ttwo.iloc[-1] * (1 + trades_df['Return'].mean()/100):.2f}")

# ── Quick ASCII equity curve ──────────────────────────────────────────────────
print("\n=== TTWO 2010–TODAY (annual returns) ===")
annual = ttwo.resample('YE').last().pct_change().dropna()
for yr, r in annual.items():
    bar = ("█" * int(abs(r * 100) / 5))[:30]
    sign = "+" if r >= 0 else "-"
    color = sign
    print(f"  {yr.year}: {sign}{abs(r)*100:5.1f}%  {bar}")
