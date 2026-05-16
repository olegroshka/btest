"""
LSE PEAD — 002_quality_filter — build_signals.py
==================================================
Preprocesses the 002_fixed composite parquet for use as an ExternalFactor.

The 002_fixed composite has timestamps at 05:00:00 (UTC offset artefact).
The PEAD prices index uses 00:00:00. This mismatch causes the engine to
produce all-NaN values for composite_raw.

Fix: normalize the composite index to midnight and reindex to the PEAD
trading calendar, forward-filling (composite is a slow-moving signal — safe
to carry last value forward until next update).

No new data is pulled. This runs in seconds.

Run from btest/ root:
    uv run python "research/generated/LSE PEAD/002_quality_filter/build_signals.py"
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BTEST_ROOT   = Path(__file__).resolve().parents[4]
PEAD_001     = BTEST_ROOT / "research/generated/LSE PEAD/001_baseline"
DIV_EPS_002  = BTEST_ROOT / "research/generated/LSE Dividend EPS/002_fixed"
DATA_DIR     = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Load source data ──────────────────────────────────────────────────────────
print("Loading 002_fixed composite and PEAD prices calendar …")
comp  = pd.read_parquet(DIV_EPS_002 / "data" / "composite.parquet")
prices = pd.read_parquet(PEAD_001 / "data" / "lse_prices.parquet")

# ── Normalize composite index to midnight ─────────────────────────────────────
# 002_fixed was built with XLON trading_dates that have 05:00:00 time components.
# Normalize to date-only (midnight) so engine alignment works.
comp.index = pd.to_datetime(comp.index.date)
print(f"  Composite: {comp.shape} | index sample: {comp.index[:2].tolist()}")

# ── Reindex to PEAD trading calendar ─────────────────────────────────────────
# The PEAD prices may have slightly different dates. Reindex + ffill carries
# quality scores forward to any trading day the PEAD strategy might need.
pead_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
comp_aligned = comp.reindex(pead_dates, method="ffill")
print(f"  After reindex to PEAD calendar: {comp_aligned.shape}")
print(f"  Non-null coverage: {comp_aligned.notna().mean().mean():.1%}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = DATA_DIR / "composite_aligned.parquet"
comp_aligned.to_parquet(out_path)
print(f"\n✓  Saved aligned composite → {out_path}")
print(f"   Index dtype: {comp_aligned.index.dtype}  tz: {comp_aligned.index.tz}")
print(f"   Index sample: {comp_aligned.index[:2].tolist()}")
