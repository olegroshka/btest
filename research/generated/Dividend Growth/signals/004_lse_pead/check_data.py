"""
LSE PEAD — 001_baseline — check_data.py
Run after build_signals.py, before strategy.py. Exits 1 on any FAIL.

    uv run python "research/generated/LSE PEAD/001_baseline/check_data.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

START = "2015-01-01"
END   = "2026-01-01"
SUE_CLIP      = 5.0
MIN_ACTIVE    = 3    # WARN threshold — PEAD is naturally sparse (holiday gaps have 1–2 tickers)
MEDIAN_ACTIVE = 15   # FAIL if median active tickers < this


def _chk(label: str, ok: bool, detail: str, level: str = "FAIL") -> bool:
    print(f"  [{'OK  ' if ok else level}] {label}: {detail}")
    return ok


def main() -> int:
    fails = 0
    print("=== LSE PEAD 001_baseline — Data Quality Check ===\n")

    sue    = pd.read_parquet(DATA_DIR / "sue_signal.parquet")
    prices = pd.read_parquet(DATA_DIR / "lse_prices.parquet")
    events = pd.read_parquet(DATA_DIR / "events.parquet")

    sue.index = pd.to_datetime(sue.index)
    prices["date"] = pd.to_datetime(prices["date"])
    events["report_date"] = pd.to_datetime(events["report_date"])

    sue_bt = sue.loc[START:END]
    active = sue_bt.notna().sum(axis=1)

    # ── Coverage ──────────────────────────────────────────────────────────────
    print("-- Coverage --")
    active_nonzero = active[active > 0]
    min_active = int(active_nonzero.min()) if not active_nonzero.empty else 0
    ok = _chk("Min active tickers/day", min_active >= MIN_ACTIVE,
              f"{min_active} (threshold {MIN_ACTIVE})", "WARN")

    med_active = active.median()
    ok = _chk("Median active tickers/day", med_active >= MEDIAN_ACTIVE,
             f"{med_active:.0f} (threshold {MEDIAN_ACTIVE})")
    if not ok:
        fails += 1

    events_bt = events[(events["report_date"] >= START) & (events["report_date"] < END)]
    by_year = events_bt.groupby(events_bt["report_date"].dt.year).size()
    worst_year, worst_n = int(by_year.idxmin()), int(by_year.min())
    ok = _chk("Min events/year", worst_n >= 20,
              f"{worst_n} in {worst_year}")
    if not ok:
        fails += 1

    # ── SUE distribution ──────────────────────────────────────────────────────
    print("\n-- SUE Distribution --")
    flat = sue_bt.stack().dropna()
    ok = _chk("SUE clipped at ±5σ", (flat.abs() <= SUE_CLIP + 0.01).all(),
              f"max_abs={flat.abs().max():.2f}")
    if not ok:
        fails += 1

    ok = _chk("No all-NaN columns in backtest window",
              sue_bt.notna().any(axis=0).all(),
              f"{sue_bt.notna().any(axis=0).sum()} / {sue_bt.shape[1]} tickers have ≥1 event")
    if not ok:
        fails += 1

    print(f"  [INFO] SUE: mean={flat.mean():.2f}  std={flat.std():.2f}  "
          f"p5={flat.quantile(0.05):.2f}  p95={flat.quantile(0.95):.2f}  n={len(flat)}")

    # ── Prices ────────────────────────────────────────────────────────────────
    print("\n-- Prices --")
    ok = _chk("No negative/zero close", (prices["close"] > 0).all(),
              f"min={prices['close'].min():.2f}")
    if not ok:
        fails += 1

    px_wide = prices.set_index(["date", "ticker"])["close"].unstack()
    spikes  = int((px_wide.pct_change(fill_method=None).abs() > 0.5).sum().sum())
    _chk("Price spikes >50% daily", spikes == 0, f"{spikes} events", "WARN")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  [INFO] Universe: {sue.shape[1]} tickers | "
          f"Price rows: {len(prices)} | Events in backtest window: {len(events_bt)}")
    print(f"  [INFO] Date range: {sue_bt.index[0].date()} → {sue_bt.index[-1].date()}")

    if fails:
        print(f"\n  ❌ {fails} FAIL(s) — fix before running strategy.py")
        return 1
    print("\n  ✅ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
