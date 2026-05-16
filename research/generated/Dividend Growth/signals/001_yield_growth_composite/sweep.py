#!/usr/bin/env python
"""
sweep.py — US_FUNDAMENTAL_DIVGROWTH_S001
=========================================
Runs all planned variants in sequence. Reads variants from plan.yaml;
each shares the same data/div_composite.parquet — no rebuild needed.

Results written to outputs/sweep/<variant_id>/summary.json
Comparison table written to outputs/sweep_results.json

Run from btest/ root:
    uv run python "research/Dividend Growth/signals/001_yield_growth_composite/sweep.py"
"""
from __future__ import annotations

import json
from pathlib import Path

from strategy import build_strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig

SIGNAL_DIR = Path("research/generated/Dividend Growth/signals/001_yield_growth_composite")
SWEEP_DIR  = SIGNAL_DIR / "outputs" / "sweep"

VARIANTS = [
    # id            long_n  short_n  rebalance  start         end
    ("v1_baseline",     50,      20,   "1w",  "2015-01-01", "2025-01-01"),
    ("v2_higher_yield", 50,      20,   "1w",  "2015-01-01", "2025-01-01"),  # build_signals change needed; run as-is for now
    ("v3_long_only",    30,       0,   "1m",  "2015-01-01", "2025-01-01"),
    ("v4_monthly",      50,      20,   "1m",  "2015-01-01", "2025-01-01"),
    ("v5_top25_bot10",  25,      10,   "1w",  "2015-01-01", "2025-01-01"),
]

summary_rows: list[dict] = []

for variant_id, long_n, short_n, rebalance, start, end in VARIANTS:
    print(f"\n{'═' * 55}")
    print(f"  Running {variant_id}  long_n={long_n}  short_n={short_n}  rebalance={rebalance}")
    print(f"{'═' * 55}")

    out_dir = SWEEP_DIR / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch output_dir for this variant
    strategy = build_strategy(long_n=long_n, short_n=short_n, rebalance=rebalance,
                               start=start, end=end)
    strategy.backtest.reporting.output_dir = str(out_dir)
    strategy.backtest.reporting.strategyAnalytics = StrategyAnalyticsConfig(
        title=f"Div Growth | {variant_id} | L{long_n}/S{short_n} {rebalance}"
    )

    result = run_backtest(strategy)
    s = result.summary()

    # Extract key metrics
    metrics = s.get("metrics", s) if isinstance(s, dict) else {}
    row = {
        "variant_id":     variant_id,
        "long_n":         long_n,
        "short_n":        short_n,
        "rebalance":      rebalance,
        "sharpe":         round(metrics.get("sharpe", float("nan")), 3),
        "cagr":           round(metrics.get("cagr", float("nan")), 4),
        "max_drawdown":   round(metrics.get("max_drawdown", float("nan")), 4),
        "calmar":         round(metrics.get("calmar", float("nan")), 3),
        "turnover_annual": round(metrics.get("turnover_annual", float("nan")), 2),
    }
    summary_rows.append(row)
    print(f"  Sharpe={row['sharpe']}  CAGR={row['cagr']:.2%}  MaxDD={row['max_drawdown']:.2%}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═' * 80}")
print(f"  SWEEP RESULTS — US_FUNDAMENTAL_DIVGROWTH_S001")
print(f"{'═' * 80}")
hdr = f"{'Variant':<22} {'LongN':>6} {'ShortN':>7} {'Reb':>5} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}"
print(f"  {hdr}")
print(f"  {'─' * 76}")
for r in summary_rows:
    print(f"  {r['variant_id']:<22} {r['long_n']:>6} {r['short_n']:>7} "
          f"{r['rebalance']:>5} {r['sharpe']:>8.3f} {r['cagr']:>8.2%} "
          f"{r['max_drawdown']:>8.2%} {r['calmar']:>8.3f}")

best = max(summary_rows, key=lambda x: x["sharpe"])
print(f"\n  Best Sharpe : {best['variant_id']} ({best['sharpe']})")

# Save
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
out_path = SIGNAL_DIR / "outputs" / "sweep_results.json"
with open(out_path, "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"  Saved → {out_path}")
