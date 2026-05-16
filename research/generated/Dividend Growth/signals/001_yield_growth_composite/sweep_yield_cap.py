"""
Yield-cap sweep for S001 — tests YIELD_CAP ∈ {0.15, 0.20, 0.25, 0.30}
Each iteration: rebuilds signals → re-runs strategy → reads summary metrics.

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/001_yield_growth_composite/sweep_yield_cap.py"
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SIGNAL_DIR = Path(__file__).resolve().parent
BTEST_ROOT = SIGNAL_DIR.parents[4]   # …/btest/  (signal/ → Dividend Growth/ → generated/ → research/ → btest/)
OUTPUTS    = SIGNAL_DIR / "outputs" / "vol_control"

BUILD_SCRIPT    = str(SIGNAL_DIR / "build_signals.py")
STRATEGY_SCRIPT = str(SIGNAL_DIR / "strategy_vol_control.py")

CAPS = [0.15, 0.20, 0.25, 0.30]

results = []

for cap in CAPS:
    label = f"{cap*100:.0f}%"
    print(f"\n{'='*60}")
    print(f"  Yield cap = {label}  →  rebuilding signals ...")
    print(f"{'='*60}")

    # 1. Rebuild signals with this cap (suppress verbose output)
    ret = subprocess.run(
        [sys.executable, BUILD_SCRIPT, "--yield-cap", str(cap)],
        cwd=str(BTEST_ROOT),
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  ⚠️  build_signals.py failed for cap={label}:\n{ret.stderr[-500:]}")
        continue

    # 2. Re-run vol-control strategy (suppress verbose output)
    print(f"  Running strategy ...")
    ret = subprocess.run(
        [sys.executable, STRATEGY_SCRIPT],
        cwd=str(BTEST_ROOT),
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  ⚠️  strategy_vol_control.py failed for cap={label}:\n{ret.stderr[-800:]}")
        continue

    # 3. Read metrics from summary.json
    raw = json.loads((OUTPUTS / "summary.json").read_text())
    m   = raw.get("metrics", raw)

    results.append({
        "Yield cap"    : label,
        "Sharpe"       : round(float(m.get("sharpe",       m.get("sharpe_ratio",      0))), 3),
        "CAGR"         : f"{float(m.get('cagr', 0))*100:.1f}%",
        "Max DD"       : f"{float(m.get('max_drawdown', 0))*100:.1f}%",
        "Calmar"       : round(float(m.get("calmar", 0)), 3),
        "Ann Vol"      : f"{float(m.get('ann_volatility', m.get('volatility', 0)))*100:.1f}%",
        "Total Return" : f"{float(m.get('total_return', 0))*100:.1f}%",
    })
    print(f"\n  ✓  cap={label}  Sharpe={results[-1]['Sharpe']}  CAGR={results[-1]['CAGR']}  MaxDD={results[-1]['Max DD']}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  YIELD CAP SWEEP — RESULTS")
print(f"{'='*60}")
if results:
    df = pd.DataFrame(results).set_index("Yield cap")
    print(df.to_string())
    print(f"\nBest Sharpe: {df['Sharpe'].astype(float).idxmax()}  ({df['Sharpe'].astype(float).max():.3f})")
else:
    print("  No results collected — check errors above.")
