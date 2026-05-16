"""
scripts/sweep_vol_threshold.py
==============================
Sweep vol_z_threshold for S001 vol-control strategy.
All other params fixed at best sweep values: yw=0.5, n=10.

Also includes baseline (no vol control = very high threshold) for comparison.

Run from btest/ root:
    uv run python "research/generated/Dividend Growth/signals/001_yield_growth_composite/sweep_vol_threshold.py"
"""
from __future__ import annotations

import sys, warnings, math
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import importlib.util as ilu
from pathlib import Path

import pandas as pd

# ── Thresholds to test (plus a "disabled" sentinel at 99 = always invested) ──
THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 99.0]
ANN = 252

# ── Load strategy module ──────────────────────────────────────────────────────
_spec = ilu.spec_from_file_location("s001_vc", str(Path("research/generated/Dividend Growth/signals/001_yield_growth_composite/strategy_vol_control.py")))
_mod  = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from quantdsl_backtest.engine.backtest_runner import run_backtest


def _metrics(result, threshold: float) -> dict:
    eq = result.equity.squeeze().dropna()
    r  = result.returns.squeeze().dropna()
    n_yrs = len(r) / ANN

    sharpe   = float(r.mean() / r.std() * math.sqrt(ANN)) if r.std() > 0 else float("nan")
    total_r  = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else float("nan")
    cagr     = float((1 + total_r) ** (1 / n_yrs) - 1) if n_yrs > 0 else float("nan")

    roll_max = eq.cummax()
    dd       = (eq - roll_max) / roll_max
    max_dd   = float(dd.min())
    calmar   = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    avg_lev  = float(result.summary().get("avg_leverage", float("nan"))) \
               if hasattr(result, "summary") else float("nan")

    label = "baseline (no filter)" if threshold >= 90 else f"vol_z < {threshold}"
    return dict(
        threshold=threshold,
        label=label,
        sharpe=round(sharpe, 3),
        cagr_pct=round(cagr * 100, 2),
        max_dd_pct=round(max_dd * 100, 2),
        calmar=round(calmar, 3),
    )


rows = []
for thr in THRESHOLDS:
    label = "baseline" if thr >= 90 else f"vz<{thr}"
    print(f"  running {label} ...", end="", flush=True)
    try:
        strat  = _mod.build_strategy(vol_z_threshold=thr, suppress_output=True)
        result = run_backtest(strat)
        row    = _metrics(result, thr)
        rows.append(row)
        print(f"  Sharpe={row['sharpe']:.3f}  DD={row['max_dd_pct']:.1f}%  Calmar={row['calmar']:.3f}")
    except Exception as e:
        print(f"  FAILED: {e}")
        rows.append(dict(threshold=thr, label=label,
                         sharpe=float("nan"), cagr_pct=float("nan"),
                         max_dd_pct=float("nan"), calmar=float("nan")))

df = pd.DataFrame(rows)

print("\n" + "=" * 72)
print("VOL THRESHOLD SWEEP  (S001 yw=0.5, n=10, with DrawdownPolicy)")
print("=" * 72)
print(df.to_string(index=False))
print()

# Best by Sharpe and Calmar
best_s = df.loc[df.sharpe.idxmax()]
best_c = df.loc[df.calmar.idxmax()]
print(f"Best Sharpe : threshold={best_s.threshold}  sharpe={best_s.sharpe}  dd={best_s.max_dd_pct}%  calmar={best_s.calmar}")
print(f"Best Calmar : threshold={best_c.threshold}  sharpe={best_c.sharpe}  dd={best_c.max_dd_pct}%  calmar={best_c.calmar}")

# Save
out_dir = Path("research/generated/Dividend Growth/signals/001_yield_growth_composite/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "vol_threshold_sweep.csv", index=False)
print(f"\nSaved → {out_dir / 'vol_threshold_sweep.csv'}")
