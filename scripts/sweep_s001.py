"""
sweep_s001.py - Parameter sweep for S001 LSE Yield + Dividend Growth
=====================================================================
Sweeps:
    yield_weight in [0.2, 0.4, 0.5, 0.6, 0.8]
    long_n       in [20, 25, 30, 40]

Optimizations:
  - Data loaded once per worker process via module-level cache in data_loader.py.
  - Runs dispatched to ProcessPoolExecutor (N_WORKERS = cpu_count // 2).
  - No temp files: strategy.py uses an ExternalFactor loader to blend ranks on-the-fly.

Run from btest/ root:
    uv run python scripts/sweep_s001.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BTEST_ROOT = Path(__file__).parent.parent
SIGNAL_DIR = BTEST_ROOT / "research/generated/Dividend Growth/signals/001_yield_growth_composite"

sys.path.insert(0, str(BTEST_ROOT / "src"))

YIELD_WEIGHTS = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
LONG_NS       = [10, 15, 20, 25, 30, 40]
ANN           = 252
N_WORKERS     = max(1, os.cpu_count() // 2)


# Module-level helpers - must be at top level for ProcessPoolExecutor pickling

def _load_strategy_module():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("strategy_s001", str(SIGNAL_DIR / "strategy.py"))
    mod  = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute_metrics(result) -> dict:
    # equity/returns can be a Series or single-column DataFrame depending on reporting config
    eq = result.equity.squeeze().dropna()
    r  = result.returns.squeeze().dropna()

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years   = len(r) / ANN
    cagr      = (1 + total_ret) ** (1 / n_years) - 1
    sharpe    = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    dd        = (eq - eq.cummax()) / eq.cummax()
    max_dd    = float(dd.min())
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0.0
    ann_vol   = float(r.std() * np.sqrt(ANN))

    if hasattr(result, "weights") and result.weights is not None and not result.weights.empty:
        w        = result.weights.fillna(0)
        n_yrs    = len(w) / ANN
        turnover = float(w.diff().abs().sum(axis=1).sum() / 2 / n_yrs) if n_yrs > 0 else float("nan")
    else:
        turnover = float("nan")

    return {
        "sharpe"  : round(sharpe,   3),
        "cagr"    : round(cagr,     4),
        "max_dd"  : round(max_dd,   4),
        "calmar"  : round(calmar,   3),
        "ann_vol" : round(ann_vol,  4),
        "turnover": round(turnover, 2),
    }


def _run_single(args: tuple) -> dict:
    """
    Worker entry-point - one (yield_weight, long_n) backtest per call.
    The module-level _DATA_CACHE in data_loader.py ensures the parquet is
    read only once per worker process, regardless of how many runs it handles.
    """
    yw, n = args
    warnings.filterwarnings("ignore")

    _src = str(Path(__file__).parent.parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    t0 = time.time()
    try:
        mod    = _load_strategy_module()
        strat  = mod.build_strategy(long_n=n, yield_weight=yw, suppress_output=True)

        from quantdsl_backtest.engine.backtest_runner import run_backtest
        result = run_backtest(strat)
        m      = _compute_metrics(result)
        return {"yield_weight": yw, "long_n": n, "elapsed": round(time.time() - t0, 1),
                "error": None, **m}
    except Exception:
        return {"yield_weight": yw, "long_n": n, "elapsed": round(time.time() - t0, 1),
                "error": traceback.format_exc(),
                "sharpe": float("nan"), "cagr": float("nan"), "max_dd": float("nan"),
                "calmar": float("nan"), "ann_vol": float("nan"), "turnover": float("nan")}


def main() -> None:
    grid    = [(yw, n) for yw in YIELD_WEIGHTS for n in LONG_NS]
    n_total = len(grid)

    print(f"Sweep: {len(YIELD_WEIGHTS)} yield_weights x {len(LONG_NS)} long_ns = {n_total} runs")
    print(f"yield_weights : {YIELD_WEIGHTS}")
    print(f"long_ns       : {LONG_NS}")
    print(f"Workers       : {N_WORKERS}  (data cached once per worker process)")
    print()

    rows    = [None] * n_total
    idx_map = {args: i for i, args in enumerate(grid)}
    done    = 0
    t0      = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single, args): args for args in grid}
        for fut in as_completed(futures):
            args  = futures[fut]
            row   = fut.result()
            rows[idx_map[args]] = row
            done += 1
            yw, n = args
            if row["error"]:
                last_line = row["error"].strip().splitlines()[-1]
                print(f"[{done:2d}/{n_total}]  yw={yw:.1f}  n={n:2d}  ERROR: {last_line}")
            else:
                print(f"[{done:2d}/{n_total}]  yw={yw:.1f}  n={n:2d}  "
                      f"sharpe={row['sharpe']:.3f}  cagr={row['cagr']:.1%}  "
                      f"dd={row['max_dd']:.1%}  {row['elapsed']:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s  ({elapsed/n_total:.1f}s/run avg)\n")

    df = pd.DataFrame(rows)
    df["cagr_%"]   = (df["cagr"]   * 100).round(2)
    df["max_dd_%"] = (df["max_dd"] * 100).round(2)
    df["gw"]       = (1 - df["yield_weight"]).round(2)

    cols = ["yield_weight", "gw", "long_n", "sharpe", "cagr_%", "max_dd_%", "calmar", "turnover"]
    print("=" * 80)
    print("FULL RESULTS  (sorted by Sharpe)")
    print("=" * 80)
    print(df[cols].sort_values("sharpe", ascending=False).to_string(index=False))

    valid = df.dropna(subset=["sharpe"])
    if valid.empty:
        print("\nAll runs failed - check errors above.")
        return

    best   = valid.loc[valid["sharpe"].idxmax()]
    best_c = valid.loc[valid["calmar"].idxmax()]

    print(f"\n{'='*80}")
    print(f"BEST by Sharpe : yield_weight={best['yield_weight']:.1f}  long_n={int(best['long_n'])}")
    print(f"  Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.1%}  MaxDD={best['max_dd']:.1%}  Calmar={best['calmar']:.3f}")
    print(f"\nBEST by Calmar : yield_weight={best_c['yield_weight']:.1f}  long_n={int(best_c['long_n'])}")
    print(f"  Sharpe={best_c['sharpe']:.3f}  CAGR={best_c['cagr']:.1%}  MaxDD={best_c['max_dd']:.1%}  Calmar={best_c['calmar']:.3f}")
    print(f"{'='*80}")

    pivot = df.pivot(index="yield_weight", columns="long_n", values="sharpe").round(3)
    print(f"\nSHARPE HEATMAP  (yield_weight x long_n)\n{pivot.to_string()}")

    out_path = SIGNAL_DIR / "outputs" / "sweep_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()