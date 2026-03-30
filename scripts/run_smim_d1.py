#!/usr/bin/env python
"""
D1-PERSISTENCE experiment runner.

Analyses gap persistence: do gaps mean-revert, and what is the half-life?
Tests H6a (expected: 4-12 quarter half-life).

Consumes A1 gap results — no new pipeline runs.

Output:
  results/metrics/level4_D1-PERSISTENCE.parquet
  results/configs/D1-PERSISTENCE.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EXP_ID = "D1-PERSISTENCE"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"

TEST_YEARS = list(range(2015, 2025))
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_gap_series() -> tuple[pd.DataFrame, dict[str, str]]:
    """Reconstruct per-actor gap series from A1 pipeline outputs.

    Since A1 stores aggregate metrics (not per-actor gaps), we re-compute
    gaps here using the L1 approach: gap = actual - (per-actor mean + spectral prediction).
    This uses the full intensity panel directly.
    """
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)

    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_data = [a for a in reg["actors"] if a["actor_id"] in actors]
    actor_sector = {a["actor_id"]: a.get("sector", "unknown") for a in actor_data}
    actor_type = {a["actor_id"]: a.get("actor_type", "unknown") for a in actor_data}
    actor_ids = [a["actor_id"] for a in actor_data]

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    wide = wide[cols]

    # Compute gaps as deviation from rolling 10yr mean (same as A1 demeaning)
    gaps = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for ty in TEST_YEARS:
        train_start = pd.Timestamp(f"{ty - 10}-01-01")
        train_end = pd.Timestamp(f"{ty - 1}-12-31")
        test_start = pd.Timestamp(f"{ty}-01-01")
        test_end = pd.Timestamp(f"{ty}-12-31")

        train = wide[(wide.index >= train_start) & (wide.index <= train_end)]
        test = wide[(wide.index >= test_start) & (wide.index <= test_end)]
        actor_mean = train.mean()
        gaps.loc[test.index] = test.values - actor_mean.values[np.newaxis, :]

    gaps = gaps.dropna(how="all")
    return gaps, actor_type


def ar1_halflife(series: np.ndarray) -> tuple[float, float]:
    """Estimate AR(1) coefficient and half-life from a series."""
    y = series[~np.isnan(series)]
    if len(y) < 4:
        return float("nan"), float("nan")
    y_lag = y[:-1]
    y_curr = y[1:]
    if np.std(y_lag) < 1e-10:
        return float("nan"), float("nan")
    # OLS: y_curr = rho * y_lag + intercept
    X = np.column_stack([y_lag, np.ones(len(y_lag))])
    try:
        beta = np.linalg.lstsq(X, y_curr, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    rho = beta[0]
    if abs(rho) >= 1.0 or rho <= 0:
        return rho, float("nan")
    halflife = np.log(0.5) / np.log(abs(rho))
    return rho, halflife


def quintile_transition_matrix(gaps: pd.DataFrame) -> np.ndarray:
    """Compute 5x5 transition matrix of gap quintiles between consecutive periods."""
    transitions = np.zeros((5, 5))
    periods = gaps.index[gaps.notna().any(axis=1)]
    for i in range(len(periods) - 1):
        row_t = gaps.loc[periods[i]].dropna()
        row_t1 = gaps.loc[periods[i + 1]].dropna()
        common = row_t.index.intersection(row_t1.index)
        if len(common) < 10:
            continue
        q_t = pd.qcut(row_t[common], 5, labels=False, duplicates="drop")
        q_t1 = pd.qcut(row_t1[common], 5, labels=False, duplicates="drop")
        for a in common:
            if pd.notna(q_t.get(a)) and pd.notna(q_t1.get(a)):
                transitions[int(q_t[a]), int(q_t1[a])] += 1
    # Normalise rows
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return transitions / row_sums


def main():
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Gap persistence analysis (H6a: half-life 4-12Q)")
    print("=" * 70)

    print("\n  Loading gap series ...", end="", flush=True)
    gaps, actor_type_map = load_gap_series()
    print(f" done. {gaps.shape[1]} actors, {gaps.shape[0]} quarters")

    # --- Per-actor AR(1) half-life ---
    rows = []
    for actor in gaps.columns:
        series = gaps[actor].values.astype(float)
        rho, hl = ar1_halflife(series)
        rows.append({
            "actor_id": actor,
            "actor_type": actor_type_map.get(actor, "unknown"),
            "rho_ar1": rho,
            "halflife_Q": hl,
        })

    df = pd.DataFrame(rows)
    valid = df[df["halflife_Q"].notna() & np.isfinite(df["halflife_Q"]) & (df["halflife_Q"] > 0)]

    print(f"\n  === PER-ACTOR AR(1) HALF-LIFE ===")
    print(f"  Actors with valid half-life: {len(valid)}/{len(df)}")
    print(f"  Median rho: {valid['rho_ar1'].median():.4f}")
    print(f"  Median half-life: {valid['halflife_Q'].median():.1f} quarters")
    print(f"  Mean half-life: {valid['halflife_Q'].mean():.1f} quarters")
    print(f"  P25-P75: [{valid['halflife_Q'].quantile(0.25):.1f}, {valid['halflife_Q'].quantile(0.75):.1f}]")

    # By actor type
    print(f"\n  By actor type:")
    for atype, group in valid.groupby("actor_type"):
        print(f"    {atype:<20} N={len(group):>3}  median_hl={group['halflife_Q'].median():.1f}Q  "
              f"mean_rho={group['rho_ar1'].mean():.3f}")

    # H6a verdict
    median_hl = valid["halflife_Q"].median()
    if 4 <= median_hl <= 12:
        verdict = "SUPPORTED (4-12Q range)"
    elif median_hl < 4:
        verdict = "NOT SUPPORTED (too fast: gaps may be noise)"
    else:
        verdict = "NOT SUPPORTED (too slow: gaps are structural constants)"
    print(f"\n  H6a verdict: median half-life = {median_hl:.1f}Q -> {verdict}")

    # --- Quintile transition matrix ---
    tm = quintile_transition_matrix(gaps)
    print(f"\n  === QUINTILE TRANSITION MATRIX ===")
    print(f"  (rows=current quintile, cols=next-quarter quintile)")
    print(f"  {'':>4}", end="")
    for j in range(5):
        print(f"  Q{j + 1:>2}", end="")
    print()
    for i in range(5):
        print(f"  Q{i + 1:>2}", end="")
        for j in range(5):
            print(f"  {tm[i, j]:>4.0%}", end="")
        print()

    diagonal_persistence = np.mean(np.diag(tm))
    print(f"\n  Diagonal persistence (stay in same quintile): {diagonal_persistence:.1%}")

    # --- Top decile tracking ---
    print(f"\n  === TOP DECILE TRACKING ===")
    periods = gaps.index[gaps.notna().any(axis=1)]
    top_decay = []
    for i in range(len(periods) - 8):
        row = gaps.loc[periods[i]].dropna()
        if len(row) < 20:
            continue
        threshold = row.abs().quantile(0.9)
        top_actors = row[row.abs() >= threshold].index
        # Track mean |gap| over next 8Q
        for lag in range(1, min(9, len(periods) - i)):
            future = gaps.loc[periods[i + lag], top_actors].dropna()
            if len(future) > 0:
                top_decay.append({"lag_Q": lag, "mean_abs_gap": future.abs().mean()})
    if top_decay:
        decay_df = pd.DataFrame(top_decay)
        print(f"  Lag   Mean |gap| of initial top-decile actors")
        for lag in range(1, 9):
            sub = decay_df[decay_df["lag_Q"] == lag]
            if len(sub) > 0:
                print(f"   {lag}Q    {sub['mean_abs_gap'].mean():.4f}")

    print("=" * 70)

    # --- Save ---
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(METRICS_DIR / f"level4_{EXP_ID}.parquet", index=False)
    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_actors": len(df),
        "n_valid_halflife": len(valid),
        "median_halflife_Q": round(float(median_hl), 2),
        "h6a_verdict": verdict,
        "diagonal_persistence": round(float(diagonal_persistence), 4),
        "transition_matrix": tm.tolist(),
    }
    with open(CONFIGS_DIR / f"{EXP_ID}.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"\n  Wrote: results/metrics/level4_{EXP_ID}.parquet")
    print(f"  Wrote: results/configs/{EXP_ID}.yaml")


if __name__ == "__main__":
    main()
