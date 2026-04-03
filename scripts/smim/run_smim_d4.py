#!/usr/bin/env python
"""
D4-EVENTS experiment runner.

Do gaps spike around known misallocation episodes?
Tests 8 historical events for gap alignment.
Target: >=6/8 events aligned.

Output:
  results/metrics/level4_D4-EVENTS.parquet
  results/configs/D4-EVENTS.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EXP_ID = "D4-EVENTS"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"

# Event catalog
EVENTS = [
    {"name": "Energy over-investment 2012-14", "date": "2013-07-01", "sector": "energy", "sign": "POSITIVE"},
    {"name": "Pre-GFC bank leverage", "date": "2007-07-01", "sector": "financials", "sign": "POSITIVE"},
    {"name": "Post-GFC tech under-invest", "date": "2010-01-01", "sector": "technology", "sign": "NEGATIVE"},
    {"name": "COVID capital reallocation", "date": "2020-04-01", "sector": "ALL", "sign": "MIXED"},
    {"name": "2022 energy super-profits", "date": "2022-04-01", "sector": "energy", "sign": "POSITIVE"},
    {"name": "Fed tightening 2022", "date": "2022-07-01", "sector": "ALL", "sign": "MIXED"},
    {"name": "AI investment surge 2023", "date": "2023-07-01", "sector": "technology", "sign": "POSITIVE"},
    {"name": "Regional bank stress 2023", "date": "2023-01-01", "sector": "financials", "sign": "NEGATIVE"},
]


def load_intensity_panel():
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_data = {a["actor_id"]: a for a in reg["actors"] if a["actor_id"] in actors}
    actor_sector = {aid: a.get("sector", "unknown") for aid, a in actor_data.items()}

    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    return wide, actor_sector


def compute_gaps(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute gaps as deviation from expanding 10yr mean."""
    gaps = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for i in range(len(wide)):
        if i < 8:  # need at least 2yr history
            continue
        lookback = max(0, i - 40)  # 10yr
        historical_mean = wide.iloc[lookback:i].mean()
        gaps.iloc[i] = wide.iloc[i] - historical_mean
    return gaps


def analyse_event(event, gaps, actor_sector):
    """Analyse one event for gap alignment."""
    event_date = pd.Timestamp(event["date"])
    sector = event["sector"]
    expected_sign = event["sign"]

    # Select actors
    if sector == "ALL":
        actors = list(gaps.columns)
    else:
        actors = [a for a in gaps.columns if actor_sector.get(a) == sector]

    if not actors:
        return {"aligned": False, "reason": "no actors in sector", "p_value": 1.0}

    sector_gaps = gaps[actors]

    # Find nearest quarter to event date
    dates = sector_gaps.index
    event_idx = dates.get_indexer([event_date], method="nearest")[0]
    if event_idx < 0 or event_idx >= len(dates):
        return {"aligned": False, "reason": "event outside data range", "p_value": 1.0}

    # Event window: [event-2Q, event+2Q]
    ev_start = max(0, event_idx - 2)
    ev_end = min(len(dates), event_idx + 3)
    # Control window: [-8Q, -3Q] before event
    ctrl_start = max(0, event_idx - 8)
    ctrl_end = max(0, event_idx - 2)

    if ctrl_end <= ctrl_start:
        return {"aligned": False, "reason": "insufficient control window", "p_value": 1.0}

    # Mean |gap| in event vs control
    event_mean_abs = sector_gaps.iloc[ev_start:ev_end].abs().mean(axis=1).mean()
    control_mean_abs = sector_gaps.iloc[ctrl_start:ctrl_end].abs().mean(axis=1).mean()

    # Gap sign in event window
    event_mean_signed = sector_gaps.iloc[ev_start:ev_end].mean(axis=1).mean()

    # t-test: event period |gaps| vs control period |gaps|
    event_vals = sector_gaps.iloc[ev_start:ev_end].abs().values.ravel()
    control_vals = sector_gaps.iloc[ctrl_start:ctrl_end].abs().values.ravel()
    event_vals = event_vals[np.isfinite(event_vals)]
    control_vals = control_vals[np.isfinite(control_vals)]

    if len(event_vals) < 2 or len(control_vals) < 2:
        return {"aligned": False, "reason": "insufficient data", "p_value": 1.0}

    t_stat, p_value = stats.ttest_ind(event_vals, control_vals, alternative="greater")

    # Sign alignment
    if expected_sign == "POSITIVE":
        sign_match = event_mean_signed > 0
    elif expected_sign == "NEGATIVE":
        sign_match = event_mean_signed < 0
    else:  # MIXED
        sign_match = True  # any sign is acceptable

    aligned = p_value < 0.10 and sign_match

    return {
        "aligned": aligned,
        "event_mean_abs": float(event_mean_abs),
        "control_mean_abs": float(control_mean_abs),
        "ratio": float(event_mean_abs / max(control_mean_abs, 1e-10)),
        "event_mean_signed": float(event_mean_signed),
        "sign_match": sign_match,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_actors": len(actors),
    }


def main():
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Historical event alignment (8 events, target: >=6/8)")
    print("=" * 70)

    print("\n  Loading data ...", end="", flush=True)
    wide, actor_sector = load_intensity_panel()
    gaps = compute_gaps(wide)
    print(f" done. {gaps.shape[1]} actors, {gaps.shape[0]} quarters")

    rows = []
    n_aligned = 0
    print(f"\n  {'Event':<35} {'Sector':<12} {'|gap| ratio':>12} {'sign':>6} {'p':>8} {'Aligned':>8}")
    print("  " + "-" * 85)

    for event in EVENTS:
        result = analyse_event(event, gaps, actor_sector)
        if result["aligned"]:
            n_aligned += 1

        sign_str = "OK" if result.get("sign_match", False) else "MISS"
        ratio_str = f"{result.get('ratio', 0):.2f}x" if "ratio" in result else "n/a"
        p_str = f"{result.get('p_value', 1.0):.4f}"
        aligned_str = "YES" if result["aligned"] else "no"

        print(f"  {event['name']:<35} {event['sector']:<12} {ratio_str:>12} "
              f"{sign_str:>6} {p_str:>8} {aligned_str:>8}")

        rows.append({
            "experiment_id": EXP_ID,
            "event_name": event["name"],
            "event_date": event["date"],
            "sector": event["sector"],
            "expected_sign": event["sign"],
            **{k: v for k, v in result.items()},
        })

    hit_rate = n_aligned / len(EVENTS)
    target_met = n_aligned >= 6
    print(f"\n  Hit rate: {n_aligned}/{len(EVENTS)} ({hit_rate:.0%})")
    print(f"  Target (>=6/8): {'MET' if target_met else 'NOT MET'}")
    print("=" * 70)

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(METRICS_DIR / f"level4_{EXP_ID}.parquet", index=False)
    config = {
        "experiment_id": EXP_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": len(EVENTS),
        "n_aligned": n_aligned,
        "hit_rate": round(hit_rate, 4),
        "target_met": target_met,
    }
    with open(CONFIGS_DIR / f"{EXP_ID}.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"\n  Wrote: results/metrics/level4_{EXP_ID}.parquet")
    print(f"  Wrote: results/configs/{EXP_ID}.yaml")


if __name__ == "__main__":
    main()
