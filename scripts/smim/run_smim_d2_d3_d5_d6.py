#!/usr/bin/env python
"""
D2+D3+D5+D6 combined runner (analysis-only, no pipeline runs).

D2-CORRECTION: Do large gaps predict subsequent CapEx revision?
D3-DIFFUSION:  Do upstream (L1) gaps predict downstream (L2) gaps?
D5-BENCHMARK-DIVERGENCE: Do benchmark classes produce different gaps?
D6-EMERGENCE-TIMING: Do emergence diagnostics lead standard risk measures?

All consume intensity panel and A1 gap series.
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

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
FRED_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "fred_signals.parquet"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


def load_panel():
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_data = {a["actor_id"]: a for a in reg["actors"] if a["actor_id"] in actors}
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [a for a in actor_data if a in wide.columns]
    return wide[cols], actor_data


def compute_gaps(wide):
    gaps = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for i in range(40, len(wide)):
        historical_mean = wide.iloc[max(0, i - 40):i].mean()
        gaps.iloc[i] = wide.iloc[i] - historical_mean
    return gaps.dropna(how="all")


# ══════════════════════════════════════════════════════════════════════════════
# D2: CORRECTION PREDICTION
# ══════════════════════════════════════════════════════════════════════════════


def run_d2(wide, gaps, actor_data):
    print("\n" + "=" * 70)
    print(f"  D2-CORRECTION  |  Do large gaps predict CapEx revision?")
    print("=" * 70)

    # Outcome: next-4Q intensity change (proxy for CapEx revision)
    outcome = wide.diff(4).shift(-4)  # y_{i,t+4} - y_{i,t}

    # Panel regression: outcome_{t+4} = beta * gap_t + FE
    gaps_flat = gaps.stack().reset_index()
    gaps_flat.columns = ["date", "actor", "gap"]
    outcome_flat = outcome.stack().reset_index()
    outcome_flat.columns = ["date", "actor", "outcome"]

    merged = gaps_flat.merge(outcome_flat, on=["date", "actor"])
    merged = merged.dropna()

    if len(merged) < 20:
        print("  Insufficient data for regression")
        return pd.DataFrame()

    X = merged["gap"].values
    y = merged["outcome"].values

    # OLS: outcome = alpha + beta * gap
    X_mat = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        resid = y - X_mat @ beta
        se = np.sqrt(np.sum(resid ** 2) / (len(y) - 2) / np.sum((X - X.mean()) ** 2))
        t_stat = beta[1] / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y) - 2))
    except Exception:
        beta = [0, 0]
        t_stat, p_value = 0, 1

    # Quintile sort
    merged["gap_quintile"] = pd.qcut(merged["gap"], 5, labels=False, duplicates="drop")
    quintile_means = merged.groupby("gap_quintile")["outcome"].mean()

    print(f"\n  Panel regression: next-4Q intensity change = alpha + beta * gap")
    print(f"    beta = {beta[1]:.4f}, t = {t_stat:.2f}, p = {p_value:.4f}")
    print(f"    N = {len(merged)}")
    print(f"\n  Quintile sort (gap -> next-4Q outcome):")
    for q, m in quintile_means.items():
        print(f"    Q{int(q) + 1}: {m:+.4f}")

    q5_minus_q1 = quintile_means.iloc[-1] - quintile_means.iloc[0] if len(quintile_means) >= 5 else 0
    print(f"    Q5 - Q1 spread: {q5_minus_q1:+.4f}")

    h6b = abs(t_stat) > 1.96
    print(f"\n  H6b verdict: {'SUPPORTED' if h6b else 'NOT SUPPORTED'} (|t|={'>' if h6b else '<'}1.96)")

    rows = [{"experiment_id": "D2-CORRECTION", "beta": beta[1], "t_stat": t_stat,
             "p_value": p_value, "n_obs": len(merged), "q5_minus_q1": q5_minus_q1,
             "h6b": h6b}]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# D3: GRAPH DIFFUSION
# ══════════════════════════════════════════════════════════════════════════════


def run_d3(gaps, actor_data):
    print("\n" + "=" * 70)
    print(f"  D3-DIFFUSION  |  Do upstream gaps predict downstream gaps?")
    print("=" * 70)

    # Layer classification
    layer_map = {aid: a.get("layer", 2) for aid, a in actor_data.items()}
    l0 = [a for a in gaps.columns if layer_map.get(a) == 0]
    l1 = [a for a in gaps.columns if layer_map.get(a) == 1]
    l2 = [a for a in gaps.columns if layer_map.get(a) == 2]

    print(f"\n  Layer distribution: L0={len(l0)}, L1={len(l1)}, L2={len(l2)}")

    # Aggregate gap by layer
    results = []
    for src_name, src_actors in [("L0", l0), ("L1", l1)]:
        for tgt_name, tgt_actors in [("L1", l1), ("L2", l2)]:
            if src_name == tgt_name or not src_actors or not tgt_actors:
                continue
            src_gap = gaps[src_actors].mean(axis=1).dropna()
            tgt_gap = gaps[tgt_actors].mean(axis=1).dropna()
            common = src_gap.index.intersection(tgt_gap.index)
            if len(common) < 10:
                continue
            src_gap = src_gap.loc[common].values
            tgt_gap = tgt_gap.loc[common].values

            # Granger-like test: does src_t predict tgt_{t+1}?
            for lag in [1, 2, 4]:
                if len(src_gap) <= lag + 2:
                    continue
                x = src_gap[:-lag]
                y = tgt_gap[lag:]
                corr = np.corrcoef(x, y)[0, 1]
                n = len(x)
                t = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-10))
                p = 2 * (1 - stats.t.cdf(abs(t), df=n - 2))
                results.append({
                    "source": src_name, "target": tgt_name, "lag_Q": lag,
                    "corr": corr, "t_stat": t, "p_value": p,
                })
                print(f"  {src_name} -> {tgt_name} (lag={lag}Q): corr={corr:.3f}, t={t:.2f}, p={p:.4f}")

    if not results:
        print("  No valid layer pairs for diffusion test")
    return pd.DataFrame(results) if results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# D5: BENCHMARK DIVERGENCE
# ══════════════════════════════════════════════════════════════════════════════


def run_d5(gaps):
    print("\n" + "=" * 70)
    print(f"  D5-BENCHMARK-DIVERGENCE  |  Do benchmark classes differ?")
    print("=" * 70)

    # At current implementation, we only have one gap series (from intensity
    # demeaning). The predictive/modal/emergence benchmarks differ in the
    # alpha (predicted vs filtered vs emergence-corrected), but we don't
    # store per-actor benchmark values from A1. Use the A1 aggregate metrics.

    a1_path = METRICS_DIR / "level1_A1-MVP-FULL.parquet"
    if not a1_path.exists():
        print("  A1 results not found — cannot compute divergence")
        return pd.DataFrame()

    a1 = pd.read_parquet(a1_path)

    print(f"\n  A1 benchmark R2 comparison (per window):")
    for wid in sorted(a1["window_id"].unique()):
        w_data = a1[a1["window_id"] == wid]
        r2s = {}
        for _, row in w_data.iterrows():
            r2s[row["benchmark_class"]] = row["oos_r2"]
        pred = r2s.get("predictive", float("nan"))
        modal = r2s.get("modal", float("nan"))
        ea = r2s.get("emergence_aware", float("nan"))
        print(f"    {wid}: pred={pred:.4f}  modal={modal:.4f}  EA={ea:.4f}  "
              f"|modal-pred|={abs(modal - pred):.4f}")

    # Aggregate
    by_bench = a1.groupby("benchmark_class")["oos_r2"].agg(["mean", "std"])
    print(f"\n  Mean R2 by benchmark:")
    for bc, row in by_bench.iterrows():
        print(f"    {bc:<20} {row['mean']:.4f} +/- {row['std']:.4f}")

    pred_vals = a1[a1["benchmark_class"] == "predictive"]["oos_r2"].values
    modal_vals = a1[a1["benchmark_class"] == "modal"]["oos_r2"].values
    if len(pred_vals) > 0 and len(modal_vals) > 0:
        mean_div = np.mean(np.abs(pred_vals - modal_vals[:len(pred_vals)]))
        corr_pm = np.corrcoef(pred_vals, modal_vals[:len(pred_vals)])[0, 1]
        print(f"\n  Pred vs Modal: mean |divergence| = {mean_div:.4f}, correlation = {corr_pm:.4f}")
        h6c = mean_div > 0.01
        print(f"  H6c verdict: {'SUPPORTED' if h6c else 'NOT SUPPORTED'} (div {'>' if h6c else '<'} 0.01)")
    else:
        h6c = False

    rows = [{"experiment_id": "D5-BENCHMARK-DIVERGENCE",
             "mean_divergence_pred_modal": float(mean_div) if 'mean_div' in dir() else 0,
             "corr_pred_modal": float(corr_pm) if 'corr_pm' in dir() else 0,
             "h6c": h6c}]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# D6: EMERGENCE TIMING
# ══════════════════════════════════════════════════════════════════════════════


def run_d6(wide, gaps):
    print("\n" + "=" * 70)
    print(f"  D6-EMERGENCE-TIMING  |  Do emergence diagnostics lead VIX?")
    print("=" * 70)

    # Load VIX from FRED
    if not FRED_PATH.exists():
        print("  FRED data not found")
        return pd.DataFrame()

    fred = pd.read_parquet(FRED_PATH)
    vix = fred[fred["signal_id"] == "VIXCLS"].copy()
    vix["event_date"] = pd.to_datetime(vix["event_date"])
    vix_q = vix.set_index("event_date")["value"].resample("QE").mean()
    vix_q.index = vix_q.index.to_period("Q").to_timestamp()

    # Simple emergence proxy: cross-sectional dispersion of gaps
    gap_dispersion = gaps.std(axis=1)  # (T,) — cross-sectional std of gaps per quarter

    # Average pairwise intensity correlation (proxy for systemic risk)
    corr_series = pd.Series(index=gaps.index, dtype=float)
    for i in range(len(gaps)):
        row = gaps.iloc[i].dropna()
        if len(row) > 5:
            c = np.corrcoef(np.column_stack([row.values, row.values]))[0, 1:]
            corr_series.iloc[i] = np.nanmean(np.abs(c)) if len(c) > 0 else 0

    # Align
    common = gap_dispersion.dropna().index.intersection(vix_q.index)
    if len(common) < 10:
        print("  Insufficient overlap between gap dispersion and VIX")
        return pd.DataFrame()

    disp = gap_dispersion.loc[common].values
    vix_vals = vix_q.loc[common].values

    # Lead-lag correlation
    results = []
    print(f"\n  Gap dispersion vs VIX lead-lag correlation:")
    for lag in range(-4, 5):
        if lag < 0:
            x = disp[:lag]
            y = vix_vals[-lag:]
        elif lag > 0:
            x = disp[lag:]
            y = vix_vals[:-lag]
        else:
            x, y = disp, vix_vals
        if len(x) < 5:
            continue
        c = np.corrcoef(x, y)[0, 1]
        results.append({"lag_Q": lag, "corr_disp_vix": c})
        marker = " <--" if abs(c) == max(abs(r["corr_disp_vix"]) for r in results) else ""
        print(f"    lag={lag:+2d}Q: corr={c:+.3f}{marker}")

    # Verdict
    pos_lead = [r for r in results if r["lag_Q"] < 0 and abs(r["corr_disp_vix"]) > 0.3]
    print(f"\n  Emergence leads VIX: {'YES' if pos_lead else 'NO'} "
          f"({'significant' if pos_lead else 'no'} negative-lag correlations > 0.3)")

    return pd.DataFrame(results) if results else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print(f"  D2+D3+D5+D6 COMBINED  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n  Loading data ...", end="", flush=True)
    wide, actor_data = load_panel()
    gaps = compute_gaps(wide)
    print(f" done. {gaps.shape[1]} actors, {gaps.shape[0]} gap quarters")

    df_d2 = run_d2(wide, gaps, actor_data)
    df_d3 = run_d3(gaps, actor_data)
    df_d5 = run_d5(gaps)
    df_d6 = run_d6(wide, gaps)

    print("\n" + "=" * 70)

    # Save all
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    for exp_id, df in [("D2-CORRECTION", df_d2), ("D3-DIFFUSION", df_d3),
                        ("D5-BENCHMARK-DIVERGENCE", df_d5), ("D6-EMERGENCE-TIMING", df_d6)]:
        if not df.empty:
            path = METRICS_DIR / f"level4_{exp_id}.parquet"
            df.to_parquet(path, index=False)
            print(f"  Wrote: {path.relative_to(PROJECT_ROOT)}")
        cfg_path = CONFIGS_DIR / f"{exp_id}.yaml"
        yaml.dump({"experiment_id": exp_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(cfg_path, "w"))

    print(f"  All D-series results saved.")


if __name__ == "__main__":
    main()
