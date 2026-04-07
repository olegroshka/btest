#!/usr/bin/env python
"""
Iteration 6.4 Gate B — Local Coherence Discovery + NCD Diagnostics.

Identifies whether any pre-registered block or train-only neighborhood
has smoother, more predictable local dynamics than the global panel.

Includes basic temporal NCD and cross-block NCD as discovery diagnostics.

Kill Rule B: no block has materially smoother dynamics than global.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4_gate_b.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

K_MAX = 15
EWM_HL = 12


# ══════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════

def load_panel_and_registry():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    actor_meta = {a["actor_id"]: a for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel, actor_meta


def ewm_demean(obs, hl=EWM_HL):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    return float(num / den) if den > 1e-12 else 0.0, bar_y


# ══════════════════════════════════════════════════════════════════════
#  Block definitions
# ══════════════════════════════════════════════════════════════════════

def define_blocks(panel, actor_meta):
    """Define all pre-registered actor blocks."""
    actors = list(panel.columns)
    blocks = {}

    # Raw sector blocks
    from collections import defaultdict
    sector_groups = defaultdict(list)
    for a in actors:
        m = actor_meta.get(a, {})
        sector_groups[m.get("sector", "unknown")].append(a)

    for sector, members in sector_groups.items():
        if len(members) >= 8:
            blocks[f"SEC_{sector}"] = members

    # Merged sector blocks (from FINAL plan)
    tech_health = [a for a in actors if actor_meta.get(a, {}).get("sector") in ("technology", "healthcare")]
    if len(tech_health) >= 10:
        blocks["MERGED_tech_health"] = tech_health

    fin = [a for a in actors if actor_meta.get(a, {}).get("sector") == "financials"]
    ind_energy = [a for a in actors if actor_meta.get(a, {}).get("sector") in ("industrials", "energy")]
    if len(ind_energy) >= 10:
        blocks["MERGED_ind_energy"] = ind_energy

    # Layer blocks
    layer_groups = defaultdict(list)
    for a in actors:
        m = actor_meta.get(a, {})
        layer_groups[m.get("layer", -1)].append(a)

    macro_inst = layer_groups.get(0, []) + layer_groups.get(1, [])
    if len(macro_inst) >= 8:
        blocks["LAYER_macro_inst"] = macro_inst
    firms = layer_groups.get(2, [])
    if len(firms) >= 10:
        blocks["LAYER_firms"] = firms

    # K-means clusters (train-only, on first training window loadings)
    # Computed fresh per window in the diagnostic loop, but we define the approach here
    blocks["_KMEANS_K2"] = None  # placeholder — computed per window
    blocks["_KMEANS_K3"] = None

    return blocks


# ══════════════════════════════════════════════════════════════════════
#  NCD computation
# ══════════════════════════════════════════════════════════════════════

def compressed_size(data_bytes):
    """Gzip compressed size of byte sequence."""
    return len(gzip.compress(data_bytes, compresslevel=9))


def tercile_encode(vec):
    """Encode a vector as tercile symbols (0, 1, 2)."""
    t1, t2 = np.percentile(vec, [33.3, 66.7])
    symbols = np.zeros(len(vec), dtype=np.uint8)
    symbols[vec > t1] = 1
    symbols[vec > t2] = 2
    return symbols.tobytes()


def ncd(x_bytes, y_bytes):
    """Normalised Compression Distance between two byte sequences."""
    cx = compressed_size(x_bytes)
    cy = compressed_size(y_bytes)
    cxy = compressed_size(x_bytes + y_bytes)
    denom = max(cx, cy)
    if denom < 1:
        return 1.0
    return (cxy - min(cx, cy)) / denom


def temporal_ncd_series(panel_values):
    """Compute NCD between consecutive cross-sectional snapshots."""
    T = panel_values.shape[0]
    ncds = []
    for t in range(T - 1):
        x = tercile_encode(panel_values[t])
        y = tercile_encode(panel_values[t + 1])
        ncds.append(ncd(x, y))
    return np.array(ncds)


def shuffled_ncd_null(panel_values, n_shuffles=100, seed=42):
    """NCD between random pairs of snapshots (null distribution)."""
    rng = np.random.default_rng(seed)
    T = panel_values.shape[0]
    ncds = []
    for _ in range(n_shuffles):
        i, j = rng.choice(T, 2, replace=False)
        x = tercile_encode(panel_values[i])
        y = tercile_encode(panel_values[j])
        ncds.append(ncd(x, y))
    return np.array(ncds)


# ══════════════════════════════════════════════════════════════════════
#  Local diagnostics
# ══════════════════════════════════════════════════════════════════════

def compute_block_diagnostics(panel, block_actors, block_name, global_diags=None):
    """Compute local coherence diagnostics for an actor block."""
    sub = panel[block_actors].dropna(how="all")
    N = len(block_actors)
    all_quarters = pd.date_range("2010-01-01", "2024-12-31", freq="QS")

    # Collect rolling residuals and bases
    residual_persistence_vals = []
    explained_var_vals = []
    effective_rank_vals = []
    geodesic_vals = []
    prev_U = None

    for q_end in all_quarters:
        q_start = q_end - pd.DateOffset(years=5)
        if q_start < pd.Timestamp("2005-01-01"):
            continue
        chunk = sub[(sub.index >= q_start) & (sub.index <= q_end)]
        valid = chunk.columns[chunk.notna().any()]
        chunk = chunk[valid].fillna(chunk[valid].mean())
        n_act = len(valid)
        if n_act < 5:
            prev_U = None
            continue
        vals = chunk.values.astype(np.float64)
        if vals.shape[0] < 4:
            prev_U = None
            continue

        # Pooled+FE residuals
        rho, bar_y = estimate_pooled_ar1(vals)
        predicted = bar_y + rho * (vals[:-1] - bar_y)
        resids = vals[1:] - predicted

        # Residual persistence
        rhos = []
        for j in range(resids.shape[1]):
            col = resids[:, j]
            if len(col) >= 3 and np.std(col[:-1]) > 1e-10:
                c = np.corrcoef(col[:-1], col[1:])[0, 1]
                if np.isfinite(c):
                    rhos.append(c)
        if rhos:
            residual_persistence_vals.append(np.mean(rhos))

        # SVD for explained variance and effective rank
        om = ewm_demean(resids, EWM_HL)
        dm = resids - om
        if dm.shape[0] >= 3 and dm.shape[1] >= 3:
            try:
                _, S, _ = np.linalg.svd(dm, full_matrices=False)
                S2 = S ** 2
                total = S2.sum()
                if total > 1e-15:
                    K_local = min(8, n_act // 3, len(S))
                    explained_var_vals.append(float(S2[:K_local].sum() / total))
                    # Effective rank
                    p = S2 / total
                    p = p[p > 1e-15]
                    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
                    effective_rank_vals.append(eff_rank)
            except Exception:
                pass

            # Local DMD basis for geodesic distance
            K_local = min(4, max(2, n_act // 5))
            try:
                mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, n_act))
                ka = min(K_local, mf.K, n_act - 2)
                U = mf.metadata["U"][:, :ka].real
                U, _ = np.linalg.qr(U)
                if prev_U is not None and prev_U.shape == U.shape:
                    # Sign-align
                    signs = np.sign(np.sum(U * prev_U, axis=0))
                    signs[signs == 0] = 1
                    U = U * signs[np.newaxis, :]
                    # Principal angles
                    svals = np.linalg.svd(prev_U.T @ U, compute_uv=False)
                    angles = np.arccos(np.clip(svals, -1.0, 1.0))
                    geodesic_vals.append(float(np.degrees(np.linalg.norm(angles))))
                prev_U = U
            except Exception:
                prev_U = None

    # Temporal NCD on residuals
    # Use the full panel residuals
    sub_filled = sub.fillna(sub.mean()).values.astype(np.float64)
    if sub_filled.shape[0] >= 20:
        ncd_temporal = temporal_ncd_series(sub_filled)
        ncd_shuffle = shuffled_ncd_null(sub_filled, n_shuffles=100)
    else:
        ncd_temporal = np.array([np.nan])
        ncd_shuffle = np.array([np.nan])

    diags = {
        "block": block_name,
        "N_actors": N,
        "resid_persistence": float(np.mean(residual_persistence_vals)) if residual_persistence_vals else np.nan,
        "explained_var": float(np.mean(explained_var_vals)) if explained_var_vals else np.nan,
        "effective_rank": float(np.mean(effective_rank_vals)) if effective_rank_vals else np.nan,
        "geodesic_deg": float(np.mean(geodesic_vals)) if geodesic_vals else np.nan,
        "n_geodesic": len(geodesic_vals),
        "ncd_temporal_mean": float(np.nanmean(ncd_temporal)),
        "ncd_shuffle_mean": float(np.nanmean(ncd_shuffle)),
        "ncd_ratio": float(np.nanmean(ncd_temporal) / np.nanmean(ncd_shuffle)) if np.nanmean(ncd_shuffle) > 0.01 else np.nan,
    }

    return diags


def compute_cross_block_ncd(panel, blocks, block_actors_map):
    """Compute NCD between pairs of blocks at each quarter."""
    results = []
    block_names = [b for b in blocks if not b.startswith("_")]

    for i in range(len(block_names)):
        for j in range(i + 1, len(block_names)):
            b1, b2 = block_names[i], block_names[j]
            a1 = block_actors_map[b1]
            a2 = block_actors_map[b2]

            sub1 = panel[a1].fillna(panel[a1].mean()).values.astype(np.float64)
            sub2 = panel[a2].fillna(panel[a2].mean()).values.astype(np.float64)

            T = min(sub1.shape[0], sub2.shape[0])
            ncds = []
            for t in range(T):
                x = tercile_encode(sub1[t])
                y = tercile_encode(sub2[t])
                ncds.append(ncd(x, y))

            results.append({
                "block1": b1, "block2": b2,
                "ncd_mean": float(np.mean(ncds)),
                "ncd_std": float(np.std(ncds)),
            })

    return results


# ══════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════

def print_diagnostics(all_diags, global_diags):
    print("\n" + "=" * 100)
    print("  LOCAL COHERENCE DIAGNOSTICS")
    print("=" * 100)

    print(f"\n  {'Block':<25s} {'N':>4s} {'Persist':>8s} {'ExplVar':>8s} {'EffRank':>8s}"
          f" {'Geod°':>7s} {'NCD_t':>7s} {'NCD_s':>7s} {'NCD_r':>7s}")
    print(f"  {'-'*90}")

    # Global first
    g = global_diags
    print(f"  {'GLOBAL':<25s} {g['N_actors']:>4d} {g['resid_persistence']:>8.4f} {g['explained_var']:>8.4f}"
          f" {g['effective_rank']:>8.1f} {g['geodesic_deg']:>7.1f}"
          f" {g['ncd_temporal_mean']:>7.3f} {g['ncd_shuffle_mean']:>7.3f} {g['ncd_ratio']:>7.3f}")
    print(f"  {'-'*90}")

    for d in sorted(all_diags, key=lambda x: x.get("ncd_ratio", 1.0)):
        b = d["block"]
        persist = d["resid_persistence"]
        ev = d["explained_var"]
        er = d["effective_rank"]
        geo = d["geodesic_deg"]
        ncd_t = d["ncd_temporal_mean"]
        ncd_s = d["ncd_shuffle_mean"]
        ncd_r = d["ncd_ratio"]

        # Flag blocks smoother than global
        flags = []
        if np.isfinite(persist) and np.isfinite(g["resid_persistence"]) and persist > g["resid_persistence"] + 0.02:
            flags.append("pers↑")
        if np.isfinite(geo) and np.isfinite(g["geodesic_deg"]) and geo < g["geodesic_deg"] - 5:
            flags.append("geo↓")
        if np.isfinite(ncd_r) and np.isfinite(g["ncd_ratio"]) and ncd_r < g["ncd_ratio"] - 0.02:
            flags.append("ncd↓")
        if np.isfinite(er) and np.isfinite(g["effective_rank"]) and er < g["effective_rank"] - 1.0:
            flags.append("rank↓")

        flag_s = " ".join(flags) if flags else ""
        persist_s = f"{persist:8.4f}" if np.isfinite(persist) else "     N/A"
        ev_s = f"{ev:8.4f}" if np.isfinite(ev) else "     N/A"
        er_s = f"{er:8.1f}" if np.isfinite(er) else "     N/A"
        geo_s = f"{geo:7.1f}" if np.isfinite(geo) else "    N/A"
        ncd_t_s = f"{ncd_t:7.3f}" if np.isfinite(ncd_t) else "    N/A"
        ncd_s_s = f"{ncd_s:7.3f}" if np.isfinite(ncd_s) else "    N/A"
        ncd_r_s = f"{ncd_r:7.3f}" if np.isfinite(ncd_r) else "    N/A"

        print(f"  {b:<25s} {d['N_actors']:>4d} {persist_s} {ev_s} {er_s} {geo_s}"
              f" {ncd_t_s} {ncd_s_s} {ncd_r_s}  {flag_s}")


def evaluate_kill_rule_b(all_diags, global_diags):
    print("\n" + "=" * 80)
    print("  KILL RULE B EVALUATION")
    print("=" * 80)

    g = global_diags
    any_smoother = False
    surviving_blocks = []

    for d in all_diags:
        reasons = []
        # Higher persistence
        if np.isfinite(d["resid_persistence"]) and np.isfinite(g["resid_persistence"]):
            if d["resid_persistence"] > g["resid_persistence"] + 0.02:
                reasons.append(f"persistence {d['resid_persistence']:.3f} > global {g['resid_persistence']:.3f}")
        # Lower geodesic
        if np.isfinite(d["geodesic_deg"]) and np.isfinite(g["geodesic_deg"]):
            if d["geodesic_deg"] < g["geodesic_deg"] - 5:
                reasons.append(f"geodesic {d['geodesic_deg']:.1f}° < global {g['geodesic_deg']:.1f}°")
        # Lower NCD ratio
        if np.isfinite(d["ncd_ratio"]) and np.isfinite(g["ncd_ratio"]):
            if d["ncd_ratio"] < g["ncd_ratio"] - 0.02:
                reasons.append(f"NCD ratio {d['ncd_ratio']:.3f} < global {g['ncd_ratio']:.3f}")
        # Lower effective rank
        if np.isfinite(d["effective_rank"]) and np.isfinite(g["effective_rank"]):
            if d["effective_rank"] < g["effective_rank"] - 1.0:
                reasons.append(f"eff.rank {d['effective_rank']:.1f} < global {g['effective_rank']:.1f}")

        if reasons:
            any_smoother = True
            surviving_blocks.append(d["block"])
            print(f"  {d['block']}: SMOOTHER — {'; '.join(reasons)}")

    print(f"\n  ═══════════════════════════════════════════════")
    if any_smoother:
        print(f"  KILL RULE B: NOT TRIGGERED — {len(surviving_blocks)} blocks smoother than global")
        print(f"  Surviving blocks: {surviving_blocks}")
        print(f"  Proceed to Gate C (local horse race).")
    else:
        print(f"  KILL RULE B: TRIGGERED — no block smoother than global")
        print(f"  Skip Gate C, proceed to Gate D (conditional gating).")
    print(f"  ═══════════════════════════════════════════════")

    return not any_smoother, surviving_blocks


# ══════════════════════════════════════════════════════════════════════
#  Save & Main
# ══════════════════════════════════════════════════════════════════════

def save_results(all_diags, global_diags, cross_block, surviving):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([global_diags] + all_diags).to_parquet(
        METRICS_DIR / "iter6_4_gate_b_diagnostics.parquet", index=False)
    if cross_block:
        pd.DataFrame(cross_block).to_parquet(
            METRICS_DIR / "iter6_4_gate_b_cross_ncd.parquet", index=False)
    print(f"\n  Saved: iter6_4_gate_b_*.parquet")


def main():
    t_start = time.time()

    print("=" * 100)
    print("  ITERATION 6.4 GATE B — LOCAL COHERENCE DISCOVERY + NCD DIAGNOSTICS")
    print("  Are any local blocks smoother than the global panel?")
    print("=" * 100)

    panel, actor_meta = load_panel_and_registry()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # Define blocks
    blocks = define_blocks(panel, actor_meta)
    block_actors = {b: a for b, a in blocks.items() if a is not None}
    print(f"\nPre-registered blocks: {len(block_actors)}")
    for b, a in sorted(block_actors.items()):
        print(f"  {b}: N={len(a)}")

    # Global diagnostics (full panel)
    print("\n  Computing GLOBAL diagnostics...")
    t0 = time.time()
    global_diags = compute_block_diagnostics(panel, list(panel.columns), "GLOBAL")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Per-block diagnostics
    all_diags = []
    for block_name, block_act in sorted(block_actors.items()):
        t0 = time.time()
        d = compute_block_diagnostics(panel, block_act, block_name, global_diags)
        all_diags.append(d)
        print(f"  {block_name}: persist={d['resid_persistence']:.3f}"
              f"  geod={d['geodesic_deg']:.1f}°"
              f"  NCD_ratio={d['ncd_ratio']:.3f}  ({time.time()-t0:.1f}s)")

    # Cross-block NCD
    print("\n  Computing cross-block NCD...")
    t0 = time.time()
    cross_block = compute_cross_block_ncd(panel, block_actors, block_actors)
    print(f"  Done ({time.time()-t0:.1f}s)")
    for cb in cross_block:
        print(f"    NCD({cb['block1']}, {cb['block2']}) = {cb['ncd_mean']:.3f} ± {cb['ncd_std']:.3f}")

    # Report
    print_diagnostics(all_diags, global_diags)

    # Kill rule
    killed, surviving = evaluate_kill_rule_b(all_diags, global_diags)

    # Save
    save_results(all_diags, global_diags, cross_block, surviving)

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    return all_diags, global_diags, cross_block, killed, surviving


if __name__ == "__main__":
    main()
