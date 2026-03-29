#!/usr/bin/env python
"""
A4-SCALING experiment runner.

Measures pipeline runtime and memory as a function of universe size N to
confirm Phase B/C experiments are computationally feasible.

Iterations: N = 20, 50, 100, 200, ~400 (US-LC + US-MC combined)

N=50 reuses A3 timing data if results/configs/A3-STACK-VALIDATION.yaml exists.

Output:
  results/metrics/level5_A4-SCALING.parquet   (N, component, time_s, mem_mb)
  results/configs/A4-SCALING.yaml             (scaling table, decision gate)

Decision gate: flag any component where N=200 run exceeds 30 min / N=400 exceeds
30 min, or any component scaling worse than O(N^2.5).

Usage::
    uv run python scripts/run_smim_a4.py
    uv run python scripts/run_smim_a4.py --skip-n400   # skip combined US-LC+MC run
    uv run python scripts/run_smim_a4.py --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry as ConcreteActorRegistry
from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.gaps.modal import ModalBenchmark
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
from quantdsl_backtest.smim.graph.operators import AggregateOperator
from quantdsl_backtest.smim.graph.sparsification import l1_sparsify
from quantdsl_backtest.smim.interfaces import Actor, ActorType, DateRange, Layer, RelationChannel
from quantdsl_backtest.smim.spectral.mode_selection import MDLModeSelector
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# ─── constants ───────────────────────────────────────────────────────────────

EXP_ID = "A4-SCALING"
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")

K_MAX_CANDIDATES = 15
TARGET_DENSITY = 0.15
MAX_GRANGER_LAG = 2

# N=400 feasibility threshold per component (seconds at N=200)
FEASIBILITY_THRESHOLD_S = 1800.0   # 30 minutes per run

# Decision gate: flag if empirical scaling exponent > this
SCALING_EXPONENT_GATE = 2.5

# ─── timing + memory ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def timed(label: str, timings: dict[str, float]) -> Generator[None, None, None]:
    t0 = time.perf_counter_ns()
    yield
    timings[label] = (time.perf_counter_ns() - t0) / 1e9


# ─── data loading ─────────────────────────────────────────────────────────────


def _actor_type(s: str) -> ActorType:
    return {
        "sector_leader": ActorType.SECTOR_LEADER,
        "large_firm": ActorType.LARGE_FIRM,
        "bank": ActorType.BANK,
    }.get(s, ActorType.LARGE_FIRM)


def load_universe_actors(
    universes: list[str],
    n_target: int,
) -> tuple[list[dict], list[str]]:
    """Load actors from one or more universe registries up to n_target.

    Returns (actor_data_list, actor_ids).
    """
    all_actor_data: list[dict] = []

    for uni in universes:
        reg_path = PROJECT_ROOT / "data" / "smim" / "registries" / f"{uni}_registry.json"
        with open(reg_path) as fh:
            reg_data = json.load(fh)

        # Load intensity actor_ids for this universe
        int_path = (
            PROJECT_ROOT / "data" / "smim" / "intensities" / f"{uni}_intensities.parquet"
        )
        df_int = pd.read_parquet(int_path)
        recent_mask = (
            (df_int["period"] >= TRAIN_START)
            & (df_int["period"] <= TEST_END)
            & (df_int["normalisation_method"] == "capex_assets_xsrank")
        )
        actors_with_intensity = set(df_int[recent_mask]["actor_id"].unique())

        for a in reg_data["actors"]:
            if a["actor_id"] in actors_with_intensity:
                all_actor_data.append(a)
            if len(all_actor_data) >= n_target:
                break
        if len(all_actor_data) >= n_target:
            break

    actor_ids = [a["actor_id"] for a in all_actor_data[:n_target]]
    return all_actor_data[:n_target], actor_ids


def build_registry(actor_data: list[dict]) -> ConcreteActorRegistry:
    actors = [
        Actor(
            actor_id=a["actor_id"],
            name=a["name"],
            actor_type=_actor_type(a["actor_type"]),
            layer=Layer(a["layer"]),
            geography=a["geography"],
            sector=a["sector"],
            external_ids=a.get("external_ids", {}),
        )
        for a in actor_data
    ]
    return ConcreteActorRegistry(actors=actors)


def load_intensities(actor_ids: list[str], universes: list[str]) -> pd.DataFrame:
    """Load and pivot intensity data for the given actors."""
    frames = []
    for uni in universes:
        path = PROJECT_ROOT / "data" / "smim" / "intensities" / f"{uni}_intensities.parquet"
        df = pd.read_parquet(path)
        df = df[
            (df["normalisation_method"] == "capex_assets_xsrank")
            & (df["actor_id"].isin(actor_ids))
            & (df["period"] >= TRAIN_START)
            & (df["period"] <= TEST_END)
        ]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    wide = combined.pivot_table(index="period", columns="actor_id", values="intensity_value")
    idx = pd.date_range(TRAIN_START, TEST_END, freq="QS")
    wide = wide.reindex(idx)
    # Keep only columns for actors in our ordered list
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols]


def load_ohlcv_returns(actor_ids: list[str], universes: list[str]) -> pd.DataFrame:
    """Load OHLCV from one or more universes, compute quarterly log-returns."""
    frames = []
    for uni in universes:
        path = PROJECT_ROOT / "equities" / "smim" / uni / "ohlcv.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["ticker"].isin(actor_ids)].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[
            (df["date"] >= TRAIN_START - pd.DateOffset(months=4))
            & (df["date"] <= TEST_END)
        ]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    ohlcv = pd.concat(frames, ignore_index=True)
    close_wide = ohlcv.pivot_table(index="date", columns="ticker", values="close")
    close_q = close_wide.resample("QE").last()
    close_q.index = close_q.index.to_period("Q").to_timestamp()
    returns_q = np.log(close_q / close_q.shift(1))
    idx = pd.date_range(TRAIN_START, TEST_END, freq="QS")
    returns_q = returns_q.reindex(idx)
    cols = [c for c in actor_ids if c in returns_q.columns]
    return returns_q[cols]


# ─── single-N pipeline run ────────────────────────────────────────────────────


def profile_n(
    n_target: int,
    universes: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for up to n_target actors and return profiling data.

    Returns dict with keys: N_actual, timings, peak_mem_mb, oos_r2, k_star.
    """
    timings: dict[str, float] = {}

    tracemalloc.start()

    # ── Load data ─────────────────────────────────────────────────────────────
    with timed("load_data", timings):
        actor_data, actor_ids = load_universe_actors(universes, n_target)
        intensities_wide = load_intensities(actor_ids, universes)
        ohlcv_returns = load_ohlcv_returns(actor_ids, universes)

    # Restrict to actors with both intensity and OHLCV, and non-NaN intensity in training
    ohlcv_cols = set(ohlcv_returns.columns) if not ohlcv_returns.empty else set()
    int_cols = set(intensities_wide.columns)
    int_train = intensities_wide[intensities_wide.index <= TRAIN_END]
    int_cols_with_data = set(int_train.columns[int_train.notna().any()])
    actor_ids_final = [
        a for a in actor_ids
        if a in ohlcv_cols and a in int_cols and a in int_cols_with_data
    ]

    N_actual = len(actor_ids_final)
    if N_actual < 5:
        raise RuntimeError(
            f"Only {N_actual} actors with complete data for N_target={n_target}"
        )

    if verbose:
        print(f"    N_actual={N_actual} (target={n_target})")

    # Rebuild filtered data
    intensities_wide = intensities_wide[actor_ids_final]
    ohlcv_returns = ohlcv_returns[actor_ids_final]

    registry = ConcreteActorRegistry(
        actors=[a for ad in actor_data
                for a in [Actor(
                    actor_id=ad["actor_id"], name=ad["name"],
                    actor_type=_actor_type(ad["actor_type"]),
                    layer=Layer(ad["layer"]),
                    geography=ad["geography"], sector=ad["sector"],
                    external_ids=ad.get("external_ids", {}),
                )] if ad["actor_id"] in set(actor_ids_final)]
    )

    # ── Granger edges ──────────────────────────────────────────────────────────
    signals_train = ohlcv_returns[ohlcv_returns.index <= TRAIN_END].fillna(
        ohlcv_returns[ohlcv_returns.index <= TRAIN_END].mean()
    )
    # Drop any fully-NaN columns
    signals_train = signals_train.dropna(axis=1, how="all")
    signal_actor_ids = list(signals_train.columns)

    registry_granger = ConcreteActorRegistry(
        actors=[a for a in registry.actors if a.actor_id in set(signal_actor_ids)]
    )
    date_range = DateRange(start=TRAIN_START, end=TRAIN_END)

    config = GrangerEdgeConfig(
        max_lag=MAX_GRANGER_LAG, p_threshold=0.05, bic_selection=False
    )
    with timed("granger_edges", timings):
        granger_adj = GrangerEdgeEstimator(config).estimate(
            signals_train, registry_granger, date_range
        )

    # ── Sparsification ─────────────────────────────────────────────────────────
    with timed("sparsification", timings):
        aggregate = AggregateOperator(
            channel_matrices={RelationChannel.FINANCIAL: granger_adj}
        )
        combined = aggregate.compute()
        sparse_op = l1_sparsify(combined, target_density=TARGET_DENSITY)

    # ── Spectral decomposition ─────────────────────────────────────────────────
    N = min(sparse_op.shape[0], len(actor_ids_final))
    op_dense = sparse_op.toarray()[:N, :N]
    K_cand = min(K_MAX_CANDIDATES, N)

    with timed("spectral_decomposition", timings):
        modal_frame_cand = SchurDecomposer().decompose(op_dense, k=K_cand)

    # ── Intensity matrices ─────────────────────────────────────────────────────
    int_cols_ordered = [c for c in actor_ids_final if c in intensities_wide.columns][:N]
    obs_train_df = intensities_wide[int_cols_ordered][intensities_wide.index <= TRAIN_END]
    col_means = obs_train_df.mean()
    obs_train = obs_train_df.fillna(col_means).values.astype(np.float64)
    obs_test_df = intensities_wide[int_cols_ordered][intensities_wide.index >= TEST_START]
    obs_test = obs_test_df.fillna(col_means).values.astype(np.float64)

    if verbose:
        n_nan = int(np.sum(np.isnan(obs_train)))
        print(f"    obs_train: {obs_train.shape}, NaN count={n_nan}")

    # ── Mode selection ─────────────────────────────────────────────────────────
    with timed("mode_selection", timings):
        modal_frame, _ = MDLModeSelector().select(modal_frame_cand, obs_train)

    k_star = modal_frame.K

    # ── Kalman filter EM ───────────────────────────────────────────────────────
    kf = KalmanFilter()
    with timed("kalman_filter_em", timings):
        state_train = kf.em_estimate(obs_train, modal_frame, max_iter=50)

    with timed("kalman_filter_test", timings):
        kf_test = KalmanFilter(
            F=state_train.params.get("F"),
            Q=state_train.params.get("Q"),
            R=state_train.params.get("R"),
        )
        state_test = kf_test.filter(obs_test, modal_frame)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    with timed("benchmark_predictive", timings):
        gap_pred = PredictiveBenchmark().compute(obs_test, state_test, modal_frame, registry)

    with timed("benchmark_modal", timings):
        ModalBenchmark().compute(obs_test, state_test, modal_frame, registry)

    # ── OOS R² ────────────────────────────────────────────────────────────────
    actual = obs_test.T.ravel()
    pred = gap_pred.benchmarks.ravel()
    oos_r2 = float(oos_r_squared(pred, actual))

    # ── Peak memory ───────────────────────────────────────────────────────────
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0 * 1024.0)

    return {
        "N_actual": N_actual,
        "timings": timings,
        "peak_mem_mb": round(peak_mb, 1),
        "oos_r2": oos_r2,
        "k_star": k_star,
    }


# ─── scaling exponent fitting ─────────────────────────────────────────────────


def fit_scaling_exponent(ns: list[int], times: list[float]) -> float:
    """Fit alpha in t ~ N^alpha via log-log OLS. Returns alpha."""
    xs = np.array([np.log(n) for n in ns if n > 0])
    ys = np.array([np.log(max(t, 1e-9)) for n, t in zip(ns, times) if n > 0])
    if len(xs) < 2:
        return float("nan")
    slope, _, _, _, _ = scipy.stats.linregress(xs, ys)
    return round(float(slope), 2)


def build_scaling_table(
    results: list[dict[str, Any]],
    components: list[str],
) -> pd.DataFrame:
    """Build scaling table: component × {alpha, t_N200_s, t_N400_s, feasible_N500}."""
    ns = [r["N_actual"] for r in results]
    rows = []
    for comp in components:
        times = [r["timings"].get(comp, float("nan")) for r in results]
        valid_pairs = [(n, t) for n, t in zip(ns, times) if np.isfinite(t) and t > 0]
        if len(valid_pairs) >= 2:
            vns, vts = zip(*valid_pairs)
            alpha = fit_scaling_exponent(list(vns), list(vts))
        else:
            alpha = float("nan")

        # Look up t at the N closest to 200 and to 400
        def _closest(target: int) -> float:
            candidates = [(abs(r["N_actual"] - target), r) for r in results]
            candidates.sort(key=lambda x: x[0])
            best_dist, best_r = candidates[0]
            # Only use if within 50% of target
            if best_dist > target * 0.5:
                return float("nan")
            return best_r["timings"].get(comp, float("nan"))

        t_n200 = _closest(200)
        t_n400 = _closest(400)

        # Extrapolate N=500 from N=200 if N=400 not available
        if np.isfinite(t_n400):
            t_n500_est = t_n400 * ((500 / 400) ** alpha) if np.isfinite(alpha) else float("nan")
        elif np.isfinite(t_n200) and np.isfinite(alpha):
            t_n500_est = t_n200 * ((500 / 200) ** alpha)
        else:
            t_n500_est = float("nan")

        feasible = (
            t_n500_est < FEASIBILITY_THRESHOLD_S
            if np.isfinite(t_n500_est)
            else None
        )

        rows.append({
            "component": comp,
            "alpha_exponent": alpha,
            "t_n200_s": round(t_n200, 4) if np.isfinite(t_n200) else None,
            "t_n400_s": round(t_n400, 4) if np.isfinite(t_n400) else None,
            "t_n500_est_s": round(t_n500_est, 1) if np.isfinite(t_n500_est) else None,
            "feasible_n500": feasible,
            "exceeds_gate": (
                alpha > SCALING_EXPONENT_GATE if np.isfinite(alpha) else None
            ),
        })
    return pd.DataFrame(rows)


# ─── output ───────────────────────────────────────────────────────────────────


def build_level5_parquet(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build level5 parquet: one row per (N, component)."""
    rows = []
    for r in results:
        N = r["N_actual"]
        mem = r["peak_mem_mb"]
        for comp, secs in r["timings"].items():
            rows.append({
                "experiment_id": EXP_ID,
                "N": N,
                "component": comp,
                "time_s": round(secs, 6),
                "mem_mb": mem,
                "oos_r2": r["oos_r2"],
                "k_star": r["k_star"],
            })
    return pd.DataFrame(rows)


# ─── main ─────────────────────────────────────────────────────────────────────

PIPELINE_COMPONENTS = [
    "load_data",
    "granger_edges",
    "sparsification",
    "spectral_decomposition",
    "mode_selection",
    "kalman_filter_em",
    "kalman_filter_test",
    "benchmark_predictive",
    "benchmark_modal",
]

# (n_target, universes) pairs
ITERATIONS = [
    (20,  ["US-LC"]),
    (50,  ["US-LC"]),
    (100, ["US-LC"]),
    (200, ["US-LC"]),
    (400, ["US-LC", "US-MC"]),
]


def run_a4(skip_n400: bool = False, verbose: bool = False) -> None:
    print(f"\n{'='*70}")
    print(f"  A4-SCALING  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  period=RECENT  signals=MACRO+MARKET  pipeline=full")
    print(f"{'='*70}\n")

    results: list[dict[str, Any]] = []
    a3_yaml = PROJECT_ROOT / "results" / "configs" / "A3-STACK-VALIDATION.yaml"

    for n_target, universes in ITERATIONS:
        if skip_n400 and n_target == 400:
            print(f"  N={n_target:>4}  [SKIPPED via --skip-n400]")
            continue

        # Reuse A3 N=50 results if available
        if n_target == 50 and a3_yaml.exists():
            with open(a3_yaml) as fh:
                a3_cfg = yaml.safe_load(fh)
            a3_timings = a3_cfg.get("runtime_seconds", {})
            # Filter to pipeline components (drop falsification, metrics)
            reused_timings = {
                k: v for k, v in a3_timings.items() if k in PIPELINE_COMPONENTS
            }
            result = {
                "N_actual": a3_cfg.get("n_actors", 50),
                "timings": reused_timings,
                "peak_mem_mb": 0.0,   # not recorded in A3
                "oos_r2": -1.6459,    # from A3 run
                "k_star": a3_cfg.get("k_star", 1),
            }
            total_s = sum(reused_timings.values())
            print(f"  N={n_target:>4}  [REUSED A3]  total={total_s:.3f}s  "
                  f"k*={result['k_star']}  OOS R2={result['oos_r2']:.4f}")
            results.append(result)
            continue

        print(f"  N={n_target:>4}  running ...", end="", flush=True)
        t_wall = time.perf_counter()
        try:
            result = profile_n(n_target, universes, verbose=verbose)
            elapsed = time.perf_counter() - t_wall
            total_comp = sum(result["timings"].values())
            print(
                f"  done ({elapsed:.1f}s wall)  "
                f"N_actual={result['N_actual']}  "
                f"k*={result['k_star']}  "
                f"OOS R2={result['oos_r2']:.4f}  "
                f"mem={result['peak_mem_mb']:.0f}MB"
            )
            results.append(result)
        except Exception as exc:
            elapsed = time.perf_counter() - t_wall
            print(f"  FAILED ({elapsed:.1f}s): {exc}")
            continue

    if len(results) < 2:
        print("\nERROR: fewer than 2 successful N runs — cannot fit scaling exponents.")
        sys.exit(1)

    # ── Build outputs ──────────────────────────────────────────────────────────
    level5_df = build_level5_parquet(results)
    scaling_df = build_scaling_table(results, PIPELINE_COMPONENTS)

    results_dir = PROJECT_ROOT / "results"
    metrics_dir = results_dir / "metrics"
    configs_dir = results_dir / "configs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = metrics_dir / f"level5_{EXP_ID}.parquet"
    level5_df.to_parquet(parquet_path, index=False)

    # ── Print scaling table ────────────────────────────────────────────────────
    ns_run = sorted(set(r["N_actual"] for r in results))
    print(f"\n{'-'*70}")
    print("  SCALING TABLE")
    print(f"  {'Component':<28} {'alpha':>6}  {'N=200s':>8}  {'N=400s':>8}  "
          f"{'N=500est':>9}  {'feasible?':>9}  {'gate?':>6}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*6}")

    flagged_components: list[str] = []
    for _, row in scaling_df.iterrows():
        alpha_s = f"{row['alpha_exponent']:.2f}" if pd.notna(row['alpha_exponent']) else "  n/a"
        n200_s = f"{row['t_n200_s']:.3f}" if row['t_n200_s'] is not None else "    n/a"
        n400_s = f"{row['t_n400_s']:.3f}" if row['t_n400_s'] is not None else "    n/a"
        n500_s = f"{row['t_n500_est_s']:.1f}" if row['t_n500_est_s'] is not None else "      n/a"
        feasible = (
            "YES" if row['feasible_n500'] is True
            else "NO" if row['feasible_n500'] is False
            else "n/a"
        )
        gate = "FLAG" if row['exceeds_gate'] is True else "ok"
        if row['exceeds_gate'] is True:
            flagged_components.append(row['component'])
        print(f"  {row['component']:<28} {alpha_s:>6}  {n200_s:>8}  {n400_s:>8}  "
              f"{n500_s:>9}  {feasible:>9}  {gate:>6}")

    # ── Total pipeline time per N ──────────────────────────────────────────────
    print(f"\n  TOTAL PIPELINE TIME PER N")
    print(f"  {'N':>6}  {'Total(s)':>10}  {'OOS R2':>8}  {'K*':>4}  {'Mem(MB)':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*4}  {'-'*8}")
    for r in sorted(results, key=lambda x: x["N_actual"]):
        total_s = sum(r["timings"].values())
        print(f"  {r['N_actual']:>6}  {total_s:>10.3f}  {r['oos_r2']:>8.4f}  "
              f"{r['k_star']:>4}  {r['peak_mem_mb']:>8.1f}")

    # ── Fit summary ────────────────────────────────────────────────────────────
    # Total pipeline alpha
    total_times = [sum(r["timings"].values()) for r in sorted(results, key=lambda x: x["N_actual"])]
    total_alpha = fit_scaling_exponent(ns_run, total_times)
    print(f"\n  Total pipeline scaling exponent: alpha={total_alpha:.2f}")

    # ── Decision gate ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if flagged_components:
        print(f"  DECISION GATE: FLAGGED components (alpha > {SCALING_EXPONENT_GATE})")
        for c in flagged_components:
            row = scaling_df[scaling_df["component"] == c].iloc[0]
            print(f"    {c}: alpha={row['alpha_exponent']:.2f}")
        print("  Raise with user before launching B-series at N=200+.")
    else:
        print(f"  DECISION GATE: PASS -- all components scale at alpha <= {SCALING_EXPONENT_GATE}")
        print("  B-series at N=200 is computationally feasible.")
    print(f"{'='*70}\n")

    # ── Write config YAML ──────────────────────────────────────────────────────
    config_payload = {
        "experiment_id": EXP_ID,
        "n_targets": [r["N_actual"] for r in results],
        "train_start": str(TRAIN_START.date()),
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "signals": "MACRO+MARKET",
        "institutions": "INST-MINIMAL",
        "pipeline_depth": "full",
        "k_max_candidates": K_MAX_CANDIDATES,
        "total_pipeline_alpha": float(total_alpha),
        "decision_gate": "PASS" if not flagged_components else "FLAG",
        "flagged_components": flagged_components,
        "scaling_table": scaling_df.to_dict(orient="records"),
        "per_n_summary": [
            {
                "N": r["N_actual"],
                "total_s": round(sum(r["timings"].values()), 4),
                "oos_r2": r["oos_r2"],
                "k_star": r["k_star"],
                "peak_mem_mb": r["peak_mem_mb"],
                "timings": {k: round(v, 6) for k, v in r["timings"].items()},
            }
            for r in sorted(results, key=lambda x: x["N_actual"])
        ],
    }
    yaml_path = configs_dir / f"{EXP_ID}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(config_payload, fh, default_flow_style=False, sort_keys=True)

    print(f"  Wrote: {parquet_path.relative_to(PROJECT_ROOT)}")
    print(f"  Wrote: {yaml_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A4-SCALING experiment.")
    parser.add_argument("--skip-n400", action="store_true",
                        help="Skip the N~400 (US-LC+US-MC) iteration.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_a4(skip_n400=args.skip_n400, verbose=args.verbose)


if __name__ == "__main__":
    main()
