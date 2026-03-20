"""Pipeline profiling harness.

Usage:
    python -m quantdsl_backtest.smim.profiling \\
        --config experiments/mvp_energy_us_uk.yaml \\
        --n-actors 50 100 200 500 \\
        --output profiling_results.json

Each component is timed independently using ``time.perf_counter_ns()``
(wall-clock).  Peak RSS is tracked via ``tracemalloc``.

Component definitions:
    edge_estimation_granger   – N² pairwise Granger tests (VAR with BIC=off,
                                max_lag=2) — O(N² × T)
    edge_estimation_narrative – TF-IDF cosine similarity, N×T text signals
    sparsification            – L1 soft-threshold to 15% density
    eigendecomposition_polar  – polar + eigh on (N, N) operator
    eigendecomposition_schur  – complex Schur on (N, N) operator
    mode_selection            – MDL criterion across K candidate modes
    kalman_filter             – Kalman filter over T steps, K modes
    kim_filter_em             – Kim EM (M regimes, 20 iterations)
    pid_synergy               – K*(K-1)/2 pairwise Gaussian PID point-estimates
    pid_bootstrap             – K*(K-1)/2 pairwise bootstrap CIs (200 draws each)
    transfer_entropy          – K*(K-1) ordered KSG TE estimates (mode pairs)
    persistent_homology       – sliding-window ripser / fallback over (T, K) states
    benchmark_computation     – ModalBenchmark: U @ alpha_filtered, gap array
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _timed() -> Generator[list[float], None, None]:
    """Yield a 1-element list; after the block list[0] = elapsed seconds."""
    out: list[float] = [0.0]
    t0 = time.perf_counter_ns()
    yield out
    out[0] = (time.perf_counter_ns() - t0) / 1_000_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data builders (mirrors tests/acceptance/smim/conftest.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_actor_registry(N: int) -> Any:
    """Return an ActorRegistry with N actors spread across four layers."""
    from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
    from quantdsl_backtest.smim.interfaces import Actor, ActorType, Layer

    _LAYER_TYPES = [
        (Layer.EXOGENOUS, ActorType.GLOBAL_SHOCK),
        (Layer.UPSTREAM, ActorType.CENTRAL_BANK),
        (Layer.TRANSMISSION, ActorType.LARGE_FIRM),
        (Layer.DOWNSTREAM, ActorType.SME),
    ]
    actors = [
        Actor(
            actor_id=f"actor_{i:04d}",
            name=f"Actor {i}",
            actor_type=_LAYER_TYPES[i % 4][1],
            layer=_LAYER_TYPES[i % 4][0],
            geography="US",
            sector="energy",
        )
        for i in range(N)
    ]
    return ActorRegistry(actors=actors)


def _make_var1_dataframe(
    N: int, T: int, seed: int = 42
) -> tuple[pd.DataFrame, Any]:
    """Return (signals DataFrame, DateRange) for Granger edge estimation.

    Generates a stable VAR(1) system with spectral radius ~0.8.
    """
    from quantdsl_backtest.smim.interfaces import DateRange

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) * 0.2 / max(float(N) ** 0.5, 1.0)
    spec_rad = float(np.max(np.abs(np.linalg.eigvals(A))))
    if spec_rad > 0.0:
        A = A / (spec_rad + 0.1) * 0.8
    data = np.zeros((T, N))
    data[0] = rng.standard_normal(N)
    for t in range(T - 1):
        data[t + 1] = A @ data[t] + 0.3 * rng.standard_normal(N)
    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    actor_ids = [f"actor_{i:04d}" for i in range(N)]
    df = pd.DataFrame(data, index=dates, columns=actor_ids)
    date_range = DateRange(start=dates[0], end=dates[-1])
    return df, date_range


def _make_text_dataframe(
    N: int, T: int, seed: int = 42
) -> tuple[pd.DataFrame, Any]:
    """Return (text signals DataFrame, DateRange) for Narrative edge estimation."""
    from quantdsl_backtest.smim.interfaces import DateRange

    rng = np.random.default_rng(seed)
    vocab = [
        "investment", "capital", "energy", "growth", "risk", "policy",
        "market", "demand", "supply", "shock", "green", "transition",
        "regulation", "firm", "sector", "output", "rate", "cost",
    ]
    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    actor_ids = [f"actor_{i:04d}" for i in range(N)]

    def rand_text() -> str:
        n_words = int(rng.integers(8, 20))
        return " ".join(str(w) for w in rng.choice(vocab, size=n_words))

    data = {aid: [rand_text() for _ in range(T)] for aid in actor_ids}
    df = pd.DataFrame(data, index=dates)
    date_range = DateRange(start=dates[0], end=dates[-1])
    return df, date_range


def _make_modal_obs(
    N: int, K: int, T: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (observations (T, N), U_true (N, K), F_true (K, K)).

    Generates from the modal state-space model:
        α_t = F α_{t-1} + η,  η ~ N(0, 0.1·I_K)
        s_t = U α_t   + ε,   ε ~ N(0, 0.1·I_N)
    """
    rng = np.random.default_rng(seed)
    U_raw = rng.standard_normal((N, K))
    U_true, _ = np.linalg.qr(U_raw)
    U_true = U_true[:, :K]
    F_true = np.diag(rng.uniform(0.7, 0.95, K))
    noise_Q = np.linalg.cholesky(np.eye(K) * 0.1)
    noise_R = np.linalg.cholesky(np.eye(N) * 0.1)
    alphas = np.zeros((T, K))
    alphas[0] = rng.standard_normal(K)
    for t in range(1, T):
        alphas[t] = F_true @ alphas[t - 1] + noise_Q @ rng.standard_normal(K)
    obs = alphas @ U_true.T + (noise_R @ rng.standard_normal((N, T))).T
    return obs, U_true, F_true


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_pipeline(
    N: int,
    T: int = 80,
    K: int = 15,
    M: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Profile all pipeline components for N actors, returning structured results.

    Args:
        N: Number of actors.
        T: Number of time steps (quarters).
        K: Number of spectral modes.
        M: Number of Markov regimes.
        seed: RNG seed for reproducibility.

    Returns:
        Dict matching the GPU_ACCELERATION_PLAN.md JSON schema:
        {N, T, K, M, components: {name: {seconds, pct}, ..., total: {seconds}},
         memory_peak_mb, scaling_vs_n50}
    """
    tracemalloc.start()
    timings: dict[str, float] = {}
    rng = np.random.default_rng(seed)

    # ── Build synthetic data ────────────────────────────────────────────────
    actor_registry = _make_actor_registry(N)
    var1_df, date_range = _make_var1_dataframe(N, T, seed)
    text_df, _ = _make_text_dataframe(N, T, seed)
    obs_modal, _U_true, _F_true = _make_modal_obs(N, K, T, seed)
    dense_op = rng.standard_normal((N, N))

    # ── 1. Granger edge estimation ──────────────────────────────────────────
    from quantdsl_backtest.smim.config import GrangerEdgeConfig
    from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator

    granger = GrangerEdgeEstimator(GrangerEdgeConfig(max_lag=2, bic_selection=False))
    with _timed() as t:
        _granger_adj = granger.estimate(var1_df, actor_registry, date_range)
    timings["edge_estimation_granger"] = t[0]

    # ── 2. Narrative edge estimation ────────────────────────────────────────
    from quantdsl_backtest.smim.config import NarrativeEdgeConfig
    from quantdsl_backtest.smim.graph.edges.narrative import NarrativeEdgeEstimator

    narrative = NarrativeEdgeEstimator(NarrativeEdgeConfig())
    with _timed() as t:
        _narrative_adj = narrative.estimate(text_df, actor_registry, date_range)
    timings["edge_estimation_narrative"] = t[0]

    # ── 3. Sparsification ───────────────────────────────────────────────────
    from quantdsl_backtest.smim.graph.sparsification import l1_sparsify

    with _timed() as t:
        _sparse_adj = l1_sparsify(sp.csr_matrix(dense_op), target_density=0.15)
    timings["sparsification"] = t[0]

    # ── 4. Eigendecomposition (Polar) ───────────────────────────────────────
    from quantdsl_backtest.smim.spectral.polar import PolarDecomposer

    polar_decomp = PolarDecomposer()
    with _timed() as t:
        polar_frame = polar_decomp.decompose(dense_op, k=K)
    timings["eigendecomposition_polar"] = t[0]

    # ── 5. Eigendecomposition (Schur) ───────────────────────────────────────
    from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

    schur_decomp = SchurDecomposer()
    with _timed() as t:
        _schur_frame = schur_decomp.decompose(dense_op, k=K)
    timings["eigendecomposition_schur"] = t[0]

    # Use polar frame as the canonical ModalFrame for downstream steps
    modal_frame = polar_frame

    # ── 6. Mode selection ───────────────────────────────────────────────────
    from quantdsl_backtest.smim.spectral.mode_selection import MDLModeSelector

    mdl = MDLModeSelector()
    with _timed() as t:
        _selected_frame, _selected_ts = mdl.select(modal_frame, obs_modal)
    timings["mode_selection"] = t[0]

    # ── 7. Kalman filter ────────────────────────────────────────────────────
    from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter

    kf = KalmanFilter()
    with _timed() as t:
        kf_state = kf.filter(obs_modal, modal_frame)
    timings["kalman_filter"] = t[0]

    # ── 8. Kim filter EM ────────────────────────────────────────────────────
    from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter

    kim = KimFilter(n_regimes=M)
    with _timed() as t:
        _kim_state = kim.em_estimate(obs_modal, modal_frame, max_iter=20)
    timings["kim_filter_em"] = t[0]

    # ── Derived quantities used by emergence diagnostics ────────────────────
    alpha_filtered = kf_state.alpha_filtered  # (T, K)
    K_act = alpha_filtered.shape[1]
    gaps_fake = rng.standard_normal((N, T)) * 0.1  # synthetic (N, T) gaps
    target_vec = gaps_fake.mean(axis=0)              # (T,) aggregate target

    # ── 9. PID synergy – K*(K-1)/2 unique mode pairs ───────────────────────
    from quantdsl_backtest.smim.emergence.pid import compute_synergy_matrix

    with _timed() as t:
        _syn_mat = compute_synergy_matrix(alpha_filtered, gaps_fake, K_act)
    timings["pid_synergy"] = t[0]

    # ── 10. PID bootstrap – K*(K-1)/2 pairs × 200 draws ────────────────────
    from quantdsl_backtest.smim.emergence.pid import bootstrap_synergy

    with _timed() as t:
        for j in range(K_act):
            for k in range(j + 1, K_act):
                _ = bootstrap_synergy(
                    alpha_filtered[:, j],
                    alpha_filtered[:, k],
                    target_vec,
                    n_bootstrap=200,
                )
    timings["pid_bootstrap"] = t[0]

    # ── 11. Transfer entropy – K*(K-1) ordered mode pairs ──────────────────
    from quantdsl_backtest.smim.emergence.transfer_entropy import ksg_transfer_entropy

    with _timed() as t:
        for j in range(K_act):
            for k in range(K_act):
                if j != k:
                    _ = ksg_transfer_entropy(
                        alpha_filtered[:, j],
                        alpha_filtered[:, k],
                    )
    timings["transfer_entropy"] = t[0]

    # ── 12. Persistent homology ─────────────────────────────────────────────
    from quantdsl_backtest.smim.emergence.tda import sliding_window_persistence

    with _timed() as t:
        _ = sliding_window_persistence(alpha_filtered, window_size=20, overlap=10)
    timings["persistent_homology"] = t[0]

    # ── 13. Benchmark computation ───────────────────────────────────────────
    from quantdsl_backtest.smim.gaps.modal import ModalBenchmark

    mb = ModalBenchmark()
    with _timed() as t:
        _ = mb.compute(obs_modal, kf_state, modal_frame, actor_registry)
    timings["benchmark_computation"] = t[0]

    # ── Peak memory ─────────────────────────────────────────────────────────
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0 * 1024.0)

    total_seconds = sum(timings.values())
    components: dict[str, dict[str, float]] = {
        name: {
            "seconds": round(secs, 4),
            "pct": round(100.0 * secs / total_seconds, 2) if total_seconds > 0 else 0.0,
        }
        for name, secs in timings.items()
    }
    components["total"] = {"seconds": round(total_seconds, 4)}

    return {
        "N": N,
        "T": T,
        "K": K,
        "M": M,
        "components": components,
        "memory_peak_mb": round(peak_mb, 1),
        "scaling_vs_n50": None,  # filled in by main()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Falsification suite profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_falsification(
    N: int,
    T: int = 80,
    K: int = 15,
    M: int = 3,
    seed: int = 99,
    B: int = 100,
) -> dict[str, Any]:
    """Profile one null-model instance: rewire + core pipeline (spectral+filter+gap).

    Args:
        N: Number of actors.
        T: Number of time steps.
        K: Number of modes.
        M: Unused (included for API consistency).
        seed: RNG seed.
        B: Number of null instances for projection.

    Returns:
        Dict with one_null_instance_seconds, component_breakdown,
        projected_B{B}_seconds, projected_B{B}_hours.
    """
    from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
    from quantdsl_backtest.smim.gaps.modal import ModalBenchmark
    from quantdsl_backtest.smim.graph.null_models import degree_preserving_rewire
    from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

    rng = np.random.default_rng(seed)
    dense_op = rng.standard_normal((N, N))
    mask = rng.random((N, N)) < 0.15
    adj = sp.csr_matrix(dense_op * mask)
    obs_modal, _U, _F = _make_modal_obs(N, K, T, seed)
    actor_registry = _make_actor_registry(N)

    schur_decomp = SchurDecomposer()
    kf = KalmanFilter()
    mb = ModalBenchmark()

    timings: dict[str, float] = {}

    with _timed() as t:
        rewired = degree_preserving_rewire(adj, rng=np.random.default_rng(seed + 1))
    timings["rewire"] = t[0]

    with _timed() as t:
        modal_frame = schur_decomp.decompose(rewired.toarray(), k=K)
    timings["spectral"] = t[0]

    with _timed() as t:
        fs = kf.filter(obs_modal, modal_frame)
    timings["filter"] = t[0]

    with _timed() as t:
        _ = mb.compute(obs_modal, fs, modal_frame, actor_registry)
    timings["benchmark"] = t[0]

    one_run_s = sum(timings.values())
    projected_s = one_run_s * B

    return {
        "one_null_instance_seconds": round(one_run_s, 4),
        "component_breakdown": {k: round(v, 4) for k, v in timings.items()},
        f"projected_B{B}_seconds": round(projected_s, 1),
        f"projected_B{B}_hours": round(projected_s / 3600.0, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(result: dict[str, Any]) -> None:
    """Print sorted component table and highlight top-3 bottlenecks."""
    N = result["N"]
    comps = result["components"]
    total_sec = comps["total"]["seconds"]

    rows = [
        (name, info["seconds"], info["pct"])
        for name, info in comps.items()
        if name != "total"
    ]
    rows.sort(key=lambda x: -x[1])

    sep = "=" * 72
    print(f"\n{sep}")
    print(
        f"  N={N}  T={result['T']}  K={result['K']}  M={result['M']}  "
        f"total={total_sec:.2f}s  peak_mem={result['memory_peak_mb']:.0f} MB"
    )
    if result.get("scaling_vs_n50") is not None:
        print(f"  Scaling vs N=50: {result['scaling_vs_n50']:.2f}×")
    print(sep)
    print(f"  {'Component':<38} {'Seconds':>8}  {'%':>6}  {'Cumul%':>7}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*6}  {'-'*7}")

    cumul = 0.0
    for name, secs, pct in rows:
        cumul += pct
        print(f"  {name:<38} {secs:>8.3f}  {pct:>6.1f}  {cumul:>7.1f}")

    print(f"  {'-'*38}  {'-'*8}")
    print(f"  {'TOTAL':<38} {total_sec:>8.3f}")

    top3 = rows[:3]
    top3_pct = sum(r[2] for r in top3)
    print(f"\n  Top-3 bottlenecks (combined {top3_pct:.1f}% of total):")
    for i, (name, secs, pct) in enumerate(top3, 1):
        print(f"    {i}. {name:<36} {pct:.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile SMIM pipeline components at multiple actor counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="experiments/mvp_energy_us_uk.yaml",
        help="Experiment YAML config path (informational; not parsed by profiler).",
    )
    parser.add_argument(
        "--n-actors",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        metavar="N",
        help="Actor counts to profile.",
    )
    parser.add_argument("--T", type=int, default=80, help="Time steps (quarters).")
    parser.add_argument("--K", type=int, default=15, help="Spectral modes.")
    parser.add_argument("--M", type=int, default=3, help="Markov regimes.")
    parser.add_argument(
        "--output",
        default="profiling_results.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--falsification-n",
        type=int,
        default=None,
        metavar="N",
        help="N for falsification profiling (defaults to max --n-actors).",
    )
    args = parser.parse_args()

    all_results: list[dict[str, Any]] = []
    baseline_total: float | None = None
    n_actors_sorted = sorted(args.n_actors)

    for N in n_actors_sorted:
        print(f"\n[profiling] N={N} — running pipeline …", flush=True)
        result = profile_pipeline(N, T=args.T, K=args.K, M=args.M)

        total_sec = result["components"]["total"]["seconds"]
        if baseline_total is None:
            baseline_total = total_sec
        result["scaling_vs_n50"] = (
            round(total_sec / baseline_total, 2) if baseline_total > 0 else 1.0
        )

        _print_summary(result)
        all_results.append(result)

    # ── Falsification ───────────────────────────────────────────────────────
    false_n = args.falsification_n if args.falsification_n is not None else max(args.n_actors)
    print(f"\n[profiling] Falsification suite (N={false_n}) …", flush=True)
    fals_result = profile_falsification(false_n, T=args.T, K=args.K, M=args.M)
    b_key_s = f"projected_B100_seconds"
    b_key_h = f"projected_B100_hours"
    print(
        f"\n  Falsification (N={false_n}):\n"
        f"    One null instance : {fals_result['one_null_instance_seconds']:.3f}s\n"
        f"    Projected B=100   : {fals_result.get(b_key_s, '?'):.1f}s "
        f"({fals_result.get(b_key_h, '?'):.2f}h)\n"
        f"    Breakdown         : {fals_result['component_breakdown']}"
    )

    output_payload = {
        "pipeline_profiles": all_results,
        "falsification_profile": fals_result,
    }
    with open(args.output, "w") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"\n[profiling] Results saved to {args.output}")


if __name__ == "__main__":
    main()
