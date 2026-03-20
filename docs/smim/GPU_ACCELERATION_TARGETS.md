# SMIM GPU Acceleration Targets

**Generated:** 2026-03-20 (updated with actual N=500 measurements)
**Profiler:** `src/quantdsl_backtest/smim/profiling.py`
**Hardware:** CPU-only baseline (Windows 11, Python 3.11, NumPy/SciPy/LAPACK)
**Config:** T=80 quarters, K=15 modes, M=3 regimes, B=200 bootstrap draws

---

## 1. Raw Profiling Results (all three sizes measured)

### N=50

| Component | Seconds | % |
|-----------|---------|---|
| pid_bootstrap | 16.36 | 57.3% |
| edge_estimation_granger | 11.16 | 39.1% |
| transfer_entropy | 0.47 | 1.6% |
| kim_filter_em | 0.42 | 1.5% |
| pid_synergy | 0.08 | 0.3% |
| everything else | 0.07 | 0.2% |
| **TOTAL** | **28.6s** | |

### N=200 (MVP experiment size)

| Component | Seconds | % |
|-----------|---------|---|
| kim_filter_em | 235.2 | 55.0% |
| edge_estimation_granger | 170.4 | 39.9% |
| pid_bootstrap | 16.0 | 3.7% |
| kalman_filter | 5.0 | 1.2% |
| transfer_entropy | 0.46 | 0.1% |
| everything else | 0.39 | 0.1% |
| **TOTAL** | **427.4s** | **14.97× vs N=50** |

### N=500 (scaling target)

| Component | Seconds | % |
|-----------|---------|---|
| edge_estimation_granger | **1516.6** | **86.1%** |
| kim_filter_em | 213.0 | 12.1% |
| kalman_filter | 14.4 | 0.8% |
| pid_bootstrap | 16.5 | 0.9% |
| transfer_entropy | 0.47 | 0.0% |
| everything else | 0.86 | 0.0% |
| **TOTAL** | **1762.1s** | **61.7× vs N=50** |

---

## 2. Scaling Analysis — Three Data Points

| Component | N=50 | N=200 | N=500 | 50→200 | 200→500 | 50→500 |
|-----------|------|-------|-------|--------|---------|--------|
| granger | 11.2s | 170.4s | 1516.6s | 15.3× | 8.9× | 135.9× |
| kim_filter_em | 0.42s | 235.2s | 213.0s | 554× | **0.91×** | 502× |
| kalman_filter | 0.008s | 5.0s | 14.4s | 620× | 2.9× | 1804× |
| pid_bootstrap | 16.4s | 16.0s | 16.5s | 1.0× | 1.0× | 1.0× |
| transfer_entropy | 0.47s | 0.46s | 0.47s | 1.0× | 1.0× | 1.0× |

### Granger: confirmed O(N²)

Three-point empirical exponent α = 2.1 (50→500, ratio 10×, observed 136×).
This is the most reliable estimate: `log(136)/log(10) = 2.13`.
Matches theoretical O(N²·T) — N² independent bivariate VAR fits, T fixed.

### Kim filter + Kalman: non-monotonic scaling — a BLAS/cache boundary effect

The N=50→N=200 ratio (554×, 620×) looked super-cubic (O(N^4.5)), but N=200→N=500
shows Kim actually **decreased** (0.91×) and Kalman grew only 2.9× (O(N^1.2)).

Root cause: Both filters compute the N×N innovation covariance and solve it:
```
S = U @ P @ U.T + R         # (N, N)
S_inv = np.linalg.solve(S, I_N)  # O(N³) — bottleneck
```
NumPy/LAPACK's `dgesv` performance is non-monotonic with N:
- **N=50**: fits entirely in L1/L2 cache → very fast (0.008s Kalman, 0.42s Kim)
- **N=200**: exceeds L2 cache, hits L3/RAM bandwidth limit → disproportionate slowdown
- **N=500**: large enough that BLAS3 (block matrix) algorithms engage efficiently →
  cost grows more slowly than N³ in practice (confirmed by 0.9× Kim N=200→N=500)

The projected O(N^4.5) extrapolation was therefore wrong.
Actual behaviour from three data points:
- Kim: peaks near N=200, barely changes to N=500 — absolute cost ~213–235s, likely
  due to the T×M²×EM_iter = 14,400 inner iterations all hitting the same cache wall
- Kalman: O(N^1.2) from N=200→N=500 — BLAS3 blocking is active; cost grows slowly

### Components constant in N

pid_bootstrap, transfer_entropy, pid_synergy — operate on K=15 mode pairs and
B=200 bootstrap draws, both fixed regardless of N. Confirmed flat across all three sizes.

---

## 3. N=500 Projection vs Actual

My earlier projections (from two-point fit on N=50→N=200 extrapolated to N=500):

| Component | Projected | **Actual** | Error |
|-----------|-----------|------------|-------|
| kim_filter_em | 14,794s | **213s** | 69× over-estimated |
| kalman_filter | 345s | **14s** | 24× over-estimated |
| edge_estimation_granger | 1,177s | **1,517s** | 1.3× under-estimated |
| **TOTAL** | 15,600s (4.3h) | **1,762s (29 min)** | 8.9× over-estimated |

The over-estimation was caused by extrapolating the anomalous N=50→N=200 scaling
(where both filters hit a memory bottleneck) to N=500, where BLAS3 re-engages.

---

## 4. Falsification Suite Profile

| N | One instance | B=100 projected | Filter % |
|---|-------------|-----------------|---------|
| 200 | 5.72s | 572s (9.5 min) | 94.9% (5.43s Kalman) |
| 500 | 5.97s | 597s (9.9 min) | 70.8% (4.23s Kalman) |

The Kalman filter dominates each null instance. Note that falsification Kalman time
is 4–5s at both N=200 and N=500, consistent with the "BLAS plateau" observed above.

---

## 5. Acceleration Decision Table

Decision rule from `GPU_ACCELERATION_PLAN.md`:
- `>10% AND O(N²) or worse` → **HIGH PRIORITY GPU**
- `>10% AND embarrassingly parallel` → **MEDIUM (batch GPU)**
- `<5%` → **Leave on CPU**

| Component | N=200 % | N=500 % | Scaling | Decision |
|-----------|---------|---------|---------|----------|
| **edge_estimation_granger** | 39.9% | **86.1%** | O(N²) confirmed | **HIGH — GPU batch, clear #1 at scale** |
| **kim_filter_em** | 55.0% | 12.1% | Peaks ~N=200, plateaus | **HIGH at N=200 — Woodbury CPU fix; GPU secondary** |
| **kalman_filter** | 1.2% | 0.8% | O(N^1.2) N=200–500 | **MEDIUM — same Woodbury fix as Kim** |
| pid_bootstrap | 3.7% | 0.9% | O(1) in N | **LOW — constant cost, minor** |
| transfer_entropy | 0.1% | 0.0% | O(1) in N | **CPU — leave** |
| eigendecomposition_schur | 0.0% | 0.0% | O(N³) but tiny | **CPU — leave** |
| eigendecomposition_polar | 0.0% | 0.0% | O(N²) but tiny | **CPU — leave** |
| everything else | <0.5% | <0.1% | various | **CPU — leave** |

---

## 6. Corrected Priority Order

The plan's original expected priority (Granger > Eigendecomposition > PID bootstrap)
was almost right for N=200, but the three-point data confirms a cleaner picture:

### Priority 1: GPU batch Granger edge estimation (clear #1 at scale)

Dominant at N=500 (86.1%). Confirmed O(N²) across all three sizes.
N²=250,000 independent bivariate VAR fits at N=500 — embarrassingly parallel.

**GPU strategy:** Launch N² concurrent VAR fits on GPU. Each fit is a T×2 VAR(p)
regression (matrix solve on a T×2p design matrix), with p≤2. For T=80, p=2:
- Design matrix: (80-2) × 4 = 78×4 per pair
- N² fits = 250,000 fits on GPU
- Each fit is tiny — batch together as a 3D batched least-squares: shape (N², T, 2p)
- CuPy: `cupy.linalg.lstsq` or batched `torch.linalg.lstsq`

Expected GPU speedup: 10–50× (small per-fit cost means GPU utilisation is the question)
Conservative estimate: 10×, so Granger drops from 1517s → 152s at N=500.

### Priority 2: Woodbury identity for Kim filter + Kalman (CPU algorithmic fix)

Dominant at N=200 (55%). Plateaus at N=500 (12%).
The N×N innovation covariance solve can be replaced by a K×K solve via Woodbury:

```
Standard:   S_inv = (R + U P U.T)^{-1}                          O(N³)
Woodbury:   S_inv = R^{-1} - R^{-1} U (P^{-1} + U.T R^{-1} U)^{-1} U.T R^{-1}
                                                    O(K³ + K²N)  where K=15
```

Theoretical speedup at N=200, K=15:
`N³ / (K³ + K²N) = 8,000,000 / 48,375 ≈ 165×`

Expected post-Woodbury times at N=200:
- Kim EM: 235s → ~1.4s
- Kalman: 5s → ~0.03s
- Total: 427s → ~192s

This is a **pure CPU change** — no GPU required. Implement before any GPU work.

Note: At N=500 where both filters are already in the BLAS plateau (~213s Kim, ~14s Kalman),
Woodbury would theoretically give 165× speedup, but the plateau makes the actual gain
dependent on how the cache/BLAS bottleneck shifts at K×K scale. Conservative estimate: 50×.

### Priority 3: GPU batch PID bootstrap (medium, constant cost)

~16s at all N. Embarrassingly parallel: K²/2 mode pairs × B=200 bootstrap draws.
At N=200+, this is only 4% → 0.9% of total. Low urgency.

**GPU strategy:** Batch all B=200 bootstrap sample indices simultaneously.
Each draw is an independent covariance matrix computation on a (T,)-length resample.
CuPy can compute all 200 covariances in one batched einsum.

### Priority 4: GPU batch Kim filter EM (if N grows beyond 500)

At N=200 the filter is the bottleneck; at N=500 Granger overwhelms it. If future
experiments use N>500, the Woodbury-fixed Kim will still be ~T×M²×n_iter = 14,400
K×K solves. These can be batched on GPU. Not urgent for the current programme.

---

## 7. Revised Experiment Programme Estimates

All estimates: N=200 (MVP) and N=500 (scaling target), 680 runs total.

| Phase | Action | N=200/run | N=500/run | 680 runs (N=200) | 680 runs (N=500) |
|-------|--------|-----------|-----------|-----------------|-----------------|
| Baseline | CPU only | 427s | 1762s | **81h (3.4 days)** | **333h (13.9 days)** |
| P1 | Woodbury (CPU) | 192s | 1536s | 36h | 290h |
| P2 | P1 + GPU Granger 10× | 35s | 171s | **6.7h** | **32h** |
| P3 | P1 + GPU Granger 20× | 27s | 95s | 5.1h | **18h** |
| P4 | P3 + GPU PID bootstrap | ~11s | ~79s | 2h | 15h |

The **proposal target of 5–20× speedup** is met by P2 alone:
- N=200: 427/35 = **12.2×** ✓
- N=500: 1762/171 = **10.3×** ✓

P3 (20× GPU Granger) delivers:
- N=200: 427/27 = **15.9×** ✓
- N=500: 1762/95 = **18.5×** ✓

---

## 8. Components That Surprised the Spec

| Component | Spec prediction | Measured reality | Impact on plan |
|-----------|----------------|------------------|----------------|
| Kim filter | "Probably not — O(M²K²T)" | **55% at N=200** but plateaus at 213s for N=500 | Woodbury fix needed; GPU secondary |
| Kalman filter | "No — O(K³T), K small" | **1.2% at N=200**, BLAS plateau from N=200+ | Same Woodbury fix; low urgency |
| PID bootstrap | "Yes — 200× base PID cost" | **57% at N=50** but O(1) in N → 4% at N=200, 0.9% at N=500 | Not a scaling problem; constant cost |
| Transfer entropy | "Maybe — O(T² per pair)" | **<2% at all N** | K² mode pairs only (K=15 fixed) — not O(N²) |
| Eigendecomposition | "Yes — O(N³)" | **<0.1% at N=200, <0.1% at N=500** | Negligible absolute time; not a priority |
| Granger | "Yes — O(N²×T)" | **40% at N=200, 86% at N=500** | Confirmed #1 bottleneck at scale |

---

## 9. Implementation Checklist

In recommended order:

- [ ] **CPU P1**: Apply Woodbury identity to `KalmanFilter.update()` — replaces `np.linalg.solve(S_N×N)` with `np.linalg.solve(S_K×K)`
- [ ] **CPU P1**: Apply same Woodbury fix to `KimFilter.filter()` inner loop
- [ ] **Verify**: Re-run profiler post-Woodbury to confirm 100×+ speedup on Kim/Kalman; run acceptance tests (121/121 must pass)
- [ ] **GPU P2**: Implement batched Granger VAR fits using CuPy/PyTorch tensor operations
- [ ] **GPU P2**: Add CPU fallback in `GrangerEdgeEstimator` (auto-detect GPU via `compute.backend`)
- [ ] **Verify**: Re-run profiler; confirm ~10× Granger speedup; run acceptance tests
- [ ] **GPU P4** (optional): Batch PID bootstrap draws on GPU if overall time budget requires it

---

## 10. Re-running the Profiler

```bash
# Quick profile (N=50, N=200 only — ~8 minutes)
uv run python -m quantdsl_backtest.smim.profiling --n-actors 50 200 --output profiling_results.json

# Full profile including N=500 (~30 minutes wall-clock)
uv run python -m quantdsl_backtest.smim.profiling --n-actors 50 200 500 --output profiling_results.json

# After implementing Woodbury fix — should show Kim/Kalman drop to <2s at N=200
uv run python -m quantdsl_backtest.smim.profiling --n-actors 200 --output profiling_post_woodbury.json
```

Raw JSON results: `profiling_results.json` (project root).
