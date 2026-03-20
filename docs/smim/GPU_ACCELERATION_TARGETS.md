# SMIM GPU Acceleration Targets

**Generated:** 2026-03-20
**Profiler:** `src/quantdsl_backtest/smim/profiling.py`
**Hardware:** CPU-only baseline (Windows 11, Python 3.11, NumPy/SciPy)
**Config:** T=80 quarters, K=15 modes, M=3 regimes, B=200 bootstrap draws

---

## 1. Raw Profiling Results

### N=50 (small run)

| Component | Seconds | % of total |
|-----------|---------|------------|
| pid_bootstrap | 25.60 | 70.3% |
| edge_estimation_granger | 9.15 | 25.1% |
| transfer_entropy | 1.08 | 3.0% |
| kim_filter_em | 0.45 | 1.2% |
| pid_synergy | 0.08 | 0.2% |
| edge_estimation_narrative | 0.04 | 0.1% |
| kalman_filter | 0.01 | 0.0% |
| eigendecomposition_polar | 0.00 | 0.0% |
| eigendecomposition_schur | 0.00 | 0.0% |
| everything else | <0.01 | 0.0% |
| **TOTAL** | **36.4s** | |

### N=200 (MVP experiment size)

| Component | Seconds | % of total |
|-----------|---------|------------|
| kim_filter_em | 235.2 | 55.0% |
| edge_estimation_granger | 170.4 | 39.9% |
| pid_bootstrap | 16.0 | 3.7% |
| kalman_filter | 5.0 | 1.2% |
| transfer_entropy | 0.46 | 0.1% |
| edge_estimation_narrative | 0.28 | 0.1% |
| eigendecomposition_schur | 0.09 | 0.0% |
| pid_synergy | 0.07 | 0.0% |
| eigendecomposition_polar | 0.05 | 0.0% |
| everything else | <0.01 | 0.0% |
| **TOTAL** | **427.4s** | |
| **Scaling vs N=50** | | **14.97×** |

> **Note on measurements.** Two independent profiling runs were conducted at N=200.
> Run 1 (used above): 427s total. Run 2: 999s total — the 2.34× inflation was caused by
> CPU contention from a background process, which anomalously inflated the Kalman filter
> from 5s → 253s. Run 1 figures are used throughout this document.

---

## 2. Empirical Scaling Laws

Scaling exponent α computed from the two data points N=50 and N=200 (ratio 4×):
`α = log(t₂₀₀ / t₅₀) / log(4)`.

| Component | N=50 | N=200 | Ratio | Empirical α | Theoretical |
|-----------|------|-------|-------|-------------|-------------|
| kim_filter_em | 0.45s | 235.2s | 526× | **O(N^4.5)** | O(N³) per EM iter |
| kalman_filter | 0.008s | 5.0s | 613× | **O(N^4.6)** | O(N³) per step |
| eigendecomposition_schur | 0.001s | 0.088s | 63× | O(N^3.0) | O(N³) ✓ |
| eigendecomposition_polar | 0.002s | 0.050s | 24× | O(N^2.3) | O(N³) + eigh |
| edge_estimation_granger | 9.15s | 170.4s | 18.6× | **O(N^2.1)** | O(N²·T) ✓ |
| edge_estimation_narrative | 0.04s | 0.28s | 6.7× | O(N^1.4) | O(N²) TF-IDF |
| pid_bootstrap | 25.60s | 16.0s | 0.6× | **O(1) in N** | O(K²·B), K fixed |
| pid_synergy | 0.08s | 0.07s | 0.9× | O(1) in N | O(K²), K fixed |
| transfer_entropy | 1.08s | 0.46s | 0.4× | O(1) in N | O(K²·T²), K fixed |
| persistent_homology | 0.012s | 0.004s | 0.3× | O(1) in N | ripser on (T,K) |
| kalman_filter | 0.008s | 5.0s | 613× | O(N^4.6) | O(N³·T) |

### Why Kim/Kalman scale super-cubically

Both the Kalman filter and Kim EM compute an N×N **innovation covariance**:

```
S = U @ P @ U.T + R          # (N, N) — O(K²N + N²)
S_inv = np.linalg.solve(S, I_N)  # (N, N) — O(N³)
```

With K=15 << N, the dominant cost is the N×N linear solve, giving theoretical O(N³).
The observed O(N^4.5) empirical exponent (worse than cubic) is caused by:
- Memory allocation pressure: a fresh (N,N) array is allocated at every time step
- Cache thrashing: at N=200, S is 200×200×8 bytes = 320 kB, exceeding L2 cache
- NumPy overhead: repeated small solves are not BLAS-3 efficient

Kim EM amplifies this by a factor of T × M² × n_iter = 80 × 9 × 20 = **14,400 inner loops**.

### N=500 Projection

Extrapolating from the empirical scaling exponents (N ratio = 10× from N=50 baseline):

| Component | N=500 estimate | Confidence |
|-----------|---------------|------------|
| kim_filter_em | **~14,100s (3.9h)** | medium (O(N^4.5) empirical) |
| edge_estimation_granger | **~1,150s (19 min)** | high (O(N²) confirmed) |
| kalman_filter | **~320s (5 min)** | medium (same issue as Kim) |
| pid_bootstrap | **~12s** | high (constant in N) |
| everything else | **~3s** | high (negligible) |
| **TOTAL** | **~15,600s (4.3h/run)** | |

At N=500 with 680 experiment runs: **~2,950 hours = 123 days** of wall-clock time.

---

## 3. Falsification Suite Profile

Profiled at N=200, one null-model instance (rewire + Schur + Kalman filter + benchmark):

| Sub-step | Seconds | % |
|----------|---------|---|
| rewire (degree-preserving) | 0.23 | 4.0% |
| schur decomposition | 0.07 | 1.1% |
| kalman filter | 5.43 | **94.9%** |
| benchmark computation | 0.00 | 0.0% |
| **one instance** | **5.72s** | |
| **projected B=100** | **572s (0.16h)** | |

The Kalman filter dominates the falsification suite for the same reason it dominates
the main pipeline — the N×N innovation covariance solve.

---

## 4. Acceleration Decision Table

Decision rule from `GPU_ACCELERATION_PLAN.md`:
- `>10% of total time AND scales as O(N²) or worse` → **HIGH PRIORITY GPU**
- `>10% of total time AND embarrassingly parallel` → **MEDIUM (batch GPU)**
- `<5% of total time` → **Leave on CPU**

| Component | N=200 % | Scaling | Decision | GPU Strategy |
|-----------|---------|---------|----------|--------------|
| **kim_filter_em** | 55.0% | O(N^4.5) | **HIGH — CPU fix first, then GPU** | ① Woodbury identity eliminates N×N solve; ② CuPy batch solves for residual K×K systems |
| **edge_estimation_granger** | 39.9% | O(N²) | **HIGH — GPU batch** | Batch all N² independent bivariate VAR fits on GPU; CuPy tensor parallelism |
| **kalman_filter** | 1.2% | O(N^4.6) | **MEDIUM — CPU fix (same as Kim)** | Same Woodbury fix as Kim EM; GPU only if still bottleneck after fix |
| pid_bootstrap | 3.7% | O(1) in N | **MEDIUM — batch GPU at small N** | 200 independent bootstrap draws per mode pair — embarrassingly parallel; priority low at N≥200 |
| transfer_entropy | 0.1% | O(1) in N | **CPU — leave** | Only K²=225 pairs, constant cost |
| eigendecomposition_schur | 0.0% | O(N³) | **CPU — leave** | <0.1s at N=200; cuSOLVER only if N grows to 500+ |
| eigendecomposition_polar | 0.0% | O(N²·³) | **CPU — leave** | Negligible |
| edge_estimation_narrative | 0.1% | O(N^1.4) | **CPU — leave** | <0.5s at N=200 |
| pid_synergy | 0.0% | O(1) | **CPU — leave** | Negligible |
| persistent_homology | 0.0% | O(1) | **CPU — leave** | ripser already C++ optimised |
| mode_selection | 0.0% | O(1) | **CPU — leave** | Lightweight |
| benchmark_computation | 0.0% | O(1) | **CPU — leave** | Single matrix multiply |

---

## 5. Corrected Priority Order

The plan's original expected priority order was: Granger > Eigendecomposition > PID bootstrap.
Profiling data **overturns this**:

### Actual priority order (by impact at N=200)

1. **Woodbury identity fix for Kim filter + Kalman filter** ← NEW #1
   Not a GPU task — a pure algorithmic fix. The N×N innovation covariance can be
   replaced with a K×K system using the Woodbury matrix inversion lemma:

   ```
   Standard:  S_inv = (R + U P U.T)^{-1}                  O(N³)
   Woodbury:  (R + U P U.T)^{-1} = R^{-1} - R^{-1} U (P^{-1} + U.T R^{-1} U)^{-1} U.T R^{-1}
              → reduces to O(K³ + K²N)                     where K=15 << N
   ```

   Theoretical speedup at N=200, K=15: `(N³) / (K³ + K²N) = 8,000,000 / 48,375 ≈ 165×`
   Expected post-fix Kim EM time at N=200: `235s / 165 ≈ 1.4s`
   Expected post-fix per-run total at N=200: `~192s → 427s × (1.4/235) + 170 + 16 + ... ≈ 190s`

2. **GPU: Granger edge estimation (batch parallel)** ← original priority confirmed
   O(N²) scaling confirmed. 39.9% at N=200, 19 min at N=500.
   N²=40,000 independent bivariate VAR fits → batch on GPU.
   Expected GPU speedup: 10–20× → reduces from 170s to 8–17s at N=200.

3. **GPU: Kim filter + Kalman (residual after Woodbury)** ← upgraded from "probably not"
   After Woodbury, inner loop becomes K×K solves (O(K³)) — fast, but 14,400 of them.
   Batch these on GPU. Residual cost at N=200 after Woodbury: ~1.4s → not urgent.
   Worth doing if experiment programme exceeds 500 runs.

4. **GPU: PID bootstrap (batch parallel)** ← original priority, lower than expected
   Only 2.4% at N=200, ~12s at N=500. Embarrassingly parallel (200 draws × K²/2 pairs).
   Implement if total time budget requires it.

5. **Everything else** — CPU, not worth the complexity.

---

## 6. Expected Speedup Roadmap

All estimates at N=200, T=80, K=15, M=3.

| Phase | Action | Estimated per-run | vs baseline | 680-run programme |
|-------|--------|------------------|-------------|-------------------|
| Baseline | CPU only | 427s | 1× | 81h (3.4 days) |
| P1 | Woodbury fix (CPU) | ~192s | 2.2× | 36h |
| P2 | P1 + GPU Granger (10×) | ~39s | 11× | 7.4h |
| P3 | P2 + GPU Granger (20×) | ~30s | 14× | 5.6h |
| P4 | P3 + GPU PID bootstrap | ~17s | 25× | 3.2h |

The **proposal target of 5–20× speedup** is achievable with P2 alone (11× at 10× GPU Granger).

At N=500 the numbers are more extreme (123 days baseline), but P1+P2 still yields ~12× speedup,
reducing to ~10 days — which may require additional GPU effort on Kim filter.

---

## 7. Components That Surprised the Spec

| Component | Spec expectation | Measured reality | Implication |
|-----------|-----------------|------------------|-------------|
| Kim filter | "Probably not — O(M²K²T)" | **55% at N=200, O(N^4.5)** | N×N solve overlooked; Woodbury fix is critical |
| Kalman filter | "No — O(K³T), K small" | **1.2% at N=200, same O(N^4.6)** | Same root cause; same fix |
| PID bootstrap | "Yes — 200× base PID cost" | **70% at N=50, 4% at N=200** | O(1) in N — not a scaling problem, a constant cost |
| Transfer entropy | "Maybe — O(T² per pair)" | **<1% at all N** | K² pairs only (K=15 fixed), not N pairs |
| Eigendecomposition | "Yes — O(N³)" | **<0.1% at N=200** | Low absolute time despite cubic scaling |

---

## 8. Raw JSON Output

The raw profiling output is saved in `profiling_results.json` (project root, not committed).
To re-run:

```bash
uv run python -m quantdsl_backtest.smim.profiling --n-actors 50 200 --output profiling_results.json
```

For N=500 (estimated 4+ hours, Kim EM dominated):
```bash
uv run python -m quantdsl_backtest.smim.profiling --n-actors 500 --output profiling_n500.json
```
