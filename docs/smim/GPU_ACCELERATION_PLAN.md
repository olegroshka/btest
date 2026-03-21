# SMIM GPU Acceleration Phase — Master Plan

## Context

Read this entire document before writing any code. This is a multi-session project
that accelerates the SMIM pipeline using PyTorch for unified CPU/CUDA execution and
a Woodbury algorithmic fix for the Kalman/Kim filter bottleneck.

The existing CPU implementations are correct (121 acceptance tests pass). All
accelerated code must produce outputs matching the acceptance test criteria.

Reference documents:
- `docs/smim/ACCEPTANCE_TESTS.md` — 121 correctness tests
- `docs/smim/EXPERIMENT_PLAN.md` — ~680 pipeline runs driving the speed requirement
- `docs/smim/GPU_ACCELERATION_TARGETS.md` — actual profiling data from Task 1.1–1.2
- `src/quantdsl_backtest/smim/interfaces.py` — existing Protocol definitions

## Profiling Results (Measured)

**Hardware:** NVIDIA GeForce RTX 4070 Ti, CUDA 12.6, PyTorch 2.10.0+cu126

Two profiling snapshots:

### Pre-Woodbury baseline (from GPU_ACCELERATION_TARGETS.md)

| Component | N=50 | N=200 | N=500 | Scaling | Priority |
|-----------|------|-------|-------|---------|----------|
| Granger edges | 39% (11s) | 40% (170s) | 86% (1517s) | O(N²) | **#1 — GPU batch** |
| Kim filter EM | 2% (0.4s) | 55% (235s) | 12% (213s) | O(N³)/BLAS plateau | **#2 — Woodbury CPU** |
| PID bootstrap | 57% (16s) | 4% (16s) | 1% (17s) | O(1) in N | **#3 — GPU batch** |
| Kalman filter | 0% (0.008s) | 1% (5s) | 0.8% (14s) | O(N³)/BLAS plateau | Fixed by #2 |
| Transfer entropy | 2% (0.5s) | 0.1% (0.5s) | <0.1% | O(1) in N | Leave on CPU |
| **TOTAL** | **28s** | **427s** | **1762s** | | |

### Post-Woodbury, pre-GPU-Granger (current state — `profiling_results.json`)

| Component | N=50 | N=200 | N=500 | Scaling | Status |
|-----------|------|-------|-------|---------|--------|
| Granger edges | 34% (8.8s) | **89% (148.6s)** | **98% (881.7s)** | O(N²) | Batch impl done, not yet wired |
| Kim filter EM | 2% (0.57s) | 0.4% (0.73s) ✅ | 0.1% (0.92s) ✅ | Woodbury fixed | **321× speedup at N=200** |
| PID bootstrap | 62% (16.1s) | 10% (16.9s) | 2% (16s) | O(1) in N | Batch impl done, not yet wired |
| Kalman filter | 0.05% (0.01s) | 0.02% (0.03s) ✅ | 0.02% (0.14s) ✅ | Woodbury fixed | **167× speedup at N=200** |
| Transfer entropy | 1.8% (0.47s) | 0.3% (0.46s) | 0.05% (0.46s) | O(1) in N | Leave on CPU |
| **TOTAL** | **26.1s** | **167s** | **900s** | | **2.6× vs baseline at N=200** |

**Key insight**: Woodbury fix (GPU-0.1) eliminated the Kim/Kalman bottleneck entirely.
Granger is now 89–98% of all pipeline time. GPU-1.2 batch implementation exists;
wiring it in (GPU-2.1) is the next highest-impact step.

## Why PyTorch, Not CuPy

PyTorch provides a **single code path** for CPU and CUDA:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.as_tensor(A_numpy, device=device, dtype=torch.float64)
U, S, Vh = torch.linalg.svd(A)
```

Benefits over the original CuPy plan:
- No separate CpuBackend/GpuBackend classes — one implementation, device-parametrised
- No parity test suite needed — same code path, same floating-point operations
- Batched operations native: `torch.linalg.solve` handles (B, N, N) batch dimensions
- Future-proof: if we add learned components (differentiable filtering, neural
  network narrative embeddings), autograd is already available

Limitation: `torch.linalg.schur` does not exist. Schur decomposition stays on
scipy (CPU). This is acceptable — Schur takes <0.1% of total time.

---

## Part 0: Woodbury Fix for Kim/Kalman (CPU, No GPU Needed) ✅ DONE

**Completed.** Commit `8db184a`. Actual results at N=200: Kim 235s → 0.73s (321×),
Kalman 5s → 0.03s (167×). Both now <0.5% of pipeline time.

### Task 0.1: Woodbury Kalman filter ✅ DONE

```
The Kalman filter currently inverts the N×N innovation covariance:
  S = U P U^T + R    (N×N)
  K = P U^T S^{-1}   (requires O(N³) inversion)

where U is N×K (modal frame), P is K×K (state covariance), R is N×N (obs noise).

Apply the Woodbury identity to compute S^{-1} via a K×K inversion instead:
  S^{-1} = R^{-1} - R^{-1} U (P^{-1} + U^T R^{-1} U)^{-1} U^T R^{-1}

The inner matrix (P^{-1} + U^T R^{-1} U) is K×K (typically 15×15).
This reduces the per-step cost from O(N³) to O(NK² + K³).

If R is diagonal (common case), R^{-1} is trivial — just 1/diag(R).

Implementation in src/quantdsl_backtest/smim/dynamics/kalman.py:

1. Add a method _woodbury_gain(U, P_pred, R) that computes the Kalman gain
   via Woodbury. Return both K and S^{-1} (needed for log-likelihood).

2. Add numerical safety: if cond(P^{-1} + U^T R^{-1} U) > 1e8, fall back
   to direct N×N inversion with a warning.

3. Replace the existing gain computation in the filter loop.

4. Do the same in the Kim filter (kim_filter.py) — each of the M² branch
   filters uses the same innovation covariance structure.

Run: uv run pytest tests/acceptance/smim/ -v --tb=short
ALL 121 tests must still pass. The outputs must be numerically identical
(within 1e-10) to the pre-Woodbury implementation.

Then re-run the profiler to measure the improvement:
uv run python -m quantdsl_backtest.smim.profiling --n-actors 50 200 500

Git add and commit:
[SMIM GPU-0.1] Apply Woodbury identity to Kalman and Kim filter gain computation
```

---

## Part 1: PyTorch Compute Layer

### Task 1.1: Create the compute module with device management ✅ DONE

Create `src/quantdsl_backtest/smim/compute/__init__.py` and
`src/quantdsl_backtest/smim/compute/torch_ops.py`:

```python
"""PyTorch-based compute operations with automatic CPU/CUDA dispatch.

Usage:
    from quantdsl_backtest.smim.compute import get_device, ensure_tensor, to_numpy

    device = get_device()  # auto-detect CUDA, or force via SMIM_DEVICE env var
    A_t = ensure_tensor(A_numpy, device=device)
    result_numpy = to_numpy(some_tensor)

All functions accept numpy arrays and return numpy arrays at the boundary.
PyTorch tensors are used internally for computation only.
"""
import os
import logging
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device(force: str | None = None) -> torch.device:
    """Get the compute device.

    Priority: force arg > SMIM_DEVICE env var > auto-detect.

    Args:
        force: "cpu", "cuda", "cuda:0", "cuda:1", or None (auto-detect).

    Returns:
        torch.device
    """
    if force is not None:
        device_str = force
    else:
        device_str = os.environ.get("SMIM_DEVICE", "auto")

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            logger.info(f"SMIM compute: using CUDA device {name}")
        else:
            device = torch.device("cpu")
            logger.info("SMIM compute: using CPU (no CUDA available)")
    elif device_str == "cpu":
        device = torch.device("cpu")
        logger.info("SMIM compute: using CPU (forced)")
    else:
        device = torch.device(device_str)
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device.index or 0)
            logger.info(f"SMIM compute: using CUDA device {name}")
    return device


def ensure_tensor(
    arr: NDArray | torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert numpy array to torch tensor on the target device.

    Always uses float64 for numerical precision matching scipy.
    """
    if device is None:
        device = get_device()
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(arr), device=device, dtype=dtype)


def to_numpy(t: torch.Tensor) -> NDArray:
    """Convert torch tensor back to numpy (on CPU)."""
    return t.detach().cpu().numpy()
```

Also create `src/quantdsl_backtest/smim/compute/linalg.py`:

```python
"""Linear algebra operations — single implementation for CPU and CUDA.

Every function: numpy in → numpy out. PyTorch used internally.
Schur decomposition falls back to scipy (not available in torch).
"""
import numpy as np
import torch
import scipy.linalg
from numpy.typing import NDArray
from .torch_ops import ensure_tensor, to_numpy, get_device


def svd(A: NDArray, k: int | None = None) -> tuple[NDArray, NDArray, NDArray]:
    """SVD with optional truncation. Works on CPU or CUDA."""
    A_t = ensure_tensor(A)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
    if k is not None:
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    return to_numpy(U), to_numpy(S), to_numpy(Vh)


def eigh(H: NDArray, k: int | None = None) -> tuple[NDArray, NDArray]:
    """Symmetric eigendecomposition. Works on CPU or CUDA."""
    H_t = ensure_tensor(H)
    vals, vecs = torch.linalg.eigh(H_t)
    if k is not None:
        # Return top-k by magnitude
        idx = torch.argsort(vals.abs(), descending=True)[:k]
        vals, vecs = vals[idx], vecs[:, idx]
    return to_numpy(vals), to_numpy(vecs)


def polar_decompose(A: NDArray) -> tuple[NDArray, NDArray]:
    """Polar decomposition A = UP via SVD. Works on CPU or CUDA."""
    A_t = ensure_tensor(A)
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)
    orth = U @ Vh
    psd = (Vh.conj().mT * S.unsqueeze(-2)) @ Vh
    return to_numpy(orth), to_numpy(psd)


def schur_decompose(A: NDArray, k: int | None = None) -> tuple[NDArray, NDArray]:
    """Schur decomposition A = QTQ^H. Always runs on CPU (scipy).

    torch.linalg.schur does not exist. This is the one operation that
    cannot be GPU-accelerated. It takes <0.1% of pipeline time, so
    the impact is negligible.
    """
    T, Q = scipy.linalg.schur(np.asarray(A), output='complex')
    if k is not None:
        eig_magnitudes = np.abs(np.diag(T))
        idx = np.argsort(eig_magnitudes)[::-1][:k]
        Q, T = Q[:, idx], T[np.ix_(idx, idx)]
    return Q, T


def hermitian_dilation_decompose(
    A: NDArray, k: int
) -> tuple[NDArray, NDArray, NDArray]:
    """Hermitian dilation: eigendecompose H = [[0,A],[A^T,0]].
    Returns (eigenvalues, U_L, U_R). Works on CPU or CUDA.
    """
    N = A.shape[0]
    A_t = ensure_tensor(A)
    zeros = torch.zeros((N, N), device=A_t.device, dtype=A_t.dtype)
    H = torch.cat([
        torch.cat([zeros, A_t], dim=1),
        torch.cat([A_t.T, zeros], dim=1),
    ], dim=0)
    vals, vecs = torch.linalg.eigh(H)
    # Select top-k positive eigenvalues
    pos_mask = vals > 1e-10
    pos_vals = vals[pos_mask]
    pos_vecs = vecs[:, pos_mask]
    # Sort descending
    idx = torch.argsort(pos_vals, descending=True)[:k]
    sigma = pos_vals[idx]
    # Extract U_L (first N rows) and U_R (last N rows)
    U_L = pos_vecs[:N, idx] * (2 ** 0.5)
    U_R = pos_vecs[N:, idx] * (2 ** 0.5)
    return to_numpy(sigma), to_numpy(U_L), to_numpy(U_R)


def batch_covariance(X: NDArray) -> NDArray:
    """Covariance matrix. Works on CPU or CUDA."""
    X_t = ensure_tensor(X)
    mean = X_t.mean(dim=0, keepdim=True)
    centered = X_t - mean
    cov = (centered.T @ centered) / (X_t.shape[0] - 1)
    return to_numpy(cov)


def batch_solve(A: NDArray, B: NDArray) -> NDArray:
    """Batched linear solve A @ X = B. Shape (batch, N, N) and (batch, N, M).
    Works on CPU or CUDA — this is the key primitive for batch Granger.
    """
    A_t = ensure_tensor(A)
    B_t = ensure_tensor(B)
    X_t = torch.linalg.solve(A_t, B_t)
    return to_numpy(X_t)
```

Tests:
- Verify all functions work on CPU
- If CUDA available, verify they also work on CUDA and produce same results
- Schur always returns CPU result regardless of SMIM_DEVICE

**Implemented.** Commits `d93cd74`, `8431c9c`. Actual files created:
- `compute/__init__.py`, `compute/torch_ops.py`, `compute/linalg.py`
- 54 unit tests in `tests/unit/smim/compute/` — 54/54 pass on both CPU and CUDA
- `hermitian_dilation_decompose` fixed for non-square A (M×M and N×N zero blocks)
- 121/121 acceptance tests pass with `SMIM_DEVICE=cuda`

Git add and commit:
[SMIM GPU-1.1] Create PyTorch compute layer with CPU/CUDA auto-dispatch

### Task 1.2: Batch Granger on PyTorch ✅ DONE

**Completed.** Commit `43d9455`. `compute/batch_granger.py` contains
`batch_granger_test` (GPU batch, fixed lag) and `_granger_sequential`
(numpy OLS reference). Constant term included (L+1 / 2L+1 params).
Chunked at 10,000 pairs/batch. 23 unit tests + 5 acceptance parity tests pass.
**Parity verified:** identical edge set to statsmodels on A-GR-1 data (CUDA + CPU).
**Not yet wired** into `GrangerEdgeEstimator` — that is Task 2.1.

This is the highest-impact GPU acceleration. Replace 250,000 sequential
statsmodels VAR fits with a single batched least-squares solve on GPU.

Create `src/quantdsl_backtest/smim/compute/batch_granger.py`:

```python
"""Batch Granger causality via PyTorch batched least-squares.

For N actors and max_lag L, the Granger test for i→j is:
  Restricted model:  y_j[t] = Σ_l a_l y_j[t-l] + ε        (L params)
  Unrestricted model: y_j[t] = Σ_l a_l y_j[t-l] + Σ_l b_l y_i[t-l] + ε  (2L params)
  F = ((RSS_r - RSS_u) / L) / (RSS_u / (T - 2L))

Strategy: construct ALL N² regression matrices in a single batch tensor
and solve via torch.linalg.lstsq. This replaces the Python loop over N²
pairs with a single GPU kernel launch.
"""
```

The implementation must:
1. For each directed pair (i, j), construct the (T-L) × 2L design matrix
   [y_j lagged, y_i lagged] and the (T-L) response vector y_j[L:]
2. Stack ALL N² design matrices into a (N², T-L, 2L) batch tensor
3. Solve via torch.linalg.lstsq in one call
4. Compute RSS_restricted (first L columns only) and RSS_unrestricted (all 2L)
   — this can also be batched
5. Compute F-statistics and p-values from scipy.stats.f.sf (on CPU after transfer)
6. Threshold and return sparse adjacency

For memory management at N=500 (250K pairs × 80 × 8 = ~150MB for design matrices):
- If GPU memory < needed, process in chunks of 10,000 pairs
- Always use float64 for numerical precision matching statsmodels

Tests:
- On the SAME synthetic VAR(1) data from acceptance test A-GR-1, verify the
  batch implementation detects the same edges as the original statsmodels version
- F-statistics agree within 1e-4 (different solver paths)
- Edge structure (which pairs are significant) is IDENTICAL

Git add and commit:
[SMIM GPU-1.2] Implement batched Granger causality via PyTorch lstsq

### Task 1.3: Batch PID Bootstrap on PyTorch

Replace 200 sequential covariance computations with batched GPU covariance.

Create `src/quantdsl_backtest/smim/compute/batch_pid.py`:

```python
"""Batch PID bootstrap via PyTorch.

For Gaussian MMI PID, synergy depends only on covariance matrices.
The bootstrap generates B resampled covariance matrices — each is a
matrix multiply, trivially parallelisable.

Strategy:
1. Generate B block-bootstrap index arrays on device
2. For each bootstrap sample, compute the covariance submatrices
   for (alpha_j, alpha_k, target) — batched as (B, 3, 3) covariances
3. Compute MI terms from covariance determinants (batched torch.linalg.det)
4. Derive R, U_j, U_k, S for all B samples in parallel
5. Return point estimate (mean) and CIs (percentiles)
"""
```

The implementation must:
1. Accept modal_states (T, K) and target (T,) as numpy
2. Generate B bootstrap index arrays of length T (block bootstrap with block
   size ~sqrt(T) for temporal dependence)
3. For each pair (j, k): stack the (alpha_j, alpha_k, target) submatrix
   across B samples → shape (B, T, 3)
4. Compute covariance: (B, 3, 3) — batched matrix multiply
5. Compute all MI terms via log-det: I(X;Y) = 0.5 * (log|Σ_X| + log|Σ_Y| - log|Σ_{XY}|)
   — batched via torch.linalg.slogdet
6. Return synergy matrix and CIs as numpy

Tests:
- Synergy point estimates match the existing CPU implementation within 1e-4
- Bootstrap CIs have same width (within 30% — different RNG but same distribution)

Git add and commit:
[SMIM GPU-1.3] Implement batched PID bootstrap via PyTorch

### Task 1.4: GPU-accelerated KNN for transfer entropy

Create `src/quantdsl_backtest/smim/compute/gpu_knn.py`:

```python
"""Brute-force k-nearest-neighbours via PyTorch.

For KSG transfer entropy, we need KNN on (T, d) point clouds where
T ≤ 10,000 and d ≤ 6. Brute-force pairwise distances on GPU is faster
than CPU KD-trees for this regime.

Strategy: compute full pairwise distance matrix on GPU via the expansion trick:
  ||a-b||² = ||a||² + ||b||² - 2 a·b
Then torch.topk for k smallest per row.
"""
```

Handle memory for large T: if T² × 8 bytes > 2GB, compute distances in row chunks.

Tests:
- KNN results match scipy.spatial.KDTree exactly (same neighbours, same distances)
- Faster than scipy KDTree for T > 2000

Git add and commit:
[SMIM GPU-1.4] Implement GPU brute-force KNN for transfer entropy

---

## Part 2: Wire Into Pipeline

### Task 2.1: Replace scipy/numpy calls with compute layer

Modify existing pipeline components to use the new compute functions.
The pattern is simple — import from compute instead of scipy:

```python
# BEFORE (in smim/spectral/polar.py):
from scipy.linalg import polar
U, P = polar(A)

# AFTER:
from quantdsl_backtest.smim.compute.linalg import polar_decompose
U, P = polar_decompose(A)
```

Files to modify:
1. `smim/spectral/schur.py` → use `compute.linalg.schur_decompose` (stays on CPU)
2. `smim/spectral/polar.py` → use `compute.linalg.polar_decompose`
3. `smim/spectral/hermitian.py` → use `compute.linalg.hermitian_dilation_decompose`
4. `smim/spectral/dmd.py` → use `compute.linalg.svd`
5. `smim/graph/edges/granger.py` → use `compute.batch_granger.batch_granger_test`
6. `smim/emergence/pid.py` → use `compute.batch_pid.batch_pid_synergy`
7. `smim/emergence/transfer_entropy.py` → use `compute.gpu_knn.knn_query`

DO NOT change any mathematical logic. Only change where the computation runs.

After wiring, run the full acceptance suite:
```bash
uv run pytest tests/acceptance/smim/ -v --tb=short
```

All 121 tests must pass. Any failure means the wiring introduced a bug.

Then test with CUDA (if available):
```bash
SMIM_DEVICE=cuda uv run pytest tests/acceptance/smim/ -v --tb=short
```

All 121 tests must also pass on CUDA. Same tests, same thresholds — because
it's the same code path, just on a different device.

Git add and commit:
[SMIM GPU-2.1] Wire compute layer into pipeline components

### Task 2.2: Add device selection to pipeline config

Extend `SmimConfig` in `smim/config.py`:

```python
class ComputeConfig(BaseModel):
    """Compute device configuration."""
    device: str = Field(
        default="auto",
        description="'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'"
    )
    granger_chunk_size: int = Field(
        default=10000,
        description="Max pairs per GPU batch for Granger (memory control)"
    )
    pid_bootstrap_on_gpu: bool = Field(
        default=True,
        description="Run PID bootstrap on GPU if available"
    )
    knn_on_gpu: bool = Field(
        default=True,
        description="Run KNN for TE on GPU if available"
    )
    float_dtype: str = Field(
        default="float64",
        description="'float64' (precise) or 'float32' (faster, less precise)"
    )
```

Add `compute: ComputeConfig` to the top-level `SmimConfig`.

Update `experiments/mvp_energy_us_uk.yaml` with the compute section.

Git add and commit:
[SMIM GPU-2.2] Add compute device configuration to SmimConfig

---

## Part 3: Verification

### Task 3.1: Device-parametrised acceptance tests

Instead of a separate parity suite, run the EXISTING 121 acceptance tests
on both CPU and CUDA. This is the PyTorch advantage — same code, both devices.

Add to `tests/acceptance/smim/conftest.py`:

```python
import os

def pytest_configure(config):
    """Register gpu marker."""
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")

@pytest.fixture(autouse=True)
def configure_device_from_env():
    """Allow SMIM_DEVICE=cuda to run acceptance suite on GPU."""
    device = os.environ.get("SMIM_DEVICE")
    if device:
        from quantdsl_backtest.smim.compute.torch_ops import get_device
        get_device.cache_clear()  # reset cached device
    yield
    if device:
        get_device.cache_clear()
```

Running on CPU (default — nothing changes):
```bash
uv run pytest tests/acceptance/smim/ -v --tb=short
```

Running on CUDA:
```bash
SMIM_DEVICE=cuda uv run pytest tests/acceptance/smim/ -v --tb=short
```

Both must show: `121/121 passed ✅`

### Task 3.2: Granger parity spot-check

Even with the same code path, the batched Granger implementation is a
DIFFERENT algorithm (batched lstsq vs sequential statsmodels VAR). Add
a dedicated cross-check:

Create `tests/acceptance/smim/test_granger_batch_parity.py`:

```python
"""Verify batched Granger produces same edges as original statsmodels."""

def test_batch_granger_matches_original():
    """On the same VAR(1) data from A-GR-1, the batch implementation
    must detect the same edges as the original statsmodels implementation."""
    # Generate known VAR(1) data (same as acceptance test A-GR-1)
    # Run BOTH: original statsmodels loop AND batch PyTorch implementation
    # Compare:
    #   - Same edges detected (identical adjacency structure)
    #   - F-statistics within 1e-3 relative tolerance
    #   - p-values agree to 2 decimal places
```

This test imports and runs both implementations side-by-side. It's the
one place where we explicitly compare old vs new, because the algorithm
changed (not just the device).

Git add and commit:
[SMIM GPU-3.1] Add device-parametrised acceptance and Granger parity tests

### Task 3.3: Determinism test

```python
def test_gpu_determinism():
    """Run pipeline 5× on CUDA with same seed. All outputs must be identical."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    # Set deterministic mode
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    results = []
    for _ in range(5):
        result = run_pipeline_once(seed=42, device="cuda")
        results.append(result)

    for i in range(1, 5):
        np.testing.assert_array_equal(results[0].gaps, results[i].gaps)
        np.testing.assert_array_equal(results[0].regimes, results[i].regimes)
```

If this fails with `torch.use_deterministic_algorithms(True)`, there's a
non-deterministic operation that needs fixing (usually a scatter/gather or
atomicAdd in a custom kernel). Document which operations require deterministic
mode and the performance cost.

Git add and commit:
[SMIM GPU-3.2] Add GPU determinism test

---

## Part 4: Performance Benchmarks

### Task 4.1: Benchmark infrastructure ✅ DONE

**Completed.** Commit `[SMIM GPU-4]`. `pytest-benchmark>=4.0` already in `benchmarks`
extra. Created `tests/benchmarks/smim/` with `conftest.py` (CPU/CUDA fixture),
4 benchmark files, and `scripts/gpu_speedup_report.py`.

Add `pytest-benchmark` to dev dependencies. Create `tests/benchmarks/smim/`:

```python
# conftest.py
import pytest
import torch

@pytest.fixture(params=["cpu", "cuda"], ids=["CPU", "CUDA"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA")
    return torch.device(request.param)
```

### Task 4.2: Component benchmarks ✅ DONE

**Completed.** Commit `[SMIM GPU-4]`.
- `test_bench_granger.py` — N in [50,100,200,500], T=80: 11–16× speedup at N≤200
- `test_bench_linalg.py` — SVD, polar, hermitian dilation; CPU wins for small N
- `test_bench_pid.py` — PID bootstrap (K in [5,10,15,20]) + KNN (T in [500..5000])
- `test_bench_pipeline.py` — full GPU-accelerated path: 8–12× at N=100–200

Run: `uv run pytest tests/benchmarks/smim/ -v --benchmark-columns=mean,stddev,rounds`

Create benchmark files that parametrise over device AND problem size:

```python
# test_bench_granger.py
@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_granger(benchmark, device, N):
    """Benchmark Granger edge estimation."""
    import os
    os.environ["SMIM_DEVICE"] = str(device)
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((N, 80))
    from quantdsl_backtest.smim.compute.batch_granger import batch_granger_test
    benchmark(batch_granger_test, signals, max_lag=4, p_threshold=0.05)

# test_bench_linalg.py
@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_svd(benchmark, device, N):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    os.environ["SMIM_DEVICE"] = str(device)
    from quantdsl_backtest.smim.compute.linalg import svd
    benchmark(svd, A, k=20)

# test_bench_pid.py
@pytest.mark.parametrize("K", [5, 10, 15, 20])
def test_bench_pid_bootstrap(benchmark, device, K):
    rng = np.random.default_rng(42)
    os.environ["SMIM_DEVICE"] = str(device)
    modal = rng.standard_normal((2000, K))
    target = rng.standard_normal(2000)
    from quantdsl_backtest.smim.compute.batch_pid import batch_pid_synergy
    benchmark(batch_pid_synergy, modal, target, n_bootstrap=200)

# test_bench_pipeline.py
@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_full_pipeline(benchmark, device, N):
    os.environ["SMIM_DEVICE"] = str(device)
    # ... generate synthetic data, run full pipeline ...
    benchmark(run_pipeline_once, ...)
```

Run:
```bash
uv run pytest tests/benchmarks/smim/ -v --benchmark-columns=mean,stddev,rounds
```

### Task 4.3: Speedup report ✅ DONE

**Completed.** Commit `[SMIM GPU-4]`. `scripts/gpu_speedup_report.py` reads
`.benchmark_results.json` and prints component speedups, Woodbury fix numbers,
experiment projection, and acceptance verification. Run with:
  `uv run python scripts/gpu_speedup_report.py`

Create `scripts/gpu_speedup_report.py` that reads benchmark results and generates:

```
SMIM GPU Speedup Report
========================
Hardware: NVIDIA RTX [model], CUDA [version], PyTorch [version]
Date: [date]

Component Speedup Table (including Woodbury fix):
| Component            | CPU Before | CPU After Woodbury | CUDA     | Total Speedup |
|----------------------|-----------|-------------------|----------|---------------|
| Granger edges (N=200) | 143s      | 143s (no change)  | ~14s     | 10.2×         |
| Kim filter EM (N=200) | 35s       | ~0.2s             | (on CPU) | 175×          |
| PID bootstrap         | 17s       | 17s (no change)   | ~2s      | 8.5×          |
| Full pipeline (N=200) | 197s      | ~163s             | ~17s     | 11.6×         |

Projected Experiment Programme:
| Scenario            | Before   | After    | Saving |
|---------------------|----------|----------|--------|
| 680 runs at N=200   | 37 hours | 3.2 hours| 91%    |
| 680 runs at N=500   | 333 hours| ~30 hours| 91%    |

Acceptance Verification:
- CPU: 121/121 passed ✅
- CUDA: 121/121 passed ✅
- Determinism: 5/5 identical runs ✅
- Granger parity: edges match ✅
```

Git add and commit:
[SMIM GPU-4.3] Generate speedup report

---

## Part 5: Execution Order

Priority-ordered (highest impact first):

1. ✅ **GPU-0.1**: Woodbury fix for Kalman/Kim — **DONE** (commit `8db184a`). Kim 235s→0.73s at N=200 (321×).
2. ✅ **GPU-1.1**: PyTorch compute layer + linalg — **DONE** (commits `d93cd74`, `8431c9c`). 54/54 unit tests, 130/130 acceptance tests on CUDA.
3. ✅ **GPU-1.2**: Batch Granger on PyTorch — **DONE** (commit `43d9455`). `batch_granger_test` parity-verified. Wired into pipeline (GPU-2.1).
4. ✅ **GPU-2.1**: Wire compute layer into pipeline — **DONE** (commit `fef7bc5`). 130/130 acceptance tests pass on CPU and CUDA.
5. ✅ **GPU-1.3**: Batch PID bootstrap — **DONE** (commit `2c58fb5`). `batch_pid_synergy` wired into pipeline.
6. ✅ **GPU-1.4**: GPU KNN for TE — **DONE** (commit `22ab5c6`). `knn_query` wired.
7. ✅ **GPU-2.2**: Config + device selection — **DONE** (commit `fef7bc5`). `ComputeConfig` added to `SmimConfig`.
8. ✅ **GPU-3.1–3.3**: Full verification — **DONE** (commit `5522fcd`). 130/130 on CPU + CUDA; determinism test passes.
9. ✅ **GPU-4.1–4.3**: Benchmarks + speedup report — **DONE** (this commit). 56 benchmarks, `scripts/gpu_speedup_report.py`.

**All GPU acceleration tasks complete.**

Measured speedups (T=80, RTX 4070 Ti):
- Granger edges: 11-16x at N=50-200 (N=500: overhead-dominated, CPU preferred)
- Full pipeline: 8-12x at N=100-200
- Kim filter EM: 321x (Woodbury, CPU-only)
- KNN for TE: 10-12x at T>=2000

### Hardware confirmed
NVIDIA GeForce RTX 4070 Ti, CUDA 12.6, PyTorch 2.10.0+cu126, 12 GB VRAM.

## Quality Gates

**Gate GPU-A (Correctness):** `uv run pytest tests/acceptance/smim/` passes
on both CPU and CUDA.
→ **✅ PASSED** — 130/130 on CPU; 130/130 with `SMIM_DEVICE=cuda`.

**Gate GPU-B (Granger Parity):** Batched Granger detects identical edges to
original statsmodels implementation on acceptance test A-GR-1 data.
→ **✅ PASSED** — `test_batch_granger_matches_statsmodels_edges` and
  `test_batch_granger_cuda_matches_cpu_edges` in `test_granger_batch_parity.py`.

**Gate GPU-C (Performance):** Pipeline achieves ≥5× end-to-end speedup at N=200.
→ **✅ PASSED** — Measured 8.3× full-pipeline speedup at N=200 (T=80).
  `uv run pytest tests/benchmarks/smim/ --benchmark-columns=mean,stddev,rounds`
  `uv run python scripts/gpu_speedup_report.py`

**Gate GPU-D (Determinism):** 5 identical CUDA runs produce bitwise identical output
with `torch.use_deterministic_algorithms(True)`.
→ **✅ PASSED** — `test_gpu_determinism` in `tests/acceptance/smim/test_gpu_determinism.py`.

---

## Dependencies

```toml
# In pyproject.toml [project.optional-dependencies]
gpu = [
    "torch>=2.2",  # includes CUDA support if installed via pip with --index-url
]
benchmarks = [
    "pytest-benchmark>=4.0",
]
```

Install PyTorch with CUDA:
```bash
# For CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Or for CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```
