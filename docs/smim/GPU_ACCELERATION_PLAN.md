# SMIM GPU Acceleration Phase — Master Prompt

## Context

Read this entire prompt before writing any code. This is a multi-session project
that adds GPU-accelerated implementations alongside the existing CPU code. The
existing CPU implementations are correct (121 acceptance tests pass). The GPU
implementations must produce identical results while being significantly faster.

Reference documents:
- `docs/smim/ACCEPTANCE_TESTS.md` — the 121 correctness tests that both backends must pass
- `docs/smim/EXPERIMENT_PLAN.md` — the experiment programme (~680 pipeline runs) driving the speed requirement
- `src/quantdsl_backtest/smim/interfaces.py` — existing Protocol definitions

## Goal

The experiment programme requires ~680 full pipeline runs. At current CPU speeds
(estimated 5–15 min per run at N=200), this is 57–170 hours. Ablation experiments
(Phase B) alone involve ~600 runs. We need a 5–20× speedup on the bottleneck
operations to make the programme practical within 1–2 weeks of wall-clock time.

The constraint: **GPU code must produce outputs matching CPU reference outputs
to within documented tolerance.** Mathematical correctness is non-negotiable.
A fast wrong answer is worse than a slow right one.

---

## Part 1: Profiling (Do This First)

Before writing any GPU code, profile the existing CPU pipeline to identify the
actual bottlenecks. Don't guess — measure.

### Task 1.1: Create the profiling harness

Create `src/quantdsl_backtest/smim/profiling.py`:

```python
"""Pipeline profiling harness.

Usage:
    python -m quantdsl_backtest.smim.profiling \
        --config experiments/mvp_energy_us_uk.yaml \
        --n-actors 50 100 200 500 \
        --output profiling_results.json
"""
```

The harness must:
1. Run the full pipeline (graph → spectral → filter → gaps → emergence) on
   synthetic data at multiple actor counts (N = 50, 100, 200, 500)
2. Time each component separately using `time.perf_counter_ns()` (wall-clock, not CPU)
3. Record peak memory via `tracemalloc`
4. Output a structured JSON with per-component timings and percentages:
   ```json
   {
     "N": 200, "T": 80, "K": 15, "M": 3,
     "components": {
       "edge_estimation_granger": {"seconds": 45.2, "pct": 38.1},
       "edge_estimation_narrative": {"seconds": 12.1, "pct": 10.2},
       "sparsification": {"seconds": 0.3, "pct": 0.3},
       "eigendecomposition_polar": {"seconds": 18.7, "pct": 15.8},
       "eigendecomposition_schur": {"seconds": 16.2, "pct": 13.7},
       "mode_selection": {"seconds": 1.1, "pct": 0.9},
       "kalman_filter": {"seconds": 2.3, "pct": 1.9},
       "kim_filter_em": {"seconds": 8.5, "pct": 7.2},
       "pid_synergy": {"seconds": 5.2, "pct": 4.4},
       "pid_bootstrap": {"seconds": 22.1, "pct": 18.6},
       "transfer_entropy": {"seconds": 6.8, "pct": 5.7},
       "persistent_homology": {"seconds": 3.2, "pct": 2.7},
       "benchmark_computation": {"seconds": 0.5, "pct": 0.4},
       "total": {"seconds": 118.6}
     },
     "memory_peak_mb": 412,
     "scaling_vs_n50": 4.2
   }
   ```
5. Print a sorted summary: component, seconds, percentage, cumulative percentage
6. Identify the top-3 bottlenecks that account for >70% of total time

Also profile the falsification suite separately (since it multiplies everything by B=100):
- Time for one null-model instance (rewire + re-run pipeline)
- Projected time for B=100

Use `make_modal_system` and `make_switching_system` from the acceptance test
fixtures to generate realistic synthetic data at each N.

Run: `uv run python -m quantdsl_backtest.smim.profiling --n-actors 50 200 500`

Save the output — it determines which components get GPU implementations.

Git add and commit:
[SMIM GPU-1.1] Create pipeline profiling harness

### Task 1.2: Profile and identify acceleration targets

Run the profiler at N=200 (the MVP experiment size) and N=500 (the scaling target).
From the results, identify the acceleration targets using this decision rule:

```
IF component takes >10% of total time AND scales as O(N²) or worse:
    → GPU acceleration candidate (high priority)
IF component takes >10% of total time AND is embarrassingly parallel:
    → GPU batch parallelisation candidate (medium priority)  
IF component takes <5% of total time:
    → Leave on CPU (not worth the complexity)
```

Expected acceleration targets (verify with profiling data):

| Component | Expected Bottleneck? | GPU Strategy |
|-----------|---------------------|--------------|
| Granger edge estimation | Yes — O(N² × T) pairwise VARs | Batch parallel: run N² independent VAR fits on GPU |
| Eigendecomposition | Yes — O(N³) dense, O(N²K) sparse | cuSOLVER via CuPy: `cupyx.scipy.linalg.schur`, SVD |
| PID bootstrap | Yes — 200× base PID cost | Batch parallel: 200 independent covariance computations |
| Falsification (B=100) | Yes — 100× full pipeline | Batch parallel across null instances |
| Kim filter | Probably not — O(M²K²T), K and M small | Leave on CPU unless profiling shows otherwise |
| Transfer entropy (KSG) | Maybe — O(T² per pair) neighbour search | cuML KNN or FAISS for nearest-neighbour queries |
| Persistent homology | Probably not — ripser is already C++ optimised | Leave on CPU |
| Kalman filter | No — O(K³T), K small | Leave on CPU |
| Mode selection | No — lightweight | Leave on CPU |
| Benchmark computation | No — matrix multiply | Leave on CPU |

Document the profiling results and acceleration targets in
`docs/smim/GPU_ACCELERATION_PLAN.md`.

Git add and commit:
[SMIM GPU-1.2] Profile pipeline and document GPU acceleration targets

---

## Part 2: Compute Backend Abstraction Layer

This is the architectural core. The abstraction must be clean enough that:
- All existing code continues to work unchanged on CPU
- GPU implementations slot in without modifying calling code
- Backend selection is automatic (GPU if available) with manual override
- Testing can force either backend

### Task 2.1: Backend abstraction

Create `src/quantdsl_backtest/smim/compute/__init__.py` and
`src/quantdsl_backtest/smim/compute/backend.py`:

```python
"""Compute backend abstraction for CPU/GPU dispatch.

Usage:
    from quantdsl_backtest.smim.compute import get_backend, ComputeBackend

    backend = get_backend()  # auto-detects GPU
    backend = get_backend(force="cpu")  # explicit CPU
    backend = get_backend(force="gpu")  # explicit GPU (raises if unavailable)

    # All numerical operations go through the backend
    result = backend.eigendecompose(operator, k=10, method="schur")
    edges = backend.granger_edges(signals, max_lag=4, p_threshold=0.05)
    synergy = backend.pid_synergy(alpha_j, alpha_k, target, n_bootstrap=200)
"""

from __future__ import annotations
import enum
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sparse


class BackendType(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


@runtime_checkable
class ComputeBackend(Protocol):
    """Interface that both CPU and GPU backends implement.

    Every method has identical signature and semantics. The ONLY
    difference is execution speed and hardware utilisation.
    """

    @property
    def backend_type(self) -> BackendType: ...

    @property
    def device_name(self) -> str: ...

    # ── Eigendecomposition ─────────────────────────────────
    def schur_decompose(
        self, A: NDArray, k: int | None = None
    ) -> tuple[NDArray, NDArray]:
        """Returns (Q, T) from Schur decomposition A = QTQ^H.
        If k is specified, return only top-k."""
        ...

    def polar_decompose(
        self, A: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Returns (U, P) from polar decomposition A = UP."""
        ...

    def hermitian_dilation_decompose(
        self, A: NDArray, k: int
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Returns (eigenvalues, U_L, U_R) from Hermitian dilation."""
        ...

    def svd(
        self, A: NDArray, k: int | None = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Returns (U, S, Vt). Truncated if k specified."""
        ...

    def eigh(
        self, H: NDArray, k: int | None = None
    ) -> tuple[NDArray, NDArray]:
        """Symmetric eigendecomposition. Returns (eigenvalues, eigenvectors)."""
        ...

    # ── Edge Estimation ────────────────────────────────────
    def batch_granger_test(
        self,
        signals: NDArray,      # (N, T)
        max_lag: int,
        p_threshold: float,
    ) -> sparse.csr_matrix:
        """Compute all N² pairwise Granger tests, return adjacency."""
        ...

    # ── PID / Information Theory ───────────────────────────
    def batch_pid_synergy(
        self,
        modal_states: NDArray,  # (T, K)
        target: NDArray,        # (T,)
        n_bootstrap: int,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute K*(K-1)/2 pairwise PID synergies with bootstrap CIs.
        Returns (synergy_matrix, ci_lower, ci_upper)."""
        ...

    def batch_covariance(
        self, X: NDArray  # (T, N)
    ) -> NDArray:
        """Batch covariance matrix computation. Shape (N, N)."""
        ...

    # ── Nearest Neighbour (for KSG TE) ────────────────────
    def knn_query(
        self,
        points: NDArray,   # (T, d)
        k: int,
    ) -> tuple[NDArray, NDArray]:
        """k-nearest-neighbour distances and indices.
        Returns (distances (T, k), indices (T, k))."""
        ...

    # ── Batch Utilities ────────────────────────────────────
    def to_device(self, arr: NDArray) -> NDArray:
        """Transfer array to compute device (no-op for CPU)."""
        ...

    def to_host(self, arr: NDArray) -> NDArray:
        """Transfer array back to CPU numpy (no-op for CPU)."""
        ...


def get_backend(force: str | None = None) -> ComputeBackend:
    """Get the active compute backend.

    Args:
        force: "cpu", "gpu", or None (auto-detect).
            If "gpu" and no GPU available, raises RuntimeError.
            If None, uses GPU if available, else CPU.

    Environment variable override: SMIM_BACKEND=cpu|gpu
    """
    ...
```

Key design principles:
- `ComputeBackend` is a Protocol — both backends implement it structurally
- Methods operate on numpy arrays (input/output always numpy on host)
- GPU backend handles device transfer internally (to_device/to_host are for
  advanced use where the caller wants to keep data on GPU across calls)
- Backend selection is: explicit arg > env var > auto-detect
- Auto-detect: try `import cupy; cupy.cuda.Device(0)` — if succeeds, use GPU

### Task 2.2: CPU backend (wraps existing implementations)

Create `src/quantdsl_backtest/smim/compute/cpu_backend.py`:

This wraps the EXISTING implementations from `smim/spectral/`, `smim/graph/`,
`smim/emergence/` etc. into the ComputeBackend interface. It should be a thin
delegation layer, NOT a reimplementation.

```python
class CpuBackend:
    """CPU compute backend — delegates to existing implementations."""

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CPU

    @property
    def device_name(self) -> str:
        return "cpu"

    def schur_decompose(self, A, k=None):
        # Delegate to scipy.linalg.schur (same as existing SchurDecomposer)
        from scipy.linalg import schur
        T, Q = schur(A, output='complex')
        if k is not None:
            # truncate to top-k by eigenvalue magnitude
            ...
        return Q, T

    def polar_decompose(self, A):
        from scipy.linalg import polar
        return polar(A)

    # ... etc for all methods
```

Tests: verify CpuBackend passes all 121 acceptance tests when used as the backend.
This is a rewiring test, not a new correctness test — the underlying implementations
are already validated.

### Task 2.3: GPU backend

Create `src/quantdsl_backtest/smim/compute/gpu_backend.py`:

```python
class GpuBackend:
    """GPU compute backend using CuPy and CUDA.

    Requirements:
        - NVIDIA GPU with CUDA support
        - CuPy installed: pip install cupy-cuda12x (match your CUDA version)
        - cuSOLVER for eigendecomposition
        - (optional) cuML for KNN, FAISS for nearest-neighbour
    """

    def __init__(self):
        import cupy as cp
        self._cp = cp
        self._device = cp.cuda.Device(0)
        # Verify CUDA is functional
        _ = cp.array([1.0])

    @property
    def backend_type(self) -> BackendType:
        return BackendType.GPU

    @property
    def device_name(self) -> str:
        return f"gpu:{self._device.id} ({self._device.attributes['DeviceName']})"

    def schur_decompose(self, A, k=None):
        cp = self._cp
        A_gpu = cp.asarray(A)
        # CuPy wraps cuSOLVER: cupyx.scipy.linalg.schur
        from cupyx.scipy.linalg import schur as cu_schur
        T_gpu, Q_gpu = cu_schur(A_gpu, output='complex')
        if k is not None:
            ...
        return cp.asnumpy(Q_gpu), cp.asnumpy(T_gpu)

    def polar_decompose(self, A):
        cp = self._cp
        A_gpu = cp.asarray(A)
        # Polar via SVD: A = U S V^H, then orthogonal = U V^H, PSD = V S V^H
        U, S, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
        orth = U @ Vh
        psd = (Vh.conj().T * S) @ Vh
        return cp.asnumpy(orth), cp.asnumpy(psd)

    def svd(self, A, k=None):
        cp = self._cp
        A_gpu = cp.asarray(A)
        U, S, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
        if k is not None:
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        return cp.asnumpy(U), cp.asnumpy(S), cp.asnumpy(Vh)

    def eigh(self, H, k=None):
        cp = self._cp
        H_gpu = cp.asarray(H)
        if k is not None and k < H.shape[0] // 2:
            # Use iterative solver for partial eigendecomposition
            from cupyx.scipy.sparse.linalg import eigsh
            H_sparse = cupyx.scipy.sparse.csr_matrix(H_gpu)
            vals, vecs = eigsh(H_sparse, k=k, which='LM')
        else:
            vals, vecs = cp.linalg.eigh(H_gpu)
        return cp.asnumpy(vals), cp.asnumpy(vecs)

    def batch_granger_test(self, signals, max_lag, p_threshold):
        # Strategy: formulate all N² VAR regressions as a batched least-squares
        # problem and solve on GPU using cuBLAS batched GEMM
        cp = self._cp
        N, T = signals.shape
        # ... construct batched regression matrices ...
        # ... solve via batched lstsq or QR ...
        # ... compute F-statistics from residuals ...
        # ... threshold and return sparse adjacency ...
        ...

    def batch_pid_synergy(self, modal_states, target, n_bootstrap):
        cp = self._cp
        T, K = modal_states.shape
        n_pairs = K * (K - 1) // 2
        # For Gaussian MMI: synergy depends only on covariance matrices
        # Compute all pairwise covariances on GPU
        # Bootstrap: generate n_bootstrap block-bootstrap index arrays on GPU
        # Compute covariance for each bootstrap sample in parallel
        # ... batched covariance computation ...
        ...

    def batch_covariance(self, X):
        cp = self._cp
        X_gpu = cp.asarray(X)
        mean = X_gpu.mean(axis=0, keepdims=True)
        centered = X_gpu - mean
        cov = (centered.T @ centered) / (X_gpu.shape[0] - 1)
        return cp.asnumpy(cov)

    def knn_query(self, points, k):
        cp = self._cp
        # Option 1: brute-force pairwise distances on GPU (fast for T < 50000)
        points_gpu = cp.asarray(points)
        # Pairwise L2 distances via the expansion trick:
        # ||a-b||² = ||a||² + ||b||² - 2 a·b
        sq_norms = cp.sum(points_gpu ** 2, axis=1, keepdims=True)
        dists = sq_norms + sq_norms.T - 2 * points_gpu @ points_gpu.T
        cp.fill_diagonal(dists, cp.inf)
        # Top-k smallest
        idx = cp.argpartition(dists, k, axis=1)[:, :k]
        # Gather distances
        d = cp.take_along_axis(dists, idx, axis=1)
        return cp.asnumpy(cp.sqrt(cp.maximum(d, 0))), cp.asnumpy(idx)

    def to_device(self, arr):
        return self._cp.asarray(arr)

    def to_host(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
```

IMPORTANT implementation notes:
- All methods accept numpy arrays and return numpy arrays (device transfer
  is internal). This is critical — calling code never touches CuPy arrays.
- For operations CuPy doesn't support directly (e.g., Schur decomposition
  may not be in all CuPy versions), fall back to CPU with a warning:
  ```python
  import warnings
  warnings.warn("Schur not available in CuPy, falling back to CPU", RuntimeWarning)
  return CpuBackend().schur_decompose(A, k)
  ```
- Memory management: use `cp.get_default_memory_pool().free_all_blocks()`
  after large operations to prevent GPU OOM.

### Task 2.4: Wire backend into existing pipeline

Modify the existing pipeline components to use the backend abstraction:

1. In `smim/spectral/schur.py`, `polar.py`, `hermitian.py`:
   Replace direct scipy calls with `backend.schur_decompose()` etc.

2. In `smim/graph/edges/granger.py`:
   Replace the pairwise VAR loop with `backend.batch_granger_test()`

3. In `smim/emergence/pid.py`:
   Replace the bootstrap loop with `backend.batch_pid_synergy()`

4. In `smim/emergence/transfer_entropy.py`:
   Replace scipy KDTree with `backend.knn_query()`

The pattern is:
```python
# Before:
from scipy.linalg import schur
T, Q = schur(A)

# After:
from quantdsl_backtest.smim.compute import get_backend
backend = get_backend()
Q, T = backend.schur_decompose(A)
```

DO NOT change any mathematical logic. Only change the compute dispatch.

### Task 2.5: Fallback and environment configuration

Create `src/quantdsl_backtest/smim/compute/config.py`:

```python
"""Compute backend configuration.

Environment variables:
    SMIM_BACKEND=cpu|gpu|auto   (default: auto)
    SMIM_GPU_DEVICE=0           (default: 0)
    SMIM_GPU_MEMORY_LIMIT=8GB   (default: no limit)
"""
```

Fallback chain:
1. If SMIM_BACKEND=gpu and GPU unavailable → raise RuntimeError (explicit failure)
2. If SMIM_BACKEND=auto and GPU unavailable → use CPU silently
3. If SMIM_BACKEND=cpu → always CPU regardless of GPU availability
4. Log the selected backend at pipeline startup: "Using compute backend: gpu:0 (NVIDIA RTX ...)"

---

## Part 3: Correctness Parity Test Suite

This is the quality gate. GPU outputs must match CPU outputs.

### Task 3.1: Create parity test infrastructure

Create `tests/acceptance/smim/test_gpu_parity.py`:

```python
"""GPU ↔ CPU correctness parity tests.

For every operation in the ComputeBackend protocol, verify that
GPU output matches CPU output to within documented tolerance.

These tests require a GPU. Skip gracefully if unavailable:
    @pytest.mark.gpu
    @pytest.mark.skipif(not gpu_available(), reason="No CUDA GPU")
"""
import pytest
import numpy as np
from quantdsl_backtest.smim.compute import get_backend, BackendType

def gpu_available() -> bool:
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False

@pytest.fixture
def cpu():
    return get_backend(force="cpu")

@pytest.fixture
def gpu():
    return get_backend(force="gpu")

PARITY_ATOL = 1e-8   # absolute tolerance for CPU-GPU agreement
PARITY_RTOL = 1e-6   # relative tolerance for CPU-GPU agreement
```

### Task 3.2: Eigendecomposition parity tests

```python
@pytest.mark.gpu
class TestEigendecompositionParity:
    """GPU eigendecomposition must match CPU to PARITY tolerance."""

    @pytest.mark.parametrize("N", [10, 50, 100, 200, 500])
    def test_schur_parity(self, cpu, gpu, N):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        Q_cpu, T_cpu = cpu.schur_decompose(A)
        Q_gpu, T_gpu = gpu.schur_decompose(A)
        # Eigenvalues must match (Q and T may differ by unitary transform)
        eig_cpu = sorted(np.diag(T_cpu), key=lambda x: (abs(x), np.angle(x)))
        eig_gpu = sorted(np.diag(T_gpu), key=lambda x: (abs(x), np.angle(x)))
        np.testing.assert_allclose(eig_cpu, eig_gpu, atol=PARITY_ATOL, rtol=PARITY_RTOL)

    @pytest.mark.parametrize("N", [10, 50, 100, 200, 500])
    def test_polar_parity(self, cpu, gpu, N):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        U_cpu, P_cpu = cpu.polar_decompose(A)
        U_gpu, P_gpu = gpu.polar_decompose(A)
        np.testing.assert_allclose(U_cpu, U_gpu, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        np.testing.assert_allclose(P_cpu, P_gpu, atol=PARITY_ATOL, rtol=PARITY_RTOL)

    @pytest.mark.parametrize("N", [10, 50, 100, 200])
    def test_hermitian_dilation_parity(self, cpu, gpu, N):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        vals_cpu, UL_cpu, UR_cpu = cpu.hermitian_dilation_decompose(A, k=min(N, 20))
        vals_gpu, UL_gpu, UR_gpu = gpu.hermitian_dilation_decompose(A, k=min(N, 20))
        np.testing.assert_allclose(sorted(vals_cpu), sorted(vals_gpu),
                                   atol=PARITY_ATOL, rtol=PARITY_RTOL)

    @pytest.mark.parametrize("N", [10, 50, 100, 200, 500])
    def test_svd_parity(self, cpu, gpu, N):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        U_cpu, S_cpu, Vt_cpu = cpu.svd(A, k=min(N, 20))
        U_gpu, S_gpu, Vt_gpu = gpu.svd(A, k=min(N, 20))
        # Singular values must match. Vectors may differ by sign.
        np.testing.assert_allclose(S_cpu, S_gpu, atol=PARITY_ATOL, rtol=PARITY_RTOL)

    @pytest.mark.parametrize("N", [10, 50, 100, 200, 500])
    def test_eigh_parity(self, cpu, gpu, N):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N, N))
        H = A + A.T  # symmetric
        vals_cpu, vecs_cpu = cpu.eigh(H)
        vals_gpu, vecs_gpu = gpu.eigh(H)
        np.testing.assert_allclose(vals_cpu, vals_gpu, atol=PARITY_ATOL, rtol=PARITY_RTOL)
```

### Task 3.3: Edge estimation parity tests

```python
@pytest.mark.gpu
class TestEdgeParity:

    @pytest.mark.parametrize("N,T", [(10, 200), (50, 200), (100, 200)])
    def test_granger_parity(self, cpu, gpu, N, T):
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((N, T))
        adj_cpu = cpu.batch_granger_test(signals, max_lag=4, p_threshold=0.05)
        adj_gpu = gpu.batch_granger_test(signals, max_lag=4, p_threshold=0.05)
        # Adjacency structure must match (same edges detected)
        np.testing.assert_array_equal(adj_cpu.toarray() > 0, adj_gpu.toarray() > 0)
        # Edge weights (F-statistics) match to loose tolerance
        # (different linear algebra paths may give slightly different F-stats)
        mask = adj_cpu.toarray() > 0
        if mask.any():
            np.testing.assert_allclose(
                adj_cpu.toarray()[mask], adj_gpu.toarray()[mask],
                atol=1e-4, rtol=1e-3
            )
```

### Task 3.4: PID and information-theoretic parity tests

```python
@pytest.mark.gpu
class TestPIDParity:

    def test_synergy_matrix_parity(self, cpu, gpu):
        rng = np.random.default_rng(42)
        T, K = 2000, 5
        modal_states = rng.standard_normal((T, K))
        target = rng.standard_normal(T)
        S_cpu, lo_cpu, hi_cpu = cpu.batch_pid_synergy(modal_states, target, n_bootstrap=50)
        S_gpu, lo_gpu, hi_gpu = gpu.batch_pid_synergy(modal_states, target, n_bootstrap=50)
        # Synergy point estimates must match closely
        np.testing.assert_allclose(S_cpu, S_gpu, atol=1e-4, rtol=1e-3)
        # Bootstrap CIs: same distributional properties (not exact — different RNG paths)
        # Verify: CI widths agree within 50%
        width_cpu = hi_cpu - lo_cpu
        width_gpu = hi_gpu - lo_gpu
        np.testing.assert_allclose(width_cpu, width_gpu, atol=0, rtol=0.5)

    def test_covariance_parity(self, cpu, gpu):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 50))
        cov_cpu = cpu.batch_covariance(X)
        cov_gpu = gpu.batch_covariance(X)
        np.testing.assert_allclose(cov_cpu, cov_gpu, atol=1e-10, rtol=1e-8)

    @pytest.mark.parametrize("T", [1000, 5000])
    def test_knn_parity(self, cpu, gpu, T):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((T, 3))
        dist_cpu, idx_cpu = cpu.knn_query(points, k=5)
        dist_gpu, idx_gpu = gpu.knn_query(points, k=5)
        # Same neighbours found
        for i in range(min(T, 100)):  # check 100 random points
            assert set(idx_cpu[i]) == set(idx_gpu[i]), f"KNN mismatch at point {i}"
        # Distances match
        np.testing.assert_allclose(
            np.sort(dist_cpu[:100], axis=1),
            np.sort(dist_gpu[:100], axis=1),
            atol=1e-8, rtol=1e-6
        )
```

### Task 3.5: Full acceptance suite on GPU backend

The ultimate parity test: run ALL 121 existing acceptance tests using the GPU
backend and verify they pass.

```python
# In conftest.py for acceptance tests, add:
@pytest.fixture(autouse=True)
def set_backend_from_env():
    """Allow running acceptance suite on GPU via env var."""
    import os
    backend = os.environ.get("SMIM_TEST_BACKEND")
    if backend:
        os.environ["SMIM_BACKEND"] = backend
    yield
    if backend:
        os.environ.pop("SMIM_BACKEND", None)
```

Run acceptance suite on GPU:
```bash
SMIM_TEST_BACKEND=gpu uv run pytest tests/acceptance/smim/ -v --tb=short
```

All 121 tests must pass on GPU backend. Any failure means the GPU implementation
has a correctness bug — fix the GPU code, not the test.

---

## Part 4: Performance Benchmark Suite

### Task 4.1: Create performance benchmark infrastructure

Create `tests/benchmarks/smim/conftest.py` and individual benchmark files.

Use `pytest-benchmark` (add to dev dependencies) for rigorous timing with
statistical confidence (multiple rounds, warmup, outlier detection).

```python
"""Performance benchmarks for CPU vs GPU backends.

Run: uv run pytest tests/benchmarks/smim/ -v --benchmark-columns=mean,stddev,rounds
Compare: uv run pytest tests/benchmarks/smim/ -v --benchmark-compare
"""
import pytest
import numpy as np
from quantdsl_backtest.smim.compute import get_backend


@pytest.fixture(params=["cpu", "gpu"], ids=["CPU", "GPU"])
def backend(request):
    try:
        return get_backend(force=request.param)
    except RuntimeError:
        pytest.skip(f"{request.param} backend not available")
```

### Task 4.2: Component-level benchmarks

Create `tests/benchmarks/smim/test_bench_eigendecomp.py`:

```python
@pytest.mark.parametrize("N", [50, 100, 200, 500, 1000])
def test_bench_schur(benchmark, backend, N):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    benchmark(backend.schur_decompose, A, k=20)

@pytest.mark.parametrize("N", [50, 100, 200, 500, 1000])
def test_bench_svd(benchmark, backend, N):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    benchmark(backend.svd, A, k=20)

@pytest.mark.parametrize("N", [50, 100, 200, 500, 1000])
def test_bench_eigh(benchmark, backend, N):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N))
    H = A + A.T
    benchmark(backend.eigh, H, k=20)
```

Create `tests/benchmarks/smim/test_bench_edges.py`:

```python
@pytest.mark.parametrize("N,T", [(20, 200), (50, 200), (100, 200), (200, 200)])
def test_bench_granger(benchmark, backend, N, T):
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((N, T))
    benchmark(backend.batch_granger_test, signals, max_lag=4, p_threshold=0.05)
```

Create `tests/benchmarks/smim/test_bench_pid.py`:

```python
@pytest.mark.parametrize("K,T", [(5, 1000), (10, 1000), (15, 2000), (20, 2000)])
def test_bench_pid_synergy(benchmark, backend, K, T):
    rng = np.random.default_rng(42)
    modal_states = rng.standard_normal((T, K))
    target = rng.standard_normal(T)
    benchmark(backend.batch_pid_synergy, modal_states, target, n_bootstrap=50)

@pytest.mark.parametrize("T", [1000, 5000, 10000])
def test_bench_knn(benchmark, backend, T):
    rng = np.random.default_rng(42)
    points = rng.standard_normal((T, 3))
    benchmark(backend.knn_query, points, k=10)
```

### Task 4.3: Pipeline-level benchmark

Create `tests/benchmarks/smim/test_bench_pipeline.py`:

```python
@pytest.mark.parametrize("N", [50, 100, 200, 500])
def test_bench_full_pipeline(benchmark, backend, N):
    """Benchmark the complete pipeline at different actor counts."""
    # Generate synthetic data matching the experiment profile
    from tests.acceptance.smim.conftest import make_modal_system
    obs, _, _ = make_modal_system(N=N, K=5, T=80, seed=42)
    # ... run full pipeline ...
    benchmark(run_pipeline_once, obs, backend=backend)
```

### Task 4.4: Speedup report generator

Create `scripts/gpu_speedup_report.py`:

```python
"""Generate GPU speedup report from benchmark results.

Usage: uv run python scripts/gpu_speedup_report.py

Output: docs/smim/reports/gpu_speedup.md
"""
```

The report should show:

```
SMIM GPU Speedup Report
========================
Hardware: NVIDIA RTX [model], CUDA [version], CuPy [version]
Date: [date]

Component Speedup Table:
| Component            | N=100 CPU | N=100 GPU | Speedup | N=500 CPU | N=500 GPU | Speedup |
|----------------------|-----------|-----------|---------|-----------|-----------|---------|
| Schur decomposition  | 0.12s     | 0.03s     | 4.0×    | 8.2s      | 0.4s      | 20.5×   |
| SVD (truncated k=20) | 0.08s     | 0.02s     | 4.0×    | 5.1s      | 0.2s      | 25.5×   |
| Polar decomposition  | 0.15s     | 0.04s     | 3.8×    | 9.8s      | 0.5s      | 19.6×   |
| Granger edges (N²)   | 2.1s      | 0.3s      | 7.0×    | 52s       | 3.1s      | 16.8×   |
| PID bootstrap (×200) | 1.5s      | 0.2s      | 7.5×    | 1.5s      | 0.2s      | 7.5×    |
| KNN (T=5000)         | 3.2s      | 0.1s      | 32.0×   | 3.2s      | 0.1s      | 32.0×   |

Projected Experiment Programme Time:
| Scenario            | CPU estimate | GPU estimate | Saving |
|---------------------|-------------|-------------|--------|
| Phase A (anchor)    | 12 hours    | 1.5 hours   | 87%    |
| Phase B (ablation)  | 85 hours    | 8 hours     | 91%    |
| Phase C (transfer)  | 15 hours    | 2 hours     | 87%    |
| Phase D (economic)  | 8 hours     | 3 hours     | 63%    |
| TOTAL               | 120 hours   | 14.5 hours  | 88%    |

Correctness Verification:
- Parity tests: XX/XX passed ✅
- Acceptance suite on GPU: 121/121 passed ✅
```

---

## Part 5: Execution Order

1. **GPU-1.1**: Profiling harness (1 session)
2. **GPU-1.2**: Profile and document targets (run profiler, write doc)
3. **GPU-2.1**: Backend abstraction Protocol (1 session)
4. **GPU-2.2**: CPU backend wrapper (1 session)
5. **GPU-2.3**: GPU backend implementation (2–3 sessions — eigendecomp first, then edges, then PID/KNN)
6. **GPU-2.4**: Wire backend into pipeline (1 session)
7. **GPU-2.5**: Fallback and config (1 session)
8. **GPU-3.1–3.4**: Parity tests (1–2 sessions)
9. **GPU-3.5**: Full acceptance suite on GPU backend (run + fix)
10. **GPU-4.1–4.3**: Performance benchmarks (1 session)
11. **GPU-4.4**: Speedup report (1 session)

Total: ~10–12 Claude Code sessions.

## Quality Gates

**Gate GPU-A (Correctness):** All 121 acceptance tests pass on GPU backend.
Not 120 — all 121. Any failure blocks the experiment programme on GPU.

**Gate GPU-B (Parity):** All parity tests pass. CPU and GPU produce the same
results to documented tolerance.

**Gate GPU-C (Performance):** GPU achieves ≥5× speedup on the top-3 bottleneck
components at N=200. If speedup is <2× on a component, that component is not
worth the GPU complexity — revert it to CPU-only.

**Gate GPU-D (Stability):** Run the full pipeline 10 times on GPU with same
input and seed. All 10 runs produce bitwise identical output.
(GPU floating-point non-determinism is a real risk — cuBLAS can reorder
operations across runs. If this fails, set CUBLAS_WORKSPACE_CONFIG and
CUDA_LAUNCH_BLOCKING appropriately.)

---

## Dependencies to Add

```toml
# In pyproject.toml [project.optional-dependencies]
gpu = [
    "cupy-cuda12x>=13.0",   # Match your CUDA version
]
benchmarks = [
    "pytest-benchmark>=4.0",
]
```

Install:
```bash
uv sync --extra gpu --extra benchmarks --extra dev
```

Verify CuPy:
```bash
uv run python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```
