"""
GPU determinism acceptance test.

Verifies that running the SMIM pipeline 5× on CUDA with the same seed
produces bitwise-identical outputs every time.

Uses ``torch.use_deterministic_algorithms(True)`` to catch any CUDA
operations that would produce non-deterministic results.  If a future
change introduces a non-deterministic GPU kernel, this test will fail
with a RuntimeError from PyTorch before the assertion is even reached.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("pipeline_sanity"),
]


def _run_pipeline_once(seed: int, device: str) -> SimpleNamespace:
    """Run a GPU-exercising pipeline step and return gaps + regimes.

    Exercises the GPU-accelerated spectral ops (``polar_decompose``, ``svd``)
    so that any non-determinism in those kernels would propagate into the
    returned arrays.  The Kalman and Kim filters are numpy-based and sit
    downstream — their outputs inherit determinism from the modal frame.

    Args:
        seed:   NumPy seed for data generation (same each call for the test).
        device: ``"cpu"`` or ``"cuda"`` — passed via ``SMIM_DEVICE`` env var.

    Returns:
        SimpleNamespace with ``gaps`` (N, T) float64 and ``regimes`` (T,) int64.
    """
    from quantdsl_backtest.smim.compute.linalg import polar_decompose, svd
    from quantdsl_backtest.smim.compute.torch_ops import get_device
    from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
    from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
    from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
    from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
    from tests.acceptance.smim.conftest import make_actor_registry

    os.environ["SMIM_DEVICE"] = device
    get_device.cache_clear()

    N, K, T = 15, 4, 120
    rng = np.random.default_rng(seed)

    # ── GPU-accelerated spectral ops ─────────────────────────────────────────
    # polar_decompose and svd both run on GPU when SMIM_DEVICE=cuda.
    # The resulting `basis` is determined by GPU floating-point arithmetic.
    A = rng.standard_normal((N, K + 4)).astype(np.float64)
    U_svd, _, _ = svd(A, k=K)                   # GPU SVD (N, K) basis
    orth, _ = polar_decompose(A[:, :K])          # GPU polar via SVD (N, K)

    # Use the polar result as the modal basis so GPU outputs feed downstream
    modal_frame = ModalFrame(
        basis=orth,
        eigenvalues=np.linspace(0.9, 0.5, K),
        method=DecompositionMethod.SCHUR,
    )

    # ── Synthetic observations seeded by GPU-derived basis ───────────────────
    F = 0.8 * np.eye(K)
    Q = 0.05 * np.eye(K)
    R = 0.1 * np.eye(N)

    alphas = np.zeros((T, K))
    alphas[0] = rng.standard_normal(K)
    for t in range(1, T):
        alphas[t] = F @ alphas[t - 1] + 0.22 * rng.standard_normal(K)

    noise_R_chol = np.linalg.cholesky(R)
    # observations depend on orth (GPU-derived), so non-determinism propagates
    obs = alphas @ orth.T + (noise_R_chol @ rng.standard_normal((N, T))).T

    # ── Kalman filter → gaps ─────────────────────────────────────────────────
    kf = KalmanFilter(F=F, Q=Q, R=R)
    state = kf.filter(obs, modal_frame)
    actors = make_actor_registry(N)
    gap_result = PredictiveBenchmark().compute(obs, state, modal_frame, actors)

    # ── Kim filter → regime labels ───────────────────────────────────────────
    P_trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    F2 = 0.3 * np.eye(K)
    kim_state = KimFilter(n_regimes=2).filter(
        obs,
        modal_frame,
        F_list=[F, F2],
        Q_list=[Q, Q],
        R=R,
        P_trans=P_trans,
    )
    regimes = np.argmax(kim_state.regime_probs, axis=1)

    # Reset device selection to avoid polluting later tests
    os.environ.pop("SMIM_DEVICE", None)
    get_device.cache_clear()

    return SimpleNamespace(
        gaps=gap_result.gaps.copy(),
        regimes=regimes.copy(),
        svd_basis=U_svd.copy(),   # direct GPU output — also checked
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_gpu_determinism():
    """Run pipeline 5× on CUDA with same seed — all outputs must be bitwise identical.

    ``torch.use_deterministic_algorithms(True)`` is enabled for the duration
    of the test.  Any non-deterministic CUDA kernel raises RuntimeError before
    the ``assert_array_equal`` check.

    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` makes cuBLAS use a fixed workspace
    size, which is required for reproducible matrix multiplications on CUDA.
    """
    # cuBLAS determinism — must be set before the first CUDA matmul in this
    # process.  If CUDA was already initialised, this env var has no effect
    # (the workspace was already allocated), but setting it here documents the
    # requirement and is harmless.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        results = [_run_pipeline_once(seed=42, device="cuda") for _ in range(5)]
    finally:
        torch.use_deterministic_algorithms(prev_deterministic)

    for i in range(1, 5):
        np.testing.assert_array_equal(
            results[0].gaps,
            results[i].gaps,
            err_msg=f"CUDA run {i}: gaps differ from run 0 — non-determinism detected",
        )
        np.testing.assert_array_equal(
            results[0].regimes,
            results[i].regimes,
            err_msg=f"CUDA run {i}: regimes differ from run 0 — non-determinism detected",
        )
        np.testing.assert_array_equal(
            results[0].svd_basis,
            results[i].svd_basis,
            err_msg=f"CUDA run {i}: SVD basis differs from run 0 — GPU SVD is non-deterministic",
        )
