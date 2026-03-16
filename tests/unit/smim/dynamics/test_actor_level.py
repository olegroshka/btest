"""Tests for actor-level state-space model (M6.2-T1)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.dynamics.actor_level import (
    ActorLevelConfig,
    ActorLevelResult,
    ActorLevelSSM,
)
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_modal_frame(N: int, K: int, rng: np.random.Generator) -> ModalFrame:
    basis, _ = np.linalg.qr(rng.standard_normal((N, K)))
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActorLevelSSM:

    def test_refuses_large_actor_set(self):
        """N=60 > max_actors=50 → ValueError."""
        rng = np.random.default_rng(42)
        N, T = 60, 30
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, 3, rng)

        ssm = ActorLevelSSM(ActorLevelConfig(max_actors=50))
        with pytest.raises(ValueError, match="Actor count"):
            ssm.fit(obs, mf)

    def test_fits_small_system(self):
        """N'=5, T=30 → ActorLevelResult with fitted=True."""
        rng = np.random.default_rng(7)
        N, T = 5, 30
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, 3, rng)

        ssm = ActorLevelSSM(ActorLevelConfig(state_dim=2, n_iter_em=3))
        result = ssm.fit(obs, mf)

        assert isinstance(result, ActorLevelResult)
        assert result.fitted is True
        assert result.n_actors == N
        assert result.n_timesteps == T

    def test_residual_variance_nonnegative(self):
        """Residual variance must be >= 0."""
        rng = np.random.default_rng(13)
        N, T = 4, 20
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, 2, rng)

        ssm = ActorLevelSSM(ActorLevelConfig(state_dim=2, n_iter_em=2))
        result = ssm.fit(obs, mf)

        assert result.residual_variance >= 0.0

    def test_incremental_r2_computed(self):
        """incremental_r2 = R²(actor-level) - R²(modal-only)."""
        rng = np.random.default_rng(21)
        N, T = 5, 25
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, 3, rng)

        ssm = ActorLevelSSM(ActorLevelConfig(state_dim=2, n_iter_em=2))
        result = ssm.fit(obs, mf)

        # Check that incremental_r2 equals the stored values
        r2_actor = result.metadata["r2_actor"]
        r2_modal = result.metadata["r2_modal"]
        expected = r2_actor - r2_modal
        assert abs(result.incremental_r2 - expected) < 1e-10

    def test_system_matrices_shape(self):
        """_build_system_matrices returns correct shapes."""
        rng = np.random.default_rng(55)
        N, d = 5, 3
        adjacency = np.eye(N)
        ssm = ActorLevelSSM(ActorLevelConfig(state_dim=d))
        F_full, C_full = ssm._build_system_matrices(N, d, adjacency)

        assert F_full.shape == (N * d, N * d)
        assert C_full.shape == (N, N * d)

    def test_fit_with_adjacency(self):
        """Fitting with a custom adjacency matrix should work."""
        rng = np.random.default_rng(33)
        N, T = 4, 20
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, 2, rng)
        adj = rng.uniform(0, 0.1, (N, N))

        ssm = ActorLevelSSM(ActorLevelConfig(state_dim=2, n_iter_em=2))
        result = ssm.fit(obs, mf, adjacency=adj)

        assert result.fitted is True

    def test_default_config(self):
        """ActorLevelConfig defaults are as specified."""
        cfg = ActorLevelConfig()
        assert cfg.state_dim == 3
        assert cfg.n_iter_em == 10
        assert cfg.max_actors == 50
