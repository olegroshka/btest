"""Tests for smim/validation/falsification.py — ShuffledEdgePlacebo, LagDestroyedPlacebo."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.validation.falsification import (
    FalsificationResult,
    LagDestroyedPlacebo,
    ShuffledEdgePlacebo,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _simple_operator(N: int = 6) -> csr_matrix:
    """Build a small sparse directed operator for testing."""
    rng = np.random.default_rng(42)
    dense = (rng.random((N, N)) > 0.6).astype(float)
    np.fill_diagonal(dense, 0.0)
    return csr_matrix(dense)


def _simple_signals(T: int = 20, N: int = 4) -> np.ndarray:
    """Build a small (T, N) signal array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, N))


# ═══════════════════════════════════════════════════════════
# ShuffledEdgePlacebo
# ═══════════════════════════════════════════════════════════

class TestShuffledEdgesTest:
    N_INSTANCES = 5

    def _run(self) -> FalsificationResult:
        op = _simple_operator()
        # pipeline_fn: count non-zero edges
        return ShuffledEdgePlacebo(n_instances=self.N_INSTANCES).run(
            op, pipeline_fn=lambda m: float(m.nnz)
        )

    def test_run_returns_falsification_result(self) -> None:
        result = self._run()
        assert isinstance(result, FalsificationResult)

    def test_null_distribution_has_n_instances_samples(self) -> None:
        result = self._run()
        assert len(result.null_distribution) == self.N_INSTANCES

    def test_test_name_is_shuffled_edge(self) -> None:
        result = self._run()
        assert result.test_name == "shuffled_edge"

    def test_p_value_in_unit_interval(self) -> None:
        result = self._run()
        assert 0.0 <= result.p_value <= 1.0

    def test_passes_is_bool(self) -> None:
        result = self._run()
        assert isinstance(result.passes, bool)


# ═══════════════════════════════════════════════════════════
# LagDestroyedPlacebo
# ═══════════════════════════════════════════════════════════

class TestLagDestroyedTest:
    N_INSTANCES = 5

    def _run(self) -> FalsificationResult:
        signals = _simple_signals()
        # pipeline_fn: mean absolute value
        return LagDestroyedPlacebo(n_instances=self.N_INSTANCES).run(
            signals, pipeline_fn=lambda s: float(np.mean(np.abs(s)))
        )

    def test_run_returns_falsification_result(self) -> None:
        result = self._run()
        assert isinstance(result, FalsificationResult)

    def test_null_distribution_has_n_instances_samples(self) -> None:
        result = self._run()
        assert len(result.null_distribution) == self.N_INSTANCES

    def test_p_value_in_unit_interval(self) -> None:
        result = self._run()
        assert 0.0 <= result.p_value <= 1.0

    def test_test_name_is_lag_destroyed(self) -> None:
        result = self._run()
        assert result.test_name == "lag_destroyed"

    def test_passes_is_bool(self) -> None:
        result = self._run()
        assert isinstance(result.passes, bool)
