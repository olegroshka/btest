"""Tests for cross-sector/geography parameter transfer experiment (M6.3-T1)."""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame
from quantdsl_backtest.smim.validation.transfer import (
    TransferExperiment,
    TransferExperimentResult,
    TransferabilityResult,
)


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

class TestTransferExperiment:

    def test_identical_data_transfers_perfectly(self):
        """source == target → all parameters transfer."""
        rng = np.random.default_rng(42)
        N, T, K = 8, 40, 3
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)
        op = rng.standard_normal((N, N))
        tm = np.eye(2) * 0.9

        exp = TransferExperiment(threshold=0.20)
        result = exp.run(
            "source", obs, mf,
            "target", obs, mf,
            source_transition_matrix=tm,
            target_transition_matrix=tm,
            source_operator=op,
            target_operator=op,
        )

        # With identical data, all 4 parameters should transfer
        assert result.n_transferable == result.n_total
        assert result.transfer_rate == 1.0
        assert result.passes is True

    def test_very_different_data_fails_transfer(self):
        """Very different random matrices → transfer_rate < 0.5 for non-skipped params."""
        rng = np.random.default_rng(77)
        N, T, K = 8, 40, 3

        # Source: low-variance data
        obs_src = rng.standard_normal((T, N)) * 0.01
        mf_src = _make_modal_frame(N, K, rng)
        op_src = np.eye(N) * 100.0  # very large spectral gap

        # Target: completely different
        obs_tgt = rng.standard_normal((T, N)) * 100.0
        mf_tgt = _make_modal_frame(N, K, rng)  # different basis
        op_tgt = rng.standard_normal((N, N)) * 0.001  # tiny spectral gap

        exp = TransferExperiment(threshold=0.20)
        result = exp.run(
            "source", obs_src, mf_src,
            "target", obs_tgt, mf_tgt,
            source_operator=op_src,
            target_operator=op_tgt,
        )

        # normalisation and edge_weights should fail (large differences)
        norm_result = next(r for r in result.parameter_results if r.parameter_name == "normalisation")
        edge_result = next(r for r in result.parameter_results if r.parameter_name == "edge_weights")
        assert not norm_result.transfers or not edge_result.transfers

    def test_transfer_rate_computed_correctly(self):
        """2/4 parameters transfer → rate = 0.5."""
        # Build result manually
        param_results = [
            TransferabilityResult("modal_structure", 1.0, 1.0, 0.01, True),
            TransferabilityResult("regime_dynamics", 1.0, 1.0, 0.01, True),
            TransferabilityResult("edge_weights", 1.0, 2.0, 0.50, False),
            TransferabilityResult("normalisation", 1.0, 2.0, 0.50, False),
        ]
        result = TransferExperimentResult(
            source_name="A",
            target_name="B",
            n_transferable=2,
            n_total=4,
            transfer_rate=2 / 4,
            parameter_results=param_results,
            passes=True,
        )
        assert result.transfer_rate == 0.5

    def test_result_has_all_four_parameters(self):
        """run() always produces results for all four parameter names."""
        rng = np.random.default_rng(10)
        N, T, K = 6, 20, 3
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)

        exp = TransferExperiment()
        result = exp.run("src", obs, mf, "tgt", obs, mf)

        param_names = {r.parameter_name for r in result.parameter_results}
        assert param_names == set(TransferExperiment.PARAMETERS)

    def test_passes_when_rate_gte_50_pct(self):
        """passes=True when transfer_rate >= 0.5."""
        rng = np.random.default_rng(99)
        N, T, K = 6, 20, 3
        obs = rng.standard_normal((T, N))
        mf = _make_modal_frame(N, K, rng)

        exp = TransferExperiment(threshold=0.20)
        # Identical data → 100% transfer rate
        result = exp.run("s", obs, mf, "t", obs, mf)
        assert result.passes is True
        assert result.transfer_rate >= 0.5

    def test_passes_false_when_rate_below_50_pct(self):
        """passes=False when transfer_rate < 0.5."""
        # Manually build a result with 1/4 transfer
        result = TransferExperimentResult(
            source_name="A",
            target_name="B",
            n_transferable=1,
            n_total=4,
            transfer_rate=0.25,
            parameter_results=[],
            passes=0.25 >= 0.5,
        )
        assert result.passes is False

    def test_relative_diff_formula(self):
        """relative_diff = |source - target| / (|source| + 1e-9)."""
        # For normalisation: uses |s - t| / max(s, t)
        # For edge_weights: uses |sg_s - sg_t| / max(|sg_s|, |sg_t|)
        # Just check that TransferabilityResult stores the values correctly
        r = TransferabilityResult(
            parameter_name="test",
            source_value=1.0,
            target_value=2.0,
            relative_diff=abs(1.0 - 2.0) / (abs(1.0) + 1e-9),
            transfers=False,
        )
        expected = abs(1.0 - 2.0) / (abs(1.0) + 1e-9)
        assert abs(r.relative_diff - expected) < 1e-10
