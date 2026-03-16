"""Integration tests for smim/emergence/ diagnostics (M4.6 tasks).

The dedicated tests are in test_pid.py, test_transfer_entropy.py, test_tda.py.
This file provides smoke tests verifying the emergence modules are importable
and produce sensible results.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdsl_backtest.smim.emergence.pid import PIDResult, gaussian_mmi_pid
from quantdsl_backtest.smim.dynamics.phase_transition import criticality_index


class TestPIDSynergy:
    def test_compute_returns_pid_result(self) -> None:
        rng = np.random.default_rng(0)
        T = 60
        alpha_j = rng.standard_normal(T)
        alpha_k = rng.standard_normal(T)
        target = rng.standard_normal(T)
        result = gaussian_mmi_pid(alpha_j, alpha_k, target)
        assert isinstance(result, PIDResult)

    def test_name_is_pid_synergy(self) -> None:
        """Verify the PIDResult dataclass has the expected fields."""
        result = PIDResult(redundancy=0.1, unique_j=0.0, unique_k=0.0, synergy=0.05)
        assert hasattr(result, "synergy")
        assert hasattr(result, "redundancy")

    def test_confidence_intervals_shape(self) -> None:
        """Bootstrap CI returns (observed, lower, upper) — length 3."""
        from quantdsl_backtest.smim.emergence.pid import bootstrap_synergy
        rng = np.random.default_rng(1)
        T = 50
        a = rng.standard_normal(T)
        b = rng.standard_normal(T)
        c = rng.standard_normal(T)
        ci = bootstrap_synergy(a, b, c, n_bootstrap=20)
        assert len(ci) == 3


class TestCriticalityIndex:
    def test_compute_returns_diagnostic_result(self) -> None:
        T = 80
        psi = np.random.default_rng(2).standard_normal(T)
        C = criticality_index(psi, window_size=8)
        assert C.shape == (T,)
        assert np.all(np.isfinite(C))
