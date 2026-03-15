"""Stub tests for smim/graph/operators.py — AggregateOperator (M2.1-T5)."""

import pytest


@pytest.mark.xfail(reason="AggregateOperator not yet implemented (M2.1-T5)", strict=False)
class TestAggregateOperator:
    def test_weighted_combination_shape(self) -> None:
        pytest.fail("not implemented")

    def test_spectral_energy_retention_above_80pct(self) -> None:
        """A3: sparsification retains >80% spectral energy."""
        pytest.fail("not implemented")

    def test_sparse_format_roundtrip(self) -> None:
        pytest.fail("not implemented")
