"""Stub tests for smim/graph/edges/ estimators (M2.1-T2 through M2.1-T4)."""

import pytest


@pytest.mark.xfail(reason="GrangerEdgeEstimator not yet implemented (M2.1-T2)", strict=False)
class TestGrangerEdgeEstimator:
    def test_estimate_returns_csr_matrix(self) -> None:
        pytest.fail("not implemented")

    def test_channel_is_financial(self) -> None:
        pytest.fail("not implemented")

    def test_bic_lag_selection(self) -> None:
        pytest.fail("not implemented")

    def test_p_value_threshold_applied(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="NarrativeEdgeEstimator not yet implemented (M2.1-T3)", strict=False)
class TestNarrativeEdgeEstimator:
    def test_estimate_returns_csr_matrix(self) -> None:
        pytest.fail("not implemented")

    def test_channel_is_narrative(self) -> None:
        pytest.fail("not implemented")

    def test_similarity_threshold_applied(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="SupplyChainEdgeEstimator not yet implemented (M2.1-T4)", strict=False)
class TestSupplyChainEdgeEstimator:
    def test_estimate_returns_csr_matrix(self) -> None:
        pytest.fail("not implemented")

    def test_channel_is_supply_chain(self) -> None:
        pytest.fail("not implemented")

    def test_coefficient_threshold_sparsifies(self) -> None:
        pytest.fail("not implemented")
