"""Import tests confirming all edge estimator modules are importable (M2.1-T3 replacement)."""

from __future__ import annotations


def test_granger_estimator_importable():
    from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
    assert GrangerEdgeEstimator is not None


def test_narrative_estimator_importable():
    from quantdsl_backtest.smim.graph.edges.narrative import NarrativeEdgeEstimator
    assert NarrativeEdgeEstimator is not None


def test_supply_chain_estimator_importable():
    from quantdsl_backtest.smim.graph.edges.supply_chain import SupplyChainEdgeEstimator
    assert SupplyChainEdgeEstimator is not None


def test_base_importable():
    from quantdsl_backtest.smim.graph.edges.base import AbstractEdgeEstimator, EdgeEstimatorFactory
    assert AbstractEdgeEstimator is not None
    assert EdgeEstimatorFactory is not None
