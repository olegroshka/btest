"""Tests for SupplyChainEdgeEstimator (M2.1-T4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import SupplyChainEdgeConfig
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.graph.edges.supply_chain import SupplyChainEdgeEstimator
from quantdsl_backtest.smim.interfaces import Actor, ActorType, DateRange, Layer, RelationChannel


def _make_registry_with_sectors(actor_ids_sectors: list[tuple[str, str]]) -> ActorRegistry:
    actors = [
        Actor(
            actor_id=aid,
            name=aid,
            actor_type=ActorType.LARGE_FIRM,
            layer=Layer.TRANSMISSION,
            geography="US",
            sector=sector,
        )
        for aid, sector in actor_ids_sectors
    ]
    return ActorRegistry(actors=actors)


def _make_io_signals() -> tuple[pd.DataFrame, DateRange]:
    """Mock I/O table: A supplies B (coeff=0.15), B supplies C (coeff=0.02)."""
    data = {
        "source_industry": ["industry_A", "industry_B"],
        "target_industry": ["industry_B", "industry_C"],
        "coefficient": [0.15, 0.02],
    }
    signals = pd.DataFrame(data)
    dr = DateRange(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-12-31"),
    )
    return signals, dr


class TestSupplyChainEdgeEstimator:
    def setup_method(self):
        self.registry = _make_registry_with_sectors([
            ("actor_A", "industry_A"),
            ("actor_B", "industry_B"),
            ("actor_C", "industry_C"),
        ])
        self.signals, self.dr = _make_io_signals()
        self.config = SupplyChainEdgeConfig(coefficient_threshold=0.01)
        self.est = SupplyChainEdgeEstimator(self.config)

    def test_output_is_csr_matrix(self):
        result = self.est.estimate(self.signals, self.registry, self.dr)
        assert isinstance(result, csr_matrix)

    def test_output_shape_is_3_by_3(self):
        result = self.est.estimate(self.signals, self.registry, self.dr)
        assert result.shape == (3, 3)

    def test_channel_is_supply_chain(self):
        assert self.est.channel == RelationChannel.SUPPLY_CHAIN

    def test_a_supplies_b_edge_present(self):
        """A→B edge: coeff=0.15 > threshold=0.01 → A[b_idx, a_idx] = 0.15."""
        result = self.est.estimate(self.signals, self.registry, self.dr)
        a_idx = self.registry.index_of("actor_A")
        b_idx = self.registry.index_of("actor_B")
        arr = result.toarray()
        assert arr[b_idx, a_idx] == pytest.approx(0.15)

    def test_b_supplies_c_edge_present(self):
        """B→C edge: coeff=0.02 > threshold=0.01 → A[c_idx, b_idx] = 0.02."""
        result = self.est.estimate(self.signals, self.registry, self.dr)
        b_idx = self.registry.index_of("actor_B")
        c_idx = self.registry.index_of("actor_C")
        arr = result.toarray()
        assert arr[c_idx, b_idx] == pytest.approx(0.02)

    def test_below_threshold_edge_absent(self):
        """With threshold=0.10, the B→C edge (0.02) should be excluded."""
        config = SupplyChainEdgeConfig(coefficient_threshold=0.10)
        est = SupplyChainEdgeEstimator(config)
        result = est.estimate(self.signals, self.registry, self.dr)
        b_idx = self.registry.index_of("actor_B")
        c_idx = self.registry.index_of("actor_C")
        arr = result.toarray()
        assert arr[c_idx, b_idx] == 0.0

    def test_missing_industry_actor_skipped_gracefully(self):
        """If an industry has no matching actor, should not crash."""
        extra_signals = pd.DataFrame({
            "source_industry": ["unknown_industry"],
            "target_industry": ["industry_A"],
            "coefficient": [0.5],
        })
        result = self.est.estimate(extra_signals, self.registry, self.dr)
        assert isinstance(result, csr_matrix)

    def test_bea_industry_external_id_mapping(self):
        """Actors with bea_industry external_id should be matched first."""
        actors_with_bea = [
            Actor(
                actor_id="actor_X",
                name="Actor X",
                actor_type=ActorType.LARGE_FIRM,
                layer=Layer.TRANSMISSION,
                geography="US",
                sector="wrong_sector",
                external_ids={"bea_industry": "bea_X"},
            ),
            Actor(
                actor_id="actor_Y",
                name="Actor Y",
                actor_type=ActorType.LARGE_FIRM,
                layer=Layer.TRANSMISSION,
                geography="US",
                sector="wrong_sector",
                external_ids={"bea_industry": "bea_Y"},
            ),
        ]
        registry = ActorRegistry(actors=actors_with_bea)
        io_signals = pd.DataFrame({
            "source_industry": ["bea_X"],
            "target_industry": ["bea_Y"],
            "coefficient": [0.25],
        })
        dr = DateRange(start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-12-31"))
        result = self.est.estimate(io_signals, registry, dr)
        x_idx = registry.index_of("actor_X")
        y_idx = registry.index_of("actor_Y")
        arr = result.toarray()
        assert arr[y_idx, x_idx] == pytest.approx(0.25)
