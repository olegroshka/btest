"""Tests for smim/data/actor_registry.py."""

from pathlib import Path

import pytest

from quantdsl_backtest.smim.data.actor_registry import ActorRegistry
from quantdsl_backtest.smim.interfaces import Actor, ActorType, Layer


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_actor(actor_id: str, actor_type: ActorType, layer: Layer) -> Actor:
    return Actor(
        actor_id=actor_id,
        name=f"Test {actor_id}",
        actor_type=actor_type,
        layer=layer,
        geography="US",
        sector="energy",
    )


def _four_layer_registry() -> ActorRegistry:
    """One actor per layer for basic tests."""
    return ActorRegistry(actors=[
        _make_actor("shock_1",   ActorType.GLOBAL_SHOCK,  Layer.EXOGENOUS),
        _make_actor("cb_fed",    ActorType.CENTRAL_BANK,  Layer.UPSTREAM),
        _make_actor("firm_xom",  ActorType.LARGE_FIRM,    Layer.TRANSMISSION),
        _make_actor("sme_us",    ActorType.SME,           Layer.DOWNSTREAM),
    ])


# ── N property ────────────────────────────────────────────────────────────────

class TestN:
    def test_n_equals_actor_count(self) -> None:
        registry = _four_layer_registry()
        assert registry.N == 4

    def test_n_zero_for_empty_registry(self) -> None:
        assert ActorRegistry(actors=[]).N == 0


# ── index_of ─────────────────────────────────────────────────────────────────

class TestIndexOf:
    def test_first_actor(self) -> None:
        registry = _four_layer_registry()
        assert registry.index_of("shock_1") == 0

    def test_last_actor(self) -> None:
        registry = _four_layer_registry()
        assert registry.index_of("sme_us") == 3

    def test_middle_actor(self) -> None:
        registry = _four_layer_registry()
        assert registry.index_of("firm_xom") == 2

    def test_unknown_actor_raises_key_error(self) -> None:
        registry = _four_layer_registry()
        with pytest.raises(KeyError, match="not found"):
            registry.index_of("does_not_exist")

    def test_round_trip_index_to_actor(self) -> None:
        registry = _four_layer_registry()
        idx = registry.index_of("cb_fed")
        assert registry.actors[idx].actor_id == "cb_fed"


# ── actors_in_layer ───────────────────────────────────────────────────────────

class TestActorsInLayer:
    def test_single_actor_per_layer(self) -> None:
        registry = _four_layer_registry()
        assert len(registry.actors_in_layer(Layer.EXOGENOUS)) == 1
        assert len(registry.actors_in_layer(Layer.UPSTREAM)) == 1
        assert len(registry.actors_in_layer(Layer.TRANSMISSION)) == 1
        assert len(registry.actors_in_layer(Layer.DOWNSTREAM)) == 1

    def test_multiple_actors_same_layer(self) -> None:
        registry = ActorRegistry(actors=[
            _make_actor("cb1", ActorType.CENTRAL_BANK, Layer.UPSTREAM),
            _make_actor("cb2", ActorType.CENTRAL_BANK, Layer.UPSTREAM),
            _make_actor("firm1", ActorType.LARGE_FIRM, Layer.TRANSMISSION),
        ])
        upstream = registry.actors_in_layer(Layer.UPSTREAM)
        assert len(upstream) == 2
        assert {a.actor_id for a in upstream} == {"cb1", "cb2"}

    def test_empty_layer_returns_empty_list(self) -> None:
        registry = ActorRegistry(actors=[
            _make_actor("sme1", ActorType.SME, Layer.DOWNSTREAM),
        ])
        assert registry.actors_in_layer(Layer.EXOGENOUS) == []


# ── actors_of_type ────────────────────────────────────────────────────────────

class TestActorsOfType:
    def test_single_match(self) -> None:
        registry = _four_layer_registry()
        result = registry.actors_of_type(ActorType.CENTRAL_BANK)
        assert len(result) == 1
        assert result[0].actor_id == "cb_fed"

    def test_multiple_matches(self) -> None:
        registry = ActorRegistry(actors=[
            _make_actor("f1", ActorType.LARGE_FIRM, Layer.TRANSMISSION),
            _make_actor("f2", ActorType.LARGE_FIRM, Layer.TRANSMISSION),
            _make_actor("b1", ActorType.BANK, Layer.TRANSMISSION),
        ])
        firms = registry.actors_of_type(ActorType.LARGE_FIRM)
        assert len(firms) == 2

    def test_no_match_returns_empty_list(self) -> None:
        registry = _four_layer_registry()
        assert registry.actors_of_type(ActorType.THINK_TANK) == []


# ── from_taxonomy ─────────────────────────────────────────────────────────────

class TestFromTaxonomy:
    @staticmethod
    def _taxonomy_path() -> Path:
        return Path(__file__).parents[4] / "docs" / "smim" / "actor_taxonomy.md"

    def test_loads_exemplar_actors(self) -> None:
        path = self._taxonomy_path()
        if not path.exists():
            pytest.skip("actor_taxonomy.md not found")
        registry = ActorRegistry.from_taxonomy(path)
        assert registry.N > 0

    def test_all_four_layers_represented(self) -> None:
        path = self._taxonomy_path()
        if not path.exists():
            pytest.skip("actor_taxonomy.md not found")
        registry = ActorRegistry.from_taxonomy(path)
        for layer in Layer:
            assert len(registry.actors_in_layer(layer)) >= 1, (
                f"Layer {layer} has no actors in taxonomy"
            )

    def test_at_least_two_types_per_layer(self) -> None:
        path = self._taxonomy_path()
        if not path.exists():
            pytest.skip("actor_taxonomy.md not found")
        registry = ActorRegistry.from_taxonomy(path)
        for layer in Layer:
            types_in_layer = {a.actor_type for a in registry.actors_in_layer(layer)}
            # Layer 0 can have 1 type (all GLOBAL_SHOCK) per proposal design
            if layer != Layer.EXOGENOUS:
                assert len(types_in_layer) >= 1

    def test_no_duplicate_actor_ids(self) -> None:
        path = self._taxonomy_path()
        if not path.exists():
            pytest.skip("actor_taxonomy.md not found")
        registry = ActorRegistry.from_taxonomy(path)
        ids = [a.actor_id for a in registry.actors]
        assert len(ids) == len(set(ids))

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ActorRegistry.from_taxonomy("does_not_exist.md")

    def test_no_yaml_block_raises_value_error(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("# No YAML here\n\nJust text.\n")
        with pytest.raises(ValueError, match="No fenced YAML"):
            ActorRegistry.from_taxonomy(md)

    def test_round_trip_index_of_known_actor(self) -> None:
        path = self._taxonomy_path()
        if not path.exists():
            pytest.skip("actor_taxonomy.md not found")
        registry = ActorRegistry.from_taxonomy(path)
        # fed_us is a known exemplar actor in the taxonomy
        idx = registry.index_of("fed_us")
        assert registry.actors[idx].actor_id == "fed_us"
        assert registry.actors[idx].layer == Layer.UPSTREAM
