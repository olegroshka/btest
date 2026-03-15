"""Tests for smim/config.py — SmimConfig Pydantic models."""

from pathlib import Path

import pytest

from quantdsl_backtest.smim.config import (
    DataConfig,
    DynamicsConfig,
    EdgeConfig,
    EmergenceConfig,
    SmimConfig,
    ScopeConfig,
    SpectralConfig,
    ValidationConfig,
)


def _make_config(**kwargs: object) -> SmimConfig:
    """Build a minimal valid SmimConfig."""
    scope = ScopeConfig(domain="test", geographies=["US"])
    return SmimConfig(scope=scope, **kwargs)  # type: ignore[arg-type]


class TestSmimConfigDefaults:
    def test_default_construction(self) -> None:
        config = _make_config()
        assert config.data.fred.cache_ttl_hours == 24
        assert config.edges.null_model_instances >= 100
        assert config.spectral.max_modes >= 1
        assert config.dynamics.em_max_iterations >= 10
        assert config.validation.falsification_null_instances >= 100

    def test_scope_config(self) -> None:
        scope = ScopeConfig(domain="energy", geographies=["US", "UK"])
        assert scope.domain == "energy"
        assert "US" in scope.geographies

    def test_edge_null_model_instances_minimum(self) -> None:
        with pytest.raises(Exception):
            EdgeConfig(null_model_instances=99)

    def test_validation_null_instances_minimum(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(falsification_null_instances=50)


class TestSmimConfigFromYaml:
    def test_load_mvp_yaml(self) -> None:
        yaml_path = Path(__file__).parents[4] / "experiments" / "mvp_energy_us_uk.yaml"
        if not yaml_path.exists():
            pytest.skip("Sample YAML not found")
        config = SmimConfig.from_yaml(yaml_path)
        assert config.scope.domain == "energy"
        assert config.scope.geographies == ["US", "UK"]
        assert config.scope.actor_universe_max == 200
        assert config.edges.null_model_instances == 100
        assert config.spectral.max_modes == 30
        assert config.dynamics.regime_candidates == [1, 2, 3, 4]

    def test_roundtrip_yaml(self, tmp_path: Path) -> None:
        config = _make_config()
        out = tmp_path / "config.yaml"
        config.to_yaml(out)
        reloaded = SmimConfig.from_yaml(out)
        assert reloaded.model_dump() == config.model_dump()

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            SmimConfig.from_yaml("does_not_exist.yaml")
