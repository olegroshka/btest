"""
SMIM Experiment Configuration

Pydantic models that parse experiment YAML files in `experiments/`.
Every tuneable parameter lives here — no hardcoded magic numbers in source code.

Usage:
    from quantdsl_backtest.smim.config import SmimConfig
    config = SmimConfig.from_yaml("experiments/mvp_energy_us_uk.yaml")

All modules read their settings from the relevant section of SmimConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════
# Scope
# ═══════════════════════════════════════════════════════════

class ScopeConfig(BaseModel):
    """Defines the empirical scope (domain, geography, actor universe)."""
    domain: str = Field(description="e.g., 'energy', 'technology'")
    geographies: list[str] = Field(description="e.g., ['US', 'UK']")
    actor_universe_max: int = Field(default=500, ge=10, le=50000)
    date_range: tuple[str, str] = Field(
        default=("2005-01-01", "2025-01-01"),
        description="(start, end) as ISO date strings",
    )


# ═══════════════════════════════════════════════════════════
# Data / Adapters
# ═══════════════════════════════════════════════════════════

class FredConfig(BaseModel):
    """FRED / ALFRED adapter settings."""
    api_key_env: str = Field(default="FRED_API_KEY", description="env var name for API key")
    enable_vintage: bool = Field(default=True, description="use ALFRED vintage retrieval")
    cache_ttl_hours: int = Field(default=24)


class EdgarConfig(BaseModel):
    """SEC EDGAR adapter settings."""
    user_agent: str = Field(default="SMIM Research research@example.com")
    xbrl_tags: list[str] = Field(
        default=["CapitalExpenditures", "ResearchAndDevelopmentExpense", "Assets"],
    )
    filing_types: list[str] = Field(default=["10-K", "10-Q"])


class GdeltConfig(BaseModel):
    """GDELT adapter settings for narrative signals."""
    themes: list[str] = Field(default=["ECON_", "ENV_", "GOV_"])
    tone_field: str = Field(default="tone")
    cache_ttl_hours: int = Field(default=6)


class ImfConfig(BaseModel):
    """IMF SDMX adapter settings."""
    datasets: list[str] = Field(default=["IFS", "BOP", "WEO"])
    format: str = Field(default="CompactData")


class OecdConfig(BaseModel):
    """OECD SDMX adapter settings."""
    datasets: list[str] = Field(default=["QNA", "SNA_TABLE1"])


class BeaConfig(BaseModel):
    """BEA I/O table adapter settings."""
    api_key_env: str = Field(default="BEA_API_KEY")
    table_ids: list[str] = Field(default=["InputOutput"])
    year_range: tuple[int, int] = Field(default=(2010, 2024))


class BisConfig(BaseModel):
    """BIS statistics adapter settings."""
    datasets: list[str] = Field(default=["CBS", "LBS", "PP"])


class DataConfig(BaseModel):
    """All data adapter configurations."""
    fred: FredConfig = Field(default_factory=FredConfig)
    edgar: EdgarConfig = Field(default_factory=EdgarConfig)
    gdelt: GdeltConfig = Field(default_factory=GdeltConfig)
    imf: ImfConfig = Field(default_factory=ImfConfig)
    oecd: OecdConfig = Field(default_factory=OecdConfig)
    bea: BeaConfig = Field(default_factory=BeaConfig)
    bis: BisConfig = Field(default_factory=BisConfig)
    pit_store_path: str = Field(default="data/smim/pit_store")
    publication_lag_buffer_days: int = Field(
        default=5,
        description="Conservative buffer for non-vintaged sources (A1 compliance)",
    )


# ═══════════════════════════════════════════════════════════
# Edges / Graph Construction
# ═══════════════════════════════════════════════════════════

class GrangerEdgeConfig(BaseModel):
    """Granger-causal edge estimation settings."""
    max_lag: int = Field(default=4, ge=1, le=12)
    p_threshold: float = Field(default=0.05, gt=0, lt=1)
    bic_selection: bool = Field(default=True)


class NarrativeEdgeConfig(BaseModel):
    """Narrative/NLP edge estimation settings."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    similarity_threshold: float = Field(default=0.3, ge=0, le=1)
    method: str = Field(default="tfidf", description="'tfidf' or 'embedding'")


class SupplyChainEdgeConfig(BaseModel):
    """Supply-chain edge estimation from I/O tables."""
    coefficient_threshold: float = Field(default=0.01)


class SparsificationConfig(BaseModel):
    """Graph sparsification settings."""
    method: str = Field(default="l1_shrinkage", description="'l1_shrinkage' or 'threshold'")
    target_density: float = Field(default=0.15, gt=0, lt=1)


class EdgeConfig(BaseModel):
    """All edge estimation configurations."""
    channels: list[str] = Field(default=["financial", "narrative", "supply_chain"])
    granger: GrangerEdgeConfig = Field(default_factory=GrangerEdgeConfig)
    narrative: NarrativeEdgeConfig = Field(default_factory=NarrativeEdgeConfig)
    supply_chain: SupplyChainEdgeConfig = Field(default_factory=SupplyChainEdgeConfig)
    sparsification: SparsificationConfig = Field(default_factory=SparsificationConfig)
    null_model_instances: int = Field(
        default=100, ge=100,
        description="B ≥ 100 null instances for comparison (proposal requirement)",
    )


# ═══════════════════════════════════════════════════════════
# Spectral Decomposition
# ═══════════════════════════════════════════════════════════

class ModeSelectionConfig(BaseModel):
    """Three-criteria mode selection (Proposal Definition 3)."""
    mdl: bool = Field(default=True, description="MDL criterion")
    compressibility_min: float = Field(
        default=0.1, ge=0, le=1,
        description="LZ compressibility threshold ρ_min",
    )
    rg_relevance: bool = Field(default=True, description="RG relevance criterion")


class SpectralConfig(BaseModel):
    """Spectral decomposition settings."""
    methods: list[str] = Field(
        default=["schur", "polar", "hermitian_dilation", "dmd"],
        description="Methods to compare in WP3",
    )
    max_modes: int = Field(default=30, ge=1, le=500)
    mode_selection: ModeSelectionConfig = Field(default_factory=ModeSelectionConfig)
    rolling_window_quarters: int = Field(
        default=20, ge=4,
        description="Rolling window for stability assessment",
    )


# ═══════════════════════════════════════════════════════════
# Dynamics / State-Space / Phase Transitions
# ═══════════════════════════════════════════════════════════

class DynamicsConfig(BaseModel):
    """State-space filtering and regime-switching settings."""
    regime_candidates: list[int] = Field(
        default=[1, 2, 3, 4],
        description="Regime counts to evaluate; selected by MDL/BIC",
    )
    regime_selection: str = Field(default="mdl_bic")
    em_max_iterations: int = Field(default=200, ge=10)
    em_convergence_tol: float = Field(default=1e-4, gt=0)
    observability_condition_max: float = Field(
        default=1e6,
        description="Max condition number κ(O_z) before reducing K*",
    )
    order_parameter_dims: list[int] = Field(
        default=[1, 2, 3],
        description="Order parameter dimensions d to evaluate for GL landscape",
    )
    criticality_window_quarters: int = Field(
        default=8, ge=4,
        description="Window size w for criticality index C_t",
    )


# ═══════════════════════════════════════════════════════════
# Emergence Diagnostics
# ═══════════════════════════════════════════════════════════

class PidConfig(BaseModel):
    """Partial Information Decomposition settings."""
    method: str = Field(default="mmi_gaussian", description="'mmi_gaussian' or 'broja'")
    bootstrap_samples: int = Field(default=200, ge=50)
    confidence_level: float = Field(default=0.95)


class TransferEntropyConfig(BaseModel):
    """Transfer entropy settings."""
    lag: int = Field(default=1, ge=1)
    ksg_neighbours: int = Field(default=5, ge=1, description="k for KSG estimator")
    conditional: bool = Field(default=True, description="Conditional TE (control for intermediaries)")


class TdaConfig(BaseModel):
    """Topological data analysis / persistent homology settings."""
    window_size: int = Field(default=20, ge=5, description="Sliding window length")
    overlap: int = Field(default=10, ge=0, description="Window overlap")
    max_homology_dim: int = Field(default=2, ge=1)
    n_windows_min: int = Field(default=50, ge=10)


class EmergenceConfig(BaseModel):
    """All emergence diagnostic configurations."""
    pid: PidConfig = Field(default_factory=PidConfig)
    transfer_entropy: TransferEntropyConfig = Field(default_factory=TransferEntropyConfig)
    tda: TdaConfig = Field(default_factory=TdaConfig)


# ═══════════════════════════════════════════════════════════
# Validation / Backtesting
# ═══════════════════════════════════════════════════════════

class ValidationConfig(BaseModel):
    """Backtesting and falsification settings."""
    holdout_strategy: str = Field(
        default="rolling_5yr",
        description="'expanding', 'rolling_5yr', 'rolling_10yr'",
    )
    falsification_null_instances: int = Field(
        default=100, ge=100,
        description="B ≥ 100 per test (proposal requirement)",
    )
    falsification_p_threshold: float = Field(default=0.05)
    event_studies: list[str] = Field(
        default=[],
        description="Named events, e.g., ['fed_rate_2018', 'covid_2020']",
    )
    event_study_window_quarters: int = Field(default=2)
    dm_test_significance: float = Field(
        default=0.10,
        description="Diebold-Mariano significance level",
    )


# ═══════════════════════════════════════════════════════════
# Top-Level Config
# ═══════════════════════════════════════════════════════════

class SmimConfig(BaseModel):
    """Top-level SMIM experiment configuration.

    Load from YAML:
        config = SmimConfig.from_yaml("experiments/mvp_energy_us_uk.yaml")

    All SMIM modules receive their settings from the relevant section.
    """

    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    edges: EdgeConfig = Field(default_factory=EdgeConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    emergence: EmergenceConfig = Field(default_factory=EmergenceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SmimConfig:
        """Load config from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Validated SmimConfig instance.

        Raises:
            FileNotFoundError: If YAML file does not exist.
            pydantic.ValidationError: If YAML content fails validation.
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
