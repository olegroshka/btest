"""
SMIM Interface Definitions (Protocols and Data Classes)

All SMIM components implement these protocols. Define interfaces FIRST,
implement SECOND. This file is the contract that enables independent
implementation of each module.

Reference: Research Proposal v5.0, Implementation Plan v1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.sparse as sparse


# ═══════════════════════════════════════════════════════════
# Enums and Constants
# ═══════════════════════════════════════════════════════════

class ActorType(str, Enum):
    """Actor type classification (Proposal Section 4.1, Table 2)."""
    # Layer 0: Exogenous environment
    GLOBAL_SHOCK = "global_shock"
    # Layer 1: Upstream institutions
    CENTRAL_BANK = "central_bank"
    REGULATOR = "regulator"
    THINK_TANK = "think_tank"
    INTL_ORG = "intl_org"
    # Layer 2: Transmission / intermediary
    LARGE_FIRM = "large_firm"
    BANK = "bank"
    SECTOR_LEADER = "sector_leader"
    # Layer 3: Downstream
    SME = "sme"
    MUNICIPALITY = "municipality"
    HOUSEHOLD = "household"
    RETAIL_INVESTOR = "retail_investor"


class Layer(int, Enum):
    """Four-layer hierarchy (Proposal Table 2)."""
    EXOGENOUS = 0
    UPSTREAM = 1
    TRANSMISSION = 2
    DOWNSTREAM = 3


class RelationChannel(str, Enum):
    """Typed influence channels (Proposal Section 4.3, C1–C7)."""
    REGULATORY = "C1"
    FINANCIAL = "C2"
    FISCAL = "C3"
    NARRATIVE = "C4"
    SUPPLY_CHAIN = "C5"
    IMITATION = "C6"
    MARKET_IMPLIED = "C7"


class BenchmarkClass(str, Enum):
    """Benchmark class labels — EVERY gap must carry one of these.
    (Proposal Section 5.5, orange box: 'never report gap without label')"""
    PREDICTIVE = "predictive"
    STRUCTURAL = "structural"
    MODAL = "modal"
    EQUILIBRIUM = "equilibrium"
    EMERGENCE_AWARE = "emergence_aware"


class DecompositionMethod(str, Enum):
    """Spectral decomposition methods (Proposal Section 5.1)."""
    SCHUR = "schur"
    DIRECTED_VARIATION = "directed_variation"
    POLAR = "polar"
    HERMITIAN_DILATION = "hermitian_dilation"
    DMD = "dmd"
    EXTENDED_DMD = "extended_dmd"


# ═══════════════════════════════════════════════════════════
# Data Classes (shared data structures)
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Actor:
    """Single actor in the typed dynamic investment system."""
    actor_id: str
    name: str
    actor_type: ActorType
    layer: Layer
    geography: str
    sector: str
    external_ids: dict[str, str] = field(default_factory=dict)  # CIK, LEI, etc.


@dataclass
class ActorRegistry:
    """Registry of all actors in the system. Provides N and indexing."""
    actors: list[Actor]

    @property
    def N(self) -> int:
        return len(self.actors)

    def index_of(self, actor_id: str) -> int:
        """Return integer index for actor_id."""
        ...

    def actors_in_layer(self, layer: Layer) -> list[Actor]:
        ...

    def actors_of_type(self, actor_type: ActorType) -> list[Actor]:
        ...


@dataclass
class DateRange:
    """Time range for estimation or evaluation."""
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: str = "QS"  # quarterly start


@dataclass
class ModalFrame:
    """Result of spectral decomposition: modal basis U_t ∈ R^{N×K}.

    Attributes:
        basis: The modal frame matrix (N × K)
        eigenvalues: Associated eigenvalues (K,) — may be complex for directed operators
        method: Which decomposition produced this
        metadata: Method-specific metadata (e.g., condition numbers, pseudospectral radii)
    """
    basis: NDArray[np.floating]          # (N, K)
    eigenvalues: NDArray[np.complexfloating]  # (K,)
    method: DecompositionMethod
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def K(self) -> int:
        """Number of retained modes."""
        return self.basis.shape[1]

    @property
    def N(self) -> int:
        """Number of actors."""
        return self.basis.shape[0]


@dataclass
class FilteredState:
    """Result of state-space filtering.

    Attributes:
        alpha_filtered: Filtered modal states α_{t|t}, shape (T, K)
        alpha_predicted: One-step predictions α_{t|t-1}, shape (T, K)
        regime_probs: Smoothed regime probabilities P(z_t=j|Y_T), shape (T, M)
        log_likelihood: Total log-likelihood
        regimes_selected: Number of regimes selected by MDL/BIC
    """
    alpha_filtered: NDArray[np.floating]     # (T, K)
    alpha_predicted: NDArray[np.floating]    # (T, K)
    regime_probs: NDArray[np.floating]       # (T, M)
    log_likelihood: float
    regimes_selected: int
    params: dict[str, Any] = field(default_factory=dict)  # F^{(z)}, Q^{(z)}, R, P


@dataclass
class GapResult:
    """Investment gaps with mandatory benchmark label.

    Attributes:
        gaps: Actor-level gaps Δ_{i,t}, shape (N, T)
        benchmark_class: Which benchmark produced these gaps
        benchmarks: The benchmark values y*_{i,t}, shape (N, T)
        modal_attribution: Per-mode gap contributions (N, T, K), optional
    """
    gaps: NDArray[np.floating]               # (N, T)
    benchmark_class: BenchmarkClass          # MANDATORY — no unlabelled gaps
    benchmarks: NDArray[np.floating]         # (N, T)
    modal_attribution: NDArray[np.floating] | None = None  # (N, T, K)


@dataclass
class DiagnosticResult:
    """Result from one emergence diagnostic."""
    name: str  # "pid_synergy", "transfer_entropy", "criticality", "topological_complexity"
    values: NDArray[np.floating]  # shape depends on diagnostic
    confidence_intervals: NDArray[np.floating] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FalsificationResult:
    """Result from one falsification test."""
    test_name: str  # "shuffled_edges", "lag_destroyed", etc.
    observed_statistic: float
    null_distribution: NDArray[np.floating]  # (B,) where B ≥ 100
    p_value: float
    passes: bool  # True if observed > 95th percentile of null


# ═══════════════════════════════════════════════════════════
# Protocols (interfaces that implementations must satisfy)
# ═══════════════════════════════════════════════════════════

@runtime_checkable
class DataAdapter(Protocol):
    """Fetches raw data from an external source.

    Implementations: FredAdapter, EdgarAdapter, GdeltAdapter, ImfAdapter, etc.
    Must respect point-in-time discipline (A1).
    """

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,  # for PIT retrieval
    ) -> pd.DataFrame:
        """Fetch data, optionally as-of a specific publication date."""
        ...

    @property
    def source_name(self) -> str:
        """Identifier for this data source (e.g., 'fred', 'edgar')."""
        ...


@runtime_checkable
class InvestmentIntensityMapper(Protocol):
    """Maps raw actor-specific observables to normalised intensity y_{i,t} ∈ [0,1].

    One implementation per ActorType (Proposal Section 4.2, Table 3).
    Must satisfy A2 (typed comparability).
    """

    def compute(
        self,
        raw_data: pd.DataFrame,
        actor: Actor,
    ) -> pd.Series:
        """Returns normalised intensity series for one actor."""
        ...

    @property
    def actor_type(self) -> ActorType:
        ...


@runtime_checkable
class EdgeEstimator(Protocol):
    """Estimates directed adjacency matrix for one relation channel.

    Implementations: GrangerEdgeEstimator, NarrativeEdgeEstimator,
                     SupplyChainEdgeEstimator, etc.
    """

    def estimate(
        self,
        signals: pd.DataFrame,  # actors × time
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> sparse.csr_matrix:
        """Returns A_t^{(r)} ∈ R^{N×N} where A[j,i] = influence of i on j."""
        ...

    @property
    def channel(self) -> RelationChannel:
        ...


@runtime_checkable
class SpectralDecomposer(Protocol):
    """Produces modal frame U_t from aggregate operator A_t.

    Implementations: SchurDecomposer, PolarDecomposer,
                     HermitianDilationDecomposer, DVBasisDecomposer
    """

    def decompose(
        self,
        operator: sparse.csr_matrix,  # A_t, shape (N, N)
        k: int,                        # number of modes to retain
    ) -> ModalFrame:
        ...

    @property
    def method(self) -> DecompositionMethod:
        ...


@runtime_checkable
class ModeSelector(Protocol):
    """Selects K* retained modes from K candidates.

    Implements the three-criteria selection (Proposal Definition 3):
    MDL, compressibility, RG relevance.
    """

    def select(
        self,
        modal_frame: ModalFrame,
        time_series: NDArray[np.floating],  # modal amplitudes (T, K)
    ) -> tuple[ModalFrame, NDArray[np.floating]]:
        """Returns (reduced frame with K* modes, reduced time series)."""
        ...


@runtime_checkable
class StateFilter(Protocol):
    """State-space filter for modal dynamics.

    Implementations: KalmanFilter (single regime), KimFilter (switching)
    """

    def filter(
        self,
        observations: NDArray[np.floating],  # s_t, shape (T, N)
        modal_frame: ModalFrame,              # U_{K*}
        n_regimes: int = 1,
    ) -> FilteredState:
        ...

    def estimate_params(
        self,
        observations: NDArray[np.floating],
        modal_frame: ModalFrame,
        n_regimes: int = 1,
        max_em_iterations: int = 200,
    ) -> FilteredState:
        """EM estimation of all parameters + filtering."""
        ...


@runtime_checkable
class BenchmarkComputer(Protocol):
    """Computes benchmark y*_{i,t} and gap Δ_{i,t}.

    Each implementation MUST tag results with its BenchmarkClass.
    """

    def compute(
        self,
        observations: NDArray[np.floating],   # y_{i,t}, shape (N, T)
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors: ActorRegistry,
    ) -> GapResult:
        ...

    @property
    def benchmark_class(self) -> BenchmarkClass:
        ...


@runtime_checkable
class EmergenceDiagnostic(Protocol):
    """Computes one emergence diagnostic.

    Implementations: PIDSynergy, TransferEntropyProfile, CriticalityIndex,
                     TopologicalComplexity, CausalEmergence
    """

    def compute(
        self,
        modal_states: FilteredState,
        modal_frame: ModalFrame,
        actors: ActorRegistry,
    ) -> DiagnosticResult:
        ...

    @property
    def name(self) -> str:
        ...


@runtime_checkable
class FalsificationTest(Protocol):
    """One of the 7 falsification tests (Proposal Section 8.8).

    Each generates B ≥ 100 null instances and computes an empirical p-value.
    """

    def run(
        self,
        pipeline_result: Any,  # SmimPipelineResult
        n_null_instances: int = 100,
    ) -> FalsificationResult:
        ...

    @property
    def test_name(self) -> str:
        ...
