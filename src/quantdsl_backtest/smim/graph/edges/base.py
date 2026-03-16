"""
Abstract base for edge estimators and the EdgeEstimatorFactory.

M2.1-T1: AbstractEdgeEstimator + EdgeEstimatorFactory
M2.1-T6: estimate_all wired through factory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.interfaces import ActorRegistry, DateRange, RelationChannel

if TYPE_CHECKING:
    from quantdsl_backtest.smim.config import EdgeConfig


class AbstractEdgeEstimator(ABC):
    """Abstract base class that implements the EdgeEstimator protocol."""

    @property
    @abstractmethod
    def channel(self) -> RelationChannel:
        """The relation channel this estimator produces."""
        ...

    @abstractmethod
    def estimate(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> csr_matrix:
        """Return A_t^{(r)} ∈ R^{N×N} where A[j,i] = influence of i on j."""
        ...

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> None:
        """Validate that signals are compatible with the actor registry.

        Raises:
            ValueError: If signal columns don't match actor IDs or any column
                        has more than 80% NaN values.
        """
        actor_ids = {a.actor_id for a in actors.actors}
        signal_cols = set(signals.columns)

        if not signal_cols.issubset(actor_ids):
            extra = signal_cols - actor_ids
            raise ValueError(
                f"Signal columns {extra} are not present in actor registry."
            )
        if not actor_ids.issubset(signal_cols):
            missing = actor_ids - signal_cols
            raise ValueError(
                f"Actor IDs {missing} have no corresponding signal columns."
            )

        # Check for >80% NaN in any column
        for col in signals.columns:
            nan_fraction = signals[col].isna().mean()
            if nan_fraction > 0.8:
                raise ValueError(
                    f"Column '{col}' has {nan_fraction:.1%} NaN values (>80%)."
                )

    def _to_sparse(
        self,
        dense_matrix: np.ndarray,
        threshold: float = 0.0,
    ) -> csr_matrix:
        """Convert N×N numpy array to csr_matrix, zeroing entries ≤ threshold."""
        arr = dense_matrix.copy()
        arr[arr <= threshold] = 0.0
        return sp.csr_matrix(arr)

    def _apply_pit_filter(
        self,
        signals: pd.DataFrame,
        date_range: DateRange,
    ) -> pd.DataFrame:
        """Filter signals rows to date_range.start ≤ date ≤ date_range.end."""
        mask = (signals.index >= date_range.start) & (signals.index <= date_range.end)
        return signals.loc[mask]


class EdgeEstimatorFactory:
    """Registry and factory for AbstractEdgeEstimator implementations.

    Usage::

        factory = EdgeEstimatorFactory()
        factory.register("financial", GrangerEdgeEstimator)
        est = factory.create("financial", config)
        results = factory.estimate_all(signals, actors, date_range, config)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[AbstractEdgeEstimator]] = {}

    def register(self, channel_name: str, cls: type[AbstractEdgeEstimator]) -> None:
        """Register an estimator class under channel_name."""
        self._registry[channel_name] = cls

    def create(self, channel_name: str, config: EdgeConfig) -> AbstractEdgeEstimator:
        """Instantiate the estimator for channel_name.

        Raises:
            KeyError: If channel_name is not registered.
        """
        if channel_name not in self._registry:
            raise KeyError(
                f"Unknown channel '{channel_name}'. "
                f"Registered channels: {list(self._registry.keys())}"
            )
        cls = self._registry[channel_name]
        # Pass the relevant sub-config to the estimator constructor
        channel_config_map = {
            "financial": config.granger,
            "narrative": config.narrative,
            "supply_chain": config.supply_chain,
        }
        sub_config = channel_config_map.get(channel_name)
        if sub_config is not None:
            return cls(sub_config)  # type: ignore[call-arg]
        return cls()  # type: ignore[call-arg]

    def estimate_all(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
        config: EdgeConfig,
    ) -> dict[RelationChannel, csr_matrix]:
        """Run all registered estimators for channels in config.channels.

        Returns:
            dict mapping RelationChannel → csr_matrix for each channel.
        """
        results: dict[RelationChannel, csr_matrix] = {}
        for channel_name in config.channels:
            estimator = self.create(channel_name, config)
            matrix = estimator.estimate(signals, actors, date_range)
            results[estimator.channel] = matrix
        return results


def make_default_factory() -> EdgeEstimatorFactory:
    """Build and return a factory pre-loaded with the standard estimators."""
    # Deferred imports to avoid circular dependencies
    from quantdsl_backtest.smim.graph.edges.granger import GrangerEdgeEstimator
    from quantdsl_backtest.smim.graph.edges.narrative import NarrativeEdgeEstimator
    from quantdsl_backtest.smim.graph.edges.supply_chain import SupplyChainEdgeEstimator

    factory = EdgeEstimatorFactory()
    factory.register("financial", GrangerEdgeEstimator)
    factory.register("narrative", NarrativeEdgeEstimator)
    factory.register("supply_chain", SupplyChainEdgeEstimator)
    return factory
