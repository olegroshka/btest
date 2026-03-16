"""Falsification tests for SMIM pipeline validation.

M5.2-T1 through M5.2-T4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.graph.null_models import (
    block_preserving_rewire,
    degree_preserving_rewire,
)


@dataclass
class FalsificationResult:
    """Result from one falsification test."""

    test_name: str
    observed_metric: float
    null_distribution: np.ndarray  # (B,)
    p_value: float
    passes: bool  # p_value < 0.05 means result is NOT due to chance
    n_instances: int
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# M5.2-T1: ShuffledEdgePlacebo
# ─────────────────────────────────────────────────────────────────────────────


class ShuffledEdgePlacebo:
    """Placebo test: rewire edges, check if structural metric degrades.

    Args:
        n_instances: Number of null rewirings.
        n_swaps_factor: Multiplier on nnz for rewiring steps.
    """

    def __init__(self, n_instances: int = 100, n_swaps_factor: int = 10) -> None:
        self.n_instances = n_instances
        self.n_swaps_factor = n_swaps_factor

    def run(
        self,
        operator: csr_matrix,
        pipeline_fn: Callable[[csr_matrix], float],
    ) -> FalsificationResult:
        """Run shuffled-edge placebo.

        Args:
            operator: Observed adjacency/operator matrix.
            pipeline_fn: Callable that maps a csr_matrix to a scalar metric.

        Returns:
            FalsificationResult
        """
        observed = float(pipeline_fn(operator))
        nnz = operator.nnz
        n_swaps = self.n_swaps_factor * max(nnz, 1)

        null_metrics = []
        rng = np.random.default_rng()
        for _ in range(self.n_instances):
            rewired = degree_preserving_rewire(operator, n_swaps=n_swaps, rng=rng)
            null_metrics.append(float(pipeline_fn(rewired)))

        null_arr = np.array(null_metrics)
        p_value = float(np.mean(null_arr >= observed))

        return FalsificationResult(
            test_name="shuffled_edge",
            observed_metric=observed,
            null_distribution=null_arr,
            p_value=p_value,
            passes=(p_value < 0.05),
            n_instances=self.n_instances,
        )


# ─────────────────────────────────────────────────────────────────────────────
# M5.2-T2: LagDestroyedPlacebo and RandomisedTypePlacebo
# ─────────────────────────────────────────────────────────────────────────────


class LagDestroyedPlacebo:
    """Placebo test: shuffle time axis to destroy temporal structure.

    Args:
        n_instances: Number of null permutations.
    """

    def __init__(self, n_instances: int = 100) -> None:
        self.n_instances = n_instances

    def run(
        self,
        signals: np.ndarray,
        pipeline_fn: Callable[[np.ndarray], float],
    ) -> FalsificationResult:
        """Run lag-destroyed placebo.

        Args:
            signals: (T, N) array of time series signals.
            pipeline_fn: Callable that maps (T, N) array to scalar metric.

        Returns:
            FalsificationResult
        """
        signals = np.asarray(signals, dtype=float)
        T = signals.shape[0]
        observed = float(pipeline_fn(signals))

        rng = np.random.default_rng()
        null_metrics = []
        for _ in range(self.n_instances):
            perm = rng.permutation(T)
            permuted = signals[perm, :]
            null_metrics.append(float(pipeline_fn(permuted)))

        null_arr = np.array(null_metrics)
        p_value = float(np.mean(null_arr >= observed))

        return FalsificationResult(
            test_name="lag_destroyed",
            observed_metric=observed,
            null_distribution=null_arr,
            p_value=p_value,
            passes=(p_value < 0.05),
            n_instances=self.n_instances,
        )


class RandomisedTypePlacebo:
    """Placebo test: shuffle actor type labels.

    Args:
        n_instances: Number of null permutations.
    """

    def __init__(self, n_instances: int = 100) -> None:
        self.n_instances = n_instances

    def run(
        self,
        actor_types: list[str],
        pipeline_fn: Callable[[list[str]], float],
    ) -> FalsificationResult:
        """Run randomised-type placebo.

        Args:
            actor_types: List of actor type strings.
            pipeline_fn: Callable that maps list of types to scalar metric.

        Returns:
            FalsificationResult
        """
        actor_types = list(actor_types)
        observed = float(pipeline_fn(actor_types))

        rng = np.random.default_rng()
        null_metrics = []
        for _ in range(self.n_instances):
            shuffled = list(rng.permutation(actor_types))
            null_metrics.append(float(pipeline_fn(shuffled)))

        null_arr = np.array(null_metrics)
        p_value = float(np.mean(null_arr >= observed))

        return FalsificationResult(
            test_name="randomised_type",
            observed_metric=observed,
            null_distribution=null_arr,
            p_value=p_value,
            passes=(p_value < 0.05),
            n_instances=self.n_instances,
        )


# ─────────────────────────────────────────────────────────────────────────────
# M5.2-T3: FrozenRegimePlacebo and SymmetricOperatorPlacebo
# ─────────────────────────────────────────────────────────────────────────────


class FrozenRegimePlacebo:
    """Placebo test: compare multi-regime to single-regime model.

    Args:
        n_instances: Not used for bootstrap (comparison-based test).
    """

    def __init__(self, n_instances: int = 100) -> None:
        self.n_instances = n_instances

    def run(
        self,
        multi_regime_metric: float,
        single_regime_metric: float,
        threshold: float = 0.01,
    ) -> FalsificationResult:
        """Compare multi-regime vs single-regime metric.

        Passes if multi_regime_metric > single_regime_metric + threshold.

        Args:
            multi_regime_metric: Metric from multi-regime model.
            single_regime_metric: Metric from single-regime model.
            threshold: Minimum improvement required (default 0.01).

        Returns:
            FalsificationResult
        """
        observed = float(multi_regime_metric)
        null_metric = float(single_regime_metric)
        passes = observed > null_metric + threshold
        p_value = 0.0 if passes else 1.0

        return FalsificationResult(
            test_name="frozen_regime",
            observed_metric=observed,
            null_distribution=np.array([null_metric]),
            p_value=p_value,
            passes=passes,
            n_instances=1,
            metadata={"single_regime_metric": null_metric, "threshold": threshold},
        )


class SymmetricOperatorPlacebo:
    """Placebo test: compare directed operator against its symmetrized version.

    Tests whether asymmetry (directionality) improves the metric.
    """

    def __init__(self) -> None:
        pass

    def run(
        self,
        operator: csr_matrix,
        pipeline_fn: Callable[[csr_matrix], float],
    ) -> FalsificationResult:
        """Compare directed operator vs symmetrized.

        Args:
            operator: Directed adjacency/operator matrix.
            pipeline_fn: Callable mapping csr_matrix to scalar.

        Returns:
            FalsificationResult
        """
        observed = float(pipeline_fn(operator))
        symmetric = (operator + operator.T) / 2.0
        null_metric = float(pipeline_fn(symmetric))

        passes = observed > null_metric
        p_value = 0.0 if passes else 1.0

        return FalsificationResult(
            test_name="symmetric_operator",
            observed_metric=observed,
            null_distribution=np.array([null_metric]),
            p_value=p_value,
            passes=passes,
            n_instances=1,
            metadata={"symmetric_metric": null_metric},
        )


# ─────────────────────────────────────────────────────────────────────────────
# M5.2-T4: BlockPreservingPlacebo, NoNetworkDFMPlacebo, FalsificationSuite
# ─────────────────────────────────────────────────────────────────────────────


class BlockPreservingPlacebo:
    """Placebo test: rewire within blocks only, testing cross-block structure.

    Args:
        n_instances: Number of null rewirings.
    """

    def __init__(self, n_instances: int = 100) -> None:
        self.n_instances = n_instances

    def run(
        self,
        operator: csr_matrix,
        layer_assignments: np.ndarray,
        pipeline_fn: Callable[[csr_matrix], float],
    ) -> FalsificationResult:
        """Run block-preserving placebo.

        Args:
            operator: Observed adjacency/operator matrix.
            layer_assignments: Integer array of length N mapping nodes to blocks.
            pipeline_fn: Callable mapping csr_matrix to scalar.

        Returns:
            FalsificationResult
        """
        observed = float(pipeline_fn(operator))

        rng = np.random.default_rng()
        null_metrics = []
        for _ in range(self.n_instances):
            rewired = block_preserving_rewire(operator, layer_assignments, rng=rng)
            null_metrics.append(float(pipeline_fn(rewired)))

        null_arr = np.array(null_metrics)
        p_value = float(np.mean(null_arr >= observed))

        return FalsificationResult(
            test_name="block_preserving",
            observed_metric=observed,
            null_distribution=null_arr,
            p_value=p_value,
            passes=(p_value < 0.05),
            n_instances=self.n_instances,
        )


class NoNetworkDFMPlacebo:
    """Placebo test: compare SMIM (with network) against no-network DFM.

    Simple comparison: passes if smim_metric > dfm_metric.
    """

    def __init__(self) -> None:
        pass

    def run(
        self,
        smim_metric: float,
        dfm_metric: float,
    ) -> FalsificationResult:
        """Compare SMIM against no-network DFM.

        Args:
            smim_metric: Performance metric from SMIM model.
            dfm_metric: Performance metric from no-network DFM baseline.

        Returns:
            FalsificationResult
        """
        observed = float(smim_metric)
        null_metric = float(dfm_metric)
        passes = observed > null_metric
        p_value = 0.0 if passes else 1.0

        return FalsificationResult(
            test_name="no_network_dfm",
            observed_metric=observed,
            null_distribution=np.array([null_metric]),
            p_value=p_value,
            passes=passes,
            n_instances=1,
            metadata={"dfm_metric": null_metric},
        )


class FalsificationSuite:
    """Run all 7 falsification tests and report results.

    Args:
        n_instances: Number of null instances for bootstrap-based tests.
    """

    def __init__(self, n_instances: int = 100) -> None:
        self.n_instances = n_instances

    def run_all(
        self,
        operator: csr_matrix | None = None,
        signals: np.ndarray | None = None,
        actor_types: list[str] | None = None,
        multi_regime_metric: float = 0.0,
        single_regime_metric: float = 0.0,
        layer_assignments: np.ndarray | None = None,
        smim_metric: float = 0.0,
        dfm_metric: float = 0.0,
        operator_pipeline_fn: Callable[[csr_matrix], float] | None = None,
        signal_pipeline_fn: Callable[[np.ndarray], float] | None = None,
        type_pipeline_fn: Callable[[list[str]], float] | None = None,
    ) -> dict[str, FalsificationResult]:
        """Run all 7 falsification tests.

        Returns:
            Dict mapping test name to FalsificationResult.
        """
        results = {}

        # Default no-op metrics if not provided
        if operator is None:
            import scipy.sparse as sp
            operator = sp.eye(4, format="csr")
        if operator_pipeline_fn is None:
            def operator_pipeline_fn(op): return float(op.nnz)
        if signals is None:
            signals = np.zeros((10, 3))
        if signal_pipeline_fn is None:
            def signal_pipeline_fn(s): return float(np.mean(s))
        if actor_types is None:
            actor_types = ["A", "B", "C"]
        if type_pipeline_fn is None:
            def type_pipeline_fn(t): return float(len(set(t)))
        if layer_assignments is None:
            layer_assignments = np.zeros(operator.shape[0], dtype=int)

        results["shuffled_edge"] = ShuffledEdgePlacebo(
            n_instances=self.n_instances
        ).run(operator, operator_pipeline_fn)

        results["lag_destroyed"] = LagDestroyedPlacebo(
            n_instances=self.n_instances
        ).run(signals, signal_pipeline_fn)

        results["randomised_type"] = RandomisedTypePlacebo(
            n_instances=self.n_instances
        ).run(actor_types, type_pipeline_fn)

        results["frozen_regime"] = FrozenRegimePlacebo(
            n_instances=self.n_instances
        ).run(multi_regime_metric, single_regime_metric)

        results["symmetric_operator"] = SymmetricOperatorPlacebo().run(
            operator, operator_pipeline_fn
        )

        results["block_preserving"] = BlockPreservingPlacebo(
            n_instances=self.n_instances
        ).run(operator, layer_assignments, operator_pipeline_fn)

        results["no_network_dfm"] = NoNetworkDFMPlacebo().run(smim_metric, dfm_metric)

        return results

    def summary_table(
        self, results: dict[str, FalsificationResult]
    ) -> pd.DataFrame:
        """Build summary table from results dict.

        Returns:
            DataFrame with columns: test_name, observed_metric, p_value, passes
        """
        rows = []
        for key, r in results.items():
            rows.append({
                "test_name": r.test_name,
                "observed_metric": r.observed_metric,
                "p_value": r.p_value,
                "passes": r.passes,
            })
        return pd.DataFrame(rows)
