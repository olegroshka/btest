"""
Channel ablation and density sweep analysis.

M2.5-T1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import EdgeConfig
from quantdsl_backtest.smim.graph.operators import AggregateOperator
from quantdsl_backtest.smim.graph.sparsification import l1_sparsify, spectral_energy_retained
from quantdsl_backtest.smim.interfaces import RelationChannel


@dataclass
class AblationResult:
    """Result of ablation analysis.

    Attributes:
        channel_ablation: Mapping from removed channel name to metric delta.
        density_sweep: Mapping from target density to spectral_energy_retained.
        summary: DataFrame summarising both analyses.
    """

    channel_ablation: dict[str, float]  # channel_removed → metric_delta
    density_sweep: dict[float, float]   # target_density → spectral_energy_retained
    summary: pd.DataFrame


def _default_metric(op: csr_matrix) -> float:
    """Default metric: Frobenius norm."""
    dense = op.toarray()
    return float(np.linalg.norm(dense, "fro"))


class AblationRunner:
    """Runs channel ablation and density sweep analyses.

    Args:
        config: EdgeConfig with channel names and sparsification settings.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    def run_channel_ablation(
        self,
        all_matrices: dict[RelationChannel, csr_matrix],
        metric_fn: Callable[[csr_matrix], float] | None = None,
    ) -> dict[str, float]:
        """Remove each channel one at a time and measure metric change.

        Args:
            all_matrices: Full set of channel matrices.
            metric_fn: Scalar metric on csr_matrix. Defaults to Frobenius norm.

        Returns:
            dict mapping channel name → (full_metric - ablated_metric).
        """
        if metric_fn is None:
            metric_fn = _default_metric

        # Compute baseline metric using all channels
        full_op = AggregateOperator(all_matrices).with_uniform_weights()
        full_metric = metric_fn(full_op.compute())

        channel_names = {
            RelationChannel.FINANCIAL: "financial",
            RelationChannel.NARRATIVE: "narrative",
            RelationChannel.SUPPLY_CHAIN: "supply_chain",
            RelationChannel.REGULATORY: "regulatory",
            RelationChannel.FISCAL: "fiscal",
            RelationChannel.IMITATION: "imitation",
            RelationChannel.MARKET_IMPLIED: "market_implied",
        }

        result: dict[str, float] = {}
        for channel, name in channel_names.items():
            if channel not in all_matrices:
                continue
            # Remove this channel
            reduced = {k: v for k, v in all_matrices.items() if k != channel}
            if not reduced:
                result[name] = full_metric
                continue
            ablated_op = AggregateOperator(reduced).with_uniform_weights()
            ablated_metric = metric_fn(ablated_op.compute())
            result[name] = full_metric - ablated_metric

        return result

    def run_density_sweep(
        self,
        operator: csr_matrix,
        densities: list[float] | None = None,
    ) -> dict[float, float]:
        """Sparsify at each target density and measure spectral_energy_retained.

        Args:
            operator: Full aggregate operator matrix.
            densities: List of target densities to sweep. Defaults to
                       [0.05, 0.10, 0.15, 0.20, 0.30, 0.50].

        Returns:
            dict mapping target_density → spectral_energy_retained.
        """
        if densities is None:
            densities = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

        result: dict[float, float] = {}
        for d in densities:
            sparsified = l1_sparsify(operator, target_density=d)
            retained = spectral_energy_retained(operator, sparsified)
            result[d] = retained

        return result

    def run(
        self,
        all_matrices: dict[RelationChannel, csr_matrix],
        metric_fn: Callable[[csr_matrix], float] | None = None,
    ) -> AblationResult:
        """Run both ablation and density sweep, build summary DataFrame.

        Args:
            all_matrices: Full set of channel matrices.
            metric_fn: Scalar metric for channel ablation.

        Returns:
            AblationResult with both analyses and a summary DataFrame.
        """
        channel_ablation = self.run_channel_ablation(all_matrices, metric_fn)

        full_op = AggregateOperator(all_matrices).with_uniform_weights()
        operator = full_op.compute()
        density_sweep = self.run_density_sweep(operator)

        # Build summary DataFrame
        ablation_df = pd.DataFrame(
            [{"analysis": "channel_ablation", "key": k, "value": v}
             for k, v in channel_ablation.items()]
        )
        sweep_df = pd.DataFrame(
            [{"analysis": "density_sweep", "key": str(k), "value": v}
             for k, v in density_sweep.items()]
        )
        summary = pd.concat([ablation_df, sweep_df], ignore_index=True)

        return AblationResult(
            channel_ablation=channel_ablation,
            density_sweep=density_sweep,
            summary=summary,
        )
