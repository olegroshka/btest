"""
Supply-chain edge estimator from BEA I/O tables (Channel C5: SUPPLY_CHAIN).

M2.1-T4

Signals are a long-format DataFrame with columns:
    source_industry, target_industry, coefficient

Actors are matched on actor.sector or actor.external_ids.get("bea_industry").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from quantdsl_backtest.smim.config import SupplyChainEdgeConfig
from quantdsl_backtest.smim.graph.edges.base import AbstractEdgeEstimator
from quantdsl_backtest.smim.interfaces import ActorRegistry, DateRange, RelationChannel


class SupplyChainEdgeEstimator(AbstractEdgeEstimator):
    """Supply-chain edge estimator using BEA I/O coefficients.

    Expects signals to be a long-format DataFrame with columns:
        ``source_industry``, ``target_industry``, ``coefficient``

    Matches actor.external_ids.get("bea_industry") first, then actor.sector.
    """

    channel = RelationChannel.SUPPLY_CHAIN

    def __init__(self, config: SupplyChainEdgeConfig) -> None:
        self.config = config

    def estimate(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> csr_matrix:
        """Estimate supply-chain adjacency matrix from I/O coefficients.

        Args:
            signals: Long-format DataFrame with columns
                     ``source_industry``, ``target_industry``, ``coefficient``.
            actors: ActorRegistry for index mapping.
            date_range: Unused for I/O tables but accepted for protocol compliance.

        Returns:
            csr_matrix of shape (N, N) where A[target_idx, source_idx] = coefficient
            if coefficient ≥ threshold.
        """
        N = actors.N
        dense = np.zeros((N, N), dtype=float)

        # Build reverse mapping: industry_code → actor index
        industry_to_idx: dict[str, int] = {}
        for actor in actors.actors:
            idx = actors.index_of(actor.actor_id)
            # Prefer explicit BEA mapping
            bea_id = actor.external_ids.get("bea_industry")
            if bea_id:
                industry_to_idx[bea_id] = idx
            # Also map by sector as fallback
            if actor.sector and actor.sector not in industry_to_idx:
                industry_to_idx[actor.sector] = idx

        required_cols = {"source_industry", "target_industry", "coefficient"}
        if not required_cols.issubset(signals.columns):
            # If signals don't have the required columns, return empty
            return csr_matrix((N, N), dtype=float)

        for _, row in signals.iterrows():
            src = row["source_industry"]
            tgt = row["target_industry"]
            coeff = float(row["coefficient"])

            if coeff < self.config.coefficient_threshold:
                continue

            src_idx = industry_to_idx.get(src)
            tgt_idx = industry_to_idx.get(tgt)

            if src_idx is None or tgt_idx is None:
                # No mapping found; skip gracefully
                continue

            dense[tgt_idx, src_idx] = coeff

        return csr_matrix(dense)
