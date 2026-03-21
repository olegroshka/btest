"""
Granger-causal edge estimator (Channel C2: FINANCIAL).

M2.1-T2

For each ordered pair (i→j), tests whether time series i Granger-causes j.
BIC lag selection uses statsmodels VAR.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from quantdsl_backtest.smim.config import GrangerEdgeConfig
from quantdsl_backtest.smim.graph.edges.base import AbstractEdgeEstimator
from quantdsl_backtest.smim.interfaces import ActorRegistry, DateRange, RelationChannel
from quantdsl_backtest.smim.compute.batch_granger import batch_granger_test


class GrangerEdgeEstimator(AbstractEdgeEstimator):
    """Granger-causal edge estimator producing a FINANCIAL channel matrix.

    A[j_idx, i_idx] = F-statistic if p < p_threshold, else 0.
    BIC is used to select the optimal lag (if config.bic_selection=True).

    When bic_selection=False the batched PyTorch implementation is used for
    all N*(N-1) pairs in a single GPU kernel pass.  When bic_selection=True
    the sequential statsmodels path (_estimate_sequential) is used to preserve
    per-pair BIC lag selection.
    """

    channel = RelationChannel.FINANCIAL

    def __init__(self, config: GrangerEdgeConfig) -> None:
        self.config = config

    def estimate(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> csr_matrix:
        """Estimate directed Granger-causal adjacency matrix.

        Routes to batched GPU implementation when bic_selection=False,
        or to the sequential statsmodels path when bic_selection=True.

        Args:
            signals: DataFrame with index=dates, columns=actor_ids.
            actors: ActorRegistry matching signal columns.
            date_range: Date range for PIT filtering.

        Returns:
            csr_matrix of shape (N, N) where A[j, i] > 0 if i Granger-causes j.
        """
        if self.config.bic_selection:
            return self._estimate_sequential(signals, actors, date_range)

        self._validate_inputs(signals, actors, date_range)
        signals_filtered = self._apply_pit_filter(signals, date_range)

        actor_ids = [a.actor_id for a in actors.actors]
        signals_filtered = signals_filtered[actor_ids]

        # Drop rows with any NaN and convert to (T, N) array
        signals_array = signals_filtered.dropna().values.astype(np.float64)

        dense = batch_granger_test(
            signals_array,
            max_lag=self.config.max_lag,
            p_threshold=self.config.p_threshold,
        )
        return self._to_sparse(dense, threshold=0.0)

    def _estimate_sequential(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> csr_matrix:
        """Fallback: original statsmodels sequential implementation with BIC selection."""
        self._validate_inputs(signals, actors, date_range)
        signals_filtered = self._apply_pit_filter(signals, date_range)

        actor_ids = [a.actor_id for a in actors.actors]
        signals_filtered = signals_filtered[actor_ids]

        N = actors.N
        dense = np.zeros((N, N), dtype=float)

        for i_idx, i_id in enumerate(actor_ids):
            for j_idx, j_id in enumerate(actor_ids):
                if i_idx == j_idx:
                    continue

                series_i = signals_filtered[i_id].dropna()
                series_j = signals_filtered[j_id].dropna()

                common_idx = series_i.index.intersection(series_j.index)
                if len(common_idx) < 10:
                    continue

                x = series_i.loc[common_idx].values
                y = series_j.loc[common_idx].values

                lag = self._select_lag(np.column_stack([y, x]))
                if lag < 1:
                    lag = 1

                data_2col = np.column_stack([y, x])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = grangercausalitytests(
                            data_2col, maxlag=lag, verbose=False
                        )
                    fstat = results[lag][0]["ssr_ftest"][0]
                    pvalue = results[lag][0]["ssr_ftest"][1]
                    if pvalue < self.config.p_threshold:
                        dense[j_idx, i_idx] = fstat
                except Exception:
                    pass

        return self._to_sparse(dense, threshold=0.0)

    def _select_lag(self, data: np.ndarray) -> int:
        """Select BIC-optimal lag for VAR, capped at max_lag."""
        if not self.config.bic_selection:
            return self.config.max_lag
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VAR(data)
                result = model.fit(ic="bic", maxlags=self.config.max_lag, trend="c")
                return max(1, result.k_ar)
        except Exception:
            return self.config.max_lag
