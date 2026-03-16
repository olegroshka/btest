"""
Narrative/NLP edge estimator (Channel C4: NARRATIVE).

M2.1-T3

Concatenates text per actor, builds TF-IDF vectors, computes cosine similarity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from quantdsl_backtest.smim.config import NarrativeEdgeConfig
from quantdsl_backtest.smim.graph.edges.base import AbstractEdgeEstimator
from quantdsl_backtest.smim.interfaces import ActorRegistry, DateRange, RelationChannel


class NarrativeEdgeEstimator(AbstractEdgeEstimator):
    """TF-IDF cosine-similarity edge estimator for the NARRATIVE channel.

    Signals are expected to be text strings (or NaN). All non-null text for
    each actor is concatenated, TF-IDF vectorized, and cosine similarity is
    computed to produce a symmetric adjacency matrix.
    """

    channel = RelationChannel.NARRATIVE

    def __init__(self, config: NarrativeEdgeConfig) -> None:
        self.config = config

    def estimate(
        self,
        signals: pd.DataFrame,
        actors: ActorRegistry,
        date_range: DateRange,
    ) -> csr_matrix:
        """Estimate narrative similarity adjacency matrix.

        Args:
            signals: DataFrame with index=dates, columns=actor_ids;
                     values are text strings or NaN.
            actors: ActorRegistry matching signal columns.
            date_range: Date range for PIT filtering.

        Returns:
            csr_matrix of shape (N, N), symmetric, with zeros on diagonal.
        """
        signals_filtered = self._apply_pit_filter(signals, date_range)

        actor_ids = [a.actor_id for a in actors.actors]
        N = actors.N

        # Concatenate text per actor (skip NaN and non-string values)
        corpus: list[str] = []
        for actor_id in actor_ids:
            if actor_id not in signals_filtered.columns:
                corpus.append("")
                continue
            texts = signals_filtered[actor_id].dropna()
            text_parts = [
                str(v) for v in texts.values
                if isinstance(v, str) and v.strip()
            ]
            corpus.append(" ".join(text_parts))

        # TF-IDF vectorization
        if self.config.method == "tfidf" or self.config.method != "embedding":
            vectorizer = TfidfVectorizer(min_df=1)
            # Handle empty corpus case
            non_empty = [t for t in corpus if t.strip()]
            if len(non_empty) < 2:
                return csr_matrix((N, N), dtype=float)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_matrix)
        else:
            # Fallback for unknown method
            sim_matrix = np.zeros((N, N), dtype=float)

        # Zero out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, 0.0)

        # Threshold
        dense = sim_matrix.copy()
        dense[dense <= self.config.similarity_threshold] = 0.0

        return csr_matrix(dense)
