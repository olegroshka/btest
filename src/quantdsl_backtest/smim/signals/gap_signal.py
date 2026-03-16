"""Bridge signals connecting SMIM gaps to the btest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdsl_backtest.smim.interfaces import FilteredState, GapResult


class GapSignal:
    """Convert a GapResult into a trading signal matrix.

    Args:
        gap_result: GapResult with .gaps (N, T) array and .actors list
        threshold: Fraction of std-dev for signal generation
        scale: If True, normalize signals proportionally to magnitude
    """

    def __init__(
        self,
        gap_result: GapResult,
        threshold: float = 0.5,
        scale: bool = True,
    ) -> None:
        self.gap_result = gap_result
        self.threshold = threshold
        self.scale = scale

    def to_signal_matrix(self) -> pd.DataFrame:
        """Convert gaps to signal matrix (actors x dates).

        Returns:
            DataFrame with index=actors, columns=dates.
            gap > threshold*std → -1 (over-invested → short)
            gap < -threshold*std → +1 (under-invested → long)
            else → 0
            If scale=True, scale non-zero values proportionally to magnitude.
        """
        gaps = np.asarray(self.gap_result.gaps, dtype=float)  # (N, T)
        N, T = gaps.shape

        # Compute per-actor std
        std = np.std(gaps, axis=1, keepdims=True)  # (N, 1)
        std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero

        threshold_vals = self.threshold * std  # (N, 1)

        signals = np.zeros_like(gaps)
        over = gaps > threshold_vals
        under = gaps < -threshold_vals

        if self.scale:
            # Scale proportionally to magnitude
            signals[over] = -(gaps[over] / std[:, 0:1].repeat(T, axis=1)[over])
            signals[under] = -(gaps[under] / std[:, 0:1].repeat(T, axis=1)[under])
            # Clip to [-1, 1]
            signals = np.clip(signals, -1.0, 1.0)
        else:
            signals[over] = -1.0
            signals[under] = 1.0

        # Determine actors and dates for index/columns
        actors = getattr(self.gap_result, "actors", None)
        if actors is None:
            actors = [str(i) for i in range(N)]

        dates = getattr(self.gap_result, "dates", None)
        if dates is None:
            dates = list(range(T))

        return pd.DataFrame(signals, index=actors, columns=dates)


class RegimeSignal:
    """Convert regime probabilities to a signal matrix.

    Args:
        regime_probs: (T, M) array from FilteredState
        actors: List of actor identifiers
        dates: List of T timestamps
    """

    def __init__(
        self,
        regime_probs: np.ndarray,
        actors: list[str],
        dates: list[pd.Timestamp],
    ) -> None:
        self.regime_probs = np.asarray(regime_probs, dtype=float)
        self.actors = actors
        self.dates = dates

    def dominant_regime(self) -> np.ndarray:
        """Return (T,) integer array of argmax regime."""
        return np.argmax(self.regime_probs, axis=1)

    def to_signal_matrix(self) -> pd.DataFrame:
        """Return 0/1 matrix where 1 = high-investment regime.

        The "high-investment regime" is defined as argmax regime index being
        the last (highest) regime index.
        """
        T = len(self.dates)
        N = len(self.actors)
        dom = self.dominant_regime()  # (T,)
        M = self.regime_probs.shape[1]
        high_regime = M - 1  # highest index = high investment

        # Broadcast across all actors
        row = (dom == high_regime).astype(float)  # (T,)
        signals = np.tile(row, (N, 1))  # (N, T)

        return pd.DataFrame(signals, index=self.actors, columns=self.dates)


class CriticalitySignal:
    """Convert criticality index to a signal matrix.

    Args:
        criticality: (T,) array from phase_transition.criticality_index
        actors: List of actor identifiers
        dates: List of T timestamps
    """

    def __init__(
        self,
        criticality: np.ndarray,
        actors: list[str],
        dates: list[pd.Timestamp],
    ) -> None:
        self.criticality = np.asarray(criticality, dtype=float)
        self.actors = actors
        self.dates = dates

    def to_signal_matrix(self) -> pd.DataFrame:
        """Broadcast criticality across all actors, normalized to [-1, 1] via min-max."""
        c = self.criticality  # (T,)
        c_min = c.min()
        c_max = c.max()

        if c_max == c_min:
            normalized = np.zeros_like(c)
        else:
            # Min-max to [0, 1] then scale to [-1, 1]
            normalized = 2.0 * (c - c_min) / (c_max - c_min) - 1.0

        N = len(self.actors)
        signals = np.tile(normalized, (N, 1))  # (N, T)

        return pd.DataFrame(signals, index=self.actors, columns=self.dates)
