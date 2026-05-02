"""Optional bridge helpers for turning SMIM-style research outputs into btest signals.

These adapters live in the DSL package because they are btest-owned integration
code, not part of the standalone SMIM scientific core.

The helpers intentionally avoid a hard runtime dependency on the standalone
`smim` package: they operate on duck-typed objects with the expected
attributes (`gaps`, optional `actors`, optional `dates`, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


class SupportsGapResult(Protocol):
    """Structural protocol for SMIM gap-like results used by the bridge."""

    gaps: Any
    benchmark_class: Any
    benchmarks: Any


@dataclass(slots=True)
class GapSignal:
    """Convert a SMIM-style gap result into a trading signal matrix.

    Args:
        gap_result: Any object exposing a `gaps` array with shape `(N, T)` and
            optionally `actors` / `dates` metadata.
        threshold: Fraction of per-actor standard deviation used as the signal
            trigger threshold.
        scale: If True, scale signals proportionally to gap magnitude and clip
            into `[-1, 1]`; otherwise produce ternary `{-1, 0, 1}` signals.
    """

    gap_result: SupportsGapResult | Any
    threshold: float = 0.5
    scale: bool = True

    def to_signal_matrix(self) -> pd.DataFrame:
        """Convert gaps to a `(actors, dates)` signal matrix."""
        gaps = np.asarray(self.gap_result.gaps, dtype=float)
        n_actors, n_dates = gaps.shape

        std = np.std(gaps, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        threshold_vals = self.threshold * std

        signals = np.zeros_like(gaps)
        over = gaps > threshold_vals
        under = gaps < -threshold_vals

        if self.scale:
            repeated_std = std[:, 0:1].repeat(n_dates, axis=1)
            signals[over] = -(gaps[over] / repeated_std[over])
            signals[under] = -(gaps[under] / repeated_std[under])
            signals = np.clip(signals, -1.0, 1.0)
        else:
            signals[over] = -1.0
            signals[under] = 1.0

        actors = getattr(self.gap_result, "actors", None) or [str(i) for i in range(n_actors)]
        dates = getattr(self.gap_result, "dates", None) or list(range(n_dates))
        return pd.DataFrame(signals, index=actors, columns=dates)


@dataclass(slots=True)
class RegimeSignal:
    """Convert regime probabilities into a broadcast signal matrix."""

    regime_probs: np.ndarray
    actors: Sequence[str]
    dates: Sequence[pd.Timestamp]

    def __post_init__(self) -> None:
        self.regime_probs = np.asarray(self.regime_probs, dtype=float)

    def dominant_regime(self) -> np.ndarray:
        """Return the dominant regime index for each date."""
        return np.argmax(self.regime_probs, axis=1)

    def to_signal_matrix(self) -> pd.DataFrame:
        """Return a `0/1` matrix where `1` denotes the highest-index regime."""
        n_dates = len(self.dates)
        n_actors = len(self.actors)
        dominant = self.dominant_regime()
        high_regime = self.regime_probs.shape[1] - 1
        row = (dominant == high_regime).astype(float)
        signals = np.tile(row, (n_actors, 1))
        return pd.DataFrame(signals, index=list(self.actors), columns=list(self.dates[:n_dates]))


@dataclass(slots=True)
class CriticalitySignal:
    """Convert a criticality time series into a broadcast signal matrix."""

    criticality: np.ndarray
    actors: Sequence[str]
    dates: Sequence[pd.Timestamp]

    def __post_init__(self) -> None:
        self.criticality = np.asarray(self.criticality, dtype=float)

    def to_signal_matrix(self) -> pd.DataFrame:
        """Broadcast a normalized criticality series across all actors."""
        criticality = self.criticality
        c_min = criticality.min()
        c_max = criticality.max()

        if c_max == c_min:
            normalized = np.zeros_like(criticality)
        else:
            normalized = 2.0 * (criticality - c_min) / (c_max - c_min) - 1.0

        signals = np.tile(normalized, (len(self.actors), 1))
        return pd.DataFrame(signals, index=list(self.actors), columns=list(self.dates))


__all__ = ["CriticalitySignal", "GapSignal", "RegimeSignal", "SupportsGapResult"]

