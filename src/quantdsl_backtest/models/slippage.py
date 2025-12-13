from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SlippageModel:
    """
    Base slippage model interface.

    Implementations should provide methods to compute slippage in basis points
    for a single order and helpers for vectorized per-bar computations.
    """

    def slippage_bps_from_participation(self, participation: float) -> float:
        raise NotImplementedError

    def slippage_bps_from_order(self, *, qty: float, volume: float) -> float:
        part = 0.0
        if volume > 0:
            part = min(1.0, abs(qty) / float(volume))
        return self.slippage_bps_from_participation(part)

    def slippage_frac_from_order(self, *, qty: float, volume: float) -> float:
        return self.slippage_bps_from_order(qty=qty, volume=volume) / 1e4

    def build_slippage_fraction_matrix_from_orders_numpy(
        self,
        *,
        net_size_arr: np.ndarray,  # signed shares traded per bar/asset
        abs_size_arr: np.ndarray,  # total absolute shares traded per bar/asset (sum of legs)
        volumes_arr: Optional[np.ndarray],  # shares traded in the bar
    ) -> np.ndarray:
        """
        Compute per-bar per-asset slippage as a fraction of price to apply on traded notional,
        using numpy arrays. Mirrors the event-driven model by:
          - computing participation on net shares
          - converting to bps via the model
          - scaling by |net|/abs (so multiple legs in a bar don't double-count)
        """
        abs_net = np.abs(net_size_arr).astype(float, copy=False)
        abs_tot = np.abs(abs_size_arr).astype(float, copy=False)

        if volumes_arr is None:
            part_net = np.zeros_like(abs_net)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                part_net = np.divide(abs_net, volumes_arr)
            part_net[~np.isfinite(part_net)] = 0.0
            np.clip(part_net, 0.0, 1.0, out=part_net)

        slip_bps = self._bps_from_participation_array(part_net)

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.divide(abs_net, abs_tot)
        scale[~np.isfinite(scale)] = 0.0
        np.clip(scale, 0.0, 1.0, out=scale)

        slip_frac = (slip_bps / 1e4) * scale

        mask = abs_tot > 0.0
        slip_frac = np.where(mask, slip_frac, 0.0)
        return slip_frac.astype("float64", copy=False)

    def build_slippage_fraction_matrix_from_orders_pandas(
        self,
        *,
        net_size: pd.DataFrame,
        abs_size: pd.DataFrame,
        volumes: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if volumes is None:
            part_net = pd.DataFrame(0.0, index=net_size.index, columns=net_size.columns, dtype="float64")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                part_net = (np.abs(net_size)) / volumes.replace(0.0, np.nan)
            part_net = part_net.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)

        slip_bps = self._bps_from_participation_df(part_net)

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (np.abs(net_size)) / abs_size.replace(0.0, np.nan)
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
        return ((slip_bps / 1e4) * scale).astype("float64").where(abs_size > 0.0, 0.0)

    # --- Helpers for vectorized bps computations ---
    def _bps_from_participation_array(self, part: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _bps_from_participation_df(self, part: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(slots=True)
class NoSlippage(SlippageModel):
    def slippage_bps_from_participation(self, participation: float) -> float:  # noqa: ARG002
        return 0.0

    def _bps_from_participation_array(self, part: np.ndarray) -> np.ndarray:
        return np.zeros_like(part, dtype="float64")

    def _bps_from_participation_df(self, part: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame(0.0, index=part.index, columns=part.columns, dtype="float64")


@dataclass(slots=True)
class PowerLawSlippage(SlippageModel):
    base_bps: float = 0.0
    k: float = 0.0
    exponent: float = 1.0

    def is_active(self) -> bool:
        return (self.base_bps != 0.0) or (self.k != 0.0)

    def slippage_bps_from_participation(self, participation: float) -> float:
        participation = float(np.clip(participation, 0.0, 1.0))
        return float(self.base_bps + self.k * (participation ** self.exponent))

    def _bps_from_participation_array(self, part: np.ndarray) -> np.ndarray:
        return (self.base_bps + self.k * np.power(part, self.exponent)).astype("float64", copy=False)

    def _bps_from_participation_df(self, part: pd.DataFrame) -> pd.DataFrame:
        return (self.base_bps + self.k * np.power(part, self.exponent)).astype("float64")


def build_slippage_model(cfg: Optional[object]) -> SlippageModel:
    """
    Factory: build a concrete SlippageModel from a DSL config
    (e.g., dsl.execution.PowerLawSlippageModel). If cfg is None, returns NoSlippage.
    """
    if cfg is None:
        return NoSlippage()

    base_bps = float(getattr(cfg, "base_bps", 0.0) or 0.0)
    k = float(getattr(cfg, "k", 0.0) or 0.0)
    exponent = float(getattr(cfg, "exponent", 1.0) or 1.0)
    return PowerLawSlippage(base_bps=base_bps, k=k, exponent=exponent)
