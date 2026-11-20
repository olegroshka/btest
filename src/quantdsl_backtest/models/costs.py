from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------
# Commission / cost models
# -----------------------------


@dataclass(slots=True)
class CostModel:
    """
    Base interface for commission/costs applied on orders.

    Implementations provide per-trade commission calculation and
    helpers to build per-bar fees fraction matrices for vectorized paths.
    """

    def commission_from_trade(self, *, qty: float, exec_price: float, shares: Optional[float] = None) -> float:
        """
        Return absolute commission dollars for a single executed trade.

        ``shares`` is the number of shares actually traded when ``qty`` is in units of an adjusted price
        (see ``Commission.share_price_field_close``); None means ``abs(qty)`` is the share count.
        """
        raise NotImplementedError

    # --- Vectorized helpers ---
    def build_fees_fraction_matrix_from_orders_numpy(
        self,
        *,
        prices_arr: np.ndarray,  # price matrix aligned with orders
        abs_size_arr: np.ndarray,  # absolute traded shares per bar/asset
    ) -> np.ndarray:
        """
        Build per-bar fraction of notional to apply as `fees` for vectorbt.
        Default implementation returns all zeros (no commission).
        """
        return np.zeros_like(prices_arr, dtype="float64")

    def build_fees_fraction_matrix_from_orders_pandas(
        self,
        *,
        prices: pd.DataFrame,
        abs_size: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype="float64")

    def vbt_single_pass_fees_frac(self) -> float:
        """
        Return a constant `fees` fraction suitable for vectorbt single-pass path
        when applicable. Defaults to 0.
        """
        return 0.0


@dataclass(slots=True)
class NoCommission(CostModel):
    def commission_from_trade(self, *, qty: float, exec_price: float, shares: Optional[float] = None) -> float:  # noqa: ARG002
        return 0.0


def _bounded(charge: float, notional: float, min_per_order: float, max_fraction: Optional[float]) -> float:
    """The charge raised to the order minimum, then capped at a fraction of the order's notional."""
    out = max(float(charge), float(min_per_order))
    if max_fraction is not None:
        out = min(out, float(max_fraction) * float(notional))
    return out


@dataclass(slots=True)
class PerShareCommission(CostModel):
    amount_per_share: float = 0.0
    min_per_order: float = 0.0
    max_fraction_of_notional: Optional[float] = None

    def commission_from_trade(self, *, qty: float, exec_price: float, shares: Optional[float] = None) -> float:
        n_shares = abs(float(qty)) if shares is None else abs(float(shares))
        charge = float(self.amount_per_share) * n_shares
        if self.min_per_order == 0.0 and self.max_fraction_of_notional is None:
            return charge
        notional = abs(float(qty)) * float(exec_price)
        if notional <= 0.0:
            return 0.0
        return _bounded(charge, notional, self.min_per_order, self.max_fraction_of_notional)

    def build_fees_fraction_matrix_from_orders_numpy(self, *, prices_arr: np.ndarray, abs_size_arr: np.ndarray) -> np.ndarray:
        # fraction = per_share / price, but only on bars with orders (abs_size>0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = self.amount_per_share / prices_arr
        frac[~np.isfinite(frac)] = 0.0
        mask = abs_size_arr > 0.0
        return np.where(mask, frac, 0.0).astype("float64", copy=False)

    def build_fees_fraction_matrix_from_orders_pandas(self, *, prices: pd.DataFrame, abs_size: pd.DataFrame) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = self.amount_per_share / prices.replace(0.0, np.nan)
        frac = frac.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
        return frac.where(abs_size > 0.0, 0.0)


@dataclass(slots=True)
class BpsNotionalCommission(CostModel):
    bps: float = 0.0
    min_per_order: float = 0.0
    max_fraction_of_notional: Optional[float] = None

    def commission_from_trade(self, *, qty: float, exec_price: float, shares: Optional[float] = None) -> float:  # noqa: ARG002
        notional = abs(float(qty)) * float(exec_price)
        if notional <= 0.0:
            return 0.0
        return _bounded((float(self.bps) / 1e4) * notional, notional, self.min_per_order, self.max_fraction_of_notional)

    def build_fees_fraction_matrix_from_orders_numpy(self, *, prices_arr: np.ndarray, abs_size_arr: np.ndarray) -> np.ndarray:
        # Constant fraction per order bar. We only apply on bars with orders.
        frac = float(self.bps) / 1e4
        mask = abs_size_arr > 0.0
        out = np.zeros_like(prices_arr, dtype="float64")
        out[mask] = frac
        return out

    def build_fees_fraction_matrix_from_orders_pandas(self, *, prices: pd.DataFrame, abs_size: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        frac = float(self.bps) / 1e4
        out = pd.DataFrame(0.0, index=abs_size.index, columns=abs_size.columns, dtype="float64")
        mask = abs_size > 0.0
        out[mask] = frac
        return out

    def vbt_single_pass_fees_frac(self) -> float:
        return float(self.bps) / 1e4


def build_cost_model(commission_cfg: Optional[object]) -> CostModel:
    """
    Factory: build a concrete CostModel from a DSL commission config
    (e.g., dsl.costs.Commission). If cfg is None, returns NoCommission.
    """
    if commission_cfg is None:
        return NoCommission()

    ctype = getattr(commission_cfg, "type", None)
    amount = float(getattr(commission_cfg, "amount", 0.0) or 0.0)
    min_per_order = float(getattr(commission_cfg, "min_per_order", 0.0) or 0.0)
    cap = getattr(commission_cfg, "max_fraction_of_notional", None)
    cap = None if cap is None else float(cap)
    if amount == 0.0 and min_per_order == 0.0:
        return NoCommission()
    if ctype == "per_share":
        return PerShareCommission(amount_per_share=amount, min_per_order=min_per_order, max_fraction_of_notional=cap)
    if ctype == "bps_notional":
        return BpsNotionalCommission(bps=amount, min_per_order=min_per_order, max_fraction_of_notional=cap)
    return NoCommission()
