from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


@dataclass(slots=True)
class VolumeLimitModel:
    """
    Interface for volume participation and min-fill rules.
    Works in notional space to decouple from unit calculations.
    """

    max_participation: float = 1.0
    mode: Literal["proportional", "clip"] = "proportional"
    min_fill_notional: float = 0.0

    def cap_notional_change(self, *, dn: float, price: float, volume: float) -> float:
        """
        Cap desired notional change `dn` based on participation of `volume * price`.
        If unlimited or invalid inputs, returns dn.
        For now 'proportional' and 'clip' behave the same at cap-level: simple clipping.
        """
        mp = float(self.max_participation)
        if (not np.isfinite(price)) or price <= 0 or (not np.isfinite(volume)) or volume <= 0:
            return dn
        if (not np.isfinite(mp)) or mp is None or mp >= 1.0 or mp <= 0.0:
            # mp<=0 considered unlimited (no fills) -> return original and min_fill will handle
            return dn
        max_notional = mp * float(volume) * float(price)
        if max_notional <= 0:
            return 0.0
        if abs(dn) <= max_notional:
            return dn
        return np.sign(dn) * max_notional

    def passes_min_fill(self, abs_notional: float) -> bool:
        return float(abs_notional) >= float(self.min_fill_notional)


def build_volume_limit_model(cfg: Optional[object]) -> VolumeLimitModel:
    if cfg is None:
        return VolumeLimitModel(max_participation=1.0, mode="proportional", min_fill_notional=0.0)
    mp = float(getattr(cfg, "max_participation", 1.0) or 1.0)
    mode = getattr(cfg, "mode", "proportional")
    mfn = float(getattr(cfg, "min_fill_notional", 0.0) or 0.0)
    return VolumeLimitModel(max_participation=mp, mode=mode, min_fill_notional=mfn)
