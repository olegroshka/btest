from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ...dsl.universe import UniverseFilter, HasHistory, MinPrice, MinDollarADV


@dataclass(slots=True)
class UniverseFilterEngine:
    """Evaluate UniverseFilter objects against a single instrument's timeseries."""

    def passes(self, df_instr: pd.DataFrame, filters: List[UniverseFilter]) -> bool:
        close = df_instr["close"] if "close" in df_instr.columns else None
        volume = df_instr["volume"] if "volume" in df_instr.columns else None

        for f in filters:
            if isinstance(f, HasHistory):
                if len(df_instr) < f.min_days:
                    return False

            elif isinstance(f, MinPrice):
                if close is None:
                    return False
                if close.min() < f.min_price:
                    return False

            elif isinstance(f, MinDollarADV):
                if close is None or volume is None:
                    return False
                dollar_adv = (close * volume).mean()
                if dollar_adv < f.min_dollar_adv:
                    return False

        return True

