from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class SelectionTraceCollector:
    rows: list[dict] = field(default_factory=list)

    def record_selection(
        self,
        *,
        dt, sig_date,
        long_names, short_names,
        long_rank_row, short_rank_row,
        long_mask, short_mask,
    ):
        for sym in long_names:
            self.rows.append({
                "dt": dt, "sig_date": sig_date, "book": "long",
                "instrument": sym, "selected": True,
                "score": float(long_rank_row.get(sym)) if sym in long_rank_row.index else None,
                "mask_passed": bool(long_mask.get(sym)) if sym in long_mask.index else None,
            })
        for sym in short_names:
            self.rows.append({
                "dt": dt, "sig_date": sig_date, "book": "short",
                "instrument": sym, "selected": True,
                "score": float(short_rank_row.get(sym)) if sym in short_rank_row.index else None,
                "mask_passed": bool(short_mask.get(sym)) if sym in short_mask.index else None,
            })

    def finalize(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
