"""GDELT 2.0 narrative intensity adapter for SMIM.

Loads pre-processed narrative signals from the canonical parquet files produced
by scripts/smim_fetch_gdelt.py.

Data pipeline:
    1. smim_fetch_gdelt.py downloads one representative GKG 2.0 CSV file per UTC
       day (slot nearest 12:00 UTC), parses it, and caches daily per-signal stats.
    2. It aggregates daily stats to a weekly panel using correct weighted formulas
       (see script docstring), writing:
         data/smim/processed/gdelt_narrative_daily.parquet  — daily panel
         data/smim/processed/gdelt_narrative.parquet         — weekly panel (canonical)
    3. This adapter reads those parquet files — it does NOT call the GDELT DOC API.

Why not DOC API:
    - Rate-limited to ~1 req/5s; common theme codes trigger 429s lasting hours.
    - Most GKG 2.0 V2EnhancedThemes codes return empty via DOC API even though
      they are present in the raw GKG files.

Available signals (theme_or_actor values):
    sector_energy, sector_technology, sector_financials, sector_healthcare,
    sector_macro, actor_FED, actor_SEC, actor_IMF, actor_BOE

GDELT data is append-only (no revisions).
pub_date = week_start + 7 days (conservative: data for week W complete by W+1 Mon).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantdsl_backtest.smim.config import GdeltConfig
from quantdsl_backtest.smim.interfaces import DateRange

# Resolve canonical data paths relative to the repo root.
# gdelt.py is at: src/quantdsl_backtest/smim/data/adapters/gdelt.py
# parents[5] = repo root
_REPO_ROOT = Path(__file__).parents[5]
_WEEKLY_PARQUET = _REPO_ROOT / "data" / "smim" / "processed" / "gdelt_narrative.parquet"
_DAILY_PARQUET  = _REPO_ROOT / "data" / "smim" / "processed" / "gdelt_narrative_daily.parquet"


@dataclass
class GdeltAdapter:
    """File-based GDELT adapter that reads from pre-processed narrative parquets.

    Reads the canonical weekly parquet produced by scripts/smim_fetch_gdelt.py.
    The weekly dataset is daily-derived and uses mathematically correct aggregation:
      - article_count = sum of daily matched article counts across the week
      - avg_tone      = article-count-weighted mean of daily tone values
      - intensity     = sum(daily_matched) / sum(daily_total_docs)

    ``series_ids`` in ``fetch()`` are SMIM signal IDs (e.g. ``"sector_energy"``,
    ``"actor_FED"``), matching the ``theme_or_actor`` column in the parquet.

    Args:
        config:       GdeltConfig from SmimConfig.data.gdelt (unused at fetch time;
                      kept for interface compatibility).
        parquet_path: Override path for the canonical weekly parquet (for testing).
        use_daily:    If True, read the daily parquet instead of the weekly one.
    """

    config: GdeltConfig = field(default_factory=GdeltConfig)
    parquet_path: Path | None = None
    use_daily: bool = False

    @property
    def source_name(self) -> str:
        return "gdelt"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch narrative intensity for the requested signals.

        Returns a wide DataFrame indexed by ``event_date`` (= ``week_start`` for
        weekly data, or ``event_date`` for daily data) with columns:
            ``{signal_id}_count``, ``{signal_id}_tone``, ``{signal_id}_intensity``

        Args:
            series_ids: SMIM signal IDs, or empty list to return all signals.
            date_range: Observation window (inclusive on both ends).
            as_of:      Upper bound on event_date for PIT compliance.
        """
        if self.use_daily:
            path       = self.parquet_path or _DAILY_PARQUET
            date_col   = "event_date"
        else:
            path       = self.parquet_path or _WEEKLY_PARQUET
            date_col   = "week_start"

        if not path.exists():
            raise FileNotFoundError(
                f"GDELT processed parquet not found: {path}\n"
                f"Run: uv run python scripts/smim_fetch_gdelt.py"
            )

        df = pd.read_parquet(path)

        effective_end = date_range.end
        if as_of is not None and as_of < effective_end:
            effective_end = as_of

        mask = (df[date_col] >= date_range.start) & (df[date_col] <= effective_end)
        df = df[mask].copy()

        if series_ids:
            df = df[df["theme_or_actor"].isin(series_ids)]

        if df.empty:
            return pd.DataFrame()

        # Pivot to wide format.
        wide: dict[pd.Timestamp, dict] = {}
        for _, row in df.iterrows():
            dt  = row[date_col]
            sig = row["theme_or_actor"]
            if dt not in wide:
                wide[dt] = {}
            wide[dt][f"{sig}_count"]     = row["article_count"]
            wide[dt][f"{sig}_tone"]      = row["avg_tone"]
            wide[dt][f"{sig}_intensity"] = row["intensity"]

        result = pd.DataFrame.from_dict(wide, orient="index")
        result.index.name = "event_date"
        result.sort_index(inplace=True)
        return result.astype("float64")
