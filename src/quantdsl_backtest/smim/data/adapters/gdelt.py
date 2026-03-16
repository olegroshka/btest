"""GDELT 2.0 narrative intensity adapter for SMIM.

Uses the GDELT DOC API v2 Timeline endpoint to retrieve article-count and
tone time series for configured CAMEO themes. Aggregates to daily frequency.

GDELT data is append-only — no revisions — so pub_date equals event_date
for the purpose of PIT compliance. The 15-minute publication lag is
negligible at daily aggregation.

Endpoint:
    https://api.gdeltproject.org/api/v2/doc/doc
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlencode

import pandas as pd

from quantdsl_backtest.smim.config import GdeltConfig
from quantdsl_backtest.smim.interfaces import DateRange

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_DT_FMT = "%Y%m%d%H%M%S"


def _ts_to_gdelt(ts: pd.Timestamp) -> str:
    return ts.strftime(_DT_FMT)


@dataclass
class GdeltAdapter:
    """GDELT DOC API v2 adapter for narrative theme intensity.

    Args:
        config: ``GdeltConfig`` from ``SmimConfig.data.gdelt``.
        http_client: Optional injectable HTTP client (for testing).
    """

    config: GdeltConfig = field(default_factory=GdeltConfig)
    http_client: object | None = None

    @property
    def source_name(self) -> str:
        return "gdelt"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch daily narrative intensity for given GDELT themes.

        ``series_ids`` are GDELT theme codes (e.g. ``["ECON_ENERGY"]``).
        If empty, uses ``config.themes``.

        Returns a wide DataFrame indexed by ``event_date`` with columns
        ``{theme}_count`` and ``{theme}_tone`` for each theme.

        Args:
            series_ids: GDELT theme codes, or empty list to use config themes.
            date_range: Observation window.
            as_of: Applied only as an upper-bound filter on ``event_date``
                since GDELT data is never revised (pub_date ≈ event_date).
        """
        themes = series_ids if series_ids else self.config.themes
        effective_end = date_range.end
        if as_of is not None and as_of < effective_end:
            effective_end = as_of

        client = self._get_client()
        frames: list[pd.DataFrame] = []

        for theme in themes:
            vol_df = self._fetch_timeline(
                client, theme, date_range.start, effective_end, mode="timelinevol"
            )
            tone_df = self._fetch_timeline(
                client, theme, date_range.start, effective_end, mode="timelinecovtone"
            )
            if vol_df is not None:
                vol_df = vol_df.rename(columns={"value": f"{theme}_count"})
            if tone_df is not None:
                tone_df = tone_df.rename(columns={"value": f"{theme}_tone"})
            parts = [df for df in [vol_df, tone_df] if df is not None]
            if not parts:
                continue
            merged = pd.concat(parts, axis=1)
            if not merged.empty:
                frames.append(merged)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result.index.name = "event_date"
        return result.astype("float64")

    def _fetch_timeline(
        self,
        client: object,
        theme: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        mode: str,
    ) -> pd.DataFrame | None:
        params = {
            "query": f"theme:{theme}",
            "mode": mode,
            "format": "json",
            "startdatetime": _ts_to_gdelt(start),
            "enddatetime": _ts_to_gdelt(end),
            "timelinesmooth": "5",
        }
        url = f"{_BASE_URL}?{urlencode(params)}"
        resp = client.get(url)  # type: ignore[attr-defined]
        if resp.status_code != 200:
            return None
        data = resp.json()
        timeline = data.get("timeline") or []
        if not timeline:
            return None

        # The timeline list may be a list of dicts or a list of series
        # GDELT returns: [{"series": [{"date": "...", "value": ...}]}]
        # or directly: [{"date": "...", "value": ...}]
        if isinstance(timeline[0], dict) and "series" in timeline[0]:
            rows = timeline[0]["series"]
        else:
            rows = timeline

        records = []
        for row in rows:
            dt_str = row.get("date", "")
            val = row.get("value")
            if dt_str and val is not None:
                # GDELT dates: YYYYMMDDHHMMSS
                try:
                    dt = pd.to_datetime(dt_str, format=_DT_FMT)
                except ValueError:
                    dt = pd.to_datetime(dt_str)
                records.append({"event_date": dt.normalize(), "value": float(val)})

        if not records:
            return None

        df = pd.DataFrame(records).set_index("event_date")
        # Aggregate to daily (sum for volume, mean for tone)
        agg = "sum" if mode == "timelinevol" else "mean"
        return df.resample("D").agg(agg)  # type: ignore[return-value]

    def _get_client(self) -> object:
        if self.http_client is not None:
            return self.http_client
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for GdeltAdapter. Install with: pip install httpx"
            ) from exc
        return httpx.Client(timeout=60.0)
