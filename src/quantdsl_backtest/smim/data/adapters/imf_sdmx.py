"""IMF SDMX-JSON adapter for SMIM.

Fetches from the IMF SDMX REST API (CompactData endpoint). Parses the
SDMX-JSON response and returns a tidy DataFrame.

Endpoint:
    http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{database}/{key}

IMF data is not vintage-archived, so A1 compliance is enforced via a
conservative publication-lag buffer (configured in SmimConfig.data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantdsl_backtest.smim.config import ImfConfig
from quantdsl_backtest.smim.interfaces import DateRange

_BASE_URL = "http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"


def _parse_sdmx_json(data: dict) -> list[dict]:
    """Parse IMF SDMX-JSON CompactData response into a flat list of records."""
    records: list[dict] = []
    try:
        dataset = data["CompactData"]["DataSet"]
    except (KeyError, TypeError):
        return records

    series_list = dataset.get("Series") or []
    if isinstance(series_list, dict):
        series_list = [series_list]

    for series in series_list:
        meta = {k.lstrip("@"): v for k, v in series.items() if k.startswith("@")}
        obs_list = series.get("Obs") or []
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        for obs in obs_list:
            period = obs.get("@TIME_PERIOD")
            value = obs.get("@OBS_VALUE")
            if period is None or value is None:
                continue
            record = {**meta, "TIME_PERIOD": period, "value": value}
            records.append(record)
    return records


def _parse_period(period: str) -> pd.Timestamp | None:
    """Parse IMF period strings (e.g. '2005-Q1', '2005-01', '2005') to Timestamp."""
    period = period.strip()
    try:
        if "Q" in period:
            year, q = period.split("-Q")
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(f"{year}-{month:02d}-01")
        return pd.to_datetime(period)
    except Exception:
        return None


@dataclass
class ImfSdmxAdapter:
    """IMF SDMX REST adapter.

    Args:
        config: ``ImfConfig`` from ``SmimConfig.data.imf``.
        http_client: Optional injectable HTTP client (for testing).
    """

    config: ImfConfig = field(default_factory=ImfConfig)
    http_client: object | None = None

    @property
    def source_name(self) -> str:
        return "imf"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch IMF series.

        ``series_ids`` are full SDMX dimension keys, e.g.:
        ``["M.US.PCPI_IX", "Q.GB.NGDP_R_K_IX"]``

        Format: ``{frequency}.{ref_area}.{indicator}``

        Returns wide DataFrame indexed by ``event_date``, one column per series_id.

        Args:
            series_ids: SDMX keys in the form ``freq.country.indicator``.
            date_range: Date range for filtering.
            as_of: Upper-bound filter on event_date (conservative A1 proxy).
        """
        client = self._get_client()
        effective_end = date_range.end
        if as_of is not None and as_of < effective_end:
            effective_end = as_of

        frames: dict[str, pd.Series] = {}
        for sid in series_ids:
            parts = sid.split(".")
            if len(parts) < 3:
                continue
            freq, ref_area, indicator = parts[0], parts[1], ".".join(parts[2:])
            dataset = self.config.datasets[0] if self.config.datasets else "IFS"
            key = f"{freq}.{ref_area}.{indicator}"
            url = f"{_BASE_URL}/{dataset}/{key}?startPeriod={date_range.start.strftime('%Y-%m-%d')}&endPeriod={effective_end.strftime('%Y-%m-%d')}"

            resp = client.get(url, headers={"Accept": "application/json"})  # type: ignore[attr-defined]
            if resp.status_code != 200:
                continue
            records = _parse_sdmx_json(resp.json())
            if not records:
                continue

            series_data: dict[pd.Timestamp, float] = {}
            for rec in records:
                ts = _parse_period(rec["TIME_PERIOD"])
                if ts is None:
                    continue
                try:
                    series_data[ts] = float(rec["value"])
                except (ValueError, TypeError):
                    pass

            s = pd.Series(series_data, name=sid, dtype="float64")
            s.index = pd.to_datetime(s.index)
            frames[sid] = s

        if not frames:
            df = pd.DataFrame(index=pd.DatetimeIndex([], name="event_date"))
            return df

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "event_date"
        df = df.sort_index()
        return df

    def _get_client(self) -> object:
        if self.http_client is not None:
            return self.http_client
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for ImfSdmxAdapter. Install with: pip install httpx"
            ) from exc
        return httpx.Client(timeout=60.0)
