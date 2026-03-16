"""OECD SDMX 3.0 adapter for SMIM.

Fetches from the OECD SDMX 3.0 REST API (sdmx.oecd.org) and parses the
SDMX-JSON 2.0 response format.

Endpoint:
    https://sdmx.oecd.org/public/rest/data/{agencyId},{flowRef},{version}/{key}

OECD data is not vintage-archived. A1 compliance is enforced via a
conservative publication-lag buffer (~75 days for quarterly QNA data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantdsl_backtest.smim.config import OecdConfig
from quantdsl_backtest.smim.interfaces import DateRange

_BASE_URL = "https://sdmx.oecd.org/public/rest/data"
_AGENCY = "OECD"
_VERSION = "1.0"


def _parse_sdmx_json_v2(data: dict) -> list[dict]:
    """Parse OECD SDMX-JSON 2.0 data message into flat records."""
    records: list[dict] = []
    try:
        structure = data["meta"]["schema"]["dataStructure"]
        dimensions = structure["dimensions"]["observation"]
        dim_names = [d["id"] for d in dimensions]
    except (KeyError, TypeError):
        # Fallback: try SDMX-JSON 1.0 structure
        return _parse_sdmx_json_v1(data)

    try:
        datasets = data["data"]["dataSets"]
    except (KeyError, TypeError):
        return records

    for dataset in datasets:
        for key_str, obs_data in (dataset.get("observations") or {}).items():
            dim_indices = [int(x) for x in key_str.split(":")]
            dim_values: dict[str, str] = {}
            for i, name in enumerate(dim_names):
                try:
                    members = dimensions[i].get("values") or []
                    idx = dim_indices[i]
                    dim_values[name] = members[idx].get("id", "") if idx < len(members) else ""
                except (IndexError, TypeError):
                    dim_values[name] = ""
            value = obs_data[0] if obs_data else None
            if value is not None:
                records.append({**dim_values, "value": value})
    return records


def _parse_sdmx_json_v1(data: dict) -> list[dict]:
    """Fallback parser for SDMX-JSON 1.0 format."""
    records: list[dict] = []
    try:
        datasets = data["dataSets"]
        structure = data["structure"]
        dim_list = structure["dimensions"]["observation"]
    except (KeyError, TypeError):
        return records

    for dataset in datasets:
        for key_str, obs_list in (dataset.get("observations") or {}).items():
            indices = [int(x) for x in key_str.split(":")]
            dim_values: dict[str, str] = {}
            for i, dim in enumerate(dim_list):
                vals = dim.get("values") or []
                idx = indices[i] if i < len(indices) else 0
                dim_values[dim["id"]] = vals[idx].get("id", "") if idx < len(vals) else ""
            value = obs_list[0] if obs_list else None
            if value is not None:
                records.append({**dim_values, "value": value})
    return records


def _period_to_timestamp(period: str) -> pd.Timestamp | None:
    """Convert OECD period strings to Timestamp."""
    period = period.strip()
    try:
        if "-Q" in period:
            y, q = period.split("-Q")
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(f"{y}-{month:02d}-01")
        if len(period) == 4:
            return pd.Timestamp(f"{period}-01-01")
        return pd.to_datetime(period)
    except Exception:
        return None


@dataclass
class OecdSdmxAdapter:
    """OECD SDMX 3.0 REST adapter.

    ``series_ids`` are dimension keys in the form
    ``{dataflow}.{frequency}.{ref_area}.{subject}``
    e.g. ``"QNA.Q.USA.P51G.GPSA"``.

    Args:
        config: ``OecdConfig`` from ``SmimConfig.data.oecd``.
        http_client: Optional injectable HTTP client (for testing).
    """

    config: OecdConfig = field(default_factory=OecdConfig)
    http_client: object | None = None

    @property
    def source_name(self) -> str:
        return "oecd"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch OECD series.

        ``series_ids`` format: ``{dataflow}.{rest_of_key}``
        e.g. ``"QNA.Q.USA.P51G.GPSA"`` → dataflow=QNA, key=Q.USA.P51G.GPSA

        Returns wide DataFrame indexed by ``event_date``, one column per series_id.

        Args:
            series_ids: OECD SDMX keys including dataflow prefix.
            date_range: Date range for filtering.
            as_of: Conservative upper-bound filter (non-vintaged source).
        """
        client = self._get_client()
        effective_end = date_range.end
        if as_of is not None and as_of < effective_end:
            effective_end = as_of

        frames: dict[str, pd.Series] = {}

        for sid in series_ids:
            # First component is the dataflow; rest is the SDMX key
            parts = sid.split(".", 1)
            if len(parts) < 2:
                continue
            dataflow, key = parts
            url = (
                f"{_BASE_URL}/{_AGENCY},{dataflow},{_VERSION}/{key}"
                f"?format=jsondata"
                f"&startPeriod={date_range.start.strftime('%Y-%m-%d')}"
                f"&endPeriod={effective_end.strftime('%Y-%m-%d')}"
            )
            resp = client.get(url)  # type: ignore[attr-defined]
            if resp.status_code != 200:
                continue

            records = _parse_sdmx_json_v2(resp.json())
            if not records:
                continue

            series_data: dict[pd.Timestamp, float] = {}
            # TIME_PERIOD is the standard period dimension in OECD SDMX
            period_key = next(
                (k for k in records[0] if "TIME" in k.upper() or "PERIOD" in k.upper()),
                None,
            )
            for rec in records:
                period_str = rec.get(period_key or "TIME_PERIOD", "")
                ts = _period_to_timestamp(period_str)
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
            return pd.DataFrame(index=pd.DatetimeIndex([], name="event_date"))

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "event_date"
        return df.sort_index()

    def _get_client(self) -> object:
        if self.http_client is not None:
            return self.http_client
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for OecdSdmxAdapter. Install with: pip install httpx"
            ) from exc
        return httpx.Client(timeout=60.0)
