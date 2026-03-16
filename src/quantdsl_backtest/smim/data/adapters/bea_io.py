"""BEA Input-Output table adapter for SMIM.

Fetches annual I/O use-table coefficients from the BEA JSON API.
Returns a tidy DataFrame suitable for supply-chain edge estimation (channel C5).

The BEA I/O tables have an 18-month publication lag (see data_source_audit.md).
A1 compliance is enforced by treating the pub_date as reference_year_end + 548 days.

Endpoint:
    https://apps.bea.gov/api/data?UserID={KEY}&method=GetData&DataSetName=InputOutput
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlencode

import pandas as pd

from quantdsl_backtest.smim.config import BeaConfig
from quantdsl_backtest.smim.interfaces import DateRange

_BASE_URL = "https://apps.bea.gov/api/data"
_PUB_LAG_DAYS = 548  # ~18 months


def _load_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"BEA API key not found. Set '{env_var}' environment variable. "
            "Get a free key at https://apps.bea.gov/API/signup/"
        )
    return key


def _parse_bea_response(data: dict) -> list[dict]:
    """Parse BEA JSON response Data array into flat records."""
    records: list[dict] = []
    try:
        rows = data["BEAAPI"]["Results"]["Data"]
    except (KeyError, TypeError):
        return records

    if isinstance(rows, dict):
        rows = [rows]

    for row in rows:
        try:
            records.append({
                "year": int(row.get("Year", 0)),
                "source_industry": row.get("RowCode", ""),
                "source_desc": row.get("RowDescription", ""),
                "target_industry": row.get("ColCode", ""),
                "target_desc": row.get("ColDescription", ""),
                "coefficient": float(
                    str(row.get("DataValue", "0")).replace(",", "") or "0"
                ),
                "table_id": row.get("TableID", ""),
            })
        except (ValueError, TypeError):
            continue
    return records


@dataclass
class BeaIoAdapter:
    """BEA Input-Output table adapter.

    ``series_ids`` for this adapter are table IDs (e.g. ``["259"]`` for the
    use-table before redefinitions).

    Args:
        config: ``BeaConfig`` from ``SmimConfig.data.bea``.
        http_client: Optional injectable HTTP client (for testing).
    """

    config: BeaConfig = field(default_factory=BeaConfig)
    http_client: object | None = None

    @property
    def source_name(self) -> str:
        return "bea"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch I/O coefficient tables for all years in the date range.

        Returns a long-format DataFrame with columns:
        ``year``, ``source_industry``, ``source_desc``, ``target_industry``,
        ``target_desc``, ``coefficient``, ``table_id``, ``pub_date``.

        ``pub_date`` is imputed as year-end + 548 days (A1 conservative buffer).
        If ``as_of`` is provided, only rows with ``pub_date <= as_of`` are returned.

        Args:
            series_ids: BEA table IDs (defaults to ``config.table_ids``).
            date_range: Used to determine which years to fetch.
            as_of: Point-in-time filter.
        """
        api_key = _load_api_key(self.config.api_key_env)
        table_ids = series_ids if series_ids else self.config.table_ids
        client = self._get_client()

        start_year = max(date_range.start.year, self.config.year_range[0])
        end_year = min(date_range.end.year, self.config.year_range[1])
        years = list(range(start_year, end_year + 1))

        all_records: list[dict] = []
        for table_id in table_ids:
            params = {
                "UserID": api_key,
                "method": "GetData",
                "DataSetName": "InputOutput",
                "TableID": table_id,
                "Year": ",".join(str(y) for y in years),
                "ResultFormat": "JSON",
            }
            url = f"{_BASE_URL}?{urlencode(params)}"
            resp = client.get(url)  # type: ignore[attr-defined]
            resp.raise_for_status()
            records = _parse_bea_response(resp.json())
            all_records.extend(records)

        if not all_records:
            return pd.DataFrame(
                columns=[
                    "year", "source_industry", "source_desc",
                    "target_industry", "target_desc", "coefficient",
                    "table_id", "pub_date",
                ]
            )

        df = pd.DataFrame(all_records)
        # Impute pub_date: year-end + 548 days
        df["pub_date"] = pd.to_datetime(
            df["year"].astype(str) + "-12-31"
        ) + pd.Timedelta(days=_PUB_LAG_DAYS)

        if as_of is not None:
            as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
            df = df[df["pub_date"] <= as_of_naive]

        return df.reset_index(drop=True)

    def _get_client(self) -> object:
        if self.http_client is not None:
            return self.http_client
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for BeaIoAdapter. Install with: pip install httpx"
            ) from exc
        return httpx.Client(timeout=60.0)
