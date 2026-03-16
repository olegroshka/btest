"""SEC EDGAR XBRL adapter for SMIM.

Fetches company-level XBRL facts (CapEx, Assets, etc.) from the SEC EDGAR
JSON API. Uses the ``filed`` date from EDGAR as ``pub_date`` — this is the
exact date the data became publicly available, enabling strict A1 compliance.

Endpoint used:
    https://data.sec.gov/api/xbrl/companyfacts/CIK{zero-padded-10}.json

Rate limit: SEC requires max 10 requests/second and a descriptive User-Agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from quantdsl_backtest.smim.config import EdgarConfig
from quantdsl_backtest.smim.interfaces import DateRange

_USER_AGENT = "SMIM Research smim@research.example.com"
_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
_MIN_REQUEST_INTERVAL = 0.11  # ~9 req/s, safely under the 10 req/s limit


def _zero_pad_cik(cik: str) -> str:
    return cik.strip().zfill(10)


@dataclass
class EdgarAdapter:
    """SEC EDGAR XBRL adapter.

    Args:
        config: ``EdgarConfig`` from ``SmimConfig.data.edgar``.
        http_client: Optional injectable HTTP client (for testing). If None,
            uses ``httpx.Client`` with the required User-Agent header.
    """

    config: EdgarConfig = field(default_factory=EdgarConfig)
    http_client: object | None = None  # httpx.Client | None

    @property
    def source_name(self) -> str:
        return "edgar"

    def fetch(
        self,
        series_ids: list[str],
        date_range: DateRange,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch XBRL facts for a list of CIK numbers.

        Returns a long-format DataFrame with columns:
        ``actor_id``, ``tag``, ``event_date``, ``pub_date``, ``value``

        where ``pub_date`` is the EDGAR filing date (A1-compliant).
        If ``as_of`` is provided, only observations with
        ``pub_date <= as_of`` are returned.

        Args:
            series_ids: CIK numbers as strings (e.g. ``["0000034088"]``).
            date_range: Date range for filtering by ``event_date``.
            as_of: If provided, filter to filings available on or before
                this date (A1 compliance).

        Returns:
            Long-format DataFrame with columns:
            ``actor_id``, ``tag``, ``event_date``, ``pub_date``, ``value``.
        """
        client = self._get_client()
        records: list[dict] = []

        for cik in series_ids:
            padded = _zero_pad_cik(cik)
            url = f"{_BASE_URL}/CIK{padded}.json"
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            self._parse_facts(data, cik, records)
            time.sleep(_MIN_REQUEST_INTERVAL)

        if not records:
            return pd.DataFrame(
                columns=["actor_id", "tag", "event_date", "pub_date", "value"]
            )

        df = pd.DataFrame(records)
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.tz_localize(None)
        df["pub_date"] = pd.to_datetime(df["pub_date"]).dt.tz_localize(None)
        df["value"] = df["value"].astype("float64")

        # Filter by date_range event_date
        df = df[
            (df["event_date"] >= date_range.start)
            & (df["event_date"] <= date_range.end)
        ]

        # A1: filter by pub_date <= as_of
        if as_of is not None:
            as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
            df = df[df["pub_date"] <= as_of_naive]

        return df.reset_index(drop=True)

    def _parse_facts(
        self, data: dict, cik: str, records: list[dict]
    ) -> None:
        """Extract configured XBRL tags from a company-facts JSON response."""
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        for tag in self.config.xbrl_tags:
            tag_data = us_gaap.get(tag, {})
            units = tag_data.get("units", {})
            # Most financial tags use USD; some use shares
            for unit_values in units.values():
                for obs in unit_values:
                    form = obs.get("form", "")
                    if form not in self.config.filing_types:
                        continue
                    end_date = obs.get("end")
                    filed_date = obs.get("filed")
                    val = obs.get("val")
                    if end_date and filed_date and val is not None:
                        records.append({
                            "actor_id": cik,
                            "tag": tag,
                            "event_date": end_date,
                            "pub_date": filed_date,  # A1: filing date = pub_date
                            "value": float(val),
                        })

    def _get_client(self) -> object:
        if self.http_client is not None:
            return self.http_client
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for EdgarAdapter. Install with: pip install httpx"
            ) from exc
        return httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=30.0,
        )
