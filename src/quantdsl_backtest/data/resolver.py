from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..dsl.data_config import DataConfig
from .requests import DataRequest


YF_PREFIX = "yf://"
FRED_PREFIX = "fred://"
PARQUET_PREFIX = "parquet://"


def resolve_source(source: str) -> str:
    """Canonicalize a user-provided `source` string.

    Today this mainly expands known aliases (e.g. FINRA) and normalizes schemes.
    """
    src = (source or "").strip()
    if not src:
        return src

    # FINRA convenience mapping routed via FRED IDs
    if src.upper().startswith("FINRA:"):
        mapped = map_finra_to_fred(src)
        if mapped is None:
            raise ValueError(
                f"Unknown FINRA source {src!r}. Supported examples: 'FINRA:HY_OAS', 'FINRA:IG_OAS'."
            )
        return mapped

    # Keep existing schemes as-is; normalize casing for known ones.
    if src.lower().startswith(FRED_PREFIX):
        return FRED_PREFIX + src[len(FRED_PREFIX):].strip()
    if src.startswith(YF_PREFIX):
        return src
    if src.startswith(PARQUET_PREFIX):
        return src

    return src


def map_finra_to_fred(source: str) -> Optional[str]:
    alias = source.split(":", 1)[1].strip().upper()
    mapping = {
        "HY_OAS": f"{FRED_PREFIX}BAMLH0A0HYM2",
        "IG_OAS": f"{FRED_PREFIX}BAMLC0A0CM",
    }
    return mapping.get(alias)


@dataclass(frozen=True, slots=True)
class DataConfigAdapter:
    """Maps existing DSL DataConfig into a provider-agnostic DataRequest."""

    @staticmethod
    def to_request(data_cfg: DataConfig, kind: str = "market_bars") -> DataRequest:
        src = resolve_source(data_cfg.source)
        return DataRequest(
            source=src,
            kind=kind,
            start=data_cfg.start,
            end=data_cfg.end,
            frequency=data_cfg.frequency,
            fields=list(data_cfg.fields or []),
            calendar=data_cfg.calendar,
            tz=data_cfg.tz,
            dataset_id=None,
        )

