from __future__ import annotations

from dataclasses import replace

from quantdsl_backtest.data.requests import DataRequest


def normalize_kind(kind: str) -> str:
    return (kind or "").strip().lower()


def normalize_frequency(freq: str) -> str:
    return (freq or "").strip().lower()


def validate_kind(kind: str) -> str:
    k = normalize_kind(kind)
    allowed = {"market_bars", "time_series"}
    if k not in allowed:
        raise ValueError(f"Unsupported kind {kind!r}. Allowed: {sorted(allowed)}")
    return k


def validate_frequency(freq: str) -> str:
    f = normalize_frequency(freq)
    aliases = {
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "h": "1h",
        "hour": "1h",
        "m": "1m",
        "min": "1m",
    }
    f = aliases.get(f, f)

    allowed = {"1d", "1h", "1m"}
    if f not in allowed:
        raise ValueError(f"Unsupported frequency {freq!r}. Allowed: {sorted(allowed)}")
    return f


def normalize_and_validate_request(req: DataRequest) -> DataRequest:
    """Normalize/validate DataRequest fields at the API boundary."""

    k = validate_kind(req.kind)
    f = validate_frequency(req.frequency)
    return replace(req, kind=k, frequency=f)

