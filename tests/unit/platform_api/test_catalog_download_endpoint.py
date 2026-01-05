from __future__ import annotations

import os
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parquet_source_uri() -> str:
    # Ensure we're pointing at a real file in the repo
    p = (_repo_root() / "equities" / "sp500_daily").resolve()
    assert p.exists(), f"Missing parquet dataset for tests: {p}"
    # The parquet provider accepts plain filesystem paths after parquet://
    return f"parquet://{p.as_posix()}"


def _client():
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "source,entities",
    [
        ("yf://AAPL", ["AAPL"]),
        (_parquet_source_uri(), ["AAPL"]),
    ],
)
def test_catalog_download_dry_run_smoke(source: str, entities: list[str]):
    """Dry-run should accept both parquet and YF sources.

    Dry-run doesn't resolve a provider; it returns a plan (best-effort) and the echoed request.
    """

    client = _client()
    resp = client.post(
        "/api/catalog/download",
        json={
            "source": source,
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2024-01-01",
            "end": "2024-01-05",
            "entities": entities,
            "dry_run": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["dry_run"] is True
    assert isinstance(data.get("request"), dict)
    assert isinstance(data.get("plan"), (list, type(None)))


def test_catalog_download_parquet_execute_returns_entities_and_stats():
    """Execute download against local parquet should return a stable response model."""

    client = _client()
    resp = client.post(
        "/api/catalog/download",
        json={
            "source": _parquet_source_uri(),
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2024-01-01",
            "end": "2024-01-10",
            "entities": ["AAPL"],
            "dry_run": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["dry_run"] is False
    assert data.get("kind") == "market_bars"
    assert "parquet://" in (data.get("source") or "")

    assert isinstance(data.get("entities"), list)
    assert "AAPL" in data.get("entities")

    if data.get("cache_stats") is not None:
        assert isinstance(data["cache_stats"], dict)


def test_catalog_download_unknown_source_dry_run_is_ok():
    """Unknown schemes are allowed in dry-run; the UI uses this to show a plan."""

    client = _client()
    resp = client.post(
        "/api/catalog/download",
        json={
            "source": "x://y",
            "kind": "market_bars",
            "frequency": "1d",
            "entities": ["AAPL"],
            "dry_run": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("dry_run") is True


@pytest.mark.parametrize(
    "payload,expected_code",
    [
        ({"source": _parquet_source_uri(), "kind": "bad_kind", "frequency": "1d", "entities": ["AAPL"], "dry_run": True}, 422),
        ({"source": _parquet_source_uri(), "kind": "market_bars", "frequency": "bad", "entities": ["AAPL"], "dry_run": True}, 422),
    ],
)
def test_catalog_download_validation_errors(payload, expected_code: int):
    client = _client()
    resp = client.post("/api/catalog/download", json=payload)
    assert resp.status_code == expected_code
