from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.manual]


@pytest.mark.skipif(
    os.getenv("QUANTDSL_RUN_NET") not in {"1", "true", "yes", "y", "on"},
    reason="Network-dependent YF test disabled by default. Set QUANTDSL_RUN_NET=1 to enable.",
)
def test_platform_api_yf_download_short_range_smoke(tmp_path, monkeypatch):
    """YF real download smoke (network).

    Uses a tiny date range and a single ticker. This is intended for manual/CI with network.
    """

    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    # isolate cache
    arctic_root = tmp_path / "arctic_yf_download"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post(
        "/api/catalog/download",
        json={
            "source": "yf://AAPL",
            "kind": "market_bars",
            "frequency": "1d",
            "start": "2024-01-02",
            "end": "2024-01-06",
            "entities": ["AAPL"],
            "dry_run": False,
        },
    )

    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("dry_run") is False
    assert data.get("frequency") == "1d"
    assert "AAPL" in (data.get("entities") or [])

