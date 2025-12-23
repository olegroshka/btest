from __future__ import annotations


def test_ui_index_returns_html():
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")
    assert "Platform UI" in r.text

    # UI elements we rely on for researcher workflow
    assert "catalogSearch" in r.text
    assert "btnCatalog" in r.text
    assert "btnCatalogClear" in r.text

    assert "btnPreview" in r.text
    assert "btnDryRun" in r.text
    assert "btnDownload" in r.text

    assert "metaSummary" in r.text
    assert "downloadSummary" in r.text

    assert "data-testid='copy-source'" in r.text or "data-testid=\"copy-source\"" in r.text

    # New workflow controls
    assert "dlSource" in r.text
    assert "dlRangeMode" in r.text
    assert "btnGuessSource" in r.text
    assert "btnCopyPayload" in r.text

    # analysis uses describe endpoint
    assert "/api/catalog/describe/" in r.text

    # Quality UI controls
    assert "btnQualityScan" in r.text
    assert "btnQualityIssues" in r.text

    # Contract: core JS functions exist (prevents dead buttons/ReferenceError)
    for fn in [
        "setPreviewTarget",
        "buildDownloadPayload",
        "updatePayloadHint",
    ]:
        assert f"function {fn}" in r.text
