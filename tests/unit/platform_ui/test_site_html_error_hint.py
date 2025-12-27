from __future__ import annotations


def test_site_html_contains_reset_hint_snippet():
    # Regression: ensure the UI includes the cache reset hint text so corrupted LMDB errors are actionable.
    # Current UI is a committed SPA shell under platform_ui/assets_dist/index.html.
    import pathlib

    index_path = pathlib.Path(__file__).resolve().parents[3] / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "index.html"
    html = index_path.read_text(encoding="utf-8")

    # The UI must contain a cache reset hint snippet so users can recover from LMDB corruption.
    assert "reset" in html.lower() and "arctic" in html.lower()
