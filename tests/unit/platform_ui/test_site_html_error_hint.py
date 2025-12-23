from __future__ import annotations


def test_site_html_contains_reset_hint_snippet():
    # This is a lightweight regression test: ensure the UI includes the cache reset hint text
    # so corrupted LMDB errors are actionable.
    from quantdsl_backtest.platform_ui.site import html_index

    html = html_index()
    # The UI must contain a cache reset hint snippet so users can recover from LMDB corruption.
    assert "reset" in html.lower() and "arctic" in html.lower()
