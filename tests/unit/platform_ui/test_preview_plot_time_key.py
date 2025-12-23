from __future__ import annotations

import re

from quantdsl_backtest.platform_ui.site import html_index


def test_ui_prefers_ts_as_time_key() -> None:
    """Regression: preview API injects index as `ts`; UI should prefer it.

    Without this, the JS can fall back to the first column and accidentally
    treat OHLC numeric columns as epoch timestamps -> 1970 axis / invisible candles.
    """

    html = html_index()

    # Ensure our preferred candidate list includes ts.
    assert "previewIndexCandidates" in html
    assert re.search(r"previewIndexCandidates\s*=\s*\[[^\]]*'ts'", html)

    # Ensure it's checked before generic timeCandidates.
    assert html.index("previewIndexCandidates") < html.index("timeCandidates")

