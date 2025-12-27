from __future__ import annotations

import re
import pathlib


def test_ui_prefers_ts_as_time_key() -> None:
    """Regression: chart UI should prefer `ts` as the time key.

    Without this, the JS can fall back to the first column and accidentally
    treat OHLC numeric columns as epoch timestamps -> 1970 axis / invisible candles.

    Current UI is bundled JS (React) emitted to platform_ui/assets_dist/assets/main.react.js.
    """

    js_path = pathlib.Path(__file__).resolve().parents[3] / "src" / "quantdsl_backtest" / "platform_ui" / "assets_dist" / "assets" / "main.react.js"
    js = js_path.read_text(encoding="utf-8")

    # Our modern plot builder explicitly looks for the `ts` key.
    assert re.search(r"\bts\b", js)
