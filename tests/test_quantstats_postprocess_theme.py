from pathlib import Path

import pandas as pd

from quantdsl_backtest.engine.metrics_quantstats import generate_quantstats_tearsheet


def test_quantstats_tearsheet_postprocess_injects_css_and_nav(tmp_path, monkeypatch):
    # Avoid importing quantstats in tests; patch a tiny stub.
    class _Reports:
        @staticmethod
        def html(rets, benchmark=None, output=None, title=None, **kwargs):
            Path(output).write_text(
                "<html><head><title>T</title></head><body><div>Q</div></body></html>",
                encoding="utf-8",
            )

    class _QS:
        reports = _Reports()

    monkeypatch.setitem(__import__("sys").modules, "quantstats", _QS)

    returns = pd.Series([0.0, 0.01], index=pd.date_range("2020-01-01", periods=2))
    out = tmp_path / "tearsheet.html"

    generate_quantstats_tearsheet(returns, output=str(out), title="My Strategy")
    html = out.read_text(encoding="utf-8")

    # Our wrapper should inject CSS and an Index link
    assert "--bg" in html  # from default_css
    assert "Index" in html
    assert "index.html" in html
