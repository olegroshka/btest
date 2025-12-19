import pandas as pd

from quantdsl_backtest.engine.analytics.types import (
    PortfolioSignalAttribution,
    SignalAnalyticsConfig,
    SignalTearsheetData,
)
from quantdsl_backtest.engine.analytics.render_tearsheets import (
    render_portfolio_signal_tearsheet_html,
    render_signal_tearsheet_html,
)


def test_signal_tearsheet_has_site_nav_links(tmp_path):
    cfg = SignalAnalyticsConfig(signals=["sig1"], horizons=[1], quantiles=5)
    rep = SignalTearsheetData(name="sig1", config=cfg)
    rep.coverage = pd.Series([0.5, 0.6], index=pd.date_range("2020-01-01", periods=2))
    rep.quantile_turnover = pd.Series([0.1, 0.2], index=rep.coverage.index)
    rep.rank_ic = {1: pd.Series([0.05, -0.02], index=rep.coverage.index)}

    out = tmp_path / "signals" / "sig1" / "signal_tearsheet.html"
    render_signal_tearsheet_html(rep, output_path=out, strategy_name="S")
    html = out.read_text(encoding="utf-8")

    # Relative linking assumes outputs/<run>/signals/<sig>/signal_tearsheet.html
    assert "../../index.html" in html
    assert "../../tearsheet.html" in html
    assert "../../attribution/sig1/portfolio_signal_tearsheet.html" in html


def test_portfolio_signal_tearsheet_has_site_nav_links(tmp_path):
    idx = pd.date_range("2020-01-01", periods=2)
    contrib_by_q = pd.DataFrame({1: [0.01, 0.02], 2: [-0.01, 0.00]}, index=idx)
    ls = pd.Series([0.02, 0.01], index=idx)
    attr = PortfolioSignalAttribution(contrib_ret_by_q=contrib_by_q, contrib_ret_ls=ls)

    out = tmp_path / "attribution" / "sig1" / "portfolio_signal_tearsheet.html"
    render_portfolio_signal_tearsheet_html(
        signal_name="sig1",
        attribution=attr,
        output_path=out,
        strategy_name="S",
    )
    html = out.read_text(encoding="utf-8")

    assert "../../index.html" in html
    assert "../../tearsheet.html" in html
    assert "../../signals/sig1/signal_tearsheet.html" in html

