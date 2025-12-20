from quantdsl_backtest.engine.analytics.render_html_utils import metric_glossary


def test_metric_glossary_has_strategy_metrics_for_tooltips():
    gl = metric_glossary()
    # Strategy KPI keys we expect to have tooltip support
    for k in [
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "tail_ratio",
        "ulcer_index",
        "var",
        "cvar",
        "win_rate",
        "skew",
        "kurtosis",
        "profit_factor",
        "turnover",
        "avg_leverage",
        "max_leverage",
        "pct_days_in_market",
    ]:
        assert k in gl
        assert isinstance(gl[k], str)
        assert gl[k].strip()

