# src/quantdsl_backtest/engine/analytics/render_html_utils.py
from __future__ import annotations

import base64
import html
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _fmt_float(x: Any, *, pct: bool = False, na: str = "—", digits: int = 3) -> str:
    try:
        if x is None:
            return na
        if isinstance(x, (float, int, np.floating, np.integer)):
            if not np.isfinite(float(x)):
                return na
            v = float(x)
            if pct:
                return f"{v * 100:.{digits}f}%"
            return f"{v:.{digits}f}"
        # pandas scalar
        v = float(x)
        if not np.isfinite(v):
            return na
        return f"{v * 100:.{digits}f}%" if pct else f"{v:.{digits}f}"
    except Exception:
        return na


def _fmt_int(x: Any, *, na: str = "—") -> str:
    try:
        if x is None:
            return na
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        v = float(x)
        if not np.isfinite(v):
            return na
        return str(int(round(v)))
    except Exception:
        return na


def _escape(s: Any) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def fig_to_base64_png(fig) -> str:
    """Return a base64 data URI for a matplotlib figure as PNG."""
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    bio.seek(0)
    b64 = base64.b64encode(bio.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _table_from_df(df: pd.DataFrame, *, float_digits: int = 3, pct_cols: Optional[set[str]] = None) -> str:
    """Render a pandas DF into a styled HTML table."""
    if df is None or df.empty:
        return '<div class="muted">No data.</div>'
    pct_cols = pct_cols or set()

    cols = list(df.columns)
    head = "".join(f"<th>{_escape(c)}</th>" for c in [""] + cols)

    rows_html = []
    for idx, row in df.iterrows():
        tds = [f"<td class='idx'>{_escape(idx)}</td>"]
        for c in cols:
            v = row[c]
            is_pct = (c in pct_cols)
            tds.append(f"<td>{_fmt_float(v, pct=is_pct, digits=float_digits)}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    return f"""
    <div class="table-wrap">
      <table class="tbl">
        <thead><tr>{head}</tr></thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def _series_summary_stats(s: pd.Series) -> Dict[str, float]:
    """Basic stats: mean, std, tstat, min, max, count."""
    s = pd.Series(s).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(s.shape[0])
    if n == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "tstat": np.nan, "min": np.nan, "max": np.nan}
    mu = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    t = mu / (sd / np.sqrt(n)) if (sd > 0 and n > 1) else np.nan
    return {"count": n, "mean": mu, "std": sd, "tstat": t, "min": float(s.min()), "max": float(s.max())}


def default_css() -> str:
    return """
    :root{
      --bg:#0b1220;
      --card:#0f1b33;
      --card2:#0d1730;
      --text:#e6edf7;
      --muted:#a9b7d0;
      --line:#1f2d4d;
      --accent:#77b8ff;
      --good:#5fe1b0;
      --bad:#ff6b8a;
      --warn:#ffd166;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
      --radius: 16px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }
    html,body{background:var(--bg); color:var(--text); margin:0; padding:0; font-family:var(--sans);}
    .page{max-width:1180px; margin:28px auto; padding:0 18px 60px;}
    .hero{display:flex; justify-content:space-between; gap:16px; align-items:flex-end; margin-bottom:18px;}
    .title{font-size:26px; font-weight:780; letter-spacing:.2px; margin:0;}
    .subtitle{margin:6px 0 0; color:var(--muted); font-size:13px;}
    .meta{display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end;}
    .pill{border:1px solid var(--line); background:rgba(255,255,255,.03); padding:8px 10px; border-radius:999px;
          font-size:12px; color:var(--muted);}
    .grid{display:grid; grid-template-columns: repeat(12, 1fr); gap:14px; align-items:stretch;}
    .card{grid-column: span 12; background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01));
          border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden;}
    .card .hd{padding:14px 16px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center;}
    .card .hd h2{margin:0; font-size:14px; letter-spacing:.3px; text-transform:uppercase;}
    .card .bd{padding:14px 16px;}
    .muted{color:var(--muted);}
    .kpis{display:grid; grid-template-columns: repeat(4, 1fr); gap:10px;}
    .kpi{background:rgba(255,255,255,.02); border:1px solid var(--line); border-radius:14px; padding:12px;}
    .kpi .k{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.25px;}
    .kpi .v{font-size:18px; font-weight:760; margin-top:4px;}
    .img{width:100%; border-radius:12px; border:1px solid var(--line); background:rgba(0,0,0,.15);}
    .row2{display:grid; grid-template-columns: 1fr 1fr; gap:12px;}
    .row3{display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px;}
    .tbl{width:100%; border-collapse:collapse; font-size:12px;}
    .tbl th, .tbl td{padding:8px 10px; border-bottom:1px solid var(--line); text-align:right;}
    .tbl th:first-child, .tbl td:first-child{text-align:left;}
    .tbl thead th{color:var(--muted); font-weight:650; text-transform:uppercase; letter-spacing:.25px; font-size:11px;}
    .tbl tbody tr:hover{background:rgba(255,255,255,.03);}
    .tbl td.idx{font-family:var(--mono); color:var(--text);}
    .table-wrap{overflow:auto; border-radius:12px; border:1px solid var(--line);}
    .toc{display:flex; gap:10px; flex-wrap:wrap;}
    .toc a{color:var(--accent); text-decoration:none; font-size:12px; border:1px solid var(--line);
           padding:7px 9px; border-radius:999px; background:rgba(255,255,255,.02);}
    .toc a:hover{background:rgba(119,184,255,.10);}
    .foot{margin-top:18px; color:var(--muted); font-size:12px;}
    .topnav{display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 0;}
    .topnav a{color:var(--accent); text-decoration:none; font-size:12px; border:1px solid var(--line);
              padding:7px 9px; border-radius:999px; background:rgba(255,255,255,.02);}
    .topnav a:hover{background:rgba(119,184,255,.10);}
    .help{margin-top:8px; font-size:12px; color:var(--muted); line-height:1.4;}
    .help code{font-family:var(--mono); font-size:11px; color:var(--text);}
    @media (max-width: 980px){
      .kpis{grid-template-columns: repeat(2, 1fr);}
      .row2, .row3{grid-template-columns: 1fr;}
    }
    """


# ---------------------------------------------------------------------------
# Report-site helpers (shared shell + glossary)
# ---------------------------------------------------------------------------


def metric_glossary() -> Dict[str, str]:
    """Short definitions for metrics shown across report pages.

    Keep these concise and practical. The index page and tearsheets can
    reference them to explain what users are looking at.
    """

    return {
        # strategy-level (common)
        "total_return": "Total return over the full backtest period (ending equity / starting equity − 1).",
        "cagr": "Compound annual growth rate of the equity curve.",
        "volatility": "Annualized volatility of returns.",
        "sharpe": "Annualized Sharpe ratio of returns (excess over risk-free, if configured).",
        "sortino": "Annualized Sortino ratio; like Sharpe but only downside volatility.",
        "calmar": "CAGR divided by absolute max drawdown; higher means better return per unit drawdown.",
        "max_drawdown": "Max peak-to-trough decline of equity.",
        "tail_ratio": "Ratio of upside tail to downside tail: |q95(returns)| / |q05(returns)|.",
        "ulcer_index": "Ulcer Index: RMS of equity drawdown magnitudes; lower is better.",
        "var": "Value at Risk (VaR) of daily returns at the default QuantStats confidence level.",
        "cvar": "Conditional VaR (CVaR): average daily return in the left tail beyond VaR.",
        "win_rate": "Fraction of days with positive returns.",
        "skew": "Skewness of daily returns; negative means more frequent/extreme left-tail moves.",
        "kurtosis": "Kurtosis of daily returns; higher implies heavier tails than a normal distribution.",
        "profit_factor": "Daily profit factor: sum(positive returns) / abs(sum(negative returns)).",
        "turnover": "Annualized turnover based on daily absolute weight changes (higher implies more trading/cost sensitivity).",
        "avg_leverage": "Average gross leverage (gross exposure / equity).",
        "max_leverage": "Maximum gross leverage observed.",
        "pct_days_in_market": "Fraction of days with non-zero gross exposure (invested days).",
        # signal ex-ante
        "coverage": "Fraction of instruments with a valid signal value each day.",
        "quantile_turnover": "Fraction of names that change quantile bucket vs previous day.",
        "rank_ic": "Spearman rank correlation between signal and forward returns.",
        "ic_tstat": "t-stat of the IC series; bigger magnitude suggests more consistent IC.",
        "ls_fwd_ret": "Forward return of top quantile minus bottom quantile (ex-ante, unconstrained).",
        # signal ex-post
        "contrib_ret_ls": "Realized portfolio contribution: (top bucket contrib − bottom bucket contrib).",
        "contrib_ret_by_q": "Realized portfolio contribution per quantile bucket in return space.",
        "cost_pnl_by_q": "Estimated costs per bucket (commission/fees + slippage proxy when available).",
    }


def _render_help(keys: List[str]) -> str:
    gl = metric_glossary()
    parts = []
    for k in keys:
        txt = gl.get(k)
        if txt:
            parts.append(f"<div><b>{_escape(k)}</b>: {_escape(txt)}</div>")
    if not parts:
        return ""
    return f"<div class='help'>{''.join(parts)}</div>"


def render_topnav(*, links: List[tuple[str, str]]) -> str:
    """Render the top navigation links.

    links: list of (label, href)
    """
    if not links:
        return ""
    return "<div class='topnav'>" + "".join(
        f"<a href='{_escape(href)}'>{_escape(label)}</a>" for label, href in links
    ) + "</div>"


def render_html_shell(
    *,
    title: str,
    body_html: str,
    nav_links: Optional[List[tuple[str, str]]] = None,
    subtitle_html: str = "",
    extra_css: str = "",
) -> str:
    """Wrap arbitrary HTML in our dark report shell.

    This is intentionally tiny and string-based (no templating dependency).
    """

    nav_links = nav_links or []
    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>{_escape(title)}</title>
<style>{default_css()}\n{extra_css}</style>
</head>
<body>
<div class='page'>
  <div class='hero'>
    <div>
      <h1 class='title'>{_escape(title)}</h1>
      <div class='subtitle'>{subtitle_html}</div>
      {render_topnav(links=nav_links)}
    </div>
  </div>
  {body_html}
  <div class='foot'>Generated by quantdsl_backtest</div>
</div>
</body>
</html>
"""

