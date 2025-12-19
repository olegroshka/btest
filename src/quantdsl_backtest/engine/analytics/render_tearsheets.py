# src/quantdsl_backtest/engine/analytics/render_tearsheets.py
from __future__ import annotations

# Ensure a non-interactive backend for headless/test environments (avoid Tk dependency).
try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
except Exception:  # pragma: no cover
    pass

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .types import SignalTearsheetData, PortfolioSignalAttribution
from .render_html_utils import (
    _escape,
    _fmt_float,
    _series_summary_stats,
    _table_from_df,
    default_css,
    fig_to_base64_png,
    # NEW: shared report-site helpers
    render_topnav,
    _render_help,
)


# ----------------------------
# Plot helpers (matplotlib)
# ----------------------------

def _plot_series(s: pd.Series, title: str, y0: Optional[float] = None) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    s = pd.Series(s).astype("float64").replace([np.inf, -np.inf], np.nan)
    ax.plot(s.index, s.values, linewidth=1.6)
    if y0 is not None:
        ax.axhline(y0, linewidth=1.0, alpha=0.6)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    uri = fig_to_base64_png(fig)
    plt.close(fig)
    return uri


def _plot_hist(s: pd.Series, title: str, bins: int = 40) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    s = pd.Series(s).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) > 0:
        ax.hist(s.values, bins=bins, alpha=0.9)
        ax.axvline(float(s.mean()), linewidth=1.4)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    uri = fig_to_base64_png(fig)
    plt.close(fig)
    return uri


def _plot_bar(values: pd.Series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    values = pd.Series(values).astype("float64")
    ax.bar([str(i) for i in values.index], values.values, alpha=0.9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    uri = fig_to_base64_png(fig)
    plt.close(fig)
    return uri


def _plot_cum_lines(df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    df = df.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    c = df.cumsum()
    for col in c.columns:
        ax.plot(c.index, c[col].values, linewidth=1.4, label=str(col))
    ax.legend(loc="best", frameon=False, ncol=3, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    uri = fig_to_base64_png(fig)
    plt.close(fig)
    return uri


# ----------------------------
# Signal tearsheet renderer
# ----------------------------

def render_signal_tearsheet_html(
    report: SignalTearsheetData,
    *,
    output_path: str | Path,
    strategy_name: Optional[str] = None,
    run_meta: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = report.config
    run_meta = run_meta or {}

    # KPI block
    k_cov = float(pd.Series(report.coverage).dropna().mean()) if report.coverage is not None else np.nan
    k_turn = float(pd.Series(report.quantile_turnover).dropna().mean()) if report.quantile_turnover is not None else np.nan

    # IC summary table
    ic_rows = []
    for h, s in (report.rank_ic or {}).items():
        st = _series_summary_stats(s)
        ic_rows.append({
            "horizon": int(h),
            "count": st["count"],
            "mean": st["mean"],
            "std": st["std"],
            "tstat": st["tstat"],
            "min": st["min"],
            "max": st["max"],
        })
    ic_summary = pd.DataFrame(ic_rows).set_index("horizon").sort_index() if ic_rows else pd.DataFrame()

    # IC decay (mean IC by horizon)
    decay = ic_summary["mean"].copy() if (not ic_summary.empty and "mean" in ic_summary.columns) else pd.Series(dtype="float64")

    # Plots
    ic_plot_uris = {}
    ic_hist_uris = {}
    ls_plot_uris = {}
    qret_bar_uris = {}

    for h in cfg.horizons:
        if h in (report.rank_ic or {}):
            ic_plot_uris[h] = _plot_series(report.rank_ic[h], f"Rank IC (Spearman) — horizon {h}d", y0=0.0)
            ic_hist_uris[h] = _plot_hist(report.rank_ic[h], f"Rank IC distribution — horizon {h}d")

        if h in (report.ls_fwd_ret or {}):
            ls_plot_uris[h] = _plot_series(report.ls_fwd_ret[h].cumsum(), f"Cumulative L–S forward return — horizon {h}d", y0=0.0)

        if h in (report.mean_fwd_ret_by_q or {}):
            avg_by_q = report.mean_fwd_ret_by_q[h].mean(axis=0, skipna=True)
            avg_by_q.index = [f"Q{int(i)}" for i in avg_by_q.index]
            qret_bar_uris[h] = _plot_bar(avg_by_q, f"Mean forward return by quantile — horizon {h}d")

    cov_uri = _plot_series(report.coverage, "Coverage (fraction of non-NaN names)", y0=None) if report.coverage is not None else ""
    xsec_std_uri = _plot_series(report.xsec_std, "Cross-sectional std (signal)", y0=None) if report.xsec_std is not None else ""
    turn_uri = _plot_series(report.quantile_turnover, "Quantile turnover (fraction changed vs prior day)", y0=None) if report.quantile_turnover is not None else ""
    decay_uri = _plot_bar(decay.rename(lambda x: f"{int(x)}d"), "IC decay (mean Rank IC by horizon)") if len(decay) else ""

    # Table HTML
    ic_tbl = _table_from_df(ic_summary, float_digits=4, pct_cols=set())

    # Build HTML
    title = f"Signal Tearsheet — {report.name}"
    sname = strategy_name or run_meta.get("strategy_name", "")

    # Site navigation (relative to outputs/<run>/signals/<signal>/signal_tearsheet.html)
    nav = render_topnav(
        links=[
            ("Index", "../../index.html"),
            ("Strategy (QuantStats)", "../../tearsheet.html"),
            ("Attribution (this signal)", f"../../attribution/{_escape(report.name)}/portfolio_signal_tearsheet.html"),
        ]
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_escape(title)}</title>
<style>{default_css()}</style>
</head>
<body>
<div class="page">
  <div class="hero">
    <div>
      <h1 class="title">{_escape(title)}</h1>
      <div class="subtitle">
        Strategy: <span class="muted">{_escape(sname)}</span>
        &nbsp;•&nbsp; Quantiles: <span class="muted">{_escape(cfg.quantiles)}</span>
        &nbsp;•&nbsp; Horizons: <span class="muted">{_escape(cfg.horizons)}</span>
        &nbsp;•&nbsp; Delay used: <span class="muted">{_escape(cfg.signal_delay_bars)} bar(s)</span>
      </div>
      {nav}
    </div>
    <div class="meta">
      <div class="pill">Avg coverage: <b>{_fmt_float(k_cov, digits=3)}</b></div>
      <div class="pill">Avg q-turnover: <b>{_fmt_float(k_turn, digits=3)}</b></div>
      <div class="pill">Mask: <b>{_escape(cfg.within_mask or "None")}</b></div>
    </div>
  </div>

  <div class="card">
    <div class="hd"><h2>What is this?</h2><div class="muted">How to read this page</div></div>
    <div class="bd">
      <div class="muted" style="line-height:1.45;">
        This is an <b>ex-ante</b> signal diagnostics page (Alphalens-style). It answers: does the signal predict
        forward returns (IC / quantile return spread), and is it stable enough to trade (coverage/turnover)?
        For realized impact under portfolio constraints and costs, use the <b>Attribution</b> page.
      </div>
      {_render_help(["coverage","quantile_turnover","rank_ic","ic_tstat","ls_fwd_ret"])}
    </div>
  </div>

  <div class="card">
    <div class="hd"><h2>Navigation</h2><div class="muted">Quick jump</div></div>
    <div class="bd">
      <div class="toc">
        <a href="#quality">Signal Quality</a>
        <a href="#ic">Information Coefficient</a>
        <a href="#quantiles">Quantile Returns</a>
        <a href="#stability">Stability & Coverage</a>
        <a href="#config">Config</a>
      </div>
    </div>
  </div>

  <div class="grid">

    <div class="card" id="quality">
      <div class="hd"><h2>Signal Quality</h2><div class="muted">High-level diagnostics</div></div>
      <div class="bd">
        <div class="kpis">
          <div class="kpi"><div class="k">Avg Coverage</div><div class="v">{_fmt_float(k_cov, digits=3)}</div></div>
          <div class="kpi"><div class="k">Avg Quantile Turnover</div><div class="v">{_fmt_float(k_turn, digits=3)}</div></div>
          <div class="kpi"><div class="k">Horizons</div><div class="v">{_escape(cfg.horizons)}</div></div>
          <div class="kpi"><div class="k">Delay Bars</div><div class="v">{_escape(cfg.signal_delay_bars)}</div></div>
        </div>
        {_render_help(["coverage","quantile_turnover"])}
        <div class="row2" style="margin-top:12px;">
          <img class="img" src="{cov_uri}" alt="coverage"/>
          <img class="img" src="{turn_uri}" alt="turnover"/>
        </div>
        <div class="row2" style="margin-top:12px;">
          <img class="img" src="{xsec_std_uri}" alt="xsec std"/>
          <img class="img" src="{decay_uri}" alt="ic decay"/>
        </div>
      </div>
    </div>

    <div class="card" id="ic">
      <div class="hd"><h2>Information Coefficient</h2><div class="muted">Rank IC (Spearman) vs forward returns</div></div>
      <div class="bd">
        {ic_tbl}
        {"".join(f'''
        <div class="row2" style="margin-top:12px;">
          <img class="img" src="{ic_plot_uris.get(h, "")}" alt="ic series {h}"/>
          <img class="img" src="{ic_hist_uris.get(h, "")}" alt="ic hist {h}"/>
        </div>
        ''' for h in cfg.horizons if h in ic_plot_uris)}
      </div>
    </div>

    <div class="card" id="quantiles">
      <div class="hd"><h2>Quantile Returns</h2><div class="muted">Forward returns by quantile + L–S spread</div></div>
      <div class="bd">
        {"".join(f'''
        <div class="row2" style="margin-top:12px;">
          <img class="img" src="{qret_bar_uris.get(h, "")}" alt="qret bar {h}"/>
          <img class="img" src="{ls_plot_uris.get(h, "")}" alt="ls cum {h}"/>
        </div>
        ''' for h in cfg.horizons if (h in qret_bar_uris and h in ls_plot_uris))}
        <div class="muted" style="margin-top:10px;">
          Notes: quantiles computed from the signal aligned to <b>execution timing</b> using delay={_escape(cfg.signal_delay_bars)}.
        </div>
      </div>
    </div>

    <div class="card" id="stability">
      <div class="hd"><h2>Stability & Coverage</h2><div class="muted">Data availability & cross-section behavior</div></div>
      <div class="bd">
        <div class="row3">
          <img class="img" src="{cov_uri}" alt="coverage 2"/>
          <img class="img" src="{xsec_std_uri}" alt="xsec std 2"/>
          <img class="img" src="{turn_uri}" alt="turnover 2"/>
        </div>
      </div>
    </div>

    <div class="card" id="config">
      <div class="hd"><h2>Config</h2><div class="muted">Exact parameters used</div></div>
      <div class="bd">
        <pre style="margin:0; font-family: var(--mono); font-size:12px; color: var(--muted); white-space:pre-wrap;">
{_escape(pd.Series(asdict(cfg)).to_string())}
        </pre>
      </div>
    </div>

  </div>

  <div class="foot">
    Generated by quantdsl_backtest • Signal tear sheet focuses on <b>ex-ante</b> predictiveness (IC / forward returns), not portfolio constraints.
  </div>
</div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


# ----------------------------
# Portfolio signal tearsheet
# ----------------------------

def render_portfolio_signal_tearsheet_html(
    *,
    signal_name: str,
    attribution: PortfolioSignalAttribution,
    output_path: str | Path,
    strategy_name: Optional[str] = None,
    run_meta: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = run_meta or {}

    contrib_by_q = attribution.contrib_ret_by_q
    ls = attribution.contrib_ret_ls

    # KPIs
    total_ls = float(pd.Series(ls).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum())
    vol_ls = float(pd.Series(ls).replace([np.inf, -np.inf], np.nan).fillna(0.0).std())
    mean_ls = float(pd.Series(ls).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())

    # Plots
    cum_q_uri = _plot_cum_lines(contrib_by_q, "Cumulative contribution by quantile (return space)")
    cum_ls_uri = _plot_series(ls.cumsum(), "Cumulative L–S contribution (Top − Bottom)", y0=0.0)

    # Summary table: total contribution by quantile
    totals = contrib_by_q.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
    tbl = pd.DataFrame({
        "total_contrib": totals,
        "avg_daily_contrib": contrib_by_q.replace([np.inf, -np.inf], np.nan).fillna(0.0).mean(axis=0),
    })
    tbl.index = [f"Q{int(i)}" for i in tbl.index]
    tbl = tbl.sort_index()

    # Costs
    cost_block = ""
    if attribution.cost_pnl_by_q is not None and not attribution.cost_pnl_by_q.empty:
        cost_by_q = attribution.cost_pnl_by_q

        # Heuristic: if typical daily values are > 0.1, they are almost certainly $ costs
        # (since return-space cost drag should be ~bps).
        try:
            daily_scale = float(cost_by_q.sum(axis=1).abs().median())
        except Exception:
            daily_scale = 0.0
        is_dollars = daily_scale > 0.1

        if is_dollars:
            cost_title = "Cumulative costs by quantile ($)"
            total_col = "total_cost_usd"
            avg_col = "avg_daily_cost_usd"
        else:
            cost_title = "Cumulative cost drag by quantile (return space approx)"
            total_col = "total_cost_drag"
            avg_col = "avg_daily_cost_drag"

        cost_cum_uri = _plot_cum_lines(cost_by_q, cost_title)
        cost_tbl = pd.DataFrame({
            total_col: cost_by_q.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0),
            avg_col: cost_by_q.replace([np.inf, -np.inf], np.nan).fillna(0.0).mean(axis=0),
        })
        cost_tbl.index = [f"Q{int(i)}" for i in cost_tbl.index]
        cost_tbl = cost_tbl.sort_index()

        unit_note = (
            "Costs are shown in $ PnL (commission/fees + slippage proxy)."
            if is_dollars
            else "Costs are shown in return space as <code>cost_pnl / equity</code> (approx)."
        )

        cost_block = f"""
        <div class=\"card\" id=\"costs\">
          <div class=\"hd\"><h2>Costs by Quantile</h2><div class=\"muted\">Commission/fees + slippage proxy if available</div></div>
          <div class=\"bd\">
            {_table_from_df(cost_tbl, float_digits=6)}
            <div style=\"margin-top:12px;\"><img class=\"img\" src=\"{cost_cum_uri}\" alt=\"cost cum\"/></div>
            <div class=\"muted\" style=\"margin-top:10px;\">
              {unit_note} For exact PnL attribution, extend the trade log with explicit slippage/borrow/financing components.
            </div>
          </div>
        </div>
        """

    title = f"Portfolio Signal Tearsheet — {signal_name}"
    sname = strategy_name or run_meta.get("strategy_name", "")

    nav = render_topnav(
        links=[
            ("Index", "../../index.html"),
            ("Strategy (QuantStats)", "../../tearsheet.html"),
            ("Signal diagnostics (this signal)", f"../../signals/{_escape(signal_name)}/signal_tearsheet.html"),
        ]
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_escape(title)}</title>
<style>{default_css()}</style>
</head>
<body>
<div class="page">
  <div class="hero">
    <div>
      <h1 class="title">{_escape(title)}</h1>
      <div class="subtitle">
        Strategy: <span class="muted">{_escape(sname)}</span>
        &nbsp;•&nbsp; This report maps <b>realized portfolio contributions</b> to the signal’s quantile buckets.
      </div>
      {nav}
    </div>
    <div class="meta">
      <div class="pill">Total L–S contrib: <b>{_fmt_float(total_ls, digits=5)}</b></div>
      <div class="pill">Mean daily L–S: <b>{_fmt_float(mean_ls, digits=6)}</b></div>
      <div class="pill">Daily L–S vol: <b>{_fmt_float(vol_ls, digits=6)}</b></div>
    </div>
  </div>

  <div class="card">
    <div class="hd"><h2>What is this?</h2><div class="muted">How to read this page</div></div>
    <div class="bd">
      <div class="muted" style="line-height:1.45;">
        This is an <b>ex-post</b> attribution page. It answers: given the actual realized weights,
        constraints, execution and costs, which signal quantile buckets contributed to portfolio returns?
        Use the <b>Signal diagnostics</b> page to see ex-ante predictiveness (IC / forward returns).
      </div>
      {_render_help(["contrib_ret_ls","contrib_ret_by_q","cost_pnl_by_q"])}
    </div>
  </div>

  <div class="card">
    <div class="hd"><h2>Navigation</h2><div class="muted">Quick jump</div></div>
    <div class="bd">
      <div class="toc">
        <a href="#contrib">Contribution</a>
        <a href="#summary">Summary</a>
        {"<a href='#costs'>Costs</a>" if cost_block else ""}
      </div>
    </div>
  </div>

  <div class="grid">

    <div class="card" id="contrib">
      <div class="hd"><h2>Contribution by Quantile</h2><div class="muted">Return space: w(t−1) × r(t)</div></div>
      <div class="bd">
        <div class="row2">
          <img class="img" src="{cum_q_uri}" alt="cum q contrib"/>
          <img class="img" src="{cum_ls_uri}" alt="cum ls contrib"/>
        </div>
        <div class="muted" style="margin-top:10px;">
          Interpretation: if the signal is working and portfolio construction is aligned, higher quantiles
          should contribute more (or at least L–S should be positive for long-top/short-bottom designs).
        </div>
      </div>
    </div>

    <div class="card" id="summary">
      <div class="hd"><h2>Summary</h2><div class="muted">Totals by bucket</div></div>
      <div class="bd">
        {_table_from_df(tbl, float_digits=6)}
      </div>
    </div>

    {cost_block}

  </div>

  <div class="foot">
    Generated by quantdsl_backtest • Portfolio tear sheet is <b>ex-post</b>: it reflects constraints, execution, and costs.
  </div>
</div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
