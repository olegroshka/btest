# src/quantdsl_backtest/engine/backtest_runner.py

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from ..dsl.strategy import Strategy
from ..engine.results import BacktestResult
from ..utils.logging import get_logger
from .data_loader import load_data_for_strategy, load_open_prices
from .factor_engine import FactorEngine
from .signal_engine import SignalEngine
from .portfolio_engine import compute_target_weights_for_date
from .execution_engine import rebalance_to_target_weights
from .accounting import (
    mark_to_market,
    apply_carry_costs,
    compute_exposures,
    compute_basic_metrics,
)
from .analytics.selection_trace import SelectionTraceCollector
from .analytics.types import (
    SignalAnalyticsConfig,
    StrategyAnalyticsConfig,
    SignalTearsheetData,
    PortfolioSignalAttribution,
    SelectionTrace,
)
from .analytics.signal_analytics import (
    compute_forward_returns,
    assign_quantiles,
    compute_rank_ic,
    mean_forward_return_by_quantile,
    quantile_turnover,
)
from .analytics.attribution import (
    contrib_return_panel,
    contrib_by_quantile,
    costs_by_instrument_day,
)
from .analytics.render_tearsheets import (
    render_signal_tearsheet_html,
    render_portfolio_signal_tearsheet_html,
)
# NEW: shared report-site utilities
from .analytics.render_html_utils import (
    _escape as _html_escape,
    _fmt_float as _html_fmt_float,
    render_html_shell,
    _render_help as _html_render_help,
    _series_summary_stats,
    metric_glossary,
)
from .metrics_advanced import compute_advanced_metrics_from_result

from pathlib import Path
from dataclasses import dataclass

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Reporting (rendering pipeline)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReportingContext:
    """Context passed to report renderers.

    Contract:
      - Renderers MUST be safe to call when `output_dir` is None.
      - Renderers MAY ignore `output_dir` if they don't write artifacts.
    """

    strategy: Strategy
    output_dir: Path | None


@runtime_checkable
class ResultsRenderer(Protocol):
    """A single reporting action.

    Renderer implementations should be small, single-purpose, and easy to test.
    """

    name: str

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ReportingPipeline:
    """Composable list of reporting actions."""

    renderers: Sequence[ResultsRenderer]

    def render_all(self, result: BacktestResult, ctx: ReportingContext) -> None:
        for r in self.renderers:
            try:
                r.render(result, ctx)
            except Exception as exc:
                # Reporting should not crash the backtest by default.
                log.warning("Reporting renderer '%s' failed: %s", getattr(r, "name", type(r).__name__), exc)


@dataclass(slots=True)
class NullReportingPipeline:
    """Null-object pipeline for tests or callers that don't want any reporting."""

    def render_all(self, result: BacktestResult, ctx: ReportingContext) -> None:
        return


@dataclass(slots=True)
class DefaultTextRenderer:
    """Minimal default output to logs (no filesystem; always safe)."""

    name: str = "text"

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        # Keep it intentionally small and stable.
        try:
            sharpe = float(result.metrics.get("sharpe", 0.0))
            max_dd = float(result.metrics.get("max_drawdown", 0.0))
        except Exception:
            sharpe, max_dd = 0.0, 0.0

        log.info(
            "Backtest complete (%s): total return %.2f%%, Sharpe %.2f, max DD %.2f%%",
            getattr(ctx.strategy.backtest, "engine", "event_driven"),
            result.total_return * 100.0,
            sharpe,
            max_dd * 100.0,
        )


@dataclass(slots=True)
class ParquetArtifactsRenderer:
    """Persist core outputs to parquet folder (delegates to BacktestResult.to_parquet)."""

    name: str = "parquet"
    include_trades: bool = True
    include_positions: bool = True
    subdir: str = ""  # optional nesting under output_dir

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        if ctx.output_dir is None:
            return
        out_dir = ctx.output_dir
        if self.subdir:
            out_dir = out_dir / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(str(out_dir), include_trades=self.include_trades, include_positions=self.include_positions)


@dataclass(slots=True)
class SignalAnalyticsHtmlRenderer:
    """Write per-signal and per-signal attribution HTML tearsheets (if attached to result)."""

    name: str = "signal_analytics_html"

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        if ctx.output_dir is None:
            return
        if render_signal_tearsheet_html is None or render_portfolio_signal_tearsheet_html is None:
            raise RuntimeError("Analytics rendering not available")

        reports = getattr(result, "signal_reports", None) or {}
        attribs = getattr(result, "signal_attribution", None) or {}

        out_dir = ctx.output_dir
        (out_dir / "signals").mkdir(parents=True, exist_ok=True)
        (out_dir / "attribution").mkdir(parents=True, exist_ok=True)

        for sname, rep in reports.items():
            render_signal_tearsheet_html(
                rep,
                output_path=out_dir / "signals" / sname / "signal_tearsheet.html",
                strategy_name=ctx.strategy.name,
                run_meta=result.metadata,
            )
        for sname, attr in attribs.items():
            render_portfolio_signal_tearsheet_html(
                signal_name=sname,
                attribution=attr,
                output_path=out_dir / "attribution" / sname / "portfolio_signal_tearsheet.html",
                strategy_name=ctx.strategy.name,
                run_meta=result.metadata,
            )


@dataclass(slots=True)
class QuantStatsHtmlRenderer:
    """Optional QuantStats metrics + HTML tearsheet writing, driven by StrategyAnalyticsConfig."""

    name: ClassVar[str] = "quantstats_html"
    config: StrategyAnalyticsConfig

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        sa = self.config
        if not sa.enabled:
            return

        # Resolve benchmark: accept Series; for string fall back to result.benchmark
        bm = None
        try:
            import pandas as _pd  # local import for isinstance check

            if isinstance(sa.benchmark, _pd.Series):
                bm = sa.benchmark
        except Exception:
            bm = None

        if bm is None and isinstance(getattr(sa, "benchmark", None), str):
            bm = result.benchmark
        if bm is None:
            bm = result.benchmark

        # QuantStats can't compute engine-derived metrics.
        qs_metric_names = [
            m
            for m in sa.metrics
            if m
            not in {
                "total_return",
                "turnover",
                "calmar",
                "win_rate",
                "profit_factor",
                "tail_ratio",
                "ulcer_index",
                "avg_leverage",
                "max_leverage",
                "pct_days_in_market",
            }
        ]

        qs_metrics = result.quantstats_metrics(
            qs_metric_names,
            benchmark=bm,
            risk_free=sa.risk_free,
            prefix=sa.prefix,
        )
        if sa.print_metrics:
            try:
                log.info("\n=== QuantStats metrics ===\n%s", qs_metrics.to_string(float_format=lambda x: f"{x:0.4f}"))
            except Exception:
                log.info("QuantStats metrics: %s", dict(qs_metrics))

        if sa.write_tearsheet:
            # output_dir belongs to reporting; StrategyAnalyticsConfig can override output_dir when needed.
            out_dir = Path(sa.output_dir) if sa.output_dir else ctx.output_dir
            if out_dir is None:
                return
            out_dir.mkdir(parents=True, exist_ok=True)
            title = sa.title or ctx.strategy.name or "Strategy"
            html_path = out_dir / sa.file_name
            result.quantstats_tearsheet(
                output=str(html_path),
                title=title,
                benchmark=bm,
                **(sa.html_kwargs or {}),
            )
            log.info("QuantStats HTML report written to: %s", html_path)


@dataclass(slots=True)
class ReportSiteIndexRenderer:
    """Write outputs/<run>/index.html that links strategy + signals + attribution.

    This is the entrypoint for the report site.
    """

    name: str = "report_site_index"

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        if ctx.output_dir is None:
            return

        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Signals summary table (kept independent of strategy KPI computation) ---
        rows = []
        reports = getattr(result, "signal_reports", None) or {}
        attribs = getattr(result, "signal_attribution", None) or {}

        for sname, rep in reports.items():
            cov = float(pd.Series(rep.coverage).dropna().mean()) if rep.coverage is not None else float("nan")
            turn = float(pd.Series(rep.quantile_turnover).dropna().mean()) if rep.quantile_turnover is not None else float("nan")

            h0 = rep.config.horizons[0] if rep.config.horizons else None
            ic_mean = float(pd.Series(rep.rank_ic.get(h0)).dropna().mean()) if (h0 is not None and h0 in rep.rank_ic) else float("nan")
            ic_t = float(_series_summary_stats(rep.rank_ic.get(h0)).get("tstat")) if (h0 is not None and h0 in rep.rank_ic) else float("nan")

            total_ls = float("nan")
            mean_ls = float("nan")
            if sname in attribs:
                ls = pd.Series(attribs[sname].contrib_ret_ls).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                total_ls = float(ls.sum())
                mean_ls = float(ls.mean())

            rows.append(
                {
                    "signal": sname,
                    "ic_mean": ic_mean,
                    "ic_tstat": ic_t,
                    "coverage": cov,
                    "q_turnover": turn,
                    "ls_total": total_ls,
                    "ls_mean": mean_ls,
                    "links": sname,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            if df["ls_total"].notna().any():
                df = df.sort_values("ls_total", ascending=False)
            else:
                df = df.sort_values("ic_mean", ascending=False)
            df = df.set_index("signal")

        table_html = '<div class="muted">No signal analytics attached to this run.</div>'
        if not df.empty:
            df2 = df.copy()
            df2["links"] = [
                f"<a href='signals/{_html_escape(s)}/signal_tearsheet.html'>signal</a> &nbsp;|&nbsp; "
                f"<a href='attribution/{_html_escape(s)}/portfolio_signal_tearsheet.html'>attrib</a>"
                for s in df2.index
            ]

            header = "".join(f"<th>{_html_escape(c)}</th>" for c in ["signal"] + list(df2.columns))
            body_rows = []
            for sig, r in df2.iterrows():
                tds = [f"<td class='idx'>{_html_escape(sig)}</td>"]
                tds.append(f"<td>{_html_fmt_float(r['ic_mean'], digits=4)}</td>")
                tds.append(f"<td>{_html_fmt_float(r['ic_tstat'], digits=3)}</td>")
                tds.append(f"<td>{_html_fmt_float(r['coverage'], digits=3)}</td>")
                tds.append(f"<td>{_html_fmt_float(r['q_turnover'], digits=3)}</td>")
                tds.append(f"<td>{_html_fmt_float(r['ls_total'], digits=6)}</td>")
                tds.append(f"<td>{_html_fmt_float(r['ls_mean'], digits=6)}</td>")
                tds.append(f"<td style='text-align:left'>{r['links']}</td>")
                body_rows.append("<tr>" + "".join(tds) + "</tr>")

            table_html = f"""
            <div class='table-wrap'>
              <table class='tbl'>
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
              </table>
            </div>
            """

        # --------------------------------------------------------------------- #
        # --- Strategy analytics config (metrics list) ---
        # --------------------------------------------------------------------- #

        # Resolve strategy analytics config (if present) to decide which metrics to show on index
        sa_cfg: StrategyAnalyticsConfig | None = None
        try:
            rep_cfg = getattr(ctx.strategy.backtest, "reporting", None)
            sa_raw = getattr(rep_cfg, "strategyAnalytics", None) if rep_cfg is not None else None
            if isinstance(sa_raw, StrategyAnalyticsConfig):
                sa_cfg = sa_raw
            elif isinstance(sa_raw, dict):
                sa_cfg = StrategyAnalyticsConfig(**sa_raw)
        except Exception:
            sa_cfg = None

        # Headline metrics for the Strategy card on index.html
        # Prefer engine-computed result.metrics when available; otherwise fall back to QuantStats.
        metrics_to_show = list(sa_cfg.metrics) if (sa_cfg is not None and sa_cfg.enabled) else [
            "total_return",
            "sharpe",
            "max_drawdown",
            "turnover",
        ]

        # Pull from result.metrics first
        metrics_map: dict[str, float] = {}
        try:
            if isinstance(getattr(result, "metrics", None), dict):
                for k, v in result.metrics.items():
                    try:
                        metrics_map[str(k)] = float(v)
                    except Exception:
                        pass
        except Exception:
            pass

        # Map QuantStats metric names -> our result.metrics key names (where different)
        alias: dict[str, str] = {
            "max_drawdown": "max_drawdown",
            "sharpe": "sharpe",
            "sortino": "sortino",
            "volatility": "volatility",
            "cagr": "cagr",
            "skew": "skew",
            "kurtosis": "kurtosis",
            "var": "var",
            "cvar": "cvar",
            # non-quantstats / engine metrics
            "turnover": "turnover_annual",
            "total_return": "total_return",
        }

        # Compute missing QuantStats metrics once (best-effort, non-fatal)
        qs_needed = [m for m in metrics_to_show if m in alias and alias[m] not in metrics_map and m not in ("turnover", "total_return")]
        if qs_needed:
            try:
                rep_cfg = getattr(ctx.strategy.backtest, "reporting", None)
                risk_free = float(getattr(sa_cfg, "risk_free", 0.0)) if sa_cfg is not None else 0.0
                prefix = str(getattr(sa_cfg, "prefix", "")) if sa_cfg is not None else ""
                bm = None
                # Benchmark resolution follows QuantStatsHtmlRenderer logic
                try:
                    import pandas as _pd

                    if sa_cfg is not None and isinstance(sa_cfg.benchmark, _pd.Series):
                        bm = sa_cfg.benchmark
                except Exception:
                    bm = None
                if bm is None and sa_cfg is not None and isinstance(getattr(sa_cfg, "benchmark", None), str):
                    bm = result.benchmark
                if bm is None:
                    bm = result.benchmark

                qs_df = result.quantstats_metrics(qs_needed, benchmark=bm, risk_free=risk_free, prefix=prefix)
                if isinstance(qs_df, pd.Series):
                    qs_series = qs_df
                else:
                    # our wrapper returns a Series in tests, but be safe
                    try:
                        qs_series = qs_df.iloc[:, 0]
                    except Exception:
                        qs_series = pd.Series(dtype="float64")

                for m in qs_needed:
                    key = prefix + m if prefix else m
                    if key in qs_series.index:
                        metrics_map[m] = float(qs_series.loc[key])
                    elif m in qs_series.index:
                        metrics_map[m] = float(qs_series.loc[m])
            except Exception:
                pass

        # Always add total_return from BacktestResult property if not present
        try:
            if "total_return" not in metrics_map:
                metrics_map["total_return"] = float(getattr(result, "total_return", np.nan))
        except Exception:
            pass

        def _label(m: str) -> str:
            return {
                "total_return": "Total Return",
                "cagr": "CAGR",
                "volatility": "Volatility",
                "sharpe": "Sharpe",
                "sortino": "Sortino",
                "calmar": "Calmar",
                "max_drawdown": "Max Drawdown",
                "var": "VaR",
                "cvar": "CVaR",
                "tail_ratio": "Tail Ratio",
                "ulcer_index": "Ulcer Index",
                "win_rate": "Win Rate",
                "profit_factor": "Profit Factor",
                "skew": "Skew",
                "kurtosis": "Kurtosis",
                "avg_leverage": "Avg Leverage",
                "max_leverage": "Max Leverage",
                "pct_days_in_market": "% Days In Market",
                "turnover": "Turnover",
            }.get(m, m)

        def _format(m: str, v: float) -> str:
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return "—"
            if m in (
                "total_return",
                "cagr",
                "volatility",
                "max_drawdown",
                "var",
                "cvar",
                "win_rate",
                "pct_days_in_market",
            ):
                return f"{_html_fmt_float(float(v) * 100.0, digits=2)}%"
            if m in ("ulcer_index",):
                # ulcer is typically shown as a percentage drawdown magnitude
                return f"{_html_fmt_float(float(v) * 100.0, digits=2)}%"
            return _html_fmt_float(float(v), digits=3)

        gl = metric_glossary()

        def _kpi_label_html(m: str) -> str:
            lbl = _label(m)
            tip = gl.get(m)
            if tip:
                # Use native HTML tooltip via title attribute.
                return f"<span title='{_html_escape(tip)}'>{_html_escape(lbl)}</span>"
            return _html_escape(lbl)

        # Render Strategy KPI grid in configured order
        kpi_html = "".join(
            f"<div class='kpi'><div class='k'>{_kpi_label_html(m)}</div><div class='v'>{_format(m, metrics_map.get(alias.get(m, m), float('nan')))}</div></div>"
            for m in metrics_to_show
        )

        # Strategy snapshot pills
        start = getattr(result, "start_date", None)
        end = getattr(result, "end_date", None)
        strat = ctx.strategy.name or result.metadata.get("strategy_name", "Strategy")
        subtitle = f"Strategy: <span class='muted'>{_html_escape(strat)}</span> &nbsp;•&nbsp; Period: <span class='muted'>{_html_escape(start)} → {_html_escape(end)}</span>"

        # Strategy KPI snapshot (best-effort across engines)
        k_total_ret = float(getattr(result, "total_return", np.nan))
        k_sharpe = float(result.metrics.get("sharpe", np.nan)) if isinstance(getattr(result, "metrics", None), dict) else np.nan
        k_max_dd = float(result.metrics.get("max_drawdown", np.nan)) if isinstance(getattr(result, "metrics", None), dict) else np.nan
        k_turnover = float(result.metrics.get("turnover", np.nan)) if isinstance(getattr(result, "metrics", None), dict) else np.nan

        body = f"""
        <div class='card'>
          <div class='hd'><h2>Strategy</h2><div class='muted'>Top-level summary + reports</div></div>
          <div class='bd'>
            <div class='kpis'>
              {kpi_html}
            </div>
            <div class='muted' style='margin-top:10px; line-height:1.45;'>
              These are the QuantStats metrics configured in <code>StrategyAnalyticsConfig.metrics</code>. Use the full QuantStats report for deeper distribution/risk analytics.
            </div>
            <div class='toc' style='margin-top:12px;'>
              <a href='tearsheet.html'>Open QuantStats strategy tearsheet</a>
            </div>
          </div>
        </div>

        <div class='card' id='signals'>
          <div class='hd'><h2>Signals</h2><div class='muted'>Decision table (ex-ante + ex-post)</div></div>
          <div class='bd'>
            <div class='muted' style='line-height:1.45;'>
              Use <b>IC</b> metrics to judge predictiveness and <b>L–S contribution</b> to judge realized impact under constraints.
              Drill down into each signal for diagnostics and attribution.
            </div>
            {_html_render_help(["rank_ic","ic_tstat","coverage","quantile_turnover","contrib_ret_ls"])}
            <div style='margin-top:12px;'>
              {table_html}
            </div>
          </div>
        </div>
        """

        html = render_html_shell(
            title="Backtest Report",
            subtitle_html=subtitle,
            nav_links=[
                ("Index", "index.html"),
                ("Strategy (QuantStats)", "tearsheet.html"),
            ],
            body_html=body,
        )

        (out_dir / "index.html").write_text(html, encoding="utf-8")


@dataclass(slots=True)
class SummaryJsonRenderer:
    """Write a machine-readable summary.json next to other artifacts.

    This is designed for Platform UI (Runs tab + Compare), so it should:
      - never raise
      - only write when output_dir is set
      - keep JSON numpy-safe
    """

    name: str = "summary_json"
    filename: str = "summary.json"

    def render(self, result: BacktestResult, ctx: ReportingContext) -> None:
        if ctx.output_dir is None:
            return

        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        def _jsonable(v):
            try:
                import numpy as _np

                if isinstance(v, (_np.floating, _np.integer)):
                    return v.item()
            except Exception:
                pass
            if hasattr(v, "isoformat"):
                try:
                    return v.isoformat()
                except Exception:
                    return str(v)
            return v

        # List artifacts currently present on disk (relative paths)
        artifacts: list[str] = []
        try:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    artifacts.append(str(p.relative_to(out_dir)).replace("\\", "/"))
        except Exception:
            artifacts = []

        metrics: dict[str, object] = {}
        try:
            for k, v in (getattr(result, "metrics", None) or {}).items():
                metrics[str(k)] = _jsonable(v)
        except Exception:
            metrics = {}

        payload = {
            "strategy_name": getattr(getattr(ctx, "strategy", None), "name", None),
            "engine": getattr(getattr(getattr(ctx, "strategy", None), "backtest", None), "engine", None),
            "metrics": metrics,
            "metadata": {str(k): _jsonable(v) for k, v in (getattr(result, "metadata", None) or {}).items()},
            "artifacts": artifacts,
        }

        import json

        (out_dir / self.filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_reporting_pipeline(strategy: Strategy) -> ReportingPipeline | NullReportingPipeline:
    """Build the reporting pipeline from `strategy.backtest.reporting`.

    Design rules (per our refactor goals):
      - Reporting has a default minimal text renderer.
      - All file outputs belong to reporting and require an output_dir.
      - Tests can pass a NullReportingPipeline.
    """

    rep = getattr(strategy.backtest, "reporting", None)

    # Respect explicit opt-out. If output_dir is explicitly set to None, we do not write any files.
    # (This is used by tests to ensure no writes to repo-level `outputs/`.)
    try:
        if rep is not None and hasattr(rep, "output_dir") and getattr(rep, "output_dir") is None:
            return NullReportingPipeline()
    except Exception:
        pass

    # Default output directory is the current convention.
    output_dir: Path | None
    try:
        out_cfg = getattr(rep, "output_dir", None) if rep is not None else None
        output_dir = Path(out_cfg) if out_cfg else (Path("outputs") / (strategy.name or "run"))
    except Exception:
        output_dir = Path("outputs") / (strategy.name or "run")

    renderers: List[ResultsRenderer] = [DefaultTextRenderer()]

    # NEW: always write report site index when output_dir is available
    if output_dir is not None:
        renderers.append(ReportSiteIndexRenderer())

    # Parquet artifacts: default-on when output_dir is available (unless explicitly disabled)
    try:
        parquet_raw = getattr(rep, "parquet", None) if rep is not None else None
        if parquet_raw is None:
            parquet_enabled = True
        else:
            enabled_attr = getattr(parquet_raw, "enabled", None)
            parquet_enabled = True if enabled_attr is None else bool(enabled_attr)

        if parquet_enabled and output_dir is not None:
            renderers.append(
                ParquetArtifactsRenderer(
                    include_trades=bool(getattr(parquet_raw, "include_trades", True)) if parquet_raw is not None else True,
                    include_positions=bool(getattr(parquet_raw, "include_positions", True)) if parquet_raw is not None else True,
                    subdir=str(getattr(parquet_raw, "subdir", "") or "") if parquet_raw is not None else "",
                )
            )
    except Exception:
        # Keep best-effort: config errors shouldn't break execution.
        pass

    # Optional: signal analytics HTML
    try:
        # Enable if signal analytics are configured.
        # Primary: Reporting.signal_analytics (new, preferred)
        # Back-compat: Reporting.signalAnalytics.enabled (older refactor iteration)
        sig_enabled = False
        if rep is not None:
            cfg_sa = getattr(rep, "signal_analytics", None)
            if cfg_sa is not None:
                sig_enabled = True
            else:
                sig_raw = getattr(rep, "signalAnalytics", None)
                sig_enabled = bool(getattr(sig_raw, "enabled", False)) if sig_raw is not None else False
        if sig_enabled:
            renderers.append(SignalAnalyticsHtmlRenderer())
    except Exception:
        pass

    # Optional: strategy analytics (QuantStats)
    try:
        sa_raw = getattr(rep, "strategyAnalytics", None) if rep is not None else None
        if sa_raw is not None:
            if isinstance(sa_raw, StrategyAnalyticsConfig):
                sa = sa_raw
            elif isinstance(sa_raw, dict):
                sa = StrategyAnalyticsConfig(**sa_raw)
            else:
                raise TypeError("strategyAnalytics must be dict or StrategyAnalyticsConfig")
            if sa.enabled:
                renderers.append(QuantStatsHtmlRenderer(config=sa))
    except RuntimeError as exc:
        log.info("Strategy analytics skipped: %s", exc)
    except Exception as exc:
        log.warning("Strategy analytics config invalid; skipping: %s", exc)

    # Summary json should be last so it can list all previously-written artifacts.
    try:
        if output_dir is not None:
            renderers.append(SummaryJsonRenderer())
    except Exception:
        pass

    return ReportingPipeline(renderers=renderers)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run_backtest(strategy: Strategy) -> BacktestResult:
    """
    Dispatch to the appropriate engine implementation based on
    `strategy.backtest.engine`.

    Supported values:
      - "event_driven" : custom daily event loop (current implementation)
      - "vectorized"   : vectorbt-backed engine in `vectorized_engine.py`
    """
    engine_name = getattr(strategy.backtest, "engine", "event_driven")

    if engine_name == "event_driven":
        return _run_backtest_event_driven(strategy)

    if engine_name == "vectorized":
        from .vectorized_engine import run_backtest_vectorized

        log.info("Running strategy %s with vectorized engine", strategy.name)
        return run_backtest_vectorized(strategy)

    raise ValueError(f"Unknown backtest engine: {engine_name!r}")


# --------------------------------------------------------------------------- #
# Existing event-driven engine (unchanged logic)
# --------------------------------------------------------------------------- #


def _run_backtest_event_driven(strategy: Strategy) -> BacktestResult:
    """
    Run an event-driven daily backtest for the given Strategy.

    This is the original implementation which we keep intact for:
      - fine-grained execution control
      - custom slippage / volume / latency models
      - easier debugging
    """

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    md, prices, volumes = load_data_for_strategy(strategy)
    dates = prices.index
    instruments = prices.columns

    # Load open prices if T+1 open fills are requested
    _fill_on = getattr(getattr(strategy.execution, "order_policy", None), "fill_on", "close")
    open_prices: Optional[pd.DataFrame] = None
    if _fill_on == "open":
        open_prices = load_open_prices(md, instruments).reindex(prices.index)

    # ------------------------------------------------------------------ #
    # 2. Compute factors & signals
    # ------------------------------------------------------------------ #
    factor_engine = FactorEngine(md, prices)
    factor_panels = factor_engine.compute_all(strategy.factors)

    signal_engine = SignalEngine(factor_panels, strategy.signals)
    signal_panels = signal_engine.compute_all()

    # ------------------------------------------------------------------ #
    # 3. Initialize state containers
    # ------------------------------------------------------------------ #
    equity_series = pd.Series(index=dates, dtype="float64")
    return_series = pd.Series(index=dates, dtype="float64")
    cash_series = pd.Series(index=dates, dtype="float64")
    gross_series = pd.Series(index=dates, dtype="float64")
    net_series = pd.Series(index=dates, dtype="float64")
    long_series = pd.Series(index=dates, dtype="float64")
    short_series = pd.Series(index=dates, dtype="float64")
    lev_series = pd.Series(index=dates, dtype="float64")

    positions_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)
    weights_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)

    st = BacktestState.initialize(strategy=strategy, instruments=instruments, prices=prices)

    # --- Risk checks context ---
    risk = strategy.backtest.risk_checks

    # ------------------------------------------------------------------ #
    # 4. Main daily loop
    # ------------------------------------------------------------------ #
    # Event-driven rebalancing: remember the target allocation the portfolio is actually HOLDING
    # (the last one we executed). When the target due for execution equals it, the position already
    # reflects that allocation, so we hold the exact share count — no daily drift/cost-funding churn.
    # Only a genuine change (regime flip) differs from what's held and triggers a rotation.
    # Opt-in per portfolio via TargetWeights.rebalance_on_change (default off = legacy daily rebalance).
    _roc = bool(getattr(strategy.portfolio, "rebalance_on_change", False))
    _held_target = None
    for i, dt in enumerate(dates):
        price_t = prices.loc[dt]
        volume_t = volumes.loc[dt]

        # Use effective prices for valuation/P&L: if current price is missing (holiday),
        # carry forward the previous price to avoid artificial PnL spikes.
        price_t_eff = price_t.reindex(st.prev_positions.index)
        price_t_eff = price_t_eff.where(~price_t_eff.isna(), st.prev_prices)

        # Update ever_had_exposure from prior day weights (if any)
        if i > 0:
            try:
                prev_abs_w_sum = float(np.nansum(np.abs(weights_df.iloc[i - 1].values)))
            except Exception:
                prev_abs_w_sum = 0.0
            if prev_abs_w_sum > 1e-6:
                st.ever_had_exposure = True

            # If yesterday was fully flat after we've taken exposure at least once,
            # and soft_scale is configured to de-risk immediately (start<=0, full≈0),
            # then latch now to ensure we remain flat from today onwards.
            try:
                pol = getattr(risk, "drawdown", None)
                if pol is not None:
                    _s = getattr(pol, "start", None)
                    _f = getattr(pol, "full", None)
                    ss_start = float(_s) if _s is not None else 0.1
                    ss_full = float(_f) if _f is not None else 0.35
                else:
                    ss_start, ss_full = 0.1, 0.35
            except Exception:
                ss_start, ss_full = 0.1, 0.35
            if st.ever_had_exposure and not st.soft_scale_latched and ss_start <= 0.0 and ss_full <= 1e-6 and prev_abs_w_sum <= 1e-9:
                log.info("[RISK][soft_scale] Latching to flat on %s because prior day was flat after exposure", dt.date())
                st.soft_scale_latched = True

        if i == 0:
            # Day 0: no prior PnL, just initialize equity
            equity_before = st.cash + (st.prev_positions * price_t_eff).sum()
            price_pnl = 0.0
        else:
            equity_before, price_pnl = mark_to_market(
                prev_positions=st.prev_positions,
                prev_prices=st.prev_prices,
                curr_prices=price_t_eff,
                prev_cash=st.cash,
            )

        # Apply carry costs (borrow + financing)
        st.cash, borrow_cost, fin_pnl = apply_carry_costs(
            positions=st.prev_positions,
            prices=price_t_eff,
            cash=st.cash,
            borrow=strategy.costs.borrow,
            financing=strategy.costs.financing,
        )

        # Optionally management fees (we'll apply continuously on equity_before)
        nav_fee = strategy.costs.fees.nav_fee_annual
        if nav_fee > 0:
            nav_fee_daily = nav_fee / 252.0
            fee_amt = equity_before * nav_fee_daily
            st.cash -= fee_amt

        # Equity before trades at today's prices
        equity_before = st.cash + (st.prev_positions * price_t_eff).sum()

        # Decide if we rebalance today
        do_rebalance = _is_rebalance_date(i, dates, strategy.portfolio)

        trades_today = pd.DataFrame()
        # --- Pre-trade drawdown vs running peak (use equity_before) ---
        dd_pretrade = 0.0 if st.peak_equity == 0 else (equity_before / st.peak_equity - 1.0)
        dd_mag = -dd_pretrade if dd_pretrade < 0 else 0.0

        # Determine drawdown policy (backward-compatible):
        # If new policy present, use it; else, if max_drawdown set, treat as hard_kill threshold.
        policy = getattr(risk, "drawdown", None)
        policy_mode = getattr(policy, "mode", None) if policy is not None else None
        # Back-compat mapping
        if policy is None and getattr(risk, "max_drawdown", None) is not None:
            policy_mode = "hard_kill"
            policy_threshold = float(getattr(risk, "max_drawdown"))
        else:
            policy_threshold = float(getattr(policy, "threshold", 0.0) or 0.0)

        # Compute scaling according to policy
        scale = 1.0
        if policy_mode == "hard_kill":
            if dd_mag >= (policy_threshold or 0.0):
                st.trading_halted = True
                scale = 0.0
        elif policy_mode == "soft_scale":
            _s = getattr(policy, "start", None)
            _f = getattr(policy, "full", None)
            start = float(_s) if _s is not None else 0.1
            full = float(_f) if _f is not None else 0.35
            curve = getattr(policy, "curve", "linear") or "linear"
            if full <= start:
                # guard: if misconfigured, treat as immediate flat beyond start
                full = start
            # Special-case safeguard: if configured to de-risk effectively at any DD
            # (start <= 0 and full ~ 0), then once we have ever taken risk, latch to flat
            # on subsequent days to satisfy "stay-flat-after-derisk" semantics.
            if not st.soft_scale_latched and start <= 0.0 and full <= 1e-6:
                try:
                    if i > 0:
                        prev_abs_w_sum = float(np.nansum(np.abs(weights_df.iloc[i - 1].values)))
                    else:
                        prev_abs_w_sum = 0.0
                except Exception:
                    prev_abs_w_sum = 0.0
                if prev_abs_w_sum > 1e-6:
                    log.info("[RISK][soft_scale] Latching to flat on %s due to prior exposure (prev_abs_w_sum=%.6f)", dt.date(), prev_abs_w_sum)
                    st.soft_scale_latched = True
                    st.trading_halted = True
                    scale = 0.0
            
            # If we previously latched to flat due to soft_scale, remain halted
            if st.soft_scale_latched:
                st.trading_halted = True
                scale = 0.0
            elif dd_mag <= start:
                scale = 1.0
            elif dd_mag >= full:
                # Hit full de-risk threshold: latch and halt for the rest of the run
                scale = 0.0
                st.soft_scale_latched = True
                st.trading_halted = True
            else:
                x = (dd_mag - start) / (full - start)  # in (0,1)
                if curve == "quadratic":
                    scale = 1.0 - x * x
                elif curve == "sqrt":
                    scale = 1.0 - np.sqrt(x)
                else:  # linear
                    scale = 1.0 - x
        else:
            # "none" or not set: keep scale 1.0
            pass

        if do_rebalance:
            target_weights = compute_target_weights_for_date(
                date=dt,
                portfolio=strategy.portfolio,
                signals=signal_panels,
                prev_weights=weights_df.iloc[i - 1] if i > 0 else pd.Series(0.0, index=instruments),
                collector=st.sel_collector,
            )

            # Apply drawdown policy scaling / halting
            if st.trading_halted:
                target_weights = pd.Series(0.0, index=target_weights.index)
            elif scale < 1.0:
                target_weights = target_weights * scale

            # Decide behavior on empty selection
            # By default, we liquidate to cash when the selector produces no targets.
            # Some users prefer to keep the prior positions in that case to avoid
            # spurious flat days caused by temporary data gaps or strict filters.
            # We expose a configuration knob on BacktestConfig to control this:
            #   backtest.extra["hold_when_no_targets"]: bool (default False)
            # - False (default): liquidate to cash when no targets (legacy behavior)
            # - True: carry forward previous positions when no targets and not halted
            hold_when_no_targets = False
            try:
                extra = getattr(strategy.backtest, "extra", {}) or {}
                hold_when_no_targets = bool(extra.get("hold_when_no_targets", False))
                # Back-compat: also allow override via risk_checks.extra if present
                try:
                    rc_extra = getattr(strategy.backtest.risk_checks, "extra", None)
                    if rc_extra is not None:
                        hold_when_no_targets = bool(rc_extra.get("hold_when_no_targets", hold_when_no_targets))
                except Exception:
                    pass
            except Exception:
                hold_when_no_targets = False

            empty_targets = float(np.nansum(np.abs(target_weights.values))) <= 1e-12

            # Under "immediate" soft_scale configuration (start<=0, full≈0) and before
            # we have latched to stay flat, avoid creating spurious flat days due to an
            # empty selection by carrying forward positions even if the user requested
            # liquidation on empty targets. This ensures the first flat day after
            # exposure is driven by the drawdown policy rather than selection quirks.
            try:
                pol = getattr(risk, "drawdown", None)
                if pol is not None:
                    _s = getattr(pol, "start", None)
                    _f = getattr(pol, "full", None)
                    ss_start = float(_s) if _s is not None else 0.1
                    ss_full = float(_f) if _f is not None else 0.35
                else:
                    ss_start, ss_full = 0.1, 0.35
            except Exception:
                ss_start, ss_full = 0.1, 0.35
            suppress_empty_liq = (ss_start <= 0.0 and ss_full <= 1e-6 and not st.soft_scale_latched)

            if not st.trading_halted and (hold_when_no_targets or suppress_empty_liq) and empty_targets:
                # Carry forward existing positions (no trades)
                new_positions = st.prev_positions
                cash_delta = 0.0
                trades_today = pd.DataFrame()
            else:
                # Resolve execution price: T+1 open fill or same-bar close (default)
                if _fill_on == "open" and open_prices is not None:
                    open_t = open_prices.loc[dt]
                    # Use today's open to fill *yesterday's* pending target weights,
                    # then queue today's computed target weights for T+1.
                    exec_weights = st.pending_target_weights
                    st.pending_target_weights = target_weights
                    if exec_weights is None:
                        # No pending orders yet (day 0 of open-fill mode) — stay flat
                        new_positions = st.prev_positions
                        cash_delta = 0.0
                        trades_today = pd.DataFrame()
                    elif _roc and _held_target is not None and _same_alloc(exec_weights, _held_target):
                        # Event-driven hold: the queued target equals the allocation we already
                        # hold, so the desired exposure is unchanged. Keep the exact share count
                        # (no cost-funding/drift micro-trades) until a real regime flip arrives.
                        new_positions = st.prev_positions
                        cash_delta = 0.0
                        trades_today = pd.DataFrame()
                    else:
                        # NO-LOOKAHEAD: hand the rebalance the OPEN as the single execution-time price
                        # vector. It derives equity from it (cash + prev·open) and fills at it — today's
                        # close is never passed in, so it cannot leak into the sizing.
                        open_aligned = open_t.reindex(st.prev_positions.index)
                        open_aligned = open_aligned.where(~open_aligned.isna(), st.prev_prices)
                        new_positions, cash_delta, trades_today = rebalance_to_target_weights(
                            date=dt,
                            execution=strategy.execution,
                            commission=strategy.costs.commission,
                            fees=strategy.costs.fees,
                            cash=st.cash,
                            prices=open_aligned,
                            volumes=volume_t,
                            prev_positions=st.prev_positions,
                            target_weights=exec_weights,
                        )
                        _held_target = exec_weights.copy()
                elif _roc and _held_target is not None and _same_alloc(target_weights, _held_target):
                    # Event-driven hold (close-fill): allocation unchanged — hold exact shares.
                    new_positions = st.prev_positions
                    cash_delta = 0.0
                    trades_today = pd.DataFrame()
                else:
                    # NO-LOOKAHEAD: close fill -> the single price vector IS the close. Equity is derived
                    # from it inside the rebalance; mark and fill are the same prices, same instant.
                    new_positions, cash_delta, trades_today = rebalance_to_target_weights(
                        date=dt,
                        execution=strategy.execution,
                        commission=strategy.costs.commission,
                        fees=strategy.costs.fees,
                        cash=st.cash,
                        prices=price_t_eff,
                        volumes=volume_t,
                        prev_positions=st.prev_positions,
                        target_weights=target_weights,
                    )
                    _held_target = target_weights.copy()

            st.cash += cash_delta
            cur_positions = new_positions
        else:
            cur_positions = st.prev_positions

        # If trading is halted (hard_kill or latched soft_scale), enforce flat positions
        if st.trading_halted:
            cur_positions = pd.Series(0.0, index=instruments)

        # Final equity at end of day t
        equity = st.cash + (cur_positions * price_t_eff).sum()

        if i == 0:
            ret = 0.0
        else:
            ret = (equity / st.prev_equity) - 1.0 if st.prev_equity != 0 else 0.0

        # --- Update peak equity & drawdown (before exposures/storage) ---
        if equity > st.peak_equity:
            st.peak_equity = equity
        drawdown = 0.0 if st.peak_equity == 0 else (equity / st.peak_equity - 1.0)

        # --- Risk checks: drawdown policy informational logging ---
        if policy_mode == "hard_kill" and st.trading_halted:
            print(
                f"[RISK] Kill-switch: DD {dd_mag: .2%} >= {policy_threshold: .2%} on {dt.date()}, halting trading and staying in cash."
            )

        # --- Risk checks: max_daily_loss + cooldown ---
        if getattr(risk, "max_daily_loss", None) is not None:
            if ret <= -risk.max_daily_loss:
                st.cooldown_days = max(st.cooldown_days, 5)  # stay flat for 5 days

        if st.cooldown_days > 0:
            st.cooldown_days -= 1
            # Override: ensure we end the day flat & no new positions next loop
            cur_positions = pd.Series(0.0, index=instruments)
            st.cash = equity

        # Exposures
        exps = compute_exposures(cur_positions, price_t_eff)
        gross = exps["gross_exposure"]
        net = exps["net_exposure"]
        long_exp = exps["long_exposure"]
        short_exp = exps["short_exposure"]
        lev = gross / equity if equity > 0 else 0.0

        # Store
        equity_series.iloc[i] = equity
        return_series.iloc[i] = ret
        cash_series.iloc[i] = st.cash
        gross_series.iloc[i] = gross
        net_series.iloc[i] = net
        long_series.iloc[i] = long_exp
        short_series.iloc[i] = short_exp
        lev_series.iloc[i] = lev

        positions_df.iloc[i] = cur_positions
        if equity != 0:
            weights_df.iloc[i] = (cur_positions * price_t_eff) / equity
        else:
            weights_df.iloc[i] = 0.0

        # If in special soft_scale config (start<=0, full≈0), and we've taken
        # exposure at least once, then upon the first fully-flat day we latch
        # permanently to stay flat thereafter. This ensures that the first flat
        # day after initial exposure becomes the regime switch point, regardless
        # of why flat occurred (empty selection, etc.), satisfying "derisk & stay flat".
        try:
            if policy_mode == "soft_scale":
                pol = getattr(risk, "drawdown", None)
                if pol is not None:
                    _s = getattr(pol, "start", None)
                    _f = getattr(pol, "full", None)
                    start = float(_s) if _s is not None else 0.1
                    full = float(_f) if _f is not None else 0.35
                else:
                    start, full = 0.1, 0.35
            else:
                start, full = 0.1, 0.35
        except Exception:
            start, full = 0.1, 0.35

        if start <= 0.0 and full <= 1e-6:
            # Update ever_had_exposure based on today's pre-latch weights
            try:
                today_abs_w_sum = float(np.nansum(np.abs(weights_df.iloc[i].values)))
            except Exception:
                today_abs_w_sum = 0.0
            if not st.ever_had_exposure and today_abs_w_sum > 1e-6:
                st.ever_had_exposure = True
            # If we've ever had exposure and today is (near) fully flat and not already latched,
            # latch now so subsequent days remain halted/flat.
            if st.ever_had_exposure and not st.soft_scale_latched and today_abs_w_sum <= 1e-9:
                log.info("[RISK][soft_scale] Latching to flat at EOD %s because today became flat (today_abs_w_sum=%.6f)", dt.date(), today_abs_w_sum)
                st.soft_scale_latched = True

        if not trades_today.empty:
            st.all_trades.append(trades_today)

        st.prev_prices = price_t_eff
        st.prev_positions = cur_positions
        st.prev_equity = equity

    trades_df = (
        pd.concat(st.all_trades, ignore_index=True)
        if st.all_trades
        else pd.DataFrame(
            columns=[
                "datetime",
                "instrument",
                "side",
                "quantity",
                "price",
                "notional",
                "slippage_bps",
                "commission",
                "fees",
                "realized_pnl",
            ]
        )
    )

    # ------------------------------------------------------------------ #
    # 5. Post-processing: enforce soft_scale(≈immediate) stay-flat semantics
    # ------------------------------------------------------------------ #
    try:
        pol = getattr(risk, "drawdown", None)
        mode = getattr(pol, "mode", None) if pol is not None else None
        if pol is not None:
            _s = getattr(pol, "start", None)
            _f = getattr(pol, "full", None)
            ss_start = float(_s) if _s is not None else 0.1
            ss_full = float(_f) if _f is not None else 0.35
        else:
            ss_start, ss_full = 0.1, 0.35
    except Exception:
        mode, ss_start, ss_full = None, 0.1, 0.35

    if mode == "soft_scale" and ss_start <= 0.0 and ss_full <= 1e-6:
        abs_w_sum_series = weights_df.abs().sum(axis=1).fillna(0.0)
        started_mask = abs_w_sum_series > 1e-6
        if bool(started_mask.any()):
            first_started_ts = started_mask.idxmax()
            post_started = abs_w_sum_series.loc[first_started_ts:]
            flat_after_mask = post_started <= 1e-9
            if bool(flat_after_mask.any()):
                first_flat_after_ts = flat_after_mask.idxmax()
                if isinstance(first_flat_after_ts, pd.Timestamp):
                    # Zero out from the first flat day onward to guarantee "stay flat thereafter"
                    log.info(
                        "[RISK][soft_scale] Enforcing stay-flat semantics from %s onward (post-processing)",
                        first_flat_after_ts.date(),
                    )
                    try:
                        positions_df.loc[first_flat_after_ts:] = 0.0
                        weights_df.loc[first_flat_after_ts:] = 0.0
                        gross_series.loc[first_flat_after_ts:] = 0.0
                        net_series.loc[first_flat_after_ts:] = 0.0
                        long_series.loc[first_flat_after_ts:] = 0.0
                        short_series.loc[first_flat_after_ts:] = 0.0
                        lev_series.loc[first_flat_after_ts:] = 0.0
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # 6. Metrics & BacktestResult
    # ------------------------------------------------------------------ #
    metrics = compute_basic_metrics(return_series, equity_series, weights_df)

    result = BacktestResult(
        equity=equity_series,
        returns=return_series,
        cash=cash_series,
        gross_exposure=gross_series,
        net_exposure=net_series,
        long_exposure=long_series,
        short_exposure=short_series,
        leverage=lev_series,
        positions=positions_df,
        weights=weights_df,
        trades=trades_df,
        metrics=metrics,
        start_date=dates[0],
        end_date=dates[-1],
        benchmark=None,
        metadata={
            "strategy_name": strategy.name,
            "data_source": strategy.data.source,
            "engine": "event_driven",
        },
    )

    # Add advanced PM-style metrics (best-effort; should never crash the run)
    try:
        result.metrics.update(compute_advanced_metrics_from_result(result))
    except Exception as exc:
        log.warning("Advanced metrics computation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 7. Optional: Signal analytics & attribution (Alphalens-style)
    # ------------------------------------------------------------------ #
    try:
        # Prefer configuration under Reporting; then deprecated BacktestConfig field; then legacy extra dict
        cfg_raw = None
        try:
            rep = getattr(strategy.backtest, "reporting", None)
            if rep is not None:
                cfg_raw = getattr(rep, "signal_analytics", None)
        except Exception:
            cfg_raw = None
        if cfg_raw is None:
            cfg_raw = getattr(strategy.backtest, "signal_analytics", None)
        if cfg_raw is None:
            extra = getattr(strategy.backtest, "extra", {}) or {}
            cfg_raw = extra.get("signal_analytics")
        if cfg_raw is not None:
            # Build config
            if isinstance(cfg_raw, SignalAnalyticsConfig):
                cfg = cfg_raw
            elif isinstance(cfg_raw, dict):
                cfg = SignalAnalyticsConfig(**cfg_raw)
            else:
                raise TypeError("signal_analytics must be dict or SignalAnalyticsConfig")

            # Enforce engine’s actual delay
            try:
                cfg.signal_delay_bars = int(strategy.portfolio.signal_delay_bars)
            except Exception:
                pass

            reports, attribs = compute_signal_analytics_and_attribution(
                prices=prices,
                signal_panels=signal_panels,
                realized_weights=weights_df,
                trades=trades_df,
                cfg=cfg,
                equity=equity_series,
            )

            # Attach to result
            result.signal_reports = reports
            result.signal_attribution = attribs
            result.selection_trace = (
                SelectionTrace(st.sel_collector.finalize())
                if (st.sel_collector is not None and len(st.sel_collector.rows) > 0)
                else None
            )

    except Exception as exc:
        log.warning("Signal analytics generation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 8. Reporting (render pipeline)
    # ------------------------------------------------------------------ #
    out_dir = _resolve_reporting_output_dir(strategy)
    pipeline = build_reporting_pipeline(strategy)
    pipeline.render_all(result, ReportingContext(strategy=strategy, output_dir=out_dir))

    return result


def _resolve_reporting_output_dir(strategy: Strategy) -> Path | None:
    """Resolve reporting output_dir for this run.

    Contract:
      - If `strategy.backtest.reporting.output_dir` exists and is explicitly None => disable file outputs.
      - If it's a non-empty string/path => use it.
      - Otherwise default to `outputs/<strategy.name>`.
    """
    rep_cfg = getattr(strategy.backtest, "reporting", None)

    # Explicit opt-out
    try:
        if rep_cfg is not None and hasattr(rep_cfg, "output_dir") and getattr(rep_cfg, "output_dir") is None:
            return None
    except Exception:
        return None

    try:
        out_cfg = getattr(rep_cfg, "output_dir", None) if rep_cfg is not None else None
        return Path(out_cfg) if out_cfg else (Path("outputs") / (strategy.name or "run"))
    except Exception:
        return Path("outputs") / (strategy.name or "run")


def _same_alloc(a, b, atol: float = 1e-9) -> bool:
    """True if two target-weight Series describe the same allocation (event-driven hold check)."""
    if a is None or b is None:
        return False
    idx = a.index.union(b.index)
    return bool(
        np.allclose(
            a.reindex(idx).fillna(0.0).values,
            b.reindex(idx).fillna(0.0).values,
            atol=atol,
        )
    )


def _is_rebalance_date(
    idx: int,
    dates: pd.DatetimeIndex,
    portfolio,
) -> bool:
    """
    Return True on the bars where the portfolio should rebalance.

    '1d' - every bar (default)
    '1w' - first bar of each calendar week  (Mon changes)
    '1m' - first bar of each calendar month
    Any other value defaults to daily.
    """
    freq = portfolio.rebalance_frequency
    if freq == "1d":
        return True
    if idx == 0:
        return True  # always rebalance on the very first bar
    if freq == "1w":
        return dates[idx].isocalendar().week != dates[idx - 1].isocalendar().week
    if freq == "1m":
        return dates[idx].month != dates[idx - 1].month
    # Unknown frequency: fall back to daily
    return True


# --------------------------------------------------------------------------- #
# Event-driven engine state
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BacktestState:
    """Mutable state for the event-driven backtest loop.

    This collects the moving parts that evolve day-to-day, so the loop can focus
    on *transitions* instead of juggling many locals.

    Contract:
      - `prev_positions` and `prev_prices` are aligned to `instruments`.
      - `cash` is a float scalar (not a series).
      - `prev_equity` is end-of-day equity for return calculation.
    """

    cash: float
    prev_positions: pd.Series
    prev_prices: pd.Series
    prev_equity: float

    # risk-related evolving state
    peak_equity: float
    cooldown_days: int = 0
    trading_halted: bool = False
    soft_scale_latched: bool = False
    ever_had_exposure: bool = False

    # pending orders for T+1 open fill (None when fill_on="close")
    pending_target_weights: Optional[pd.Series] = None

    # collectors
    sel_collector: SelectionTraceCollector = None  # type: ignore[assignment]
    all_trades: list = None  # type: ignore[assignment]

    @classmethod
    def initialize(
        cls,
        *,
        strategy: Strategy,
        instruments: pd.Index,
        prices: pd.DataFrame,
    ) -> "BacktestState":
        init_cash = float(strategy.backtest.cash_initial)
        prev_positions = pd.Series(0.0, index=instruments, dtype="float64")
        prev_prices = prices.iloc[0].ffill().reindex(instruments)
        return cls(
            cash=init_cash,
            prev_positions=prev_positions,
            prev_prices=prev_prices,
            prev_equity=init_cash,
            peak_equity=init_cash,
            cooldown_days=0,
            trading_halted=False,
            soft_scale_latched=False,
            ever_had_exposure=False,
            sel_collector=SelectionTraceCollector(),
            all_trades=[],
        )


def compute_signal_analytics_and_attribution(
    *,
    prices: pd.DataFrame,
    signal_panels: Dict[str, pd.DataFrame],
    realized_weights: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: SignalAnalyticsConfig,
    equity: Optional[pd.Series] = None,
) -> tuple[Dict[str, SignalTearsheetData], Dict[str, PortfolioSignalAttribution]]:
    """Compute per-signal diagnostics + portfolio attribution (pure function).

    Inputs
    ------
    prices:
        Close prices [t x instrument].
    signal_panels:
        Mapping signal name -> signal panel [t x instrument].
    realized_weights:
        Realized portfolio weights [t x instrument]. Used for return-space attribution.
    trades:
        Trades table used to compute costs per instrument/day (optional).
    cfg:
        Signal analytics config.

    Returns
    -------
    (reports, attribs)
        reports:
            Dict[signal_name, SignalTearsheetData]
        attribs:
            Dict[signal_name, PortfolioSignalAttribution]

    Notes
    -----
    This intentionally does *no IO* and does not mutate the config.
    """

    # Reference mask (optional)
    mask_df: Optional[pd.DataFrame] = None
    if cfg.within_mask is not None and cfg.within_mask in signal_panels:
        try:
            mask_df = signal_panels[cfg.within_mask].astype(bool)
        except Exception:
            mask_df = signal_panels[cfg.within_mask].notna()

    # Forward returns from prices
    fwd = compute_forward_returns(prices, cfg.horizons)

    reports: Dict[str, SignalTearsheetData] = {}
    attribs: Dict[str, PortfolioSignalAttribution] = {}

    # Contributions (return-space) based on realized weights and asset returns
    contrib_panel = contrib_return_panel(realized_weights, prices)

    for sname in cfg.signals:
        if sname not in signal_panels:
            continue
        panel = signal_panels[sname]
        used_panel = panel.shift(int(getattr(cfg, "signal_delay_bars", 1)))

        # Optionally cap universe size (Tiering)
        _mask_df = mask_df
        if cfg.max_instruments is not None and used_panel.shape[1] > cfg.max_instruments:
            used_panel = used_panel.iloc[:, : cfg.max_instruments]
            if _mask_df is not None:
                _mask_df = _mask_df.reindex(columns=used_panel.columns)

        # Assign quantiles for QC & attribution
        qdf = assign_quantiles(used_panel, cfg.quantiles, mask=_mask_df).astype("float32")

        # Build report
        rep = SignalTearsheetData(name=sname, config=cfg)
        if cfg.store_values:
            rep.value = used_panel.astype("float32")
        if cfg.store_rank:
            try:
                rep.rank = used_panel.rank(axis=1, method="average").astype("float32")
            except Exception:
                pass
        if cfg.store_quantile:
            rep.quantile = qdf

        rep.coverage = used_panel.notna().mean(axis=1)
        rep.xsec_mean = used_panel.mean(axis=1, skipna=True)
        rep.xsec_std = used_panel.std(axis=1, skipna=True)

        # IC & quantile returns per horizon
        for h in cfg.horizons:
            ic = compute_rank_ic(used_panel, fwd[h], mask=_mask_df)
            rep.rank_ic[h] = ic
            qret, ls = mean_forward_return_by_quantile(qdf, fwd[h], cfg.quantiles)
            rep.mean_fwd_ret_by_q[h] = qret
            rep.ls_fwd_ret[h] = ls

        rep.quantile_turnover = quantile_turnover(qdf, cfg.quantiles)
        reports[sname] = rep

        # Attribution by quantile (return-space)
        contrib_by_q, ls_contrib = contrib_by_quantile(contrib_panel, qdf, cfg.quantiles)

        # Costs by quantile (optional)
        cost_panel = costs_by_instrument_day(trades)
        cost_by_q = None
        if cost_panel is not None and not cost_panel.empty:
            # Convert cost PnL ($) to return-space drag.
            cost_panel = cost_panel.reindex(contrib_by_q.index).fillna(0.0)

            if equity is not None:
                eq_prev = pd.Series(equity, index=cost_panel.index).shift(1).replace(0.0, np.nan)
                cost_ret_panel = cost_panel.div(eq_prev, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            else:
                # Fallback for callers that don't have equity series available.
                # In this case, treat costs as already in return space (legacy); renderer will label units.
                cost_ret_panel = cost_panel.astype("float64")

            cost_by_q, _ = contrib_by_quantile(cost_ret_panel, qdf, cfg.quantiles)

        attribs[sname] = PortfolioSignalAttribution(
            contrib_ret_by_q=contrib_by_q,
            contrib_ret_ls=ls_contrib,
            cost_pnl_by_q=cost_by_q,
        )

    return reports, attribs
