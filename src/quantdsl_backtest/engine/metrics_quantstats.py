# src/quantdsl_backtest/engine/metrics_quantstats.py

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
import warnings

def compute_quantstats_metrics(
    returns: pd.Series,
    metric_names: Iterable[str],
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.0,
) -> Dict[str, float]:
    """Compute a configurable set of metrics via quantstats.stats.

    Parameters
    ----------
    returns:
        Strategy returns as a pandas Series (daily by convention).
    metric_names:
        Names of functions in ``quantstats.stats`` to call, e.g.
        ["cagr", "sharpe", "sortino", "max_drawdown", "volatility"].
    benchmark:
        Optional benchmark returns Series for metrics that support it
        (e.g. alpha, beta, r_squared, information_ratio, etc.).
    risk_free:
        Risk-free rate used for Sharpe/Sortino where relevant. Interpreted
        in the same way quantstats does (per-period or annualised depending
        on the metric).

    Returns
    -------
    Dict[str, float]
        Mapping from metric name to float value.

    Notes
    -----
    This is a thin adapter around :mod:`quantstats.stats`. It introspects
    each function's signature and only passes arguments it supports, so you
    can safely mix metrics that do and do not take ``benchmark`` or ``rf``.
    """
    try:
        import quantstats as qs  # type: ignore
    except ImportError as exc:  # pragma: no cover - only hit if qs missing
        raise RuntimeError(
            "quantstats is not installed. Install it with `pip install quantstats` "
            "to enable advanced analytics."
        ) from exc

    import inspect

    returns = returns.astype("float64")
    if benchmark is not None:
        benchmark = benchmark.astype("float64")

    out: Dict[str, float] = {}

    for name in metric_names:
        fn = getattr(qs.stats, name, None)
        if fn is None:
            raise ValueError(f"Unknown quantstats metric '{name}'")

        sig = inspect.signature(fn)
        kwargs = {}
        if "benchmark" in sig.parameters and benchmark is not None:
            kwargs["benchmark"] = benchmark
        if "rf" in sig.parameters:
            kwargs["rf"] = risk_free

        # Many quantstats stats emit RuntimeWarnings (divide by zero, empty slices)
        # on degenerate inputs (e.g., all‑zero returns). Suppress those from
        # third‑party libs to keep our test suite noise‑free while preserving
        # the numeric output semantics used by quantstats.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"numpy\._core\._methods",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"numpy\.lib\._function_base_impl",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"scipy\.stats\._distn_infrastructure",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"quantstats\.stats",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"quantstats\.stats",
                message=r"No non-zero returns found.*",
            )
            value = fn(returns, **kwargs)
        try:
            out[name] = float(value)
        except Exception:
            # Some metrics may return arrays/Series; take scalar if possible.
            try:
                out[name] = float(getattr(value, "item", lambda: value)())
            except Exception:
                raise TypeError(
                    f"Quantstats metric '{name}' returned non-scalar value {value!r}"
                )

    return out


def _inject_report_site_shell(
    html_text: str,
    *,
    title: str,
    nav_links: list[tuple[str, str]],
    subtitle_html: str = "",
) -> str:
    """Best-effort wrapper for QuantStats HTML.

    We keep QuantStats' internal body content intact but:
      - inject our CSS (dark theme)
      - inject a small header + navigation

    This is intentionally string-based and tolerant of minor QuantStats changes.
    """

    from .analytics.render_html_utils import default_css, render_topnav

    css = default_css()
    extra_css = """
    /* QuantStats overrides: make default light theme usable on dark. */
    body { background: var(--bg) !important; color: var(--text) !important; }
    a { color: var(--accent) !important; }

    /* Tables (QuantStats uses plain tables + pandas/stats tables) */
    table { color: var(--text) !important; background: transparent !important; }
    thead, tbody { background: transparent !important; }
    th, td { border-color: var(--line) !important; background: transparent !important; }
    th { color: var(--muted) !important; }

    /* Some QuantStats templates/embedded styles set white backgrounds */
    .table, .dataframe { background: transparent !important; }
    .table thead th, .dataframe thead th { background: rgba(255,255,255,.03) !important; }

    /* Containers */
    .container, .content, .main { background: transparent !important; }

    /* Common 'metrics' header rows (best-effort) */
    .metrics thead th, .metrics th { background: rgba(255,255,255,.03) !important; }
    """

    header = f"""
<div class='page'>
  <div class='hero'>
    <div>
      <h1 class='title'>{title}</h1>
      <div class='subtitle'>{subtitle_html}</div>
      {render_topnav(links=nav_links)}
    </div>
  </div>
</div>
"""

    # Inject CSS first
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", f"<style>{css}\n{extra_css}</style></head>", 1)
    else:
        html_text = f"<style>{css}\n{extra_css}</style>" + html_text

    # Insert header right after <body ...>
    import re

    m = re.search(r"<body[^>]*>", html_text, flags=re.IGNORECASE)
    if m:
        insert_at = m.end()
        html_text = html_text[:insert_at] + header + html_text[insert_at:]
    else:
        html_text = header + html_text

    return html_text


def generate_quantstats_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    output: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> None:
    """Generate an HTML tearsheet via ``quantstats.reports.html``.

    Parameters
    ----------
    returns:
        Strategy returns Series.
    benchmark:
        Optional benchmark returns Series, or ticker string understood by
        quantstats (e.g. "SPY").
    output:
        Path to HTML file. If None, quantstats' default is used (it will
        generate a temporary file and open it in a browser).
    title:
        Optional report title.
    **kwargs:
        Passed through to :func:`quantstats.reports.html`.
    """
    # Ensure a non-interactive matplotlib backend for headless/test environments
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")  # safe in headless CI; no-op if already set
        except Exception:
            pass
    except Exception:
        pass

    try:
        import quantstats as qs  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "quantstats is not installed. Install it with `pip install quantstats` "
            "to enable HTML tearsheets."
        ) from exc

    rets = returns.astype("float64")
    bench = benchmark.astype("float64") if isinstance(benchmark, pd.Series) else benchmark

    # Suppress noisy warnings from seaborn/numpy/scipy/quantstats that can occur
    # for degenerate inputs (e.g., constant/zero-variance returns) when rendering
    # plots inside the HTML report.
    # Additionally, forward a flag to quantstats to disable 0-variance density warnings.
    kwargs.setdefault("warn_singular", False)

    with warnings.catch_warnings():
        # seaborn API deprecation used by quantstats' internal plotting
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            module=r"seaborn\.categorical",
            message=r".*vert: bool will be deprecated.*",
        )
        # numerical runtime warnings from downstream libs during plotting/stats
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"numpy\..*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"scipy\..*",
        )
        # stats warnings emitted within quantstats itself on degenerate inputs
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"quantstats\.stats",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"quantstats\.stats",
            message=r"No non-zero returns found.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"quantstats\._plotting\.core",
            message=r"Dataset has 0 variance; skipping density estimate.*",
        )

        qs.reports.html(
            rets,
            benchmark=bench,
            output=output,
            title=title,
            **kwargs,
        )

    # Optional: post-process generated HTML to match our dark report theme.
    # Keep it best-effort and skip if output is None or file missing.
    try:
        if output:
            from pathlib import Path

            p = Path(output)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="ignore")
                wrapped = _inject_report_site_shell(
                    txt,
                    title=title or "Strategy",
                    nav_links=[
                        ("Index", "index.html"),
                    ],
                    subtitle_html="QuantStats strategy tearsheet",
                )
                p.write_text(wrapped, encoding="utf-8")
    except Exception:
        # Never fail the backtest because of HTML theming.
        pass
