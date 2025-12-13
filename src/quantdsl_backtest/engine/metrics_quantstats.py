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
    try:
        import quantstats as qs  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "quantstats is not installed. Install it with `pip install quantstats` "
            "to enable HTML tearsheets."
        ) from exc

    rets = returns.astype("float64")
    bench = benchmark.astype("float64") if isinstance(benchmark, pd.Series) else benchmark

    # Suppress seaborn's PendingDeprecationWarning about 'vert' -> 'orientation'
    # which is triggered inside quantstats when it calls seaborn's old plotting API.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            module=r"seaborn\.categorical",
            message=r".*vert: bool will be deprecated.*",
        )
        qs.reports.html(
            rets,
            benchmark=bench,
            output=output,
            title=title,
            **kwargs,
        )