# src/quantdsl_backtest/engine/portfolio_optimizer.py
"""
Risk-based / optimised weighting for a set of selected instruments.

One pure function — :func:`optimize_weights` — takes a trailing window of asset
returns and produces long-only weights summing to 1.0 via one of several
schemes. It backs the DSL's :class:`~quantdsl_backtest.dsl.portfolio.OptimizedWeight`
book weighting, but it is deliberately engine-agnostic so sfera strategies,
lab research and the R04 sleeve combiner can call it directly on any returns
panel.

Backend: `skfolio <https://skfolio.org>`_ (sklearn-API, cvxpy/CLARABEL solver)
when installed. ``equal`` and ``inverse_vol`` have pure-numpy implementations
and never need skfolio; the convex/hierarchical methods raise a clear
ImportError if it is missing.

Methods
-------
equal          1/N — reference point.
inverse_vol    w_i ∝ 1/σ_i — the classic vol-scaled book (what most sfera
               strategies hand-roll today).
min_variance   Markowitz global-minimum-variance, Ledoit-Wolf shrunk covariance.
risk_parity    Equal risk contribution (RiskBudgeting on variance).
hrp            Hierarchical Risk Parity (López de Prado) — no matrix inversion,
               robust on short windows / correlated sleeves.
max_sharpe     Mean-variance tangency portfolio. Uses the sample mean, so it is
               the noisiest option; prefer risk-based methods for live books.
min_cvar       Minimise 95% CVaR (tail-risk aware; the R-DIV / bond-book choice).

All methods honour ``min_weight`` / ``max_weight`` per-name bounds. Degenerate
inputs (too few rows, a single asset, all-NaN columns, solver failure) fall
back to ``inverse_vol`` → ``equal`` so a rebalance never fails on a bad window.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

OptMethod = Literal[
    "equal",
    "inverse_vol",
    "min_variance",
    "risk_parity",
    "hrp",
    "max_sharpe",
    "min_cvar",
]

_SKFOLIO_METHODS = {"min_variance", "risk_parity", "hrp", "max_sharpe", "min_cvar"}


def _skfolio_available() -> bool:
    try:
        import skfolio  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def _equal(cols: pd.Index) -> pd.Series:
    n = len(cols)
    return pd.Series(1.0 / n if n else 0.0, index=cols, dtype="float64")


def _inverse_vol(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std(ddof=1)
    vol = vol.replace(0.0, np.nan)
    if vol.isna().all():
        return _equal(returns.columns)
    inv = (1.0 / vol).fillna(0.0)
    tot = float(inv.sum())
    return inv / tot if tot > 0 else _equal(returns.columns)


def _apply_bounds(w: pd.Series, min_weight: float, max_weight: float) -> pd.Series:
    """Clip to [min, max] and renormalise to sum 1 (iterative so clipping holds)."""
    w = w.astype("float64").fillna(0.0)
    if len(w) == 0:
        return w
    lo, hi = float(min_weight), float(max_weight)
    # Infeasible bounds (e.g. 10 names with max 0.05) → best effort: cap and let sum < 1.
    if hi * len(w) < 1.0 - 1e-12:
        return w.clip(lower=lo, upper=hi)
    for _ in range(50):
        w = w.clip(lower=lo, upper=hi)
        s = float(w.sum())
        if s <= 0:
            return _equal(w.index).clip(lower=lo, upper=hi)
        w = w / s
        if (w <= hi + 1e-12).all() and (w >= lo - 1e-12).all():
            break
    return w


def _skfolio_weights(
    returns: pd.DataFrame,
    method: str,
    min_weight: float,
    max_weight: float,
    cvar_beta: float,
) -> pd.Series:
    from skfolio import RiskMeasure
    from skfolio.moments import LedoitWolf
    from skfolio.optimization import (
        HierarchicalRiskParity,
        MeanRisk,
        ObjectiveFunction,
        RiskBudgeting,
    )
    from skfolio.prior import EmpiricalPrior

    prior = EmpiricalPrior(covariance_estimator=LedoitWolf())
    common = dict(min_weights=min_weight, max_weights=max_weight)

    if method == "min_variance":
        est = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=prior,
            **common,
        )
    elif method == "risk_parity":
        est = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, prior_estimator=prior, **common)
    elif method == "hrp":
        est = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE, prior_estimator=prior, **common)
    elif method == "max_sharpe":
        est = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=prior,
            **common,
        )
    elif method == "min_cvar":
        est = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
            cvar_beta=cvar_beta,
            **common,
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(method)

    est.fit(returns)
    return pd.Series(np.asarray(est.weights_, dtype="float64"), index=returns.columns)


def optimize_weights(
    returns: pd.DataFrame,
    method: OptMethod = "inverse_vol",
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    min_periods: int = 20,
    cvar_beta: float = 0.95,
    fallback: Optional[OptMethod] = "inverse_vol",
) -> pd.Series:
    """
    Long-only weights (sum = 1) for the columns of ``returns``.

    Parameters
    ----------
    returns
        Trailing window of simple returns, ``[date x instrument]``. Columns that
        are entirely NaN are dropped (weight 0); remaining NaNs are filled with 0.
    method
        One of :data:`OptMethod`.
    min_weight, max_weight
        Per-name bounds, applied inside the optimiser for skfolio methods and
        by clip-and-renormalise for the numpy ones.
    min_periods
        Fewer valid rows than this → fall straight back to ``fallback``.
    cvar_beta
        Confidence level for ``min_cvar``.
    fallback
        Method used when the requested one cannot run (missing skfolio, solver
        failure, degenerate window). ``None`` → raise instead.

    Returns
    -------
    pd.Series indexed like ``returns.columns`` (dropped columns get 0.0).
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        return pd.Series(dtype="float64")

    all_cols = returns.columns
    r = returns.replace([np.inf, -np.inf], np.nan)
    r = r.dropna(axis=1, how="all")
    r = r.dropna(axis=0, how="all").fillna(0.0)

    def _finish(w: pd.Series) -> pd.Series:
        w = _apply_bounds(w, min_weight, max_weight)
        return w.reindex(all_cols).fillna(0.0)

    if r.shape[1] == 0:
        return pd.Series(0.0, index=all_cols, dtype="float64")
    if r.shape[1] == 1:
        return _finish(pd.Series(1.0, index=r.columns))

    if method == "equal":
        return _finish(_equal(r.columns))

    if len(r) < min_periods:
        log.debug(
            "optimize_weights: %d rows < min_periods=%d; using %s", len(r), min_periods, fallback
        )
        if fallback is None:
            raise ValueError("optimize_weights: insufficient history and no fallback")
        return optimize_weights(
            r, fallback, min_weight=min_weight, max_weight=max_weight, min_periods=0, fallback=None
        )

    if method == "inverse_vol":
        return _finish(_inverse_vol(r))

    if method not in _SKFOLIO_METHODS:
        raise ValueError(f"optimize_weights: unknown method {method!r}")

    if not _skfolio_available():
        msg = f"optimize_weights: method {method!r} requires `skfolio` (pip install skfolio)"
        if fallback is None:
            raise ImportError(msg)
        log.warning("%s; falling back to %s", msg, fallback)
        return optimize_weights(
            r, fallback, min_weight=min_weight, max_weight=max_weight, min_periods=0, fallback=None
        )

    try:
        w = _skfolio_weights(r, method, min_weight, max_weight, cvar_beta)
        if not np.isfinite(w.values).all() or w.sum() <= 0:
            raise RuntimeError("non-finite or zero weights from solver")
        return _finish(w)
    except Exception as exc:
        if fallback is None:
            raise
        log.warning("optimize_weights[%s] failed (%s); falling back to %s", method, exc, fallback)
        return optimize_weights(
            r, fallback, min_weight=min_weight, max_weight=max_weight, min_periods=0, fallback=None
        )


def optimize_weights_series(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    method: OptMethod = "inverse_vol",
    *,
    lookback_bars: int = 252,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience: roll :func:`optimize_weights` over ``rebalance_dates`` using
    the ``lookback_bars`` of returns ending at (and including) each date.

    Returns a ``[rebalance_date x instrument]`` weights matrix suitable for
    :class:`~quantdsl_backtest.dsl.portfolio.TargetWeights(weights=...)`.
    No look-ahead: the window for date *t* ends at *t*; the caller decides the
    execution lag via ``signal_delay_bars``.
    """
    rets = prices.pct_change()
    out = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns, dtype="float64")
    for dt in rebalance_dates:
        if dt not in rets.index:
            continue
        pos = rets.index.get_loc(dt)
        if isinstance(pos, slice):
            pos = pos.start
        win = rets.iloc[max(0, int(pos) - lookback_bars + 1) : int(pos) + 1]
        out.loc[dt] = optimize_weights(win, method, **kwargs)
    return out


__all__ = ["OptMethod", "optimize_weights", "optimize_weights_series"]
