# src/quantdsl_backtest/engine/accounting.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..dsl.costs import BorrowCost, FinancingCost

#: Per-run cache of borrow-rate files: (path, instrument column, rate column) -> Series by instrument.
_BORROW_RATE_FILES: Dict[Tuple[str, str, str], pd.Series] = {}


def _load_borrow_rate_file(path: str, instrument_column: str, rate_column: str) -> pd.Series:
    key = (str(path), instrument_column, rate_column)
    if key in _BORROW_RATE_FILES:
        return _BORROW_RATE_FILES[key]
    file = Path(path)
    frame = pd.read_csv(file) if file.suffix.lower() == ".csv" else pd.read_parquet(file)
    for column in (instrument_column, rate_column):
        if column not in frame.columns:
            raise ValueError(
                f"borrow rates file {file} has no column {column!r}; columns are {list(frame.columns)}"
            )
    series = (
        frame[[instrument_column, rate_column]]
        .dropna()
        .drop_duplicates(instrument_column, keep="last")
        .set_index(instrument_column)[rate_column]
        .astype("float64")
    )
    _BORROW_RATE_FILES[key] = series
    return series


def resolve_borrow_rates(borrow: BorrowCost, instruments: pd.Index) -> pd.Series:
    """Annual borrow rate per instrument: its own (``rates`` first, then the file), else the default."""
    out = pd.Series(float(borrow.default_annual_rate), index=pd.Index(instruments), dtype="float64")
    path = getattr(borrow, "rates_path", None)
    if path:
        file_rates = _load_borrow_rate_file(
            path, getattr(borrow, "instrument_column", "ticker"), getattr(borrow, "rate_column", "fee_rate")
        )
        hit = out.index.intersection(file_rates.index)
        out.loc[hit] = file_rates.reindex(hit).to_numpy()
    mapping = getattr(borrow, "rates", None)
    if mapping:
        for name, rate in mapping.items():
            if name in out.index:
                out.loc[name] = float(rate)
    return out


def has_per_instrument_borrow_rates(borrow: BorrowCost) -> bool:
    return bool(getattr(borrow, "rates", None)) or bool(getattr(borrow, "rates_path", None))
from ..utils.logging import get_logger

try:  # shared platform stats (quant_models/perf) — optional so the engine still runs without it
    from perf.advanced import probabilistic_sharpe_ratio as _qp_psr
    from perf.metrics import cagr as _qp_cagr
    from perf.robustness import fitness as _qp_fitness, persistence as _qp_persistence
    _HAS_QP = True
except Exception:  # pragma: no cover
    _HAS_QP = False


log = get_logger(__name__)


def mark_to_market(
    prev_positions: pd.Series,
    prev_prices: pd.Series,
    curr_prices: pd.Series,
    prev_cash: float,
) -> Tuple[float, float]:
    """
    Mark positions to current prices, returning:

        equity_before_trades, price_pnl

    Price PnL is computed as sum(positions * (curr - prev)).
    """
    # Align and ensure numeric types
    prev_positions = prev_positions.reindex_like(prev_positions).fillna(0.0).astype("float64")
    prev_prices = prev_prices.reindex(prev_positions.index).astype("float64")
    curr_prices = curr_prices.reindex(prev_positions.index).astype("float64")

    # Only compute P&L where both previous and current prices are valid
    valid_mask = prev_prices.notna() & curr_prices.notna()
    if valid_mask.any():
        p_prev = prev_prices[valid_mask]
        p_curr = curr_prices[valid_mask]
        pos_prev = prev_positions[valid_mask]
        pnl_vector = (p_curr - p_prev) * pos_prev
        price_pnl = float(pnl_vector.sum())
    else:
        price_pnl = 0.0

    # Equity before trades: previous cash plus previous marks for all instruments
    # (where previous prices are known), plus today's price P&L on valid instruments.
    prev_mark_mask = prev_prices.notna()
    prev_notional = float((prev_positions[prev_mark_mask] * prev_prices[prev_mark_mask]).sum())
    equity_before = float(prev_cash + prev_notional + price_pnl)
    return equity_before, float(price_pnl)


def apply_coupon_income(
    prev_positions: pd.Series,
    coupon_per_unit: pd.Series,
) -> float:
    """Credit coupon (bond income) cash for one bar: sum(positions * coupon_per_unit).

    For fixed-income strategies marked at DIRTY price (clean + accrued), the dirty mark
    already books the daily accrual; on a coupon date the dirty price drops by the coupon
    (accrued resets) and THIS credits the offsetting cash, so total return stays correct and
    smooth. `coupon_per_unit` is the coupon paid per unit held on this date (0 elsewhere);
    it may come from the explicit schedule or the MOEX accrued-drop inference.
    """
    if coupon_per_unit is None:
        return 0.0
    pos = prev_positions.fillna(0.0).astype("float64")
    cpn = coupon_per_unit.reindex(pos.index).fillna(0.0).astype("float64")
    return float((pos * cpn).sum())


def apply_carry_costs(
    positions: pd.Series,
    prices: pd.Series,
    cash: float,
    borrow: BorrowCost,
    financing: FinancingCost,
    dt_years: float = 1.0 / 252.0,
) -> Tuple[float, float, float]:
    """
    Apply simple borrow and financing costs over a time step.

    Returns:
        new_cash, borrow_cost, financing_pnl
    """
    positions = positions.fillna(0.0)
    prices = prices.reindex(positions.index)

    long_notional = float((positions.clip(lower=0.0) * prices).sum())
    short_by_name = -positions.clip(upper=0.0) * prices
    short_notional = float(short_by_name.sum())

    # Borrow cost on shorts: each name at its own annual rate, the rest at the default
    if has_per_instrument_borrow_rates(borrow):
        rates = resolve_borrow_rates(borrow, positions.index)
        borrow_cost = float((short_by_name.fillna(0.0) * rates).sum()) * dt_years
    else:
        borrow_cost = short_notional * borrow.default_annual_rate * dt_years

    # Financing on cash: project baseline tests expect spread applied on cash
    # balance (positive earns, negative pays).
    fin_rate = float(financing.spread_bps) / 1e4
    financing_pnl = cash * fin_rate * dt_years

    new_cash = cash - borrow_cost + financing_pnl
    return new_cash, borrow_cost, financing_pnl


def apply_funding(
    prev_positions: pd.Series,
    prices: pd.Series,
    funding_rate: pd.Series | None,
) -> float:
    """Perp funding PAID over one bar (positive = cash out): sum(q * price * Σrate).

    `funding_rate` is the summed per-event rate of the funding events settling inside this bar's
    holding window, per instrument. rate > 0 → longs pay (cost > 0 for q > 0), shorts receive.
    """
    if funding_rate is None:
        return 0.0
    pos = prev_positions.fillna(0.0).astype("float64")
    px = prices.reindex(pos.index).fillna(0.0).astype("float64")
    fr = funding_rate.reindex(pos.index).fillna(0.0).astype("float64")
    return float((pos * px * fr).sum())


def compute_exposures(
    positions: pd.Series,
    prices: pd.Series,
) -> Dict[str, float]:
    """
    Compute long/short/gross/net exposures and leverage (assuming eq>0 provided elsewhere).
    """
    positions = positions.fillna(0.0)
    prices = prices.reindex(positions.index)

    notional = positions * prices
    long_exp = float(notional.clip(lower=0.0).sum())
    short_exp = float(notional.clip(upper=0.0).sum())  # negative
    gross = long_exp + abs(short_exp)
    net = long_exp + short_exp

    return {
        "long_exposure": long_exp,
        "short_exposure": short_exp,
        "gross_exposure": gross,
        "net_exposure": net,
    }


def compute_basic_metrics(
    returns: pd.Series,
    equity: pd.Series,
    weights: pd.DataFrame | None,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """
    Compute basic performance metrics using robust, comparable logic:
      - total_return
      - sharpe (annualized by `periods_per_year`: 252 exchange days, 365 for 24/7 crypto daily)
      - sortino (annualized)
      - max_drawdown
      - turnover_annual (if weights provided)
    Plus, when `perf` (quant_models/perf) is installed, the robustness layer every consumer (critic, sfera
    Review/monitor) reads off the same JSON:
      - turnover_daily, cagr
      - psr            P(true Sharpe > 0), skew/kurtosis-corrected
      - fitness        WorldQuant BRAIN: sharpe * sqrt(|cagr| / max(turnover_daily, 0.125))
      - sharpe_recent, persistence_ratio, persistence  (ALIVE / SOFTENING / DECAYING —
                       does the edge still earn in the last 2y at the full-window rate?)
    """
    metrics: Dict[str, float] = {}

    # Build and clean returns series for metrics
    rets_raw = returns
    if rets_raw is None or len(rets_raw) == 0:
        try:
            # Fallback: derive from equity if available
            rets_raw = equity.pct_change() if equity is not None else pd.Series(dtype="float64")
        except Exception:
            rets_raw = pd.Series(dtype="float64")

    # Hygiene: replace inf with NaN, then treat NaNs as 0.0 (flat day)
    rets_raw = rets_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Clip absurd returns for metrics only (keep raw equity path for total_return)
    lower, upper = -0.30, 0.30
    mask_lower = rets_raw < lower
    mask_upper = rets_raw > upper
    clip_count = int(mask_lower.sum() + mask_upper.sum())
    rets_for_metrics = rets_raw.clip(lower=lower, upper=upper)

    if clip_count > 0:
        # Log a concise warning with a few example dates
        try:
            clipped_idx = rets_raw.index[mask_lower | mask_upper]
            examples = ", ".join(str(d) for d in list(clipped_idx[:3]))
            log.warning(
                "[METRICS] Clipped %d daily returns outside [%.0f%%, %.0f%%] for metrics (examples: %s)",
                clip_count,
                lower * 100,
                upper * 100,
                examples,
            )
        except Exception:
            log.warning(
                "[METRICS] Clipped %d daily returns outside [%.0f%%, %.0f%%] for metrics",
                clip_count,
                lower * 100,
                upper * 100,
            )

    # Total return from equity if available (raw, not clipped)
    try:
        if equity is not None and len(equity) >= 2:
            total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        else:
            total_ret = 0.0
    except Exception:
        total_ret = 0.0

    if len(rets_for_metrics) == 0:
        metrics["total_return"] = float(total_ret)
        metrics["sharpe"] = 0.0
        metrics["sortino"] = 0.0
        metrics["max_drawdown"] = 0.0
        metrics["turnover_annual"] = 0.0
        return metrics

    mean_ret = float(rets_for_metrics.mean())
    std_ret = float(rets_for_metrics.std())
    neg_ret = rets_for_metrics[rets_for_metrics < 0]

    ann_factor = float(np.sqrt(periods_per_year))
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    if len(neg_ret) > 0:
        downside_std = float(neg_ret.std())
        sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown
    cum = (1.0 + rets_for_metrics).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # Turnover: 0.5 * sum(|w_t - w_{t-1}|) per day, annualized
    turnover_daily = 0.0
    try:
        if weights is not None and len(weights) > 1:
            diff = weights.diff().abs().sum(axis=1) * 0.5
            turnover_daily = float(diff.mean())
    except Exception:
        turnover_daily = 0.0
    turnover_annual = float(turnover_daily * periods_per_year)

    metrics["total_return"] = float(total_ret)
    metrics["sharpe"] = float(sharpe)
    metrics["sortino"] = float(sortino)
    metrics["max_drawdown"] = float(max_dd)
    metrics["turnover_annual"] = float(turnover_annual)
    metrics["turnover_daily"] = float(turnover_daily)

    # Robustness layer (perf). Defensive like everything above: a stats failure must never
    # take down a backtest — it just leaves the field NaN, which downstream gates treat as FAIL.
    if _HAS_QP:
        try:
            ann_ret = _qp_cagr(rets_for_metrics, periods=periods_per_year)        # perf.cagr takes RETURNS, periods pinned
            metrics["cagr"] = float(ann_ret)
            metrics["psr"] = float(_qp_psr(rets_for_metrics))
            metrics["fitness"] = float(_qp_fitness(sharpe, ann_ret, turnover_daily))
            pers = _qp_persistence(rets_for_metrics, periods=periods_per_year)
            metrics["sharpe_recent"] = float(pers["sharpe_recent"]) if pers["sharpe_recent"] == pers["sharpe_recent"] else float("nan")
            metrics["persistence_ratio"] = float(pers["ratio"]) if pers["ratio"] is not None else float("nan")
            metrics["persistence"] = str(pers["label"])
        except Exception as exc:  # pragma: no cover
            log.warning("[METRICS] perf robustness layer failed: %s", exc)
    return metrics
