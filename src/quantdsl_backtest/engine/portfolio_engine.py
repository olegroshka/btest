# src/quantdsl_backtest/engine/portfolio_engine.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN
from ..utils.logging import get_logger


log = get_logger(__name__)


def compute_target_weights_for_date(
    date: pd.Timestamp,
    portfolio: LongShortPortfolio,
    signals: Dict[str, pd.DataFrame],
    prev_weights: pd.Series,
) -> pd.Series:
    """
    Compute target weights on a rebalance date, based on the DSL portfolio spec.

    - `signals` should contain at least:
        "rank", "long_candidates", "short_candidates"
      matching our sample strategy.

    - `prev_weights` is a Series indexed by instrument.

    Returns: Series of target weights indexed by instrument.
    """
    # Use delayed signals
    delay = portfolio.signal_delay_bars
    sig_df = signals["rank"]  # we know sample strategy uses this
    all_dates = sig_df.index
    date_pos = all_dates.get_loc(date)

    if isinstance(date_pos, slice):
        date_pos = date_pos.start  # should not happen for unique index

    sig_pos = date_pos - delay
    if sig_pos < 0:
        # Not enough history to generate signals yet; keep previous weights
        return prev_weights.copy()

    sig_date = all_dates[sig_pos]

    rank_row = sig_df.loc[sig_date]

    long_mask = signals["long_candidates"].loc[sig_date].astype(bool)
    short_mask = signals["short_candidates"].loc[sig_date].astype(bool)

    # Select instruments for long and short books using selectors
    long_names = _select_book_names(
        date=sig_date,
        book=portfolio.long_book,
        rank_row=rank_row,
        mask=long_mask,
    )
    short_names = _select_book_names(
        date=sig_date,
        book=portfolio.short_book,
        rank_row=rank_row,
        mask=short_mask,
    )

    instruments = sig_df.columns
    target = pd.Series(0.0, index=instruments, dtype="float64")

    n_long = len(long_names)
    n_short = len(short_names)

    # Compute long/short gross exposures from leverage + net target
    L = portfolio.target_gross_leverage
    N = portfolio.target_net_exposure
    long_gross = max(0.0, (L + N) / 2.0)
    short_gross = max(0.0, (L - N) / 2.0)

    if n_long > 0 and long_gross > 0.0:
        w_long = long_gross / n_long
        for name in long_names:
            target[name] = w_long

    if n_short > 0 and short_gross > 0.0:
        w_short = -short_gross / n_short
        for name in short_names:
            target[name] = w_short

    # Enforce max abs weight per name
    max_w = portfolio.max_abs_weight_per_name
    if max_w is not None and max_w > 0:
        target = target.clip(lower=-max_w, upper=max_w)

    # NOTE: We skip precise sector neutrality for now; that requires sector data.
    if portfolio.sector_neutral is not None:
        log.debug("SectorNeutral requested, but not enforced in this skeleton engine.")

    # Turnover limit: scale towards target if necessary
    if portfolio.turnover_limit is not None:
        target = _apply_turnover_limit(
            prev_weights=prev_weights,
            target_weights=target,
            max_fraction=portfolio.turnover_limit.max_fraction,
        )

    return target


def _select_book_names(
    date: pd.Timestamp,
    book: Book,
    rank_row: pd.Series,
    mask: pd.Series,
) -> list[str]:
    """
    Based on the Book.selector type, choose instrument names at a given date.
    Currently supports TopN and BottomN.
    """
    # Filter out instruments failing mask or with NaN rank
    valid = rank_row[mask & rank_row.notna()]

    if valid.empty:
        return []

    if isinstance(book.selector, TopN):
        n = book.selector.n
        # sort descending by rank (higher = better)
        selected = valid.sort_values(ascending=False).head(n).index.tolist()
        return selected

    if isinstance(book.selector, BottomN):
        n = book.selector.n
        # sort ascending by rank (lower = worse)
        selected = valid.sort_values(ascending=True).head(n).index.tolist()
        return selected

    raise TypeError(f"Unsupported selector type in Book: {type(book.selector)}")


def _apply_turnover_limit(
    prev_weights: pd.Series,
    target_weights: pd.Series,
    max_fraction: float,
) -> pd.Series:
    """
    Apply a simple turnover limit:

        turnover = 0.5 * sum(|target - prev|)

    If turnover > max_fraction, scale the move towards target
    by factor = max_fraction / turnover.

    Returns scaled target weights.
    """
    prev = prev_weights.reindex(target_weights.index).fillna(0.0)
    delta = target_weights - prev
    turnover = 0.5 * np.abs(delta).sum()

    if turnover <= max_fraction or turnover == 0:
        return target_weights

    scale = max_fraction / turnover
    log.debug("Turnover %.4f > %.4f, scaling moves by %.4f", turnover, max_fraction, scale)
    return prev + delta * scale
