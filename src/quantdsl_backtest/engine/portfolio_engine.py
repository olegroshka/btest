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
    # Use delayed signals. Make this generic by resolving factor/mask names
    # from the portfolio books rather than hardcoding specific keys.
    delay = portfolio.signal_delay_bars

    long_sel = portfolio.long_book.selector
    short_sel = portfolio.short_book.selector

    def _get_df_opt(name: Optional[str]) -> Optional[pd.DataFrame]:
        if name is None:
            return None
        return signals.get(name)

    # Resolve rank/score DataFrames per book
    _long_rank_try = _get_df_opt(getattr(long_sel, "factor_name", None))
    long_rank_df = _long_rank_try if _long_rank_try is not None else signals.get("rank")

    _short_rank_try = _get_df_opt(getattr(short_sel, "factor_name", None))
    short_rank_df = _short_rank_try if _short_rank_try is not None else signals.get("rank")

    # Resolve masks per book (default to all True over the rank_df shape)
    _long_mask_try = _get_df_opt(getattr(long_sel, "mask_name", None))
    long_mask_df = _long_mask_try if _long_mask_try is not None else signals.get("long_candidates")

    _short_mask_try = _get_df_opt(getattr(short_sel, "mask_name", None))
    short_mask_df = _short_mask_try if _short_mask_try is not None else signals.get("short_candidates")

    # Choose a reference index that contains the requested date
    ref_df: Optional[pd.DataFrame] = None
    for df in (long_rank_df, short_rank_df):
        if df is not None and date in df.index:
            ref_df = df
            break
    if ref_df is None:
        # Fallback: try any signal frame containing the date
        for df in signals.values():
            if isinstance(df, pd.DataFrame) and date in df.index:
                ref_df = df
                break

    if ref_df is None:
        # No signal aligned to this date; keep previous weights
        log.debug("No signals available for %s; keeping previous weights.", date)
        return prev_weights.copy()

    all_dates = ref_df.index
    date_pos = all_dates.get_loc(date)
    if isinstance(date_pos, slice):
        date_pos = date_pos.start  # should not happen for unique index

    sig_pos = date_pos - delay
    if sig_pos < 0:
        # Not enough history to generate signals yet; keep previous weights
        return prev_weights.copy()

    sig_date = all_dates[sig_pos]

    # Pull rows for the signal date, with graceful fallbacks
    def _safe_row(df_opt: Optional[pd.DataFrame], like: Optional[pd.DataFrame]) -> pd.Series:
        if df_opt is None:
            # Create an all-NaN row matching the 'like' columns if available
            cols = like.columns if like is not None else (prev_weights.index if prev_weights is not None else [])
            return pd.Series(index=cols, dtype="float64")
        # If date missing in df, return NaNs over its columns
        if sig_date not in df_opt.index:
            return pd.Series(index=df_opt.columns, dtype="float64")
        row = df_opt.loc[sig_date]
        # Ensure Series
        if isinstance(row, pd.DataFrame):
            # Unexpected multi-index/slice; take first row
            row = row.iloc[0]
        return row

    long_rank_row = _safe_row(long_rank_df, short_rank_df)
    short_rank_row = _safe_row(short_rank_df, long_rank_df)

    def _safe_mask_row(mask_df_opt: Optional[pd.DataFrame], idx_like: pd.Index) -> pd.Series:
        if mask_df_opt is None:
            return pd.Series(True, index=idx_like, dtype="bool")
        if sig_date not in mask_df_opt.index:
            return pd.Series(True, index=idx_like, dtype="bool")
        m = mask_df_opt.loc[sig_date].astype(bool)
        # Align to idx_like; any missing defaults to False (conservative)
        return m.reindex(idx_like).fillna(False).astype(bool)

    long_mask = _safe_mask_row(long_mask_df, long_rank_row.index)
    short_mask = _safe_mask_row(short_mask_df, short_rank_row.index)

    # Instruments universe comes from any available rank df; fallback to prev_weights index
    instruments: pd.Index
    if long_rank_df is not None:
        instruments = long_rank_df.columns
    elif short_rank_df is not None:
        instruments = short_rank_df.columns
    else:
        instruments = prev_weights.index

    target = pd.Series(0.0, index=instruments, dtype="float64")

    # Optional gating: if there aren't enough valid ranks to populate both books,
    # keep target at zeros (baseline reference behavior in integration tests).
    try:
        long_n_req = getattr(portfolio.long_book.selector, "n", 0)
        short_n_req = getattr(portfolio.short_book.selector, "n", 0)
        n_required = int(long_n_req) + int(short_n_req)
    except Exception:
        n_required = 0

    if n_required > 0:
        valid_long = long_rank_row[long_rank_row.notna()].index if long_rank_row is not None else pd.Index([])
        valid_short = short_rank_row[short_rank_row.notna()].index if short_rank_row is not None else pd.Index([])
        # union of symbols with a valid rank in either book
        valid_union = valid_long.union(valid_short)
        if int(valid_union.size) < n_required:
            # Not enough valid names to form both books: return zeros after turnover cap
            if portfolio.turnover_limit is not None:
                target = _apply_turnover_limit(
                    prev_weights=prev_weights,
                    target_weights=target,
                    max_fraction=portfolio.turnover_limit.max_fraction,
                )
            return target

    # Select instruments for long and short books using selectors
    long_names = _select_book_names(
        date=sig_date,
        book=portfolio.long_book,
        rank_row=long_rank_row,
        mask=long_mask,
    )
    short_names = _select_book_names(
        date=sig_date,
        book=portfolio.short_book,
        rank_row=short_rank_row,
        mask=short_mask,
    )

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
    masked_valid = rank_row[mask & rank_row.notna()]
    unmasked_valid = rank_row[rank_row.notna()]

    if isinstance(book.selector, TopN):
        n = book.selector.n
        # sort descending by rank (higher = better)
        selected = masked_valid.sort_values(ascending=False).head(n).index.tolist()
        # If mask yields fewer than n, fill the remainder from the unmasked pool
        if len(selected) < n:
            remaining = n - len(selected)
            filler = (
                unmasked_valid.drop(index=selected, errors="ignore")
                .sort_values(ascending=False)
                .head(remaining)
                .index.tolist()
            )
            selected.extend(filler)
        return selected

    if isinstance(book.selector, BottomN):
        n = book.selector.n
        # sort ascending by rank (lower = worse)
        selected = masked_valid.sort_values(ascending=True).head(n).index.tolist()
        # If mask yields fewer than n, fill the remainder from the unmasked pool
        if len(selected) < n:
            remaining = n - len(selected)
            filler = (
                unmasked_valid.drop(index=selected, errors="ignore")
                .sort_values(ascending=True)
                .head(remaining)
                .index.tolist()
            )
            selected.extend(filler)
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
