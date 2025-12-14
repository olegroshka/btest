"""
Quick data validation script for long-format market data stored in a Parquet file.

Expected schema (case-insensitive columns are accepted and will be lower-cased):
- date:     trading date (any parseable datetime)
- ticker:   instrument identifier
- close:    close price (float)

What it does:
- Loads the parquet file into a DataFrame
- Normalizes column names and parses dates
- Builds a wide price panel (date x ticker)
- Computes daily log returns
- Prints:
  - Dataset shape, unique tickers, date range
  - Duplicates summary
  - Counts of non-positive/zero prices
  - Missing data ratios overall and per-ticker (top offenders)
  - Basic stats for returns (describe)
  - Top-N largest absolute daily returns
- Optionally writes detected outliers to CSV (abs(return) > threshold)

Usage examples (from project root):
  python scripts/validate_parquet.py equities\\indicies.parquet
  python scripts/validate_parquet.py --top 50 --ret-threshold 0.15 equities\\indicies.parquet

Notes:
- The script is read-only; it does not modify source files.
- If you pass no path, it will try common defaults for indices file names.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _find_default_path(path: Optional[str]) -> str:
    if path:
        return path
    # Try a few common spellings
    candidates = [
        os.path.join("equities", "indicies.parquet"),
        os.path.join("equities", "indecies.parquet"),
        os.path.join("equities", "indices.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fall back to the first candidate
    return candidates[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _ensure_required_columns(df: pd.DataFrame, date_col: str, id_col: str, price_col: str) -> None:
    missing = [c for c in (date_col, id_col, price_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Available columns: {list(df.columns)}"
        )


def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    return df


def _basic_overview(df: pd.DataFrame, date_col: str, id_col: str, price_col: str) -> None:
    nrows = len(df)
    ntickers = df[id_col].nunique(dropna=True)
    dmin, dmax = df[date_col].min(), df[date_col].max()
    print("\n=== Dataset overview ===")
    print(f"Rows: {nrows:,}")
    print(f"Tickers: {ntickers:,}")
    print(f"Date range: {dmin} .. {dmax}")

    # Duplicates
    dup_mask = df.duplicated(subset=[date_col, id_col], keep=False)
    dup_count = int(dup_mask.sum())
    print(f"Duplicate (date, ticker) rows: {dup_count:,}")
    if dup_count > 0:
        print("Sample duplicates (top 10):")
        print(df.loc[dup_mask, [date_col, id_col, price_col]].head(10))

    # Non-positive prices
    nonpos = (df[price_col] <= 0).sum()
    zeros = (df[price_col] == 0).sum()
    nans = df[price_col].isna().sum()
    print(f"Non-positive prices: {nonpos:,} (zeros: {zeros:,}, NaNs: {nans:,})")


def _build_prices(df: pd.DataFrame, date_col: str, id_col: str, price_col: str) -> pd.DataFrame:
    # Sort to have stable order
    df_sorted = df.sort_values([date_col, id_col])
    prices = df_sorted.pivot(index=date_col, columns=id_col, values=price_col)
    prices = prices.sort_index()
    return prices


def _missingness_report(prices: pd.DataFrame, top: int = 10) -> None:
    total = prices.size
    missing = int(prices.isna().sum().sum())
    ratio = missing / total if total > 0 else np.nan
    print("\n=== Missing data ===")
    print(f"Total cells: {total:,}; missing: {missing:,} ({ratio:.2%})")

    per_ticker = prices.isna().mean().sort_values(ascending=False)
    print(f"Top {min(top, len(per_ticker))} tickers by missing ratio:")
    print(per_ticker.head(top))


def _returns(prices: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    if use_log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets


def _ret_stats(rets: pd.DataFrame, top_abs: int = 20) -> None:
    print("\n=== Basic return stats (daily) ===")
    print(rets.describe())

    print(f"\nTop {top_abs} largest abs daily returns:")
    stacked = rets.abs().stack().sort_values(ascending=False).head(top_abs)
    print(stacked)


def _find_outliers(
    prices: pd.DataFrame, rets: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    if threshold <= 0:
        return pd.DataFrame(columns=["date", "ticker", "price", "prev_price", "ret"])
    mask = rets.abs() > threshold
    if not mask.any().any():
        return pd.DataFrame(columns=["date", "ticker", "price", "prev_price", "ret"])

    idx = mask.stack()
    idx = idx[idx].index  # MultiIndex [(date, ticker), ...]

    out = []
    for d, t in idx:
        p = prices.at[d, t]
        p_prev = prices.shift(1).at[d, t]
        r = rets.at[d, t]
        out.append((d, t, p, p_prev, r))

    out_df = pd.DataFrame(out, columns=["date", "ticker", "price", "prev_price", "ret"])
    out_df = out_df.sort_values("ret", key=lambda s: s.abs(), ascending=False)
    return out_df


def validate(
    path: str,
    date_col: str = "date",
    id_col: str = "ticker",
    price_col: str = "close",
    top_abs: int = 20,
    top_missing: int = 10,
    ret_threshold: float = 0.2,
    write_outliers: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Reading parquet: {path}")
    df = pd.read_parquet(path)
    df = _normalize_columns(df)
    _ensure_required_columns(df, date_col, id_col, price_col)
    df = _parse_dates(df, date_col)

    _basic_overview(df, date_col, id_col, price_col)

    prices = _build_prices(df, date_col, id_col, price_col)
    _missingness_report(prices, top=top_missing)

    rets = _returns(prices, use_log=True)
    _ret_stats(rets, top_abs=top_abs)

    outliers = _find_outliers(prices, rets, threshold=ret_threshold)
    print("\n=== Return outliers (abs > {:.2%}) ===".format(ret_threshold))
    if outliers.empty:
        print("No outliers found above the threshold.")
    else:
        print(outliers.head(50).to_string(index=False))

    if write_outliers and not outliers.empty:
        base, _ = os.path.splitext(path)
        out_path = base + f"_outliers_gt_{int(ret_threshold*100)}pct.csv"
        outliers.to_csv(out_path, index=False)
        print(f"Outliers written to: {out_path}")

    return df, prices, rets


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate long-format market data parquet file")
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to parquet file (default: tries equities/indicies.parquet and common variants)",
    )
    p.add_argument("--date-col", default="date", help="Date column name (default: date)")
    p.add_argument("--id-col", default="ticker", help="Instrument id column (default: ticker)")
    p.add_argument("--price-col", default="close", help="Price column (default: close)")
    p.add_argument("--top", type=int, default=20, help="Top-N largest abs returns to display (default: 20)")
    p.add_argument(
        "--top-missing",
        type=int,
        default=10,
        help="Top-N tickers by missing ratio to display (default: 10)",
    )
    p.add_argument(
        "--ret-threshold",
        type=float,
        default=0.20,
        help="Return outlier threshold on abs value (e.g., 0.2 = 20%%). Set <=0 to disable.",
    )
    p.add_argument(
        "--write-outliers",
        action="store_true",
        help="Write detected outliers to a CSV next to the parquet file",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    path = _find_default_path(args.path)

    validate(
        path=path,
        date_col=args.date_col,
        id_col=args.id_col,
        price_col=args.price_col,
        top_abs=args.top,
        top_missing=args.top_missing,
        ret_threshold=args.ret_threshold,
        write_outliers=args.write_outliers,
    )


if __name__ == "__main__":
    main()
