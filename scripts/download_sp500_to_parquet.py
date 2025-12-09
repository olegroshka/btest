"""
Download S&P 500 daily OHLCV data and save it in the parquet format expected by
quantdsl_backtest's parquet adapter.

Output schema (long-form table):
  - date: datetime64[ns]
  - ticker: str
  - open, high, low, close: float
  - volume: int
  - sector: str (optional, from Wikipedia)

By default, writes to relative path: equities/sp500_daily
This matches DataConfig(source="parquet://equities/sp500_daily").

Requirements:
  - pandas
  - vectorbt (for Yahoo Finance downloader)
  - pyarrow (for parquet IO)

Usage examples (PowerShell):
  uv run python scripts/download_sp500_to_parquet.py
  uv run python scripts/download_sp500_to_parquet.py --start 2015-01-01 --end 2025-01-01 --out equities/sp500_daily
  # Use your own tickers CSV (columns: ticker[,sector]) to avoid Wikipedia/lxml dependency:
  uv run python scripts/download_sp500_to_parquet.py --tickers-csv data/sp500_tickers.csv
"""

from __future__ import annotations

import argparse
import os
import time
import random
from typing import List, Tuple, Optional, Dict

import pandas as pd
import vectorbt as vbt
import yfinance as yf


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_constituents() -> pd.DataFrame:
    """
    Fetch current S&P 500 constituents and sectors from Wikipedia.
    Returns a DataFrame with at least columns: [ticker, security, sector].
    """
    tables = pd.read_html(WIKI_SP500_URL)
    if not tables:
        raise RuntimeError("Failed to read S&P 500 table from Wikipedia")

    # The first table typically contains the constituents
    df = tables[0].copy()

    # Normalize column names and tickers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Common columns: Symbol, Security, GICS Sector, GICS Sub-Industry, Headquarters Location, etc.
    # Try to map variations robustly
    rename_map = {
        "symbol": "ticker",
        "security": "security",
        "gics_sector": "sector",
    }
    for k, v in list(rename_map.items()):
        if k not in df.columns:
            # Try alternative spellings
            if k == "gics_sector":
                for alt in ("gics sector", "gics_sector", "gicssector"):
                    if alt in df.columns:
                        rename_map[alt] = v
                        del rename_map[k]
                        break
    df = df.rename(columns=rename_map)

    if "ticker" not in df.columns:
        # Try other possible names
        for alt in ("symbol", "ticker_symbol", "ticker"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break
    if "ticker" not in df.columns:
        raise RuntimeError("Could not locate ticker column in Wikipedia S&P 500 table")

    # Some tickers contain dots which Yahoo represents with hyphens (e.g., BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)

    # Ensure sector exists; if not, fill with NA
    if "sector" not in df.columns:
        df["sector"] = pd.NA

    return df[["ticker", "sector"]].dropna(subset=["ticker"]).reset_index(drop=True)


def load_tickers_from_csv(path: str) -> pd.DataFrame:
    """Load tickers (and optional sector) from a CSV file.
    Expected columns:
      - ticker (required)
      - sector (optional)
    """
    df = pd.read_csv(path)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "ticker" not in df.columns:
        # Try alternative names
        for alt in ("symbol", "ticker_symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ticker"})
                break
    if "ticker" not in df.columns:
        raise ValueError("CSV must contain a 'ticker' column")
    # Normalize Yahoo-compatible tickers
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    if "sector" not in df.columns:
        df["sector"] = pd.NA
    return df[["ticker", "sector"]].dropna(subset=["ticker"]).reset_index(drop=True)


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _sleep_backoff(try_idx: int, base: float = 1.0, jitter: float = 0.25) -> None:
    delay = base * (2 ** try_idx) * (1 + random.random() * jitter)
    time.sleep(min(delay, 30))


def _download_chunk_vbt(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    yfdata = vbt.YFData.download(
        symbols,
        start=start,
        end=end,
        interval="1d",
        missing_index="drop",
        missing_columns="drop",
    )
    fields = {
        "Open": yfdata.get("Open"),
        "High": yfdata.get("High"),
        "Low": yfdata.get("Low"),
        "Close": yfdata.get("Close"),
        "Volume": yfdata.get("Volume"),
    }
    out: Dict[str, pd.DataFrame] = {}
    for k, df in fields.items():
        if df is None:
            out[k] = pd.DataFrame()
        else:
            # Convert Series to DataFrame if necessary
            if isinstance(df, pd.Series):
                df = df.to_frame(symbols[0])
            out[k] = df.sort_index()
    return out


def _fallback_yf_per_symbol(symbols: List[str], start: str, end: str, retries: int = 2) -> Dict[str, pd.DataFrame]:
    kept: List[str] = []
    failed: List[str] = []
    cols: Dict[str, List[pd.Series]] = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}

    for sym in symbols:
        ok = False
        for t in range(retries + 1):
            try:
                df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Standardize columns
                    for fld in ["Open", "High", "Low", "Close", "Volume"]:
                        if fld in df.columns and not pd.Series(df[fld]).dropna().empty:
                            s = df[fld].copy()
                            s.name = sym
                            cols[fld].append(s)
                    if "Close" in df.columns and not pd.Series(df["Close"]).dropna().empty:
                        kept.append(sym)
                        ok = True
                        break
            except Exception:
                pass
            _sleep_backoff(t, base=0.5)
        if not ok:
            failed.append(sym)

    def _concat(series_list: List[pd.Series]) -> pd.DataFrame:
        if not series_list:
            return pd.DataFrame()
        df = pd.concat(series_list, axis=1)
        df.sort_index(inplace=True)
        return df

    result = {
        "Open": _concat(cols["Open"]),
        "High": _concat(cols["High"]),
        "Low": _concat(cols["Low"]),
        "Close": _concat(cols["Close"]),
        "Volume": _concat(cols["Volume"]),
    }
    result["__kept__"] = kept  # type: ignore
    result["__failed__"] = failed  # type: ignore
    return result


def _merge_panels_union(dst: Optional[pd.DataFrame], src: pd.DataFrame) -> pd.DataFrame:
    if dst is None or dst.empty:
        return src.copy()
    if src is None or src.empty:
        return dst.copy()
    merged = pd.concat([dst, src], axis=1)
    # Drop duplicate columns keeping first occurrence
    if hasattr(merged, "columns"):
        merged = merged.loc[:, ~merged.columns.duplicated()]
    merged.sort_index(inplace=True)
    return merged


def download_ohlcv(
    symbols: List[str],
    start: str,
    end: str,
    chunk_size: int = 50,
    retries: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download OHLCV panels resiliently.

    1) Try to download in chunks via vectorbt with retries and backoff.
    2) For any chunk that fails or yields empty Close, fall back to yfinance per-symbol.
    3) Merge all chunks using union of columns (symbols).
    """
    chunks = _chunk_list(symbols, max(1, int(chunk_size)))

    open_all: Optional[pd.DataFrame] = None
    high_all: Optional[pd.DataFrame] = None
    low_all: Optional[pd.DataFrame] = None
    close_all: Optional[pd.DataFrame] = None
    vol_all: Optional[pd.DataFrame] = None

    total_kept: List[str] = []
    total_failed: List[str] = []

    for idx, ch in enumerate(chunks, start=1):
        print(f"Chunk {idx}/{len(chunks)}: {len(ch)} symbols")
        success = False
        last_err: Optional[Exception] = None
        for t in range(retries + 1):
            try:
                frames = _download_chunk_vbt(ch, start, end)
                close_df = frames["Close"]
                # If close is empty or has zero columns, treat as failure
                if close_df is None or close_df.empty or (hasattr(close_df, "columns") and len(close_df.columns) == 0):
                    raise RuntimeError("Empty Close returned")

                # Merge
                open_all = _merge_panels_union(open_all, frames["Open"]) if frames["Open"] is not None else open_all
                high_all = _merge_panels_union(high_all, frames["High"]) if frames["High"] is not None else high_all
                low_all = _merge_panels_union(low_all, frames["Low"]) if frames["Low"] is not None else low_all
                close_all = _merge_panels_union(close_all, frames["Close"]) if frames["Close"] is not None else close_all
                vol_all = _merge_panels_union(vol_all, frames["Volume"]) if frames["Volume"] is not None else vol_all

                if hasattr(close_df, "columns"):
                    total_kept.extend([c for c in close_df.columns if c not in total_kept])
                success = True
                print(f"  ✓ Downloaded via vectorbt for chunk with {len(ch)} symbols; kept {len(close_df.columns) if hasattr(close_df,'columns') else 1}")
                break
            except Exception as e:
                last_err = e
                if t < retries:
                    print(f"  vectorbt chunk failed (try {t+1}/{retries+1}): {e}. Retrying...")
                    _sleep_backoff(t)
                else:
                    print(f"  vectorbt chunk failed after retries: {e}. Falling back to per-symbol yfinance...")

        if not success:
            fb = _fallback_yf_per_symbol(ch, start, end)
            # Merge fallback frames
            open_all = _merge_panels_union(open_all, fb.get("Open", pd.DataFrame()))
            high_all = _merge_panels_union(high_all, fb.get("High", pd.DataFrame()))
            low_all = _merge_panels_union(low_all, fb.get("Low", pd.DataFrame()))
            close_all = _merge_panels_union(close_all, fb.get("Close", pd.DataFrame()))
            vol_all = _merge_panels_union(vol_all, fb.get("Volume", pd.DataFrame()))
            kept_syms = fb.get("__kept__", [])  # type: ignore
            fail_syms = fb.get("__failed__", [])  # type: ignore
            total_kept.extend([s for s in kept_syms if s not in total_kept])
            total_failed.extend(fail_syms)
            print(f"  ✓ Fallback kept {len(kept_syms)}; dropped {len(fail_syms)}")

    # If nothing kept, raise
    if close_all is None or close_all.empty or (hasattr(close_all, "columns") and len(close_all.columns) == 0):
        raise RuntimeError("All chunks failed or returned empty data; cannot build dataset.")

    # Ensure indices sorted
    for _df in (open_all, high_all, low_all, close_all, vol_all):
        if _df is not None and not _df.empty:
            _df.sort_index(inplace=True)

    # Replace None with empty DataFrames for return type stability
    def _ensure(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return df if df is not None else pd.DataFrame()

    return _ensure(open_all), _ensure(high_all), _ensure(low_all), _ensure(close_all), _ensure(vol_all)


def build_long_table(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    vol_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge wide OHLCV panels into a long-form table with columns:
    date, ticker, open, high, low, close, volume

    Robust to missing fields per symbol: we build from the UNION of symbols
    across available panels and only require 'close' to be present for a row
    to be kept. Other fields may be NaN.
    """
    # Ensure inputs are DataFrames (not Series)
    def ensure_df(df: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        return df.to_frame(df.name) if isinstance(df, pd.Series) else df

    open_df = ensure_df(open_df)
    high_df = ensure_df(high_df)
    low_df = ensure_df(low_df)
    close_df = ensure_df(close_df)
    vol_df = ensure_df(vol_df)

    # Gather union of symbols across panels
    symbol_sets = []
    for _df in (open_df, high_df, low_df, close_df, vol_df):
        if not _df.empty:
            symbol_sets.append(set(map(str, _df.columns)))
    all_syms = set().union(*symbol_sets) if symbol_sets else set()

    # Helper to convert panel to long form with explicit value name
    def to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "ticker", value_name])
        # Ensure all_syms present as columns (missing ones will be added with NaN)
        df = df.copy()
        missing = [s for s in all_syms if s not in df.columns]
        if missing:
            for m in missing:
                df[m] = pd.NA
        # Order columns consistently
        cols = sorted(list(all_syms)) if all_syms else list(df.columns)
        df = df[cols]
        df = df.sort_index()
        out = df.stack(dropna=False).reset_index()
        out.columns = ["date", "ticker", value_name]
        return out

    open_long = to_long(open_df, "open")
    high_long = to_long(high_df, "high")
    low_long = to_long(low_df, "low")
    close_long = to_long(close_df, "close")
    vol_long = to_long(vol_df, "volume")

    # Outer-merge all fields on ['date','ticker']
    long = close_long  # base
    for part in (open_long, high_long, low_long, vol_long):
        if part is not None and not part.empty:
            long = long.merge(part, on=["date", "ticker"], how="outer")

    # Keep rows that have a close value
    long = long.dropna(subset=["close"], how="any")

    # Enforce dtypes
    if not long.empty:
        long["date"] = pd.to_datetime(long["date"], errors="coerce")
        long = long.dropna(subset=["date"])  # drop any malformed dates
        long["ticker"] = long["ticker"].astype(str)
        for c in ("open", "high", "low", "close"):
            if c in long.columns:
                long[c] = pd.to_numeric(long[c], errors="coerce")
        if "volume" in long.columns:
            long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
        else:
            long["volume"] = pd.NA
        # Fill volume NaN with 0 to simplify downstream ADV calc (optional)
        long["volume"] = long["volume"].fillna(0).astype("int64")

        # Sort for nice layout
        long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

    return long


def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 daily OHLCV to parquet (long form)")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD, exclusive)")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("equities", "sp500_daily"),
        help="Output parquet file path (matches DataConfig source without 'parquet://')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of symbols per chunk for vectorbt downloads (default: 50)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per chunk for vectorbt downloads (default: 3)",
    )
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=20,
        help="Minimum number of symbols with data required to proceed (default: 20)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=100,
        help="Minimum number of days with non-null close per symbol to count towards min-symbols (default: 100)",
    )
    parser.add_argument(
        "--tickers-csv",
        type=str,
        default=None,
        help="Optional path to CSV with columns: ticker[,sector]. If provided, Wikipedia is not used.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Optional comma-separated list of tickers to download (overrides Wikipedia).",
    )
    args = parser.parse_args()

    # Resolve tickers source
    if args.tickers:
        syms = [s.strip() for s in args.tickers.split(",") if s.strip()]
        sp500 = pd.DataFrame({"ticker": syms})
        sp500["sector"] = pd.NA
        print(f"Using tickers from --tickers: {len(syms)} symbols")
    elif args.tickers_csv:
        print(f"Loading tickers from CSV: {args.tickers_csv}")
        sp500 = load_tickers_from_csv(args.tickers_csv)
        print(f"Loaded {len(sp500)} symbols from CSV")
    else:
        print("Fetching S&P 500 constituents from Wikipedia...")
        sp500 = get_sp500_constituents()
    symbols = sp500["ticker"].tolist()
    print(f"Found {len(symbols)} symbols")

    print(f"Downloading Yahoo Finance OHLCV from {args.start} to {args.end} (daily)...")
    open_df, high_df, low_df, close_df, vol_df = download_ohlcv(
        symbols, args.start, args.end, chunk_size=args.chunk_size, retries=args.retries
    )

    print("Building long-form table...")
    long = build_long_table(open_df, high_df, low_df, close_df, vol_df)

    # Safety: don't write empty parquet silently
    if long.empty:
        raise RuntimeError(
            "No usable rows were assembled (missing 'close' for all date-ticker pairs). "
            "This can happen if Yahoo returned no data for your selection. "
            "Try a shorter date range, a different ticker set, or rerun later."
        )

    # Enforce minimum data thresholds for robustness
    counts = long["close"].notna().groupby(long["ticker"]).sum()
    good_symbols = counts[counts >= args.min_days].index.tolist()
    n_good = len(good_symbols)
    if n_good < args.min_symbols:
        sample = ", ".join(good_symbols[:10])
        raise RuntimeError(
            f"Only {n_good} symbols have >= {args.min_days} days of data; "
            f"minimum required is {args.min_symbols}. Example kept: {sample}"
        )

    # Attach sector info where available
    long = long.merge(sp500, on="ticker", how="left")

    # Ensure output directory exists
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Log brief summary
    n_syms = long["ticker"].nunique()
    dt_min = pd.to_datetime(long["date"]).min()
    dt_max = pd.to_datetime(long["date"]).max()
    n_rows = len(long)
    n_close = long["close"].notna().sum()
    print(f"Assembled {n_rows} rows across {n_syms} tickers from {dt_min.date()} to {dt_max.date()} (non-null close: {n_close}).")

    # Write parquet; note: path intentionally may not have .parquet extension to match config
    print(f"Writing parquet to: {out_path}")
    long.to_parquet(out_path, index=False)

    # Persist kept/dropped symbol lists next to parquet for transparency
    base = out_path
    kept_path = base + "_kept_symbols.txt"
    dropped_path = base + "_dropped_symbols.txt"
    try:
        # Kept: all symbols present in final table
        kept_final = sorted(long["ticker"].unique().tolist())
        with open(kept_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_final))
        # Dropped: from input list minus kept
        dropped_final = sorted([s for s in symbols if s not in set(kept_final)])
        with open(dropped_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dropped_final))
        print(f"Saved kept/dropped symbol lists: {kept_path}, {dropped_path}")
    except Exception as ex:
        print(f"Warning: failed to write kept/dropped symbols files: {ex}")

    print("Done. Example preview:")
    print(long.head())


if __name__ == "__main__":
    main()
