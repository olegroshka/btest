from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _normalize_header(h: str) -> str:
    """
    Normalize a header string: strip, collapse multiple spaces to one, unify case for matching.
    Keeps original case/spacing for output elsewhere.
    """
    s = str(h)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _parse_date(series: pd.Series) -> pd.Series:
    """Parse Date column handling Excel serials and strings."""
    # If numeric, likely Excel serials
    if pd.api.types.is_numeric_dtype(series):
        ser = series.dropna().astype(float)
        if not ser.empty:
            m = ser.median()
            if 10000 < m < 100000:
                return pd.to_datetime(series, unit="d", origin="1899-12-30", errors="coerce")
    # Else, try direct parsing
    return pd.to_datetime(series, errors="coerce")


def _find_columns_for_sheet(sheet_name: str, df: pd.DataFrame) -> Dict[str, str]:
    """
    Given a single-sheet DataFrame that corresponds to a single ticker (sheet_name),
    find the exact columns for date, open, high, low, close, volume.

    Expected columns on each tab:
      - Date
      - "<SHEET> Index - Volume  (L1)"  (spacing around (L1) may vary)
      - "(R1) Open"
      - "(R1) High"
      - "(R1) Low"
      - "(R1) Close"
    """
    # Build robust matchers
    norm_cols = {c: _normalize_header(c) for c in df.columns}
    inv = {v.lower(): c for c, v in norm_cols.items()}  # normalized lower -> original

    def find_exact(label: str) -> Optional[str]:
        key = _normalize_header(label).lower()
        return inv.get(key)

    # Primary OHLC
    col_open = find_exact("(R1) Open") or find_exact("Open")
    col_high = find_exact("(R1) High") or find_exact("High")
    col_low = find_exact("(R1) Low") or find_exact("Low")
    col_close = find_exact("(R1) Close") or find_exact("Close")

    # Date
    col_date = find_exact("Date") or find_exact("date") or next((c for c in df.columns if str(c).strip().lower() in {"date", "dt", "timestamp"}), None)

    # Volume: allow flexible spacing and ticker-specific prefix
    # Example seen: "UKX Index - Volume  (L1)" (double space before (L1))
    # Build regex using sheet_name
    sn = re.escape(_normalize_header(sheet_name))
    vol_regexes = [
        rf"^{sn}\s+Index\s*-\s*Volume\s*\(L1\)$",
        rf"^{sn}\s+Index\s*-\s*Volume\s*\(\s*L1\s*\)$",
        rf"^{sn}\s+Index\s*-\s*Volume\s*$",
    ]
    col_volume: Optional[str] = None
    for c, n in norm_cols.items():
        lc = n
        for pattern in vol_regexes:
            if re.match(pattern, lc, flags=re.IGNORECASE):
                col_volume = c
                break
        if col_volume is not None:
            break

    # Fallback volume names if ticker-prefixed header not present
    if col_volume is None:
        for c, n in norm_cols.items():
            if n.lower() in {"volume", "vol"} or re.match(r"^.*index\s*-\s*volume\s*(\(L1\))?$", n, flags=re.IGNORECASE):
                col_volume = c
                break

    # Broader fallback: any header containing the word "volume" (case-insensitive).
    # If multiple candidates exist, pick the one with the most digit-containing cells.
    if col_volume is None:
        vol_candidates: List[str] = [c for c, n in norm_cols.items() if "volume" in n.lower()]
        if vol_candidates:
            def _digit_count(col: str) -> int:
                try:
                    s = df[col]
                    # Count cells that include at least one digit
                    return int(s.astype(str).str.contains(r"\d", regex=True).sum())
                except Exception:
                    return -1
            vol_candidates.sort(key=_digit_count, reverse=True)
            col_volume = vol_candidates[0]

    cols: Dict[str, Optional[str]] = {
        "date": col_date,
        "open": col_open,
        "high": col_high,
        "low": col_low,
        "close": col_close,
        "volume": col_volume,
    }

    # Require minimally date and close to proceed
    missing = [k for k, v in cols.items() if k in ("date", "close") and v is None]
    if missing:
        raise RuntimeError(f"Sheet '{sheet_name}': missing required columns: {missing}. Found headers: {list(df.columns)}")

    return {k: v for k, v in cols.items() if v is not None}


def _standardize_sheet(sheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert one sheet (one ticker) into standardized OHLCV long-form rows."""
    colmap = _find_columns_for_sheet(sheet_name, df)

    out = pd.DataFrame()
    # Date
    out["date"] = _parse_date(df[colmap["date"]])

    # OHLC
    for k in ("open", "high", "low", "close"):
        if k in colmap:
            out[k] = pd.to_numeric(df[colmap[k]], errors="coerce")
        else:
            out[k] = pd.NA

    # Volume
    if "volume" in colmap:
        vol_raw = df[colmap["volume"]]
        # Robust parsing for volume strings (e.g., "1,234", "1 234", "-", " ")
        if pd.api.types.is_numeric_dtype(vol_raw):
            vol_series = pd.to_numeric(vol_raw, errors="coerce")
        else:
            s = vol_raw.astype(str)
            # Normalize spaces (regular, NBSP, thin spaces) and separators
            s = s.str.replace(r"[\u2009\u202F\u00A0]", "", regex=True)  # thin/NNBSP
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace(" ", "", regex=False)
            s = s.str.strip().str.lower()
            # Treat blanks and non-numeric tokens as NaN
            s = s.replace({"": pd.NA, "-": pd.NA, "na": pd.NA, "nan": pd.NA})
            # If still non-numeric, coerce
            vol_series = pd.to_numeric(s, errors="coerce")
        # No negative volumes
        vol_series = vol_series.mask(vol_series < 0, 0)
        out["volume"] = vol_series.fillna(0)
    else:
        out["volume"] = 0

    # Ticker from sheet name
    out["ticker"] = str(sheet_name)

    # Clean rows: drop bad dates and rows with null close
    out = out.dropna(subset=["date"])  # invalid dates
    out = out.loc[out["close"].notna()].copy()  # enforce valid close

    # Final dtypes/order
    out = out[["date", "ticker", "open", "high", "low", "close", "volume"]]
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _read_xlsx(path: str, sheet: Optional[str] = None) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    """Read the Excel file. If a sheet is specified, return only that; else all sheets."""
    if sheet is None:
        dfs = pd.read_excel(path, sheet_name=None)
        dfs = {k: v for k, v in dfs.items() if v is not None and not v.empty}
        return dfs
    else:
        df = pd.read_excel(path, sheet_name=sheet)
        return df


def _validate_basic(df: pd.DataFrame) -> None:
    """Basic sanity checks: ensure required columns and no nulls in key fields."""
    required = ["date", "ticker", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"Parquet missing required columns: {missing}")
    if df["date"].isna().any():
        raise AssertionError("Null dates present after conversion")
    if df["ticker"].isna().any():
        raise AssertionError("Null tickers present after conversion")
    if df["close"].isna().any():
        n = int(df["close"].isna().sum())
        raise AssertionError(f"Found {n} rows with null close after conversion")


def _print_stats(df: pd.DataFrame) -> None:
    try:
        n_rows = len(df)
        n_cols = df.shape[1]
        tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist()) if "ticker" in df.columns else []
        n_tickers = len(tickers)
        dmin = df["date"].min() if "date" in df.columns else None
        dmax = df["date"].max() if "date" in df.columns else None
        nulls = df.isna().sum()
        print("Conversion stats:")
        print(f"  rows: {n_rows}, cols: {n_cols}")
        if dmin is not None:
            print(f"  date range: {dmin} .. {dmax}")
        print(f"  tickers: {n_tickers}" + (f" (sample: {', '.join(tickers[:10])}{'...' if n_tickers>10 else ''})" if n_tickers else ""))
        print("  nulls per column:")
        for c, v in nulls.items():
            print(f"    {c}: {v}")
    except Exception as e:
        print(f"Warning: failed to print stats: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert Indices XLSX to parquet (long form like sp500_daily)")
    parser.add_argument("--xlsx", type=str, default=os.path.join("data", "Indices.xlsx"), help="Path to input XLSX")
    parser.add_argument("--sheet", type=str, default=None, help="Optional single sheet name to read (defaults to all)")
    parser.add_argument("--out", type=str, default=os.path.join("equities", "indicies.parquet"), help="Output parquet path (directory/file without extension)")
    parser.add_argument("--validate", action="store_true", help="Run basic value checks after writing (no null close)")
    args = parser.parse_args()

    print(f"Reading XLSX: {args.xlsx} (sheet={args.sheet or 'ALL'})")
    raw_any = _read_xlsx(args.xlsx, args.sheet)
    if raw_any is None:
        raise RuntimeError("Failed to read input XLSX. Check the path/sheet.")

    print("Converting sheets (each sheet is one ticker)...")
    if isinstance(raw_any, dict):
        parts: List[pd.DataFrame] = []
        for sheet_name, df in raw_any.items():
            if df is None or df.empty:
                continue
            std_part = _standardize_sheet(sheet_name, df)
            if not std_part.empty:
                parts.append(std_part)
        if not parts:
            raise RuntimeError("No non-empty sheets found in the XLSX after standardization.")
        std = pd.concat(parts, axis=0, ignore_index=True)
    else:
        raw = raw_any
        if raw is None or raw.empty:
            raise RuntimeError("Input XLSX produced an empty DataFrame. Check the path/sheet.")
        # Use provided sheet name as ticker if single sheet requested
        sheet_name = args.sheet or os.path.splitext(os.path.basename(args.xlsx))[0]
        std = _standardize_sheet(sheet_name, raw)

    # Basic required columns check
    req_missing = [c for c in ["date", "ticker", "close"] if c not in std.columns]
    if req_missing:
        raise RuntimeError(f"Missing required columns after standardization: {req_missing}")
    if std[["date", "ticker", "close"]].empty:
        raise RuntimeError("Standardized data is empty after filtering required columns (date, ticker, close).")

    # Ensure output directory exists
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Writing parquet to: {out_path}")
    std.to_parquet(out_path, index=False)
    _print_stats(std)

    if args.validate:
        print("Reloading parquet and running basic validation...")
        pq = pd.read_parquet(out_path)
        _validate_basic(pq)
        print("Validation passed: no nulls in date/ticker/close and required columns present.")

    print("Done.")


if __name__ == "__main__":
    main()
