from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd

from .catalog_meta import get_meta_library, read_catalog_index


class _ArcticLibLike(Protocol):
    def read(self, symbol: str) -> Any: ...


class _ArcticLike(Protocol):
    def get_library(self, name: str, create_if_missing: bool = ...): ...


@dataclass(frozen=True, slots=True)
class SymbolMetaRecord:
    symbol: str
    row: dict[str, Any]


def get_symbol_meta(*, arctic: _ArcticLike, symbol: str) -> Optional[SymbolMetaRecord]:
    """Lookup one symbol in the catalog_index_v1 metadata table."""

    symbol = str(symbol)
    meta_lib = get_meta_library(arctic=arctic)
    df = read_catalog_index(meta_lib=meta_lib)
    if df.empty or "symbol" not in df.columns:
        return None

    out = df[df["symbol"].astype(str) == symbol]
    if out.empty:
        return None

    # Take the last row (latest write wins)
    row = out.iloc[-1].where(pd.notna(out.iloc[-1])).to_dict()
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return SymbolMetaRecord(symbol=symbol, row=row)


def preview_frame(
    *,
    lib: _ArcticLibLike,
    symbol: str,
    head: int = 5,
    tail: int = 5,
) -> dict[str, Any]:
    """Return a lightweight preview for a cached frame.

    Output includes:
      - library (best-effort / may be None)
      - columns
      - index start/end
      - head/tail records (row-oriented)

    Best-effort: will raise if lib.read fails.
    """

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*BlockManagerUnconsolidated.*",
            category=DeprecationWarning,
        )
        obj = lib.read(symbol)

    data = getattr(obj, "data", obj)

    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "value")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    # Ensure a sane, sortable time index when possible.
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    h = int(head) if head is not None else 5
    t = int(tail) if tail is not None else 5
    h = max(h, 0)
    t = max(t, 0)

    def _infer_ohlc_aliases(frame: pd.DataFrame) -> pd.DataFrame:
        """Return a frame that contains canonical 'open/high/low/close/volume' columns when inferable.

        Strategy:
          1) If provider used variants like 'Open'/'Close'/'Adj Close', rename them to canonical lowercase.
          2) If canonical columns still missing but a source column exists, add alias columns.

        This is preview-only normalization (does not affect stored data).
        """

        if frame.empty:
            return frame

        cols = list(frame.columns)
        lower_to_col: dict[str, str] = {str(c).strip().lower(): c for c in cols}

        # rename map from original column -> canonical
        rename_map: dict[Any, str] = {}
        for canon, variants in {
            "open": ["open"],
            "high": ["high"],
            "low": ["low"],
            "close": ["close", "adj close", "adj_close", "adjclose", "settle", "px_last"],
            "volume": ["volume", "vol"],
        }.items():
            for v in variants:
                src = lower_to_col.get(str(v).strip().lower())
                if src is not None:
                    rename_map[src] = canon
                    break

        out = frame
        if rename_map:
            try:
                out = out.rename(columns=rename_map)
            except Exception:
                out = frame

        # Refresh mapping after rename
        lower_to_col = {str(c).strip().lower(): c for c in list(out.columns)}

        # If canonical columns still missing, add them as aliases using existing columns.
        def _add_alias(canon: str, variants: list[str]) -> None:
            nonlocal out, lower_to_col
            if canon in lower_to_col:
                return
            for v in variants:
                src = lower_to_col.get(str(v).strip().lower())
                if src is not None:
                    out = out.assign(**{canon: out[src]})
                    lower_to_col = {str(c).strip().lower(): c for c in list(out.columns)}
                    return

        _add_alias("open", ["open"])
        _add_alias("high", ["high"])
        _add_alias("low", ["low"])
        _add_alias("close", ["close", "adj close", "adj_close", "adjclose", "settle", "px_last"])
        _add_alias("volume", ["volume", "vol"])

        return out

    def _records(d: pd.DataFrame) -> list[dict[str, Any]]:
        # include index as 'ts' for UI
        out = _infer_ohlc_aliases(d)

        # Ensure `ts` is always a clean ISO-8601 string that JS Date.parse can handle.
        # Arctic/pandas indices can carry timezone-naive values; `strftime("%z")` yields empty string
        # and can produce timestamps that some JS engines parse inconsistently.
        o = out.copy()
        o.insert(0, "ts", o.index)

        try:
            if isinstance(o.index, pd.DatetimeIndex):
                # Always emit strict UTC ISO timestamps for robust JS parsing.
                # Using a trailing 'Z' avoids inconsistencies across JS engines with '+00:00'.
                idx = pd.DatetimeIndex(o.index)
                try:
                    if getattr(idx, "tz", None) is None:
                        idx = idx.tz_localize("UTC")  # type: ignore[attr-defined]
                    else:
                        idx = idx.tz_convert("UTC")  # type: ignore[attr-defined]
                except Exception:
                    pass

                o["ts"] = [pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ") for x in idx]
            else:
                o["ts"] = o["ts"].astype(str)
        except Exception:
            o["ts"] = o["ts"].astype(str)

        # Convert NaN/NA to None for JSON.
        o = o.replace({pd.NA: None})
        try:
            # pandas type stubs are strict; runtime supports None replacement.
            o = o.where(pd.notna(o), None)  # type: ignore[arg-type]
        except Exception:
            pass
        return o.to_dict(orient="records")  # type: ignore[return-value]

    idx_min = None
    idx_max = None
    try:
        idx_min = None if len(df) == 0 else str(df.index.min())
        idx_max = None if len(df) == 0 else str(df.index.max())
    except Exception:
        pass

    # Preview columns should reflect aliasing and include the 'ts' index we injected.
    df_for_cols = _infer_ohlc_aliases(df)
    cols = ["ts"] + [str(c) for c in df_for_cols.columns]

    return {
        "library": str(getattr(lib, "name", None) or getattr(lib, "library", None) or getattr(lib, "_name", None) or "") or None,
        "symbol": str(symbol),
        "columns": cols,
        "rows": int(len(df)),
        "index_start": idx_min,
        "index_end": idx_max,
        "head": _records(df.head(h)) if h else [],
        "tail": _records(df.tail(t)) if t else [],
    }


def describe_frame(*, lib: _ArcticLibLike, symbol: str) -> dict[str, Any]:
    """Compute accurate summary stats for a cached frame.

    This is intended for platform UI analytics (Phase 3.1): it reads the full cached
    DataFrame and returns a lightweight JSON-serializable summary.

    Output keys:
      - library, symbol
      - rows, columns
      - index_start, index_end
      - dtypes: {col: dtype_str}
      - missing: {col: missing_count}
      - numeric: {col: {min,max,mean,std}}

    Best-effort: will raise if lib.read fails.
    """

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*BlockManagerUnconsolidated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Passing a BlockManagerUnconsolidated to DataFrame is deprecated.*",
            category=DeprecationWarning,
        )
        obj = lib.read(symbol)

    data = getattr(obj, "data", obj)

    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "value")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # index range
    idx_min = None
    idx_max = None
    try:
        idx_min = None if len(df) == 0 else str(df.index.min())
        idx_max = None if len(df) == 0 else str(df.index.max())
    except Exception:
        pass

    # dtypes / missing / uniques / non-null
    dtypes: dict[str, str] = {}
    missing: dict[str, int] = {}
    non_null_pct: dict[str, float] = {}
    unique: dict[str, int] = {}

    nrows = int(len(df))
    # Avoid expensive unique counts on huge frames (still bounded for platform use)
    max_unique_scan = 250_000

    for c in df.columns:
        cs = str(c)
        s = df[c]
        try:
            dtypes[cs] = str(s.dtype)
        except Exception:
            dtypes[cs] = "unknown"

        miss = 0
        try:
            miss = int(s.isna().sum())
        except Exception:
            miss = 0
        missing[cs] = miss

        try:
            non_null_pct[cs] = float(0.0 if nrows == 0 else (1.0 - (miss / max(nrows, 1))))
        except Exception:
            non_null_pct[cs] = 0.0

        try:
            if nrows and nrows <= max_unique_scan:
                unique[cs] = int(s.nunique(dropna=True))
            else:
                unique[cs] = -1
        except Exception:
            unique[cs] = -1

    # numeric stats (only for numeric dtypes)
    numeric: dict[str, dict[str, Any]] = {}
    try:
        num_df = df.select_dtypes(include=["number"])
    except Exception:
        num_df = pd.DataFrame(index=df.index)

    for c in num_df.columns:
        cs = str(c)
        s = num_df[c]
        try:
            s_non = s.dropna()
            numeric[cs] = {
                "min": None if s_non.empty else float(s_non.min()),
                "max": None if s_non.empty else float(s_non.max()),
                "mean": None if s_non.empty else float(s_non.mean()),
                "std": None if s_non.empty else float(s_non.std(ddof=1)),
            }
        except Exception:
            numeric[cs] = {"min": None, "max": None, "mean": None, "std": None}

    # time index gap detection (best-effort; only if datetime-like)
    gaps: dict[str, Any] = {
        "expected_freq": None,
        "missing_periods": 0,
        "missing_timestamps_sample": [],
        "missing_intervals_sample": [],
        "duplicate_timestamps": 0,
        "duplicate_timestamps_sample": [],
        "max_gap_periods": 0,
        "max_gap_days": 0,
    }
    try:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 1:
            idx_sorted = df.index.sort_values()

            # duplicates
            try:
                dup_mask = idx_sorted.duplicated(keep="first")
                gaps["duplicate_timestamps"] = int(dup_mask.sum())
                if gaps["duplicate_timestamps"]:
                    gaps["duplicate_timestamps_sample"] = [str(ts) for ts in idx_sorted[dup_mask][:10]]
            except Exception:
                gaps["duplicate_timestamps"] = 0
                gaps["duplicate_timestamps_sample"] = []

            # max observed gaps (in calendar days) between consecutive timestamps
            if len(idx_sorted) >= 2:
                try:
                    deltas = idx_sorted.to_series().diff().dropna()
                    if not deltas.empty:
                        max_days = float(deltas.max() / pd.Timedelta(days=1))
                        # gaps are periods/days beyond the expected 1-step distance
                        gaps["max_gap_days"] = int(max(0, round(max_days - 1)))
                except Exception:
                    pass

                inferred = pd.infer_freq(pd.DatetimeIndex(idx_sorted))
                gaps["expected_freq"] = inferred

                # Fallback: if infer_freq fails (e.g., because there are gaps), estimate a daily-like cadence.
                inferred_eff = inferred
                if inferred_eff is None:
                    try:
                        deltas = idx_sorted.to_series().diff().dropna()
                        if not deltas.empty:
                            # Use median delta as a robust estimator.
                            med = deltas.median()
                            try:
                                # pandas typing stubs can treat Timedelta median as NaTType; normalize defensively.
                                med_days = float(med / pd.Timedelta(days=1)) if pd.notna(med) else 0.0  # type: ignore[arg-type]
                            except Exception:
                                med_days = 0.0
                            if 0.9 <= med_days <= 1.1:
                                inferred_eff = "D"
                            else:
                                inferred_eff = None
                    except Exception:
                        inferred_eff = None

                # For now, support daily-like only (platform focuses on 1d but should be extensible)
                if inferred_eff in ("D", "B"):
                    full = pd.date_range(idx_sorted.min(), idx_sorted.max(), freq=inferred_eff)
                    missing_idx = full.difference(pd.DatetimeIndex(idx_sorted))
                    gaps["missing_periods"] = int(len(missing_idx))
                    gaps["missing_timestamps_sample"] = [str(ts) for ts in missing_idx[:10]]

                    # Compute max gap in terms of missing periods between consecutive points.
                    try:
                        pos = pd.Index(full).get_indexer(idx_sorted.unique())
                        if len(pos) >= 2:
                            step = pd.Series(pos).diff().dropna()
                            max_step = int(step.max())
                            gaps["max_gap_periods"] = int(max(0, max_step - 1))
                    except Exception:
                        pass

                    # Missing intervals sample for UI: show a few contiguous missing runs as [start,end]
                    try:
                        if len(missing_idx):
                            miss = missing_idx.sort_values()
                            run_start = miss[0]
                            prev = miss[0]
                            intervals = []
                            for ts in miss[1:]:
                                if inferred_eff == "D":
                                    expected_next = prev + pd.Timedelta(days=1)
                                else:  # "B"
                                    expected_next = prev + pd.tseries.offsets.BDay(1)
                                if ts == expected_next:
                                    prev = ts
                                    continue
                                intervals.append([str(run_start), str(prev)])
                                if len(intervals) >= 5:
                                    break
                                run_start = ts
                                prev = ts
                            if len(intervals) < 5:
                                intervals.append([str(run_start), str(prev)])
                            gaps["missing_intervals_sample"] = intervals
                    except Exception:
                        pass

    except Exception:
        pass

    return {
        "library": str(getattr(lib, "name", None) or getattr(lib, "library", None) or getattr(lib, "_name", None) or "") or None,
        "symbol": str(symbol),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "index_start": idx_min,
        "index_end": idx_max,
        "dtypes": dtypes,
        "missing": missing,
        "numeric": numeric,
        "gaps": gaps,
        "non_null_pct": non_null_pct,
        "unique": unique,
    }

