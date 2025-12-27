from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import pandas as pd

from .catalog_meta import get_meta_library, read_catalog_index


class _MetaLibLike(Protocol):
    def has_symbol(self, symbol: str) -> bool: ...

    def read(self, symbol: str) -> Any: ...

    def write(self, symbol: str, data: Any) -> Any: ...


class _ArcticLike(Protocol):
    def get_library(self, name: str, create_if_missing: bool = ...): ...


ISSUES_SYMBOL = "catalog_quality_issues"
SCANS_SYMBOL = "catalog_quality_scans"


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=timezone.utc))


def _normalize_created_at_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure created_at column is consistently tz-aware UTC.

    ArcticDB can fail to normalize frames if a column mixes tz-aware and tz-naive
    timestamps. We store created_at as UTC tz-aware always.
    """

    if df is None or df.empty:
        return df
    if "created_at" not in df.columns:
        return df

    out = df.copy()
    try:
        # Coerce everything to UTC. If values are tz-naive, interpret them as UTC.
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
        # For tz-naive, localize to UTC; for tz-aware, convert to UTC.
        if getattr(out["created_at"].dt, "tz", None) is None:
            out["created_at"] = out["created_at"].dt.tz_localize("UTC")
        else:
            out["created_at"] = out["created_at"].dt.tz_convert("UTC")
    except Exception:
        # Best effort: leave as-is
        return df

    return out


def _is_lmdb_corruption_error(exc: Exception) -> bool:
    msg = str(exc)
    # Known LMDB/arcticdb low-level corruption patterns
    return (
        "MDB_PAGE_NOTFOUND" in msg
        or "MDB_CORRUPTED" in msg
        or "MDB_INVALID" in msg
        or "LMDBError" in msg
        or "File is not an LMDB file" in msg
    )


def _reset_meta_library(arctic: _ArcticLike) -> None:
    """Best-effort reset for the meta library used by quality registry.

    ArcticDB stores the meta library under platform_meta/catalog.
    If the environment is corrupted, recreating just this library is usually enough.

    Note: this is intentionally defensive and should never raise.
    """

    try:
        # Re-open with create_if_missing to trigger re-init.
        arctic.get_library("platform_meta/catalog", create_if_missing=True)
    except Exception:
        pass


def _safe_read_df(meta_lib: _MetaLibLike, symbol: str, columns: list[str]) -> pd.DataFrame:
    try:
        if not meta_lib.has_symbol(symbol):
            return pd.DataFrame(columns=columns)
        obj = meta_lib.read(symbol)
        data = getattr(obj, "data", obj)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(columns=columns)

        # Defensive: keep created_at consistent to avoid Arctic normalization failures.
        df = _normalize_created_at_utc(df)
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


def _safe_write_df(meta_lib: _MetaLibLike, symbol: str, df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        df = _normalize_created_at_utc(df)
    meta_lib.write(symbol, df)


def _safe_write_df_resilient(*, arctic: _ArcticLike, meta_lib: _MetaLibLike, symbol: str, df: pd.DataFrame) -> None:
    """Write with a single retry if the underlying LMDB store is corrupted."""

    try:
        _safe_write_df(meta_lib, symbol, df)
        return
    except Exception as e:
        if not _is_lmdb_corruption_error(e):
            raise
        # Attempt to recover once.
        _reset_meta_library(arctic)
        meta_lib2 = get_meta_library(arctic=arctic)
        _safe_write_df(meta_lib2, symbol, df)


@dataclass(frozen=True, slots=True)
class QualityIssue:
    issue_id: str
    created_at: pd.Timestamp

    provider: str
    frequency: str
    kind: str
    dataset: str
    entity: str
    library: str
    symbol: str

    severity: str  # info|warning|error
    issue_type: str  # gaps|duplicates|non_monotonic|other

    # summary metrics
    missing_periods: int
    duplicate_timestamps: int
    max_gap_periods: int

    # samples
    missing_intervals_sample: list[list[str]]
    duplicate_timestamps_sample: list[str]


@dataclass(frozen=True, slots=True)
class QualityScan:
    scan_id: str
    created_at: pd.Timestamp
    status: str  # succeeded|failed

    provider: Optional[str] = None
    frequency: Optional[str] = None
    kind: Optional[str] = None
    dataset: Optional[str] = None
    entity: Optional[str] = None

    scanned_symbols: int = 0
    issues_created: int = 0
    error: Optional[str] = None


def _issue_to_record(i: QualityIssue) -> dict[str, Any]:
    return {
        "issue_id": i.issue_id,
        "created_at": pd.Timestamp(i.created_at).tz_convert("UTC") if pd.Timestamp(i.created_at).tzinfo is not None else pd.Timestamp(i.created_at).tz_localize("UTC"),
        "provider": i.provider,
        "frequency": i.frequency,
        "kind": i.kind,
        "dataset": i.dataset,
        "entity": i.entity,
        "library": i.library,
        "symbol": i.symbol,
        "severity": i.severity,
        "issue_type": i.issue_type,
        "issue": i.issue_type,  # UI alias
        "missing_periods": int(i.missing_periods),
        "duplicate_timestamps": int(i.duplicate_timestamps),
        "max_gap_periods": int(i.max_gap_periods),
        "missing_intervals_sample": i.missing_intervals_sample,
        "duplicate_timestamps_sample": i.duplicate_timestamps_sample,
    }


def _scan_to_record(s: QualityScan) -> dict[str, Any]:
    return {
        "scan_id": s.scan_id,
        "created_at": pd.Timestamp(s.created_at).tz_convert("UTC") if pd.Timestamp(s.created_at).tzinfo is not None else pd.Timestamp(s.created_at).tz_localize("UTC"),
        "status": s.status,
        "provider": s.provider,
        "frequency": s.frequency,
        "kind": s.kind,
        "dataset": s.dataset,
        "entity": s.entity,
        "scanned_symbols": int(s.scanned_symbols),
        "issues_created": int(s.issues_created),
        "error": s.error,
    }


def list_quality_issues(
    *,
    arctic: _ArcticLike,
    provider: str | None = None,
    frequency: str | None = None,
    dataset: str | None = None,
    kind: str | None = None,
    entity: str | None = None,
    issue_type: str | None = None,
    severity: str | None = None,
) -> pd.DataFrame:
    meta_lib = get_meta_library(arctic=arctic)
    df = _safe_read_df(
        meta_lib,
        ISSUES_SYMBOL,
        columns=[
            "issue_id",
            "created_at",
            "provider",
            "frequency",
            "kind",
            "dataset",
            "entity",
            "library",
            "symbol",
            "severity",
            "issue_type",
            "missing_periods",
            "duplicate_timestamps",
            "max_gap_periods",
            "missing_intervals_sample",
            "duplicate_timestamps_sample",
        ],
    )

    if df.empty:
        return df

    # filters
    def _eq(col: str, v: str | None) -> None:
        nonlocal df
        if v is None or v == "":
            return
        if col in df.columns:
            df = df[df[col].astype(str) == str(v)]

    _eq("provider", provider)
    _eq("frequency", frequency)
    _eq("dataset", dataset)
    _eq("kind", kind)
    _eq("entity", entity)
    _eq("issue_type", issue_type)
    _eq("severity", severity)

    if "created_at" in df.columns:
        try:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            df = df.sort_values(["created_at"], ascending=False)
        except Exception:
            pass

    return df


def get_quality_issue(*, arctic: _ArcticLike, issue_id: str) -> Optional[dict[str, Any]]:
    df = list_quality_issues(arctic=arctic)
    if df.empty:
        return None
    out = df[df["issue_id"].astype(str) == str(issue_id)]
    if out.empty:
        return None
    row = out.iloc[0].where(pd.notna(out.iloc[0]), None).to_dict()
    return row


def _mk_issue_id(*, provider: str, frequency: str, kind: str, dataset: str, entity: str, issue_type: str) -> str:
    # deterministic-ish id; enough for local-first
    stamp = _utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"{provider}/{frequency}/{kind}/{dataset}/{entity}/{issue_type}"
    return f"qi_{stamp}_{abs(hash(key)) % 10**10}"


def _mk_scan_id() -> str:
    stamp = _utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return f"qs_{stamp}"


def run_quality_scan(
    *,
    arctic: _ArcticLike,
    provider: str | None = None,
    frequency: str | None = None,
    dataset: str | None = None,
    kind: str | None = None,
    entity: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Scan cached symbols for quality issues and record them.

    Current implementation is local-first and in-process. It uses:
      - catalog_index meta to locate symbols
      - describe endpoint logic (services.catalog_symbol.describe_frame) to compute gaps/dupes metrics

    Returns scan record + created issue ids + scan results for UI.
    """

    from .catalog_symbol import describe_frame

    meta_lib = get_meta_library(arctic=arctic)

    scan_id = _mk_scan_id()
    created_at = _utcnow()

    created_issue_ids: list[str] = []
    results: list[dict[str, Any]] = []
    scanned = 0
    issues_created = 0

    try:
        # Load meta index
        idx_df = read_catalog_index(meta_lib=meta_lib)

        # Apply filters on meta index
        if not idx_df.empty:
            if provider is not None and "provider" in idx_df.columns:
                idx_df = idx_df[idx_df["provider"].astype(str).str.upper() == str(provider).upper()]
            if frequency is not None and "frequency" in idx_df.columns:
                idx_df = idx_df[idx_df["frequency"].astype(str) == str(frequency)]
            if dataset is not None and "dataset" in idx_df.columns:
                idx_df = idx_df[idx_df["dataset"].astype(str) == str(dataset)]
            if kind is not None and "kind" in idx_df.columns:
                idx_df = idx_df[idx_df["kind"].astype(str) == str(kind)]
            if entity is not None and "entity" in idx_df.columns:
                idx_df = idx_df[idx_df["entity"].astype(str) == str(entity)]

        if idx_df.empty:
            scan = QualityScan(
                scan_id=scan_id,
                created_at=created_at,
                status="succeeded",
                provider=provider,
                frequency=frequency,
                kind=kind,
                dataset=dataset,
                entity=entity,
                scanned_symbols=0,
                issues_created=0,
                error=None,
            )
            scans_df = _safe_read_df(
                meta_lib,
                SCANS_SYMBOL,
                columns=[
                    "scan_id",
                    "created_at",
                    "status",
                    "provider",
                    "frequency",
                    "kind",
                    "dataset",
                    "entity",
                    "scanned_symbols",
                    "issues_created",
                    "error",
                ],
            )
            scan_rec_df = pd.DataFrame([_scan_to_record(scan)])
            if scans_df.empty:
                scans_df = scan_rec_df
            else:
                scans_df = pd.concat([scans_df, scan_rec_df], ignore_index=True)
            _safe_write_df(meta_lib, SCANS_SYMBOL, scans_df)
            return {"scan": _scan_to_record(scan), "created_issue_ids": [], "results": []}

        # Read existing issues table (append-only)
        issues_df = _safe_read_df(
            meta_lib,
            ISSUES_SYMBOL,
            columns=[
                "issue_id",
                "created_at",
                "provider",
                "frequency",
                "kind",
                "dataset",
                "entity",
                "library",
                "symbol",
                "severity",
                "issue_type",
                "missing_periods",
                "duplicate_timestamps",
                "max_gap_periods",
                "missing_intervals_sample",
                "duplicate_timestamps_sample",
            ],
        )

        initial_issues_count = len(issues_df)

        # Iterate symbols
        for _, row in idx_df.head(int(limit)).iterrows():
            prov = str(row.get("provider") or "")
            freq = str(row.get("frequency") or "")
            knd = str(row.get("kind") or "")
            dset = str(row.get("dataset") or "")
            ent = str(row.get("entity") or "")
            sym = str(row.get("symbol") or "")

            # Prefer explicit library from meta index.
            library = str(row.get("library") or "")
            if not library:
                library = f"market_data/{prov.upper()}/{str(freq).lower()}" if prov and freq else ""

            if not library:
                continue

            try:
                lib = arctic.get_library(library)
            except Exception:
                continue

            scanned += 1
            desc = describe_frame(lib=lib, symbol=sym)
            gaps = desc.get("gaps") or {}

            missing_periods = int(gaps.get("missing_periods") or 0)
            dupes = int(gaps.get("duplicate_timestamps") or 0)
            max_gap_periods = int(gaps.get("max_gap_periods") or 0)

            has_issues = (missing_periods > 0 or dupes > 0 or max_gap_periods > 0)
            
            results.append({
                "symbol": sym,
                "rows": desc.get("rows", 0),
                "missing_periods": missing_periods,
                "duplicate_timestamps": dupes,
                "max_gap_periods": max_gap_periods,
                "has_issues": has_issues,
                "issues": []  # UI might check this
            })

            if not has_issues:
                continue

            severity = "warning"
            if dupes > 0:
                severity = "error"
            if missing_periods > 0 and max_gap_periods > 10:
                severity = "error"

            issue_type = "gaps" if missing_periods > 0 or max_gap_periods > 0 else "duplicates"

            issue_id = _mk_issue_id(
                provider=prov,
                frequency=freq,
                kind=knd,
                dataset=dset,
                entity=ent,
                issue_type=issue_type,
            )

            issue = QualityIssue(
                issue_id=issue_id,
                created_at=_utcnow(),
                provider=prov,
                frequency=freq,
                kind=knd,
                dataset=dset,
                entity=ent,
                library=library,
                symbol=sym,
                severity=severity,
                issue_type=issue_type,
                missing_periods=missing_periods,
                duplicate_timestamps=dupes,
                max_gap_periods=max_gap_periods,
                missing_intervals_sample=list(gaps.get("missing_intervals_sample") or []),
                duplicate_timestamps_sample=list(gaps.get("duplicate_timestamps_sample") or []),
            )

            rec_df = pd.DataFrame([_issue_to_record(issue)])
            if issues_df.empty:
                issues_df = rec_df
            else:
                issues_df = pd.concat([issues_df, rec_df], ignore_index=True)

            created_issue_ids.append(issue_id)
            issues_created += 1

        # Only write if we actually added something
        if len(issues_df) > initial_issues_count:
            _safe_write_df_resilient(arctic=arctic, meta_lib=meta_lib, symbol=ISSUES_SYMBOL, df=issues_df)

        scan = QualityScan(
            scan_id=scan_id,
            created_at=created_at,
            status="succeeded",
            provider=provider,
            frequency=frequency,
            kind=kind,
            dataset=dataset,
            entity=entity,
            scanned_symbols=scanned,
            issues_created=issues_created,
            error=None,
        )

        scans_df = _safe_read_df(
            meta_lib,
            SCANS_SYMBOL,
            columns=[
                "scan_id",
                "created_at",
                "status",
                "provider",
                "frequency",
                "kind",
                "dataset",
                "entity",
                "scanned_symbols",
                "issues_created",
                "error",
            ],
        )
        scan_rec_df = pd.DataFrame([_scan_to_record(scan)])
        if scans_df.empty:
            scans_df = scan_rec_df
        else:
            scans_df = pd.concat([scans_df, scan_rec_df], ignore_index=True)
        _safe_write_df_resilient(arctic=arctic, meta_lib=meta_lib, symbol=SCANS_SYMBOL, df=scans_df)

        return {"scan": _scan_to_record(scan), "created_issue_ids": created_issue_ids, "results": results}

    except Exception as e:
        scan = QualityScan(
            scan_id=scan_id,
            created_at=created_at,
            status="failed",
            provider=provider,
            frequency=frequency,
            kind=kind,
            dataset=dataset,
            entity=entity,
            scanned_symbols=scanned,
            issues_created=issues_created,
            error=str(e),
        )
        scans_df = _safe_read_df(
            meta_lib,
            SCANS_SYMBOL,
            columns=[
                "scan_id",
                "created_at",
                "status",
                "provider",
                "frequency",
                "kind",
                "dataset",
                "entity",
                "scanned_symbols",
                "issues_created",
                "error",
            ],
        )
        scan_rec_df = pd.DataFrame([_scan_to_record(scan)])
        if scans_df.empty:
            scans_df = scan_rec_df
        else:
            scans_df = pd.concat([scans_df, scan_rec_df], ignore_index=True)
        _safe_write_df_resilient(arctic=arctic, meta_lib=meta_lib, symbol=SCANS_SYMBOL, df=scans_df)
        return {"scan": _scan_to_record(scan), "created_issue_ids": created_issue_ids, "results": results}
