"""
LSE PEAD — 001_baseline — discover.py
======================================
Pre-flight data profiler. Run BEFORE build_signals.py.

Connects directly to sfera postgres and profiles the raw source tables.
Flags contamination, gaps, embargoes, and distribution issues UPFRONT
so you don't waste time building a broken signal.

Inspired by the BMLL "Data Discovery Summary" panel pattern:
  – column-level null rates + distribution type
  – universe contamination (investment trusts, ETFs, cross-listed shells)
  – coverage by year (embargo / thin history detection)
  – signal preview (can SUE actually be computed from these sources?)

Outputs:
  data/discovery.json — structured, machine-readable; read by plan.yaml + sweep.py

Run from btest/ root:
    uv run python "research/generated/LSE PEAD/001_baseline/discover.py"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg

SFERA_ROOT = Path(r"c:\Personal\Business & Investments\Python codes\sfera")
sys.path.insert(0, str(SFERA_ROOT))
import site
_sfera_venv = SFERA_ROOT / ".venv" / "Lib" / "site-packages"
if _sfera_venv.exists():
    site.addsitedir(str(_sfera_venv))
from data.config.database_config import DB_CONFIG

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Constants (must mirror build_signals.py) ───────────────────────────────
START           = "2015-01-01"
END             = "2026-01-01"
EXCHANGE        = "LSE"
MIN_EVENTS      = 4
MIN_PRICE       = 100.0   # GBX
# Known contamination categories — investment trusts / ETFs report NAV-based EPS,
# not operating earnings. Their SUE is meaningless for PEAD.
EXCLUDE_INDUSTRIES = ["Asset Management", "Exchange Traded Fund", "Closed End Fund"]


def _conn():
    return psycopg.connect(**{k: DB_CONFIG[k] for k in ["host", "port", "dbname", "user", "password"]})


def _print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def profile_earnings(conn) -> dict:
    """Profile eodhd.earnings_history for LSE."""
    _print_section("1. Source Table: eodhd.earnings_history (LSE)")

    df = pd.read_sql("""
        SELECT ticker, date, report_date,
               eps_actual, eps_estimate, eps_difference,
               before_after_market
        FROM eodhd.earnings_history
        WHERE exchange = %(ex)s AND deprecated_at IS NULL
        ORDER BY ticker, date
    """, conn, params={"ex": EXCHANGE})

    total_rows   = len(df)
    n_tickers    = df["ticker"].nunique()
    date_range   = (df["report_date"].min(), df["report_date"].max())

    # Null rates per column
    null_rates = {}
    for col in ["eps_actual", "eps_estimate", "eps_difference", "before_after_market", "report_date"]:
        null_rates[col] = round(df[col].isna().mean() * 100, 1)

    # Coverage by year
    df["report_date"] = pd.to_datetime(df["report_date"])
    by_year = df[
        (df["report_date"] >= START) & (df["report_date"] < END)
    ].groupby(df["report_date"].dt.year).agg(
        events=("ticker", "count"),
        tickers=("ticker", "nunique"),
        pct_with_estimate=("eps_estimate", lambda x: round(x.notna().mean() * 100, 1))
    ).to_dict(orient="index")

    # Distribution of eps_difference
    diff = df["eps_difference"].dropna()
    dist = {
        "mean": round(float(diff.mean()), 4),
        "std":  round(float(diff.std()), 4),
        "p5":   round(float(diff.quantile(0.05)), 4),
        "p25":  round(float(diff.quantile(0.25)), 4),
        "p50":  round(float(diff.quantile(0.50)), 4),
        "p75":  round(float(diff.quantile(0.75)), 4),
        "p95":  round(float(diff.quantile(0.95)), 4),
    }

    print(f"  Rows: {total_rows:,}  |  Tickers: {n_tickers:,}")
    print(f"  Date range: {date_range[0]} → {date_range[1]}")
    print(f"  Null rates:")
    for col, pct in null_rates.items():
        flag = " ⚠" if pct > 20 else ""
        print(f"    {col:<30} {pct:>5.1f}%{flag}")
    print(f"  eps_difference dist: mean={dist['mean']}  std={dist['std']}  "
          f"p5={dist['p5']}  p95={dist['p95']}")
    print(f"\n  Coverage by year (backtest window {START} → {END}):")
    print(f"  {'Year':<6} {'Events':>8} {'Tickers':>8} {'% w/ estimate':>14}")
    for yr, row in sorted(by_year.items()):
        thin_flag = " ⚠ THIN" if row["events"] < 50 else ""
        print(f"  {yr:<6} {row['events']:>8,} {row['tickers']:>8,} {row['pct_with_estimate']:>13.1f}%{thin_flag}")

    return {
        "total_rows": total_rows,
        "n_tickers": n_tickers,
        "null_rates": null_rates,
        "coverage_by_year": by_year,
        "eps_difference_dist": dist,
    }


def profile_contamination(conn, earnings_tickers: list[str]) -> dict:
    """Check how many tickers are investment trusts, ETFs, or cross-listed shells."""
    _print_section("2. Universe Contamination Check")

    df = pd.read_sql("""
        SELECT ticker, name, sector, industry
        FROM eodhd.instruments
        WHERE exchange_code = %(ex)s
          AND ticker = ANY(%(tickers)s)
    """, conn, params={"ex": EXCHANGE, "tickers": earnings_tickers})

    total = len(df)

    # Investment trusts / ETFs (SUE is meaningless for these)
    contaminated = df[df["industry"].isin(EXCLUDE_INDUSTRIES)]

    # Cross-listed shells: non-GB primary, blank sector/industry
    shells = df[(df["sector"] == "") & (~df["ticker"].isin(contaminated["ticker"]))]

    # Clean operating companies
    clean = df[
        (~df["ticker"].isin(contaminated["ticker"])) &
        (~df["ticker"].isin(shells["ticker"]))
    ]

    print(f"  Total instruments profiled: {total:,}")
    print(f"  ✅ Clean operating companies:  {len(clean):,}  ({len(clean)/max(total,1)*100:.1f}%)")
    print(f"  ⚠  Investment trusts/ETFs:     {len(contaminated):,}  ({len(contaminated)/max(total,1)*100:.1f}%)")
    print(f"  ⚠  Cross-listed shells:        {len(shells):,}  ({len(shells)/max(total,1)*100:.1f}%)")

    if not contaminated.empty:
        print(f"\n  Investment Trust/ETF tickers (top 20):")
        for _, r in contaminated.head(20).iterrows():
            print(f"    {r['ticker']:<8} {r['name'][:45]:<45} [{r['industry']}]")

    if not shells.empty:
        print(f"\n  Cross-listed shell tickers (top 10, blank sector):")
        for _, r in shells.head(10).iterrows():
            print(f"    {r['ticker']:<8} {r['name'][:45]}")

    return {
        "total_profiled": total,
        "clean_count": len(clean),
        "contaminated_count": len(contaminated),
        "shell_count": len(shells),
        "contaminated_tickers": contaminated["ticker"].tolist(),
        "shell_tickers": shells["ticker"].tolist(),
        "clean_pct": round(len(clean) / max(total, 1) * 100, 1),
    }


def profile_prices(conn, n_sample: int = 20) -> dict:
    """Quick price sanity check — coverage and min-price filter impact."""
    _print_section("3. Price Data Quality (eodhd.prices, LSE, adj_close)")

    df = pd.read_sql("""
        SELECT ticker,
               COUNT(*) AS days,
               MIN(adj_close_price) AS min_px,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY adj_close_price) AS median_px,
               MAX(adj_close_price) AS max_px,
               SUM(CASE WHEN adj_close_price <= 0 THEN 1 ELSE 0 END) AS zero_days
        FROM eodhd.prices
        WHERE exchange = %(ex)s
          AND trade_date >= '2012-01-01'
          AND trade_date < %(end)s
          AND deprecated_at IS NULL
        GROUP BY ticker
    """, conn, params={"ex": EXCHANGE, "end": END})

    total = len(df)
    below_min_price = (df["median_px"] < MIN_PRICE).sum()
    short_history   = (df["days"] < 252).sum()
    zero_price      = (df["zero_days"] > 0).sum()

    print(f"  Tickers with price history: {total:,}")
    print(f"  Below MIN_PRICE ({MIN_PRICE} GBX median):  {below_min_price:,}  ({below_min_price/total*100:.1f}%)")
    print(f"  Insufficient history (<252d):             {short_history:,}  ({short_history/total*100:.1f}%)")
    print(f"  Zero/negative price days:                 {zero_price:,} tickers affected")

    return {
        "total_tickers_with_prices": total,
        "pct_below_min_price": round(below_min_price / total * 100, 1),
        "pct_short_history": round(short_history / total * 100, 1),
        "zero_price_tickers": int(zero_price),
    }


def compute_sue_preview(conn) -> dict:
    """
    Compute a quick SUE preview directly in SQL to validate the signal
    before running the full build_signals pipeline.
    """
    _print_section("4. SUE Signal Preview (SQL preview, no parquet write)")

    df = pd.read_sql("""
        SELECT ticker, date, report_date, eps_actual, eps_estimate, eps_difference
        FROM eodhd.earnings_history
        WHERE exchange = %(ex)s
          AND deprecated_at IS NULL
          AND report_date >= %(start)s
          AND report_date < %(end)s
          AND eps_estimate IS NOT NULL
          AND eps_difference IS NOT NULL
        ORDER BY ticker, date
    """, conn, params={"ex": EXCHANGE, "start": START, "end": END})

    # Compute SUE (same formula as build_signals.py)
    def _sue(grp):
        vol = grp["eps_actual"].shift(1).rolling(8, min_periods=4).std()
        return (grp["eps_difference"] / vol).clip(-5, 5)

    df["sue"] = df.groupby("ticker", group_keys=False).apply(_sue, include_groups=False)
    df_valid  = df.dropna(subset=["sue"])

    n_events  = len(df_valid)
    n_tickers = df_valid["ticker"].nunique()

    # Quintile distribution of SUE
    quint = pd.qcut(df_valid["sue"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
    quint_counts = quint.value_counts().sort_index().to_dict()

    # Fraction with |SUE| > thresholds (signal strength filter candidates)
    for thr in [0.5, 1.0, 2.0]:
        pct = (df_valid["sue"].abs() > thr).mean() * 100
        print(f"  |SUE| > {thr:.1f}: {pct:.1f}% of events")

    dist = {
        "n_events": n_events,
        "n_tickers": n_tickers,
        "mean": round(float(df_valid["sue"].mean()), 3),
        "std":  round(float(df_valid["sue"].std()), 3),
        "pct_above_1sig": round((df_valid["sue"].abs() > 1.0).mean() * 100, 1),
        "pct_above_2sig": round((df_valid["sue"].abs() > 2.0).mean() * 100, 1),
        "quintile_counts": {k: int(v) for k, v in quint_counts.items()},
    }

    # Events per year
    df_valid["year"] = pd.to_datetime(df_valid["report_date"]).dt.year
    by_year = df_valid.groupby("year").size().to_dict()
    dist["events_by_year"] = {int(k): int(v) for k, v in by_year.items()}

    print(f"\n  Valid SUE events: {n_events:,} across {n_tickers} tickers")
    print(f"  SUE: mean={dist['mean']}  std={dist['std']}")
    print(f"\n  Events by year:")
    for yr, n in sorted(by_year.items()):
        bar = "█" * (n // 10)
        thin_flag = " ⚠ THIN" if n < 30 else ""
        print(f"  {yr}: {n:>5}  {bar}{thin_flag}")

    return dist


def write_discovery(result: dict):
    out_path = DATA_DIR / "discovery.json"
    result["generated_at"] = datetime.utcnow().isoformat()
    result["exchange"] = EXCHANGE
    result["start"] = START
    result["end"] = END
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  ✅ Wrote {out_path}")


def main():
    print("=" * 60)
    print("  LSE PEAD · 001_baseline — Data Discovery")
    print("=" * 60)

    conn = _conn()

    # 1. Profile earnings source table
    earnings_result = profile_earnings(conn)

    # 2. Check contamination
    all_tickers = pd.read_sql(
        "SELECT DISTINCT ticker FROM eodhd.earnings_history WHERE exchange = %s AND deprecated_at IS NULL",
        conn, params=(EXCHANGE,)
    )["ticker"].tolist()
    contamination = profile_contamination(conn, all_tickers)

    # 3. Price quality
    prices = profile_prices(conn)

    # 4. SUE preview
    sue_preview = compute_sue_preview(conn)

    conn.close()

    # ── Summary flags ──────────────────────────────────────────────────────
    _print_section("5. Discovery Summary")

    flags = []
    if contamination["contaminated_count"] > 0:
        flags.append({
            "level": "WARN",
            "message": f"{contamination['contaminated_count']} investment trust/ETF tickers will contaminate SUE",
            "action": "Add industry exclusion filter to build_signals.py"
        })
    if contamination["shell_count"] > 0:
        flags.append({
            "level": "WARN",
            "message": f"{contamination['shell_count']} cross-listed shells (blank sector) in universe",
            "action": "Consider filtering on sector != '' in build_signals.py"
        })
    if prices["zero_price_tickers"] > 0:
        flags.append({
            "level": "WARN",
            "message": f"{prices['zero_price_tickers']} tickers have zero/negative price days",
            "action": "adj_close_price > 0 filter in build_signals.py handles this"
        })
    thin_years = [yr for yr, n in sue_preview.get("events_by_year", {}).items() if n < 30]
    if thin_years:
        flags.append({
            "level": "WARN",
            "message": f"Thin event coverage in years: {thin_years}",
            "action": "Consider extending START date or accepting sparse coverage"
        })

    for f in flags:
        print(f"  [{f['level']}] {f['message']}")
        print(f"         → {f['action']}")

    if not flags:
        print("  ✅ No major issues detected")

    # ── Variant suggestions ────────────────────────────────────────────────
    _print_section("6. Suggested Variants (for sweep.py)")
    variants = [
        {"id": "v1_baseline",   "hold_days": 21, "top_n": 8,  "rebalance": "1d",  "sue_min_abs": None, "notes": "current baseline"},
        {"id": "v2_weekly",     "hold_days": 21, "top_n": 8,  "rebalance": "1w",  "sue_min_abs": None, "notes": "reduce daily churn"},
        {"id": "v3_strong_sue", "hold_days": 21, "top_n": 8,  "rebalance": "1d",  "sue_min_abs": 1.0,  "notes": "only |SUE|>1σ events"},
        {"id": "v4_short_hold", "hold_days": 10, "top_n": 8,  "rebalance": "1d",  "sue_min_abs": None, "notes": "capture early drift only"},
        {"id": "v5_long_hold",  "hold_days": 42, "top_n": 8,  "rebalance": "1d",  "sue_min_abs": None, "notes": "longer drift window"},
        {"id": "v6_wider_port", "hold_days": 21, "top_n": 12, "rebalance": "1w",  "sue_min_abs": None, "notes": "wider portfolio"},
        {"id": "v7_clean_univ", "hold_days": 21, "top_n": 8,  "rebalance": "1d",  "sue_min_abs": None,
         "notes": "exclude Asset Management + blank sector (trusts/shells filtered out)",
         "exclude_industries": EXCLUDE_INDUSTRIES, "exclude_blank_sector": True},
    ]
    for v in variants:
        print(f"  {v['id']:<20} hold={v.get('hold_days','?'):>3}d  "
              f"top_n={v.get('top_n','?'):>3}  "
              f"rebal={v.get('rebalance','?'):<3}  "
              f"|sue|>{v.get('sue_min_abs') or '—':<4}  "
              f"// {v['notes']}")

    # Write discovery.json
    result = {
        "earnings": earnings_result,
        "contamination": contamination,
        "prices": prices,
        "sue_preview": sue_preview,
        "flags": flags,
        "suggested_variants": variants,
    }
    write_discovery(result)
    print("\n  Run next: uv run python \"research/generated/LSE PEAD/001_baseline/build_signals.py\"")


if __name__ == "__main__":
    main()
