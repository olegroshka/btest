"""
universe.py — Shared LSE Universe Definition
=============================================
Single source of truth for the canonical LSE dividend-growth universe.
Import this in any build_signals.py or discover.py within this strategy.

Usage
-----
from research.generated.Dividend_Growth.universe import LSEUniverse

u = LSEUniverse(conn)
u.introspect()          # print schema of all source tables — call once at script top
tickers = u.tickers()  # list[str] — canonical universe tickers for backtest window
prices  = u.prices()   # pd.DataFrame — OHLCV for all universe tickers, cleaned

Design notes
------------
- All filters are defined here and nowhere else. To change the universe, change this file.
- introspect() is cheap (information_schema queries) — call it at the top of every script
  so you always see the current schema and catch column renames early.
- prices() applies the bad-tick filter and currency/price/history guards.
- Individual signals may use a *subset* of these tickers (e.g. only dividend payers,
  or only tickers with analyst EPS coverage) — that narrowing happens in each signal's
  build_signals.py, not here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ── Database connection ───────────────────────────────────────────────────────
DB = dict(host="localhost", dbname="sfera", user="postgres", password="lokomotiv")

# ── Canonical universe parameters ─────────────────────────────────────────────
# Change these here; all signals pick them up automatically.
EXCHANGE        = "LSE"
CURRENCIES      = ("GBX", "GBP")   # exclude USD/EUR cross-listings and GDRs
MIN_PRICE_GBX   = 100.0            # 1 GBP floor — excludes shells, warrants, pennies
MIN_HISTORY_DAYS = 252             # ~1 trading year before a ticker enters the universe
BAD_TICK_THRESHOLD = 0.50          # |1-day adj_close return| > 50% → treat as data error

PULL_START      = "2013-01-01"     # pull earlier than backtest start for warmup
PULL_END        = "2026-06-01"
BACKTEST_START  = "2015-01-01"
BACKTEST_END    = "2026-01-01"

# ── Shared output paths (used by all signals in this strategy group) ──────────
# lse_prices.parquet is written ONCE by the first signal that needs it.
# All strategy.py and notebooks reference this single file.
STRATEGY_ROOT    = Path(__file__).resolve().parent
SHARED_DATA_DIR  = STRATEGY_ROOT / "shared_data"
SHARED_PRICES    = SHARED_DATA_DIR / "lse_prices.parquet"
SHARED_PRICES_REL = "research/generated/Dividend Growth/shared_data/lse_prices.parquet"

# ── Tables profiled by introspect() ───────────────────────────────────────────
_SOURCE_TABLES = [
    ("eodhd", "prices"),
    ("eodhd", "dividends"),
    ("eodhd", "earnings_history"),
    ("eodhd", "instruments"),
]


class LSEUniverse:
    """
    Wraps the canonical LSE universe definition.

    Parameters
    ----------
    conn : psycopg2 connection (caller owns it — we never close it here)
    verbose : bool — print progress messages (default True)
    """

    def __init__(self, conn, verbose: bool = True):
        self._conn    = conn
        self._verbose = verbose
        self._prices_cache: pd.DataFrame | None = None
        self._tickers_cache: list[str] | None   = None

    # ── Schema introspection ──────────────────────────────────────────────────

    def introspect(self) -> dict[str, list[str]]:
        """
        Print columns for every source table this strategy touches.
        Call this at the TOP of every build_signals.py and discover.py.
        Returns dict of {schema.table: [col, ...]} for programmatic use.
        """
        cur = self._conn.cursor()
        schema_map: dict[str, list[str]] = {}

        print("=" * 65)
        print("  SCHEMA INTROSPECTION - sfera postgres")
        print("=" * 65)

        for schema, table in _SOURCE_TABLES:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            rows = cur.fetchall()
            key  = f"{schema}.{table}"
            cols = [r[0] for r in rows]
            schema_map[key] = cols

            if self._verbose:
                print(f"\n  {key}")
                print(f"  {'-' * 55}")
                for col, dtype, nullable in rows:
                    null_flag = " (nullable)" if nullable == "YES" else ""
                    print(f"    {col:<35} {dtype}{null_flag}")

        print("=" * 65)
        return schema_map

    # ── Universe tickers ──────────────────────────────────────────────────────

    def tickers(self) -> list[str]:
        """
        Return canonical universe tickers: LSE equities in GBX/GBP,
        with MinPrice >= 100 GBX and MinHistory >= 252 days in backtest window.
        Result is cached after first call.
        """
        if self._tickers_cache is not None:
            return self._tickers_cache

        if self._verbose:
            print(f"\nBuilding canonical universe  [{BACKTEST_START} to {BACKTEST_END}] ...")

        q = """
        SELECT
            p.ticker,
            i.currency,
            COUNT(DISTINCT p.trade_date)         AS trading_days,
            AVG(p.adj_close_price)               AS avg_price_gbx,
            MIN(p.trade_date)                    AS first_date,
            MAX(p.trade_date)                    AS last_date
        FROM eodhd.prices p
        LEFT JOIN eodhd.instruments i ON p.ticker = i.ticker
        WHERE p.exchange     = %(exchange)s
          AND p.trade_date  >= %(start)s
          AND p.trade_date  <= %(end)s
          AND p.adj_close_price > 0
          AND p.close_price    > 0
          AND p.deprecated_at IS NULL
        GROUP BY p.ticker, i.currency
        HAVING
            AVG(p.adj_close_price)      >= %(min_price)s
            AND COUNT(DISTINCT p.trade_date) >= %(min_history)s
        """
        df = pd.read_sql(
            q, self._conn,
            params={
                "exchange":    EXCHANGE,
                "start":       BACKTEST_START,
                "end":         BACKTEST_END,
                "min_price":   MIN_PRICE_GBX,
                "min_history": MIN_HISTORY_DAYS,
            },
        )

        # Currency filter — remove USD/EUR cross-listings
        before = len(df)
        df = df[df["currency"].isin(CURRENCIES) | df["currency"].isna()]
        after  = len(df)

        self._tickers_cache = sorted(df["ticker"].tolist())

        if self._verbose:
            print(f"  Raw LSE tickers (price+history filter) : {before:,}")
            print(f"  After currency filter {CURRENCIES}    : {after:,}")
            print(f"  Canonical universe size                : {len(self._tickers_cache):,} tickers")

        return self._tickers_cache

    # ── Prices ────────────────────────────────────────────────────────────────

    def prices(self) -> pd.DataFrame:
        """
        Load and return cleaned OHLCV for all universe tickers.
        Applies bad-tick filter (|1d return| > BAD_TICK_THRESHOLD → NaN → ffill).
        Index is DatetimeIndex(tz=None, time=05:00:00) — matches QuantDSL convention.
        Result is cached after first call.
        """
        if self._prices_cache is not None:
            return self._prices_cache

        tickers = self.tickers()
        if self._verbose:
            print(f"\nLoading prices for {len(tickers):,} universe tickers  [{PULL_START} to {PULL_END}] ...")

        placeholders = ",".join(["%s"] * len(tickers))
        q = f"""
        SELECT
            ticker,
            trade_date::date        AS trade_date,
            open_price              AS open,
            high_price              AS high,
            low_price               AS low,
            adj_close_price         AS close,
            close_price             AS close_unadj,
            volume
        FROM eodhd.prices
        WHERE exchange      = %s
          AND ticker        IN ({placeholders})
          AND trade_date   >= %s
          AND trade_date   <= %s
          AND close_price     > 0
          AND adj_close_price > 0
          AND deprecated_at IS NULL
        ORDER BY ticker, trade_date
        """
        df = pd.read_sql(
            q, self._conn,
            params=[EXCHANGE] + tickers + [PULL_START, PULL_END],
            parse_dates=["trade_date"],
        )
        if self._verbose:
            print(f"  Raw rows: {len(df):,}  |  tickers: {df['ticker'].nunique():,}")

        # Bad-tick filter
        df = df.sort_values(["ticker", "trade_date"])
        pct = df.groupby("ticker")["close"].pct_change().abs()
        bad = pct > BAD_TICK_THRESHOLD
        n_bad = bad.sum()
        if n_bad > 0:
            df.loc[bad, "close"] = np.nan
            df["close"] = df.groupby("ticker")["close"].ffill()
        if self._verbose:
            print(f"  Bad ticks cleaned (|1d return|>{BAD_TICK_THRESHOLD:.0%}): {n_bad:,} rows")

        # QuantDSL timestamp convention: date + 05:00:00 UTC-naive
        df["date"] = (
            pd.to_datetime(df["trade_date"])
            .dt.tz_localize("UTC").dt.normalize() + pd.Timedelta(hours=5)
        ).dt.tz_localize(None)

        self._prices_cache = df
        if self._verbose:
            print(f"  Final: {len(df):,} rows  |  {df['ticker'].nunique():,} tickers")
        return df

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a one-line universe summary (no data load)."""
        print(
            f"LSEUniverse  exchange={EXCHANGE}  "
            f"currencies={CURRENCIES}  "
            f"min_price={MIN_PRICE_GBX} GBX  "
            f"min_history={MIN_HISTORY_DAYS}d  "
            f"window={BACKTEST_START}→{BACKTEST_END}"
        )
