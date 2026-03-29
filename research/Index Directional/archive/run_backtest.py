"""
Index Directional — standalone backtest runner
===============================================
Run from the btest folder:
    uv run python "research/Index Directional/run_backtest.py"
    uv run python "research/Index Directional/run_backtest.py" --chart

Signals tested (all combinations on both CAC-TR and LVC 2×):
  • TKAN v3          – long when predicted 5d cumulative log-return ≥ threshold
  • IVol fixed       – long when CAC implied vol < 45
  • IVol z-score     – long when rolling 126d z-score < 1.0
  • TKAN + IVol      – both conditions must hold

Output: rich metrics table printed to console, saved to outputs/idx_directional_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = pathlib.Path(__file__).resolve().parent
_BTEST_ROOT = _HERE.parents[1]   # btest/
_TKAN_V3    = _HERE / "tkan" / "v3"
_WEIGHTS    = _TKAN_V3 / "weights"
_OUTPUT     = _BTEST_ROOT / "outputs" / "idx_directional"
_OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BTEST_ROOT.parent))   # workspace root → finds signum, sfera-db

# ── Model config (MUST match the training notebook) ───────────────────────────
WINDOW_SIZE      = 30
PREDICTION_DAYS  = 5
TARGET_TYPE      = "path5d_dm"
FEATURE_COLS     = [
    "log_return_1d", "high_low_range", "close_to_high",
    "close_vs_sma15",
    "ivol_zscore", "ivol_ema_ratio", "ivol_pctl", "ivol_roc5",
    "rvol_park20_zscore", "vol_spread",
    "return_5d", "return_20d",
]
BACKTEST_START   = "2015-01-01"
IVOL_EXIT_WINDOW = 126

# ── IVol thresholds to sweep ──────────────────────────────────────────────────
IVOL_FIXED_THR   = 45.0   # default for the "fixed" signal
TKAN_THR         = 0.0    # default TKAN signal cutoff


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_features, lvc) — df_features has close/OHLC/ivol + all feature cols,
    lvc has a single 'close' col indexed by date."""
    import sfera_db

    cactr = sfera_db.query(
        "SELECT trade_date AS date, close_price AS close "
        "FROM bbgidx.index_total_return WHERE ticker = 'CACT' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")

    cac_ohlc = sfera_db.query(
        "SELECT trade_date AS date, open_price AS open, high_price AS high, "
        "low_price AS low, close_price AS cac_close "
        "FROM bbgidx.index_prices WHERE ticker = 'CAC' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")

    ivol_raw = sfera_db.query(
        'SELECT trade_date AS date, "3m_50d_ivol" AS ivol '
        "FROM bbgidx.index_implied_vol WHERE ticker = 'CAC' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")[["ivol"]]

    common = cactr.index.intersection(cac_ohlc.index).intersection(ivol_raw.index)
    df = cac_ohlc.loc[common].copy()
    df["close"] = cactr.loc[common, "close"]
    df["ivol"]  = ivol_raw.loc[common, "ivol"]

    # ── feature engineering (same as notebook) ────────────────────────────────
    df["log_return_1d"]  = np.log(df["close"] / df["close"].shift(1))
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan)
    df["close_to_high"]  = (df["high"] - df["cac_close"]) / (df["high"] - df["low"] + 1e-9)
    df["sma15"]          = df["close"].rolling(15).mean()
    df["close_vs_sma15"] = df["close"] / df["sma15"] - 1
    df["return_5d"]      = np.log(df["close"] / df["close"].shift(5))
    df["return_20d"]     = np.log(df["close"] / df["close"].shift(20))

    hl_sq  = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    park   = np.sqrt((1 / (4 * np.log(2))) * hl_sq.rolling(20).mean() * 252)
    close_rvol = df["log_return_1d"].rolling(20).std() * np.sqrt(252)
    df["rvol_park20"] = park.where(park.notna() & (park > 0), close_rvol)

    ivol = df["ivol"]
    df["ivol_ewma20"]        = ivol.ewm(span=20).mean()
    df["ivol_zscore"]        = (ivol - ivol.rolling(IVOL_EXIT_WINDOW).mean()) / \
                                (ivol.rolling(IVOL_EXIT_WINDOW).std() + 1e-9)
    df["ivol_ema_ratio"]     = ivol / (df["ivol_ewma20"] + 1e-9)
    df["ivol_pctl"]          = ivol.rolling(IVOL_EXIT_WINDOW).apply(
                                   lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df["ivol_roc5"]          = ivol.pct_change(5)
    df["rvol_park20_zscore"] = (df["rvol_park20"] - df["rvol_park20"].rolling(IVOL_EXIT_WINDOW).mean()) / \
                                (df["rvol_park20"].rolling(IVOL_EXIT_WINDOW).std() + 1e-9)
    df["vol_spread"]         = ivol - df["rvol_park20"]

    # ── LVC (2× CAC) from sfera_db ────────────────────────────────────────────
    try:
        lvc_raw = sfera_db.query(
            "SELECT trade_date AS date, close_price AS close "
            "FROM instruments.etf_prices WHERE ticker = 'LVC' ORDER BY trade_date"
        ).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")
    except Exception:
        # Fall back to the cached CSV in btest/data/
        _csv = _BTEST_ROOT / "data" / "lvc_ohlcv.csv"
        if _csv.exists():
            lvc_raw = pd.read_csv(_csv, parse_dates=["date"]).set_index("date")[["close"]]
        else:
            lvc_raw = None

    return df, lvc_raw


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TKAN predictions (load from cache, recompute if stale)
# ─────────────────────────────────────────────────────────────────────────────

def _config_fingerprint() -> str:
    import hashlib
    sig = f"{WINDOW_SIZE}|{PREDICTION_DAYS}|{TARGET_TYPE}|{sorted(FEATURE_COLS)}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def load_tkan_predictions(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Returns pred_df with columns r1..r5, DatetimeIndex; or None if weights not found."""
    manifest_path = _WEIGHTS / "manifest.json"
    cache_path    = _WEIGHTS / "pred_cache.pkl"

    if not manifest_path.exists():
        print("⚠   No TKAN weights found — TKAN signals will be skipped.")
        return None

    current_fp = _config_fingerprint()

    # Try cache first
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _cached = pickle.load(f)
        if (isinstance(_cached, tuple) and len(_cached) == 3
                and isinstance(_cached[0], pd.DataFrame)
                and _cached[2] == current_fp):
            pred_df, retrain_dates, _ = _cached
            print(f"✅  TKAN predictions loaded from cache  "
                  f"({len(pred_df):,} rows  {pred_df.index[0].date()} → {pred_df.index[-1].date()})")
            return pred_df
        else:
            print("⚠   Pred cache stale — recomputing (this may take a minute)…")

    # Recompute from weights
    import tensorflow as tf
    from tkan import TKAN
    from sklearn.preprocessing import RobustScaler

    X_all = df[FEATURE_COLS].copy()
    y_all = pd.DataFrame({
        f"r{i}": np.log(df["close"].shift(-i) / df["close"].shift(-(i - 1)))
        for i in range(1, PREDICTION_DAYS + 1)
    }, index=df.index)
    valid = X_all.notna().all(axis=1) & y_all.notna().all(axis=1)
    X_all = X_all.loc[valid]
    dates = X_all.index

    with open(manifest_path) as f:
        manifest = json.load(f)

    def _build_model():
        mdl = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, len(FEATURE_COLS))),
            TKAN(100, return_sequences=True,  use_bias=True),
            tf.keras.layers.Dropout(0.2),
            TKAN(100, return_sequences=True,  use_bias=True),
            tf.keras.layers.Dropout(0.2),
            TKAN(100, return_sequences=False, use_bias=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(PREDICTION_DAYS),
        ])
        mdl.compile(optimizer="adam", loss="mse")
        return mdl

    def _load_cycle(key):
        mdl = _build_model()
        mdl.load_weights(str(_WEIGHTS / f"{key}.weights.h5"))
        with open(_WEIGHTS / f"{key}_scaler_X.pkl", "rb") as ff:
            scaler = pickle.load(ff)
        ym_path = _WEIGHTS / f"{key}_ymean.npy"
        y_mean  = np.load(str(ym_path)) if ym_path.exists() else np.zeros(PREDICTION_DAYS, "float32")
        return mdl, scaler, y_mean

    cycles = sorted(manifest["cycles"].items(), key=lambda x: x[1]["cycle_num"])
    cycle_boundaries = [(info["train_end_idx"], key) for key, info in cycles]

    bt_start_idx = dates.searchsorted(pd.Timestamp(BACKTEST_START))
    cycle_i = 0; cur_key = None; retrain_dates = []
    cycle_for_idx: dict[int, str] = {}
    for i in range(bt_start_idx, len(dates)):
        if cycle_i < len(cycle_boundaries) and i >= cycle_boundaries[cycle_i][0]:
            new_key = cycle_boundaries[cycle_i][1]
            if new_key != cur_key:
                retrain_dates.append(dates[i]); cur_key = new_key
            cycle_i += 1
        if cur_key is None or i < WINDOW_SIZE:
            continue
        cycle_for_idx[i] = cur_key

    from collections import defaultdict
    cycle_groups: dict[str, list[int]] = defaultdict(list)
    for idx, ck in cycle_for_idx.items():
        cycle_groups[ck].append(idx)

    X_sc_full = None  # lazy per-cycle
    preds_dict: dict = {}
    for ck, idxs in sorted(cycle_groups.items()):
        mdl, scl, ym = _load_cycle(ck)
        X_sc = scl.transform(X_all).astype("float32")
        Xw   = np.stack([X_sc[i - WINDOW_SIZE:i] for i in idxs])
        preds_batch = mdl(Xw, training=False).numpy() + ym
        for idx, p in zip(idxs, preds_batch):
            preds_dict[dates[idx]] = p
        print(f"  {ck}: {len(idxs):,} rows  {dates[idxs[0]].date()} → {dates[idxs[-1]].date()}")

    cols    = [f"r{i+1}" for i in range(PREDICTION_DAYS)]
    pred_df = pd.DataFrame.from_dict(preds_dict, orient="index", columns=cols)
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    pred_df = pred_df.sort_index()

    with open(cache_path, "wb") as f:
        pickle.dump((pred_df, retrain_dates, current_fp), f)
    print(f"✅  {len(pred_df)} predictions cached")
    return pred_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    position: pd.Series,
    label: str,
) -> dict:
    """Compute a rich metrics dict from daily return series."""
    sr   = strat_returns.fillna(0)
    br   = bench_returns.reindex(sr.index).fillna(0)
    pos  = position.reindex(sr.index).fillna(0)

    n_days  = len(sr)
    n_years = n_days / 252

    ann_ret = sr.mean() * 252
    ann_vol = sr.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside vol only)
    down    = sr[sr < 0]
    sortino = ann_ret / (down.std() * np.sqrt(252)) if len(down) > 1 else 0.0

    # Max drawdown
    equity  = (1 + sr).cumprod()
    mdd     = float((equity / equity.cummax() - 1).min()) * 100

    # CAGR
    total_mult = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr       = (total_mult ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    # Calmar
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    # Beta / Alpha (OLS)
    valid_both = br.notna() & sr.notna()
    if valid_both.sum() > 30:
        slope, intercept, *_ = scipy_stats.linregress(br[valid_both], sr[valid_both])
        beta  = float(slope)
        alpha = float(intercept) * 252 * 100   # annualized %, per year
    else:
        beta = alpha = float("nan")

    # Win rate (active days only)
    active = sr[pos > 0]
    win_rate = float((active > 0).mean() * 100) if len(active) > 0 else float("nan")

    # % time in market
    pct_in_market = float(pos.mean() * 100)

    total_pct = (total_mult - 1) * 100

    return {
        "Label":        label,
        "CAGR %":       round(cagr,        2),
        "Sharpe":       round(sharpe,       3),
        "Sortino":      round(sortino,      3),
        "Ann. Vol %":   round(ann_vol*100,  2),
        "Max DD %":     round(mdd,          2),
        "Calmar":       round(calmar,       3) if not np.isnan(calmar) else "—",
        "Beta":         round(beta,         3) if not np.isnan(beta)   else "—",
        "Alpha %/yr":   round(alpha,        2) if not np.isnan(alpha)  else "—",
        "Win Rate %":   round(win_rate,     1) if not np.isnan(win_rate) else "—",
        "In-Mkt %":     round(pct_in_market,1),
        "Total %":      round(total_pct,    1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Signal definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame, pred_df: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    """Returns dict of signal_name → binary position Series (0 or 1), pre-shift."""
    ivol = df["ivol"]

    # IVol-based signals
    sig_ivol_fixed  = (ivol < IVOL_FIXED_THR).astype(float)
    sig_ivol_z      = ((ivol - ivol.rolling(IVOL_EXIT_WINDOW).mean()) /
                       (ivol.rolling(IVOL_EXIT_WINDOW).std() + 1e-9) < 1.0).astype(float)
    sig_ema_cross   = (ivol.ewm(span=5).mean() < ivol.ewm(span=40).mean()).astype(float)

    signals: dict[str, pd.Series] = {
        "Always-In":        pd.Series(1.0, index=df.index),
        "IVol < 45":        sig_ivol_fixed,
        "IVol Z < 1.0":     sig_ivol_z,
        "IVol EMA5<EMA40":  sig_ema_cross,
    }

    if pred_df is not None:
        pred_cum = pred_df.sum(axis=1).reindex(df.index).fillna(0)
        sig_tkan = (pred_cum >= TKAN_THR).astype(float)
        signals["TKAN v3"]              = sig_tkan
        signals["TKAN + IVol<45"]       = (sig_tkan * sig_ivol_fixed).astype(float)
        signals["TKAN + IVol Z<1.0"]    = (sig_tkan * sig_ivol_z).astype(float)
        signals["TKAN + IVol EMA-cross"]= (sig_tkan * sig_ema_cross).astype(float)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Print table
# ─────────────────────────────────────────────────────────────────────────────

def _col_width(col: str, rows: list[dict]) -> int:
    return max(len(col), max(len(str(r.get(col, ""))) for r in rows)) + 2


def print_table(rows: list[dict], title: str = "") -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: _col_width(c, rows) for c in cols}
    sep  = "+" + "+".join("-" * widths[c] for c in cols) + "+"
    head = "|" + "|".join(f" {c:<{widths[c]-1}}" for c in cols) + "|"
    if title:
        print(f"\n{'─'*len(sep)}")
        print(f"  {title}")
        print(f"{'─'*len(sep)}")
    print(sep)
    print(head)
    print(sep)
    for r in rows:
        print("|" + "|".join(f" {str(r.get(c,'')):<{widths[c]-1}}" for c in cols) + "|")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print("\n── Loading data from sfera_db ──────────────────────────────────────────────")
    df, lvc_raw = load_data()
    bt_start    = pd.Timestamp(BACKTEST_START)
    df_bt       = df.loc[df.index >= bt_start].copy()
    print(f"CAC feats : {df_bt.index[0].date()} → {df_bt.index[-1].date()}  ({len(df_bt):,} rows)")

    print("\n── Loading TKAN v3 predictions ─────────────────────────────────────────────")
    pred_df = load_tkan_predictions(df)

    print("\n── Building signals ────────────────────────────────────────────────────────")
    signals = build_signals(df_bt, pred_df.reindex(df_bt.index) if pred_df is not None else None)

    # ── Instruments ──────────────────────────────────────────────────────────
    instruments: dict[str, pd.Series] = {
        "CAC-TR (1×)": np.log(df_bt["close"] / df_bt["close"].shift(1)),
    }
    if lvc_raw is not None:
        lvc_common = lvc_raw.index.intersection(df_bt.index)
        lvc_close  = lvc_raw.loc[lvc_common, "close"]
        lvc_ret    = np.log(lvc_close / lvc_close.shift(1))
        instruments["LVC (2×)"] = lvc_ret.reindex(df_bt.index)
    else:
        print("⚠   LVC data not found — only CAC-TR will be shown")

    # ── Run matrix ──────────────────────────────────────────────────────────
    all_rows: dict[str, list[dict]] = {instr: [] for instr in instruments}

    for instr_name, daily_ret in instruments.items():
        for sig_name, raw_sig in signals.items():
            sig_aligned = raw_sig.reindex(daily_ret.index).fillna(0)
            # 1-day execution lag: signal at close[T] → trade at close[T+1]
            pos = sig_aligned.shift(1).fillna(0)
            strat_ret = daily_ret * pos
            row = compute_metrics(strat_ret, daily_ret, pos, sig_name)
            all_rows[instr_name].append(row)

    # ── Print ────────────────────────────────────────────────────────────────
    for instr_name, rows in all_rows.items():
        print_table(rows, title=f"Instrument: {instr_name}  |  Backtest: {BACKTEST_START} → today")

    # ── Save ─────────────────────────────────────────────────────────────────
    combined = []
    for instr_name, rows in all_rows.items():
        for r in rows:
            combined.append({"Instrument": instr_name, **r})
    out_path = _OUTPUT / "metrics.csv"
    pd.DataFrame(combined).to_csv(out_path, index=False)
    print(f"\n💾  Results saved → {out_path}")

    # ── Optional chart ───────────────────────────────────────────────────────
    if args.chart:
        _show_chart(df_bt, pred_df, instruments, signals)


def _show_chart(
    df_bt: pd.DataFrame,
    pred_df: Optional[pd.DataFrame],
    instruments: dict[str, pd.Series],
    signals: dict[str, pd.Series],
) -> None:
    try:
        from signum import Chart, Dashboard
    except ImportError:
        print("⚠   signum not installed — skipping chart")
        return

    # Show TKAN + IVol<45 on CAC-TR
    sig_name = "TKAN + IVol<45" if "TKAN + IVol<45" in signals else "IVol < 45"
    raw_sig  = signals[sig_name]
    pos      = raw_sig.reindex(df_bt.index).fillna(0).shift(1).fillna(0)
    daily_ret = list(instruments.values())[0]
    strat_ret = daily_ret * pos
    bh_eq     = (1 + daily_ret.fillna(0)).cumprod()
    st_eq     = (1 + strat_ret.fillna(0)).cumprod()

    def _ts(series):
        return pd.DataFrame({"time": series.index.strftime("%Y-%m-%d"), "value": series.values})

    price_series = np.exp(daily_ret.fillna(0).cumsum())  # re-based price
    shade        = pd.DataFrame({"time": df_bt.index.strftime("%Y-%m-%d"), "position": pos.values})

    p1 = Chart(height=360, watermark="CAC40 TR").line(_ts(price_series), name="CAC40 TR", color="#4169E1")
    p1.shade(shade, color="#00CC96", opacity=0.15)

    if pred_df is not None:
        pred_cum = pred_df.sum(axis=1).reindex(df_bt.index).fillna(0)
        p2 = Chart(height=140, watermark="TKAN 5d predicted return").baseline(_ts(pred_cum), base_value=0)
    else:
        ivol_ts = _ts(df_bt["ivol"])
        p2 = Chart(height=140, watermark="CAC IVol").line(ivol_ts, name="IVol", color="#F59E0B")

    p3 = (
        Chart(height=180, watermark="Equity")
        .line(_ts(bh_eq), name="B&H", color="#4169E1")
        .line(_ts(st_eq), name=sig_name, color="#00CC96")
    )

    Dashboard(
        panes=[p1, p2, p3],
        titles=["CAC40 TR — signal shading", "TKAN predicted 5d return", "Strategy vs B&H"],
        theme="dark",
    ).show()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Directional backtest runner")
    parser.add_argument(
        "--chart", action="store_true",
        help="Open Signum chart after printing metrics",
    )
    parser.add_argument(
        "--tkan-thr", type=float, default=TKAN_THR,
        help=f"TKAN entry threshold (default {TKAN_THR})",
    )
    parser.add_argument(
        "--ivol-thr", type=float, default=IVOL_FIXED_THR,
        help=f"IVol fixed exit threshold (default {IVOL_FIXED_THR})",
    )
    args = parser.parse_args()
    TKAN_THR       = args.tkan_thr
    IVOL_FIXED_THR = args.ivol_thr
    main(args)
