"""TKAN v3 — Daily Signal Runner
================================
Pulls latest CACT total return + CAC OHLC + CAC IVol from Sfera DB.
Builds 12 stationary features (matching TKAN_v3_research.ipynb exactly).
Loads the most recent walk-forward cycle from ./weights/.
Predicts next 5 signed daily log returns via MC Dropout (50 passes).
Applies IVol exit gate. Prints buy signal. Shows 2-panel chart.

Run daily:
    python TKAN_Index_v3.py
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pandas.tseries.offsets import BDay
import tensorflow as tf
from tkan import TKAN
from sklearn.preprocessing import RobustScaler

# ── sfera_db path discovery ───────────────────────────────────────────────────
# 7 levels up from this file = "Python codes" root; sfera-db lives there
_HERE = os.path.dirname(os.path.abspath(__file__))
_SFERA_DB = os.path.normpath(os.path.join(_HERE, *(['..'] * 7), 'sfera-db'))
if os.path.isdir(_SFERA_DB) and _SFERA_DB not in sys.path:
    sys.path.insert(0, _SFERA_DB)
import sfera_db

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — must match TKAN_v3_research.ipynb exactly
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_DIR     = os.path.join(_HERE, 'weights')
OUTPUT_DIR      = os.path.join(_HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE      = 30     # days of context per prediction
PREDICTION_DAYS  = 5      # forward signed daily log returns
IVOL_EXIT_WINDOW = 126    # rolling window for IVol percentile
IVOL_EXIT_PCTL   = 80     # percentile threshold to block buy
ENTRY_THRESHOLD  = 0.0    # min predicted 5d cumulative log return to generate signal
MC_SAMPLES       = 50     # MC Dropout passes for uncertainty bands

FEATURE_COLS = [
    'log_return_1d', 'high_low_range', 'close_to_high',
    'close_vs_sma15',
    'ivol_zscore', 'ivol_ema_ratio', 'ivol_pctl', 'ivol_roc5',
    'rvol_park20_zscore', 'vol_spread',
    'return_5d', 'return_20d',
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """Pull all market data from Sfera DB."""
    print("Loading data from Sfera DB...")

    cactr = sfera_db.query(
        "SELECT trade_date AS date, close_price AS close "
        "FROM bbgidx.index_total_return WHERE ticker = 'CACT' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d['date'])).set_index('date')

    cac_ohlc = sfera_db.query(
        "SELECT trade_date AS date, open_price AS open, high_price AS high, "
        "low_price AS low, close_price AS cac_close "
        "FROM bbgidx.index_prices WHERE ticker = 'CAC' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d['date'])).set_index('date')

    ivol_raw = sfera_db.query(
        'SELECT trade_date AS date, "3m_50d_ivol" AS ivol '
        "FROM bbgidx.index_implied_vol WHERE ticker = 'CAC' ORDER BY trade_date"
    ).assign(date=lambda d: pd.to_datetime(d['date'])).set_index('date')[['ivol']]

    print(f"  CACT TR:  {cactr.index[0].date()} -> {cactr.index[-1].date()} ({len(cactr)} rows)")
    print(f"  CAC OHLC: {cac_ohlc.index[0].date()} -> {cac_ohlc.index[-1].date()} ({len(cac_ohlc)} rows)")
    print(f"  IVol:     {ivol_raw.index[0].date()} -> {ivol_raw.index[-1].date()} ({len(ivol_raw)} rows)")

    return cactr, cac_ohlc, ivol_raw


def build_features(cactr, cac_ohlc, ivol_raw):
    """Build 12 stationary features — identical to TKAN_v3_research.ipynb."""
    # Forward-fill ivol so slight staleness never truncates the dataset
    ivol_filled = ivol_raw.reindex(ivol_raw.index.union(cactr.index)).ffill()

    common = cactr.index.intersection(cac_ohlc.index).intersection(ivol_filled.index)
    df = cac_ohlc.loc[common].copy()
    df['close'] = cactr.loc[common, 'close']
    df['ivol']  = ivol_filled.loc[common, 'ivol']

    # Price features
    df['log_return_1d']  = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close'].shift(1).replace(0, np.nan)
    df['close_to_high']  = (df['high'] - df['cac_close']) / (df['high'] - df['low'] + 1e-9)
    df['sma15']          = df['close'].rolling(15).mean()
    df['close_vs_sma15'] = df['close'] / df['sma15'] - 1
    df['return_5d']      = np.log(df['close'] / df['close'].shift(5))
    df['return_20d']     = np.log(df['close'] / df['close'].shift(20))

    # Parkinson RVol (falls back to close-based RVol on bad data)
    hl_sq      = np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2
    park       = np.sqrt((1 / (4 * np.log(2))) * hl_sq.rolling(20).mean() * 252)
    close_rvol = df['log_return_1d'].rolling(20).std() * np.sqrt(252)
    df['rvol_park20'] = park.where(park.notna() & (park > 0), close_rvol)

    # IVol features
    ivol = df['ivol']
    ivol_ewma20          = ivol.ewm(span=20).mean()
    df['ivol_ewma20']    = ivol_ewma20
    df['ivol_zscore']    = (ivol - ivol.rolling(IVOL_EXIT_WINDOW).mean()) / \
                           (ivol.rolling(IVOL_EXIT_WINDOW).std() + 1e-9)
    df['ivol_ema_ratio'] = ivol / (ivol_ewma20 + 1e-9)
    df['ivol_pctl']      = ivol.rolling(IVOL_EXIT_WINDOW).apply(
                               lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df['ivol_roc5']      = ivol.pct_change(5)

    # RVol z-score and vol spread
    df['rvol_park20_zscore'] = (df['rvol_park20'] - df['rvol_park20'].rolling(IVOL_EXIT_WINDOW).mean()) / \
                               (df['rvol_park20'].rolling(IVOL_EXIT_WINDOW).std() + 1e-9)
    df['vol_spread']         = ivol - df['rvol_park20']

    X     = df[FEATURE_COLS].copy()
    valid = X.notna().all(axis=1)
    X     = X.loc[valid]
    df    = df.loc[valid]

    print(f"Features: {X.index[0].date()} -> {X.index[-1].date()} ({len(X)} rows, {len(FEATURE_COLS)} features)")
    return X, df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model(n_features=len(FEATURE_COLS)):
    """Architecture must match TKAN_v3_research.ipynb exactly."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, n_features)),
        TKAN(100, return_sequences=True,  use_bias=True),
        tf.keras.layers.Dropout(0.2),
        TKAN(100, return_sequences=True,  use_bias=True),
        tf.keras.layers.Dropout(0.2),
        TKAN(100, return_sequences=False, use_bias=True),  # collapses time axis
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(PREDICTION_DAYS),            # 5 demeaned daily returns
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def load_latest_cycle():
    """Load the most recently trained walk-forward cycle."""
    manifest_path = os.path.join(WEIGHTS_DIR, 'manifest.json')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run TKAN_v3_research.ipynb training section first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    cycles     = sorted(manifest['cycles'].items(), key=lambda x: x[1]['cycle_num'])
    cycle_key, cycle_info = cycles[-1]

    w_path  = os.path.join(WEIGHTS_DIR, f'{cycle_key}.weights.h5')
    s_path  = os.path.join(WEIGHTS_DIR, f'{cycle_key}_scaler_X.pkl')
    ym_path = os.path.join(WEIGHTS_DIR, f'{cycle_key}_ymean.npy')

    model = build_model()
    model.load_weights(w_path)

    with open(s_path, 'rb') as f:
        scaler = pickle.load(f)

    y_mean = np.load(ym_path) if os.path.isfile(ym_path) else np.zeros(PREDICTION_DAYS, dtype='float32')

    print(f"Loaded: {cycle_key}  (trained through {cycle_info['train_end_date']}, "
          f"{cycle_info['n_samples']} samples)")
    print(f"  y_mean (training-window mean logret): {y_mean}")
    return model, scaler, y_mean, cycle_key, cycle_info


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_next(model, scaler, y_mean, X_df):
    """
    MC Dropout inference on the last WINDOW_SIZE rows of X.
    Returns (mean, lo_95, hi_95) — each shape (PREDICTION_DAYS,) in log-return space.
    """
    X_win   = X_df.iloc[-WINDOW_SIZE:]
    X_sc    = scaler.transform(X_win).reshape(1, WINDOW_SIZE, -1)
    X_batch = np.repeat(X_sc, MC_SAMPLES, axis=0)
    raw     = model(X_batch, training=True).numpy()  # Dropout active → distributional
    preds   = raw + y_mean                            # add back training-window mean
    m = preds.mean(axis=0)
    s = preds.std(axis=0)
    return m, m - 1.96 * s, m + 1.96 * s


def log_returns_to_price_path(last_price, log_returns):
    """Convert array of daily log returns to price path anchored at last_price."""
    return last_price * np.exp(np.cumsum(log_returns))


# ══════════════════════════════════════════════════════════════════════════════
# IVOL EXIT GATE
# ══════════════════════════════════════════════════════════════════════════════
def check_ivol_gate(ivol_series):
    iv          = ivol_series.sort_index().dropna()
    latest      = float(iv.iloc[-1])
    pctl_series = iv.rolling(IVOL_EXIT_WINDOW, min_periods=20).quantile(IVOL_EXIT_PCTL / 100)
    threshold   = float(pctl_series.iloc[-1])
    pctl_rank   = float(iv.rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True).iloc[-1])
    is_blocked  = latest > threshold
    details = (f"IVol={latest:.1f}, {IVOL_EXIT_PCTL}th-pctl={threshold:.1f} "
               f"(current rank={pctl_rank:.0%}) → {'BLOCKED' if is_blocked else 'OK'}")
    return is_blocked, details, threshold


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(df, pred_mean, pred_lo, pred_hi, ivol_threshold,
                 last_date, future_dates, has_signal, cycle_key, cycle_info):
    plt.style.use('dark_background')

    window     = 60
    recent     = df['close'].iloc[-window:]
    last_price = float(recent.iloc[-1])
    price_mean = log_returns_to_price_path(last_price, pred_mean)
    price_lo   = log_returns_to_price_path(last_price, pred_lo)
    price_hi   = log_returns_to_price_path(last_price, pred_hi)
    cum_ret    = pred_mean.sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f'TKAN v3 — CAC40 TR  |  {cycle_key} (trained to {cycle_info["train_end_date"]})  '
        f'|  Predicted 5d return: {cum_ret:+.3f} ({np.exp(cum_ret)-1:+.2%})',
        color='#e0e0e0', fontsize=13, fontweight='bold')

    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaa')
        ax.grid(True, alpha=0.15, color='#555')
        for spine in ax.spines.values():
            spine.set_color('#333')

    # ── Top: CACT TR price + predictions ──────────────────────────────────
    ax1.plot(recent.index, recent.values, color='#4A90D9', linewidth=2.5, label='CACT TR actual')
    ax1.plot(future_dates, price_mean, color='#E74C3C', linewidth=2, linestyle='--',
             marker='D', markersize=5, label='Predicted (mean)')
    ax1.fill_between(future_dates, price_lo, price_hi, alpha=0.2, color='#E74C3C', label='95% CI')
    if has_signal:
        ax1.axvspan(future_dates[0], future_dates[-1], alpha=0.08, color='#2ECC71')
        ax1.set_title('★ BUY SIGNAL ACTIVE — LVC.PA', color='#2ECC71', fontsize=11, pad=4)
    else:
        ax1.set_title('No buy signal', color='#aaa', fontsize=10, pad=4)
    ax1.set_ylabel('CACT TR Index', color='#ccc')
    ax1.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ddd')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # ── Bottom: IVol + gate threshold ─────────────────────────────────────
    iv_recent = df['ivol'].iloc[-window * 2:]
    ax2.plot(iv_recent.index, iv_recent.values, color='#F39C12', linewidth=2,
             label='CAC IVol (3M 50D)')
    ax2.axhline(ivol_threshold, color='#E74C3C', linewidth=1.5, linestyle='--',
                label=f'{IVOL_EXIT_PCTL}th pctl = {ivol_threshold:.1f}')
    ax2.fill_between(iv_recent.index, iv_recent.values, ivol_threshold,
                     where=iv_recent.values > ivol_threshold,
                     alpha=0.3, color='#E74C3C', label='Above threshold (blocked)')
    ax2.set_ylabel('Implied Vol', color='#ccc')
    ax2.set_xlabel('Date', color='#ccc')
    ax2.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ddd')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'tkan_v3_daily_{datetime.now().strftime("%Y%m%d")}.png')
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("TKAN v3 — Daily Signal Runner")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    cactr, cac_ohlc, ivol_raw = load_data()
    X, df = build_features(cactr, cac_ohlc, ivol_raw)

    # ── Model ─────────────────────────────────────────────────────────────
    model, scaler, y_mean, cycle_key, cycle_info = load_latest_cycle()

    # ── Predict ───────────────────────────────────────────────────────────
    pred_mean, pred_lo, pred_hi = predict_next(model, scaler, y_mean, X)

    last_date    = X.index[-1]
    future_dates = [last_date + BDay(i + 1) for i in range(PREDICTION_DAYS)]
    last_price   = float(df['close'].iloc[-1])

    print(f"\n{'─'*65}")
    print(f"Predictions from {last_date.date()}  (CACT TR close = {last_price:.2f})")
    print(f"{'─'*65}")
    cum = 0.0
    for i, (fd, r, lo, hi) in enumerate(zip(future_dates, pred_mean, pred_lo, pred_hi)):
        cum += r
        price = last_price * np.exp(cum)
        print(f"  Day {i+1}  {str(fd.date())}:  r={r:+.4f}  [lo={lo:+.4f}, hi={hi:+.4f}]"
              f"  →  price ~{price:.1f}")

    cum_5d = float(pred_mean.sum())
    print(f"\n  Cumulative 5d log-return: {cum_5d:+.4f}  ({np.exp(cum_5d)-1:+.2%})")

    # ── IVol gate ─────────────────────────────────────────────────────────
    ivol_blocked, ivol_details, ivol_threshold = check_ivol_gate(df['ivol'])
    print(f"\nIVol gate: {ivol_details}")

    # ── Signal ────────────────────────────────────────────────────────────
    has_signal = (cum_5d >= ENTRY_THRESHOLD) and not ivol_blocked
    print(f"\n{'─'*65}")
    if ivol_blocked:
        print("  ⛔  IVol EXIT GATE ACTIVE — signal suppressed")
    elif has_signal:
        print(f"  ✅  BUY SIGNAL — predicted {np.exp(cum_5d)-1:+.2%} over 5 days")
        print(f"      Enter: LVC.PA at open on {future_dates[0].date()}")
        print(f"      Exit:  after {PREDICTION_DAYS} business days ({future_dates[-1].date()})")
    else:
        print(f"  —   No signal (5d pred={cum_5d:+.4f}, threshold={ENTRY_THRESHOLD})")
    print(f"{'─'*65}")

    # ── Chart ─────────────────────────────────────────────────────────────
    plot_results(df, pred_mean, pred_lo, pred_hi, ivol_threshold,
                 last_date, future_dates, has_signal, cycle_key, cycle_info)


if __name__ == "__main__":
    main()
