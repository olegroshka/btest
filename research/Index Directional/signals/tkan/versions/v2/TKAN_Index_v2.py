import os
import csv
import tensorflow as tf
from tkan import TKAN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta
import psycopg
import warnings
warnings.filterwarnings('ignore')

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONFIGURATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR   = os.path.join(ROOT_DIR, "Data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")

LVC_CSV      = os.path.join(DATA_DIR, "lvc_ohlcv.csv")
CAC_IVOL     = os.path.join(DATA_DIR, "cac_ivol.csv")
CAC_OHLCV    = os.path.join(DATA_DIR, "cac_ohlcv.csv")
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "tkan_v2_weights.weights.h5")

# Sfera Postgres (bbgidx schema)
SFERA_ENV = os.path.normpath(os.path.join(
    ROOT_DIR, "..", "..", "..", "RU Market Data", "Python", "Sfera", ".env"
))

WINDOW_SIZE     = 10
PREDICTION_DAYS = 10
THRESHOLD       = 1.0   # â‚¬ increase for buy signal
EPOCHS          = 50
BATCH_SIZE      = 32

# IVol exit gate params (from percentile analysis)
IVOL_EXIT_WINDOW  = 126   # rolling window for percentile
IVOL_EXIT_PCTL    = 80    # percentile threshold
IVOL_EXIT_CONFIRM = 2     # days above to confirm exit



# ── SFERA DB HELPERS ─────────────────────────────────────────────────

def _sfera_connstr():
    """Read Sfera .env and return a psycopg connection string."""
    if not os.path.isfile(SFERA_ENV):
        return None
    cfg = {}
    with open(SFERA_ENV, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
    return (f"host={cfg.get('DB_HOST','localhost')} "
            f"port={cfg.get('DB_PORT','5432')} "
            f"dbname={cfg.get('DB_NAME','sfera')} "
            f"user={cfg.get('DB_USER','postgres')} "
            f"password={cfg.get('DB_PASSWORD','')}")


def load_cac_ohlcv_from_db():
    """Pull CAC 40 OHLCV from bbgidx.index_prices."""
    connstr = _sfera_connstr()
    if connstr is None:
        return None
    try:
        with psycopg.connect(connstr) as conn:
            df = pd.read_sql(
                "SELECT trade_date AS date, open_price AS open, high_price AS high, "
                "low_price AS low, close_price AS close, volume "
                "FROM bbgidx.index_prices WHERE ticker = 'CAC' ORDER BY trade_date",
                conn, parse_dates=['date'], index_col='date',
            )
        print(f"  CAC OHLCV from Sfera DB: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  Sfera DB (CAC OHLCV) failed: {e}")
        return None


def load_cac_ivol_from_db():
    """Pull CAC 3M 50D implied vol from bbgidx.index_implied_vol."""
    connstr = _sfera_connstr()
    if connstr is None:
        return None
    try:
        with psycopg.connect(connstr) as conn:
            df = pd.read_sql(
                'SELECT trade_date AS date, "3m_50d_ivol" AS ivol '
                "FROM bbgidx.index_implied_vol WHERE ticker = 'CAC' ORDER BY trade_date",
                conn, parse_dates=['date'], index_col='date',
            )
        print(f"  CAC IVol from Sfera DB: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  Sfera DB (CAC IVol) failed: {e}")
        return None

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DATA LOADING & FEATURE ENGINEERING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def update_lvc_data(lvc_path, ticker="LVC.PA"):
    """Download latest LVC data from Yahoo Finance and append to CSV."""
    print("Updating LVC data...")
    lvc = pd.read_csv(lvc_path, parse_dates=['date'], index_col='date').sort_index()
    last_date = lvc.index[-1]

    start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    yf_data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if yf_data.empty:
        print(f"  LVC up to date ({last_date.date()})")
        return lvc

    if yf_data.columns.nlevels > 1:
        yf_data.columns = yf_data.columns.droplevel(1)
    yf_data.index.name = 'date'
    yf_data.columns = [c.lower() for c in yf_data.columns]
    yf_data = yf_data[['open', 'high', 'low', 'close', 'volume']]

    combined = pd.concat([lvc, yf_data[~yf_data.index.isin(lvc.index)]]).sort_index()
    combined.to_csv(lvc_path)
    print(f"  LVC updated: {lvc.index[-1].date()} -> {combined.index[-1].date()} (+{len(combined)-len(lvc)} rows)")
    return combined


def update_cac_ohlcv(cac_path, ticker="^FCHI"):
    """Download latest CAC 40 index data from Yahoo Finance."""
    print("Updating CAC OHLCV...")
    cac = pd.read_csv(cac_path, parse_dates=['date'], index_col='date').sort_index()
    last_date = cac.index[-1]

    start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    yf_data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if yf_data.empty:
        print(f"  CAC OHLCV up to date ({last_date.date()})")
        return cac

    if yf_data.columns.nlevels > 1:
        yf_data.columns = yf_data.columns.droplevel(1)
    yf_data.index.name = 'date'
    yf_data.columns = [c.lower() for c in yf_data.columns]
    yf_data = yf_data[['open', 'high', 'low', 'close', 'volume']]

    combined = pd.concat([cac, yf_data[~yf_data.index.isin(cac.index)]]).sort_index()
    combined.to_csv(cac_path)
    print(f"  CAC updated: {cac.index[-1].date()} -> {combined.index[-1].date()} (+{len(combined)-len(cac)} rows)")
    return combined


def load_and_build_features(auto_update=True):
    """
    Load LVC prices, CAC IVol, CAC OHLCV. Align on common dates.
    Build feature set with vol-awareness.

    Features (all lagged by 1 day):
      1. Prior Close
      2. Prior High
      3. Prior Low
      4. Prior Volume
      5. Prior SMA(15)
      6. Prior CAC IVol (3M 50D implied vol)
      7. Prior IVol EMA(20) -- short-term vol trend
      8. Prior IVol percentile rank (126d rolling)
      9. Prior IVol rate of change (5d)
     10. Prior Parkinson RVol (20d, on CAC underlying)
     11. Prior Vol Risk Premium (IVol - RVol)

    Returns: X, y, market_data DataFrame, ivol_series (for exit gate)
    """
    # Load data
    if auto_update:
        lvc = update_lvc_data(LVC_CSV)
        cac_ohlcv = update_cac_ohlcv(CAC_OHLCV)
    else:
        lvc = pd.read_csv(LVC_CSV, parse_dates=['date'], index_col='date').sort_index()
        cac_ohlcv = pd.read_csv(CAC_OHLCV, parse_dates=['date'], index_col='date').sort_index()

    # CAC OHLCV & IVol: try Sfera DB first, fall back to CSV
    cac_ohlcv_db = load_cac_ohlcv_from_db()
    if cac_ohlcv_db is not None and not cac_ohlcv_db.empty:
        cac_ohlcv = cac_ohlcv_db
        print("  Using CAC OHLCV from Sfera DB")
    else:
        print("  Using CAC OHLCV from CSV (DB unavailable)")

    ivol_db = load_cac_ivol_from_db()
    if ivol_db is not None and not ivol_db.empty:
        ivol_raw = ivol_db
        print("  Using CAC IVol from Sfera DB")
    else:
        ivol_raw = pd.read_csv(CAC_IVOL, parse_dates=['date'], index_col='date').sort_index()
        print("  Using CAC IVol from CSV (DB unavailable)")

    # Align on common dates
    common = lvc.index.intersection(ivol_raw.index).intersection(cac_ohlcv.index)
    df = pd.DataFrame({
        'open':   lvc.loc[common, 'open'],
        'high':   lvc.loc[common, 'high'],
        'low':    lvc.loc[common, 'low'],
        'close':  lvc.loc[common, 'close'],
        'volume': lvc.loc[common, 'volume'],
        'ivol':   ivol_raw.loc[common, 'ivol'],
        'cac_high':  cac_ohlcv.loc[common, 'high'],
        'cac_low':   cac_ohlcv.loc[common, 'low'],
        'cac_close': cac_ohlcv.loc[common, 'close'],
    }).sort_index()

    # Derived features
    df['sma15'] = df['close'].rolling(15, min_periods=1).mean()
    df['ivol_ema20'] = df['ivol'].ewm(span=20).mean()
    df['ivol_pctl'] = df['ivol'].rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True)
    df['ivol_roc5'] = df['ivol'].pct_change(5)

    # Parkinson RVol (20d) on CAC underlying
    log_hl = np.log(df['cac_high'] / df['cac_low'])
    park_factor = 1.0 / (4.0 * np.log(2))
    df['rvol_park20'] = np.sqrt(park_factor * (log_hl ** 2).rolling(20).mean() * 252) * 100

    # Vol risk premium (IVol - RVol)
    df['vol_spread'] = df['ivol'] - df['rvol_park20']

    # Create lagged features (shift by 1 day = no lookahead)
    feature_cols = ['close', 'high', 'low', 'volume', 'sma15',
                    'ivol', 'ivol_ema20', 'ivol_pctl', 'ivol_roc5',
                    'rvol_park20', 'vol_spread']
    for col in feature_cols:
        df[f'prior_{col}'] = df[col].shift(1)

    df.dropna(inplace=True)

    print(f"Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print(f"Features: {len(feature_cols)} (all lagged by 1 day)")

    prior_cols = [f'prior_{c}' for c in feature_cols]
    X = df[prior_cols]
    y = df['close']

    return X, y, df, ivol_raw


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SEQUENCES, MODEL, TRAINING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i:i + window_size])
    return np.array(X_seq), np.array(y_seq)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        TKAN(100, return_sequences=True, use_bias=True),
        TKAN(100, return_sequences=True, use_bias=True),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)
    return history


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PREDICTION & SIGNAL
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def predict_future(model, X, scaler_X, scaler_y, market_data, window_size=WINDOW_SIZE):
    """Predict next 10 business days from most recent data."""
    X_last = X.iloc[-window_size:]
    X_scaled = scaler_X.transform(X_last).reshape(1, window_size, -1)
    preds_scaled = model.predict(X_scaled).flatten()
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    last_date = market_data.index[-1]
    future_dates = [last_date + BDay(i + 1) for i in range(PREDICTION_DAYS)]

    return pd.DataFrame({'Date': future_dates, 'Predicted Close Price': preds})


def check_ivol_exit_gate(ivol_series):
    """
    Check if IVol is in spike territory (rolling percentile above threshold).
    Returns: (is_blocked, details_str)
    """
    iv = ivol_series.sort_index()
    latest_iv = iv.iloc[-1]
    pctl_rank = iv.rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True).iloc[-1]

    pctl_series = iv.rolling(IVOL_EXIT_WINDOW, min_periods=20).quantile(IVOL_EXIT_PCTL / 100)
    above = (iv > pctl_series).iloc[-20:]
    consec = 0
    for v in reversed(above.values):
        if v:
            consec += 1
        else:
            break

    is_blocked = consec >= IVOL_EXIT_CONFIRM
    thr_val = pctl_series.iloc[-1]

    details = (f"CAC IVol={latest_iv:.1f}, {IVOL_EXIT_PCTL}p threshold={thr_val:.1f} "
               f"(rank={pctl_rank:.0%}), {consec}d above (need {IVOL_EXIT_CONFIRM})")

    return is_blocked, details


def flag_investment_signal(future_df, last_close, threshold=THRESHOLD, ivol_blocked=False):
    """
    Check if predictions warrant a buy signal.
    Blocked if IVol exit gate is active.
    """
    target = last_close + threshold
    signal_df = future_df[future_df['Predicted Close Price'] >= target]
    has_signal = not signal_df.empty

    if ivol_blocked:
        print(f"\n  TKAN predicts upside (target {target:.2f}), but IVol EXIT GATE is ACTIVE")
        print(f"    Signal SUPPRESSED -- high implied vol regime, stay flat")
        return signal_df, False

    if has_signal:
        print(f"\n  Investment Signal: Predicted price >= {target:.2f}")
        print(signal_df[['Date', 'Predicted Close Price']])
        _save_signal(signal_df.iloc[0:1], last_close, threshold, suppressed=False)
    else:
        print(f"\n  No signal -- no predicted price >= {target:.2f}")

    return signal_df, has_signal


def _save_signal(signal_df, last_close, threshold, suppressed=False):
    """Save signal to local CSV + ForgeFolio."""
    ticker = "LVC"
    signal_date = datetime.now().strftime('%Y-%m-%d')
    signal_value = 0 if suppressed else 1
    max_pred = signal_df['Predicted Close Price'].max()
    increase = max_pred - last_close
    confidence = min(0.95, 0.5 + (increase / threshold) * 0.3) if not suppressed else 0.0
    confidence = round(confidence, 2)
    source = "TKAN_v2_ivol"

    local_path = os.path.join(DATA_DIR, "signal.csv")
    forgefolio_path = os.path.join(os.path.dirname(ROOT_DIR),
                                   "ForgeFolio", "data", "Andrey", "strategies", "S08", "signal.csv")

    # Check duplicate
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0] == ticker and row[1] == signal_date:
                        print(f"  Signal for {ticker} on {signal_date} already exists -- skipping")
                        return
        except Exception:
            pass

    row = [ticker, signal_date, signal_value, confidence, source]
    for path, name in [(local_path, "Local"), (forgefolio_path, "ForgeFolio")]:
        try:
            exists = os.path.exists(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(['ticker', 'date', 'signal', 'confidence', 'source'])
                w.writerow(row)
            print(f"  Signal saved to {name}: {path}")
        except Exception as e:
            print(f"  Error saving to {name}: {e}")

    # Detailed signal
    detail_path = os.path.join(DATA_DIR, "signal_detailed.csv")
    try:
        det = signal_df.copy()
        det['Ticker'] = ticker
        det['Signal_Date'] = signal_date
        det['Signal_Value'] = signal_value
        det['Confidence'] = confidence
        det['Source'] = source
        det['Last_Actual_Price'] = last_close
        det['Price_Increase'] = det['Predicted Close Price'] - last_close
        det['Threshold_Used'] = threshold
        cols = ['Ticker', 'Signal_Date', 'Date', 'Signal_Value', 'Confidence', 'Source',
                'Last_Actual_Price', 'Predicted Close Price', 'Price_Increase', 'Threshold_Used']
        det = det[cols]
        exists = os.path.exists(detail_path)
        det.to_csv(detail_path, mode='a', header=not exists, index=False)
    except Exception as e:
        print(f"  Error saving detailed signal: {e}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PLOTTING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def plot_past_and_future(market_data, future_df, ivol_series, signal_df=None, window=20):
    """2-panel matplotlib popup: LVC price + predictions, CAC IVol with exit threshold."""
    plt.style.use('dark_background')
    recent = market_data[['close']].iloc[-window:]

    # Shared x-axis range
    x_min = recent.index[0]
    x_max = future_df['Date'].max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaa')
        ax.grid(True, alpha=0.15, color='#555')
        for spine in ax.spines.values():
            spine.set_color('#333')

    # ── Top panel: price ─────────────────────────────────────────────────
    ax1.plot(recent.index, recent['close'], color='#4A90D9', linewidth=2.5,
             label='Actual LVC')
    ax1.plot(future_df['Date'], future_df['Predicted Close Price'],
             color='#E74C3C', linewidth=2.5, linestyle='--', marker='D',
             markersize=5, label='Predicted')
    if signal_df is not None and not signal_df.empty:
        ax1.scatter(signal_df['Date'], signal_df['Predicted Close Price'],
                    marker='*', color='#F1C40F', s=200, zorder=5,
                    edgecolors='#B7950B', linewidths=0.8, label='Signal')
    ax1.set_ylabel('LVC Price (EUR)', color='#ccc', fontsize=11)
    ax1.set_title('TKAN v2 \u2014 LVC Price Prediction (with CAC IVol awareness)',
                  color='#e0e0e0', fontsize=15, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', fontsize=9, facecolor='#1a1a2e',
               edgecolor='#444', labelcolor='#ddd')
    ax1.set_xlim(x_min, x_max)

    # ── Bottom panel: IVol ───────────────────────────────────────────────
    iv = ivol_series.loc[x_min:].sort_index()
    pctl_thr = ivol_series.rolling(IVOL_EXIT_WINDOW, min_periods=20).quantile(IVOL_EXIT_PCTL / 100)
    thr = pctl_thr.loc[iv.index]

    ax2.plot(iv.index, iv.values, color='#F39C12', linewidth=2,
             label='CAC IVol (3M 50D)')
    ax2.plot(thr.index, thr.values, color='#E74C3C', linewidth=1.5,
             linestyle='--', label=f'{IVOL_EXIT_PCTL}th pctl ({IVOL_EXIT_WINDOW}d)')
    ax2.fill_between(iv.index, iv.values, thr.values,
                     where=iv.values > thr.values,
                     alpha=0.25, color='#E74C3C', label='Above threshold')
    ax2.set_ylabel('Implied Vol', color='#ccc', fontsize=11)
    ax2.set_xlabel('Date', color='#ccc', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9, facecolor='#1a1a2e',
               edgecolor='#444', labelcolor='#ddd')
    ax2.set_xlim(x_min, x_max)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main():
    retrain = False
    auto_update = True

    print("=== TKAN v2 Stock Prediction (with CAC IVol) ===")
    print(f"Retrain: {retrain} | Auto-update: {auto_update}")
    print(f"IVol exit gate: {IVOL_EXIT_PCTL}th pctl over {IVOL_EXIT_WINDOW}d, confirm={IVOL_EXIT_CONFIRM}d\n")

    # Load & feature engineer
    X, y, market_data, ivol_raw = load_and_build_features(auto_update=auto_update)

    # Check if new data was added (compare weights date vs data date)
    if auto_update and os.path.exists(WEIGHTS_PATH):
        weights_mtime = datetime.fromtimestamp(os.path.getmtime(WEIGHTS_PATH))
        data_end = market_data.index[-1].to_pydatetime().replace(tzinfo=None)
        if data_end > weights_mtime:
            print(f"  New data detected (weights: {weights_mtime.date()}, data: {data_end.date()}) -- will retrain")
            retrain = True

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)
    if len(X_seq) == 0:
        raise ValueError("No sequences created -- not enough data")

    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X_seq, y_seq, market_data.index[WINDOW_SIZE:len(X_seq)+WINDOW_SIZE],
        test_size=0.05, shuffle=False)

    # Build model
    input_shape = X_train.shape[1:]  # (10, 11)
    model = build_model(input_shape)
    model.build((None, *input_shape))
    print(f"Model built: input shape {input_shape}")

    # Load or train
    weights_loaded = False
    if not retrain and os.path.exists(WEIGHTS_PATH):
        try:
            model.load_weights(WEIGHTS_PATH)
            print(f"Weights loaded from {WEIGHTS_PATH}")
            weights_loaded = True
        except Exception as e:
            print(f"Failed to load weights ({e}) -- will retrain")

    if not weights_loaded:
        print("Training model...")
        train_model(model, X_train, y_train, X_test, y_test)
        model.save_weights(WEIGHTS_PATH)
        print(f"Weights saved to {WEIGHTS_PATH}")

    # Predict next 10 days
    future_df = predict_future(model, X, scaler_X, scaler_y, market_data)

    print(f"\nPredictions from {market_data.index[-1].date()}:")
    for _, row in future_df.iterrows():
        print(f"  {row['Date'].date()}  EUR {row['Predicted Close Price']:.2f}")

    # IVol exit gate
    ivol_blocked, ivol_details = check_ivol_exit_gate(ivol_raw['ivol'])
    print(f"\nIVol gate: {ivol_details}")
    if ivol_blocked:
        print("  >>> EXIT GATE ACTIVE -- buy signals will be suppressed <<<")

    # Signal
    last_close = market_data['close'].iloc[-1]
    signal_df, has_signal = flag_investment_signal(future_df, last_close,
                                                   ivol_blocked=ivol_blocked)

    # Plot
    plot_past_and_future(market_data, future_df, ivol_raw['ivol'],
                         signal_df=signal_df if has_signal else None)


if __name__ == "__main__":
    main()

