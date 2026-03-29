"""TKAN v2.6 Walk-Forward Model Training
=======================================
Stationary feature set (log returns, ratios, z-scores).
Target: price ratios (future_price / anchor_price).
Scaling: RobustScaler on X only (no clipping). No scaler on Y.
Loss: MSE — penalizes large errors (drawdowns) quadratically.
Dropout layers for future MC inference.

Trains TKAN models at each retrain point and caches:
  - Model weights (.weights.h5)
  - Fitted scaler (scaler_X.pkl)
  - Training metadata (manifest.json)

Subsequent runs skip cycles that already have cached weights.
Delete the weights folder to force full retrain.
"""
import os, json, hashlib, pickle, time
import tensorflow as tf
from tkan import TKAN
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (must match prediction_chart / backtest)
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR    = os.path.join(ROOT_DIR, "Data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")

LVC_CSV   = os.path.join(DATA_DIR, "lvc_ohlcv.csv")
CAC_IVOL  = os.path.join(DATA_DIR, "cac_ivol.csv")
CAC_OHLCV = os.path.join(DATA_DIR, "cac_ohlcv.csv")

WINDOW_SIZE     = 10
PREDICTION_DAYS = 10
EPOCHS          = 50
BATCH_SIZE      = 32
BACKTEST_START  = '2015-01-01'
RETRAIN_FREQ    = 126
IVOL_EXIT_WINDOW = 126
DROPOUT_RATE    = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG FINGERPRINT — if any of these change, cached weights are invalid
# ══════════════════════════════════════════════════════════════════════════════
def compute_config_hash():
    config_str = json.dumps({
        'version': 'v2.6_mse',
        'WINDOW_SIZE': WINDOW_SIZE,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'BACKTEST_START': BACKTEST_START,
        'RETRAIN_FREQ': RETRAIN_FREQ,
        'IVOL_EXIT_WINDOW': IVOL_EXIT_WINDOW,
        'DROPOUT_RATE': DROPOUT_RATE,
        'model': '3xTKAN100_seq_Dropout_Dense1',
        'features': 'stationary_13',
        'target': 'forward_price_ratio',
        'loss': 'mse',
        'clipping': 'none',
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# DATA & STATIONARY FEATURES
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'log_return_1d', 'high_low_range', 'close_to_high',
    'volume_ratio', 'close_vs_sma15',
    'ivol_zscore', 'ivol_ema_ratio', 'ivol_pctl', 'ivol_roc5',
    'rvol_park20_zscore', 'vol_spread',
    'return_5d', 'return_20d',
]


def load_and_build_features():
    """Build stationary feature set. All features are lagged by 1 day."""
    lvc = pd.read_csv(LVC_CSV, parse_dates=['date'], index_col='date').sort_index()
    cac_ohlcv = pd.read_csv(CAC_OHLCV, parse_dates=['date'], index_col='date').sort_index()
    ivol_raw = pd.read_csv(CAC_IVOL, parse_dates=['date'], index_col='date').sort_index()

    common = lvc.index.intersection(ivol_raw.index).intersection(cac_ohlcv.index)
    df = pd.DataFrame({
        'open':      lvc.loc[common, 'open'],
        'high':      lvc.loc[common, 'high'],
        'low':       lvc.loc[common, 'low'],
        'close':     lvc.loc[common, 'close'],
        'volume':    lvc.loc[common, 'volume'],
        'ivol':      ivol_raw.loc[common, 'ivol'],
        'cac_high':  cac_ohlcv.loc[common, 'high'],
        'cac_low':   cac_ohlcv.loc[common, 'low'],
        'cac_close': cac_ohlcv.loc[common, 'close'],
    }).sort_index()

    # ── Stationary features (all computed at time t) ──
    # Price features → log returns / ratios
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = np.log(df['high'] / df['low'])
    df['close_to_high'] = np.log(df['close'] / df['high'])

    # Volume → relative to 20d average
    vol_ma20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_ma20

    # SMA → log distance from SMA
    sma15 = df['close'].rolling(15, min_periods=1).mean()
    df['close_vs_sma15'] = np.log(df['close'] / sma15)

    # IVol → z-score and ratio
    ivol_mean63 = df['ivol'].rolling(63).mean()
    ivol_std63 = df['ivol'].rolling(63).std()
    df['ivol_zscore'] = (df['ivol'] - ivol_mean63) / ivol_std63
    ivol_ema20 = df['ivol'].ewm(span=20).mean()
    df['ivol_ema_ratio'] = df['ivol'] / ivol_ema20
    df['ivol_pctl'] = df['ivol'].rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True)
    df['ivol_roc5'] = df['ivol'].pct_change(5)

    # Realized vol → z-score
    log_hl = np.log(df['cac_high'] / df['cac_low'])
    park_factor = 1.0 / (4.0 * np.log(2))
    rvol_park20 = np.sqrt(park_factor * (log_hl ** 2).rolling(20).mean() * 252) * 100
    rvol_mean63 = rvol_park20.rolling(63).mean()
    rvol_std63 = rvol_park20.rolling(63).std()
    df['rvol_park20_zscore'] = (rvol_park20 - rvol_mean63) / rvol_std63

    # Vol spread (already stationary)
    df['vol_spread'] = df['ivol'] - rvol_park20

    # Momentum at longer horizons
    df['return_5d'] = np.log(df['close'] / df['close'].shift(5))
    df['return_20d'] = np.log(df['close'] / df['close'].shift(20))

    # ── Lag all features by 1 day (use t-1 to predict t..t+10) ──
    lagged = {}
    for col in FEATURE_COLS:
        lagged[col] = df[col].shift(1)
    X = pd.DataFrame(lagged, index=df.index)

    # ── Target: log return from close today ──
    y = df['close']  # we'll compute forward log returns in create_sequences

    df.dropna(inplace=True)
    valid = df.index
    X = X.loc[valid]
    y = y.loc[valid]

    # Drop any remaining NaN from lagged features
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y, df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING & TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def create_sequences(X, y_close, window_size):
    """Create sequences. Target is price ratio: future_price / anchor_price.
    Anchor = last close in the input window (y_close[i + window_size - 1]).
    Ratios are level-independent (~1.0) so no scaler needed."""
    X_seq, y_seq = [], []
    for i in range(len(X) - 2 * window_size):
        X_seq.append(X[i:i + window_size])
        anchor = y_close[i + window_size - 1]
        fwd_prices = y_close[i + window_size:i + 2 * window_size]
        y_seq.append(fwd_prices / anchor)
    return np.array(X_seq), np.array(y_seq)


def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, n_features)),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_cycle(X_train_df, y_close_series):
    """Scale X with RobustScaler, targets are price ratios (no scaling needed).
       Returns (model, scaler_X, val_loss)."""
    scaler_X = RobustScaler()
    X_sc = scaler_X.fit_transform(X_train_df)

    y_vals = y_close_series.values
    X_seq, y_seq = create_sequences(X_sc, y_vals, WINDOW_SIZE)
    if len(X_seq) < 50:
        raise ValueError(f"Only {len(X_seq)} sequences — need more training data")

    # y_seq is already price ratios (~1.0), no scaling needed
    model = build_model(X_seq.shape[2])
    history = model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, verbose=0)
    val_loss = history.history['val_loss'][-1]
    return model, scaler_X, val_loss


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def run_training():
    config_hash = compute_config_hash()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Check if existing manifest matches current config
    manifest_path = os.path.join(WEIGHTS_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if manifest.get('config_hash') != config_hash:
            print(f"Config changed (old={manifest.get('config_hash')}, new={config_hash})")
            print("Cached weights are INVALID. Delete weights/ folder and rerun to retrain all.")
            print("Or press Enter to retrain missing cycles with new config...")
            input()
    else:
        manifest = {'config_hash': config_hash, 'cycles': {}}

    X, y, df = load_and_build_features()
    dates = X.index
    bt_start = pd.Timestamp(BACKTEST_START)
    bt_start_idx = dates.searchsorted(bt_start)

    print(f"Data: {dates[0].date()} to {dates[-1].date()} ({len(dates)} rows)")
    print(f"Features: {len(FEATURE_COLS)} stationary inputs → predicting price ratios")
    print(f"Backtest starts at idx {bt_start_idx} ({dates[bt_start_idx].date()})")
    print(f"Config hash: {config_hash}")
    print("=" * 70)

    # Determine all retrain points
    retrain_points = []
    i = bt_start_idx
    while i < len(dates):
        retrain_points.append(i)
        i += RETRAIN_FREQ
    print(f"Total retrain cycles: {len(retrain_points)}")

    trained = 0
    skipped = 0
    total_time = 0

    for cycle_num, train_end_idx in enumerate(retrain_points, 1):
        cycle_date = dates[train_end_idx].strftime('%Y-%m-%d')
        cycle_key = f"cycle_{cycle_num:03d}_{cycle_date}"
        weights_path = os.path.join(WEIGHTS_DIR, f"{cycle_key}.weights.h5")
        scaler_x_path = os.path.join(WEIGHTS_DIR, f"{cycle_key}_scaler_X.pkl")

        # Check if already cached
        if (os.path.exists(weights_path) and
            os.path.exists(scaler_x_path) and
            cycle_key in manifest.get('cycles', {})):
            skipped += 1
            print(f"  [{cycle_num:3d}/{len(retrain_points)}] {cycle_date} — cached (skip)")
            continue

        # Train
        t0 = time.time()
        print(f"  [{cycle_num:3d}/{len(retrain_points)}] {cycle_date} — training on {train_end_idx} rows...", end=" ", flush=True)
        model, scaler_X, val_loss = train_cycle(
            X.iloc[:train_end_idx], y.iloc[:train_end_idx]
        )
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"val_loss={val_loss:.6f} ({elapsed:.1f}s)")

        # Save weights + scaler
        model.save_weights(weights_path)
        with open(scaler_x_path, 'wb') as f:
            pickle.dump(scaler_X, f)

        # Update manifest
        manifest['cycles'][cycle_key] = {
            'cycle_num': cycle_num,
            'date': cycle_date,
            'train_end_idx': int(train_end_idx),
            'train_rows': int(train_end_idx),
            'val_loss': float(val_loss),
            'train_time_s': round(elapsed, 1),
        }
        manifest['config_hash'] = config_hash

        # Save manifest after each cycle (resumable)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        trained += 1

    print("=" * 70)
    print(f"Done. Trained: {trained} | Skipped (cached): {skipped} | Total: {len(retrain_points)}")
    if trained > 0:
        print(f"Training time: {total_time:.0f}s ({total_time/trained:.1f}s avg per cycle)")
    print(f"Weights saved to: {WEIGHTS_DIR}")


if __name__ == "__main__":
    run_training()
