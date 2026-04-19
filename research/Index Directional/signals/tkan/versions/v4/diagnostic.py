# TKAN v4 - Diagnostic Script
# Loads cycle_000 weights + scaler, runs inference day-by-day for its
# entire OOS period, prints raw X_win last row and pred for each day.
# Also checks pred_cache for flat/identical rows.
#
# Run:
#   $env:PYTHONUTF8="1"; $env:TF_ENABLE_ONEDNN_OPTS="0"
#   & "C:/Users/Andrey/miniconda3/envs/ml/python.exe" diagnostic.py
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tkan import TKAN
from sklearn.preprocessing import RobustScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, *(['..'] * 7)) + os.sep + 'sfera-db'))
import sfera_db

WEIGHTS_DIR     = os.path.join(_HERE, 'weights')
WINDOW_SIZE     = 10
IVOL_WINDOW     = 126
FEATURE_COLS = [
    'log_return_1d', 'high_low_range', 'close_to_high',
    'close_vs_sma15',
    'ivol_zscore', 'ivol_ema_ratio', 'ivol_pctl', 'ivol_roc5',
    'rvol_park20_zscore', 'vol_spread',
    'return_5d', 'return_20d',
]
TARGET_COLS = [f'd{k}' for k in range(1, 11)]


# ─── 1. PRED_CACHE STATS ──────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: pred_cache.pkl stats")
print("=" * 70)
cache_path = os.path.join(WEIGHTS_DIR, 'pred_cache.pkl')
pred_df = pickle.load(open(cache_path, 'rb'))
print(f"Shape: {pred_df.shape}   {pred_df.index[0].date()} → {pred_df.index[-1].date()}")
print()
print("First 15 rows:")
print(pred_df.head(15).to_string())
print()
print("d1 std per year:")
print(pred_df['d1'].groupby(pred_df.index.year).std().to_string())
print()
diff = pred_df['d1'].diff().abs()
print(f"d1 max day-to-day change : {diff.max():.8f}")
print(f"d1 rows with diff < 1e-8 : {(diff < 1e-8).mean():.1%}  (should be near 0%)")
print(f"d1 overall std           : {pred_df['d1'].std():.8f}")


# ─── 2. REBUILD DATA ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 2: Rebuilding features from DB")
print("=" * 70)

def _q(sql):
    return (sfera_db.query(sql)
            .assign(date=lambda d: pd.to_datetime(d['date']))
            .set_index('date'))

cactr    = _q("SELECT trade_date AS date, close_price AS close FROM bbgidx.index_total_return WHERE ticker='CACT' ORDER BY trade_date")
cac_ohlc = _q("SELECT trade_date AS date, open_price AS open, high_price AS high, low_price AS low, close_price AS cac_close FROM bbgidx.index_prices WHERE ticker='CAC' ORDER BY trade_date")
ivol_raw = _q("SELECT trade_date AS date, \"3m_50d_ivol\" AS ivol FROM bbgidx.index_implied_vol WHERE ticker='CAC' ORDER BY trade_date")[['ivol']]

ivol_filled = ivol_raw.reindex(ivol_raw.index.union(cactr.index)).ffill()
common = cactr.index.intersection(cac_ohlc.index).intersection(ivol_filled.index)

df = cac_ohlc.loc[common].copy()
df['close'] = cactr.loc[common, 'close']
df['ivol']  = ivol_filled.loc[common, 'ivol']

df['log_return_1d']  = np.log(df['close'] / df['close'].shift(1))
df['high_low_range'] = (df['high'] - df['low']) / df['close'].shift(1).replace(0, np.nan)
df['close_to_high']  = (df['high'] - df['cac_close']) / (df['high'] - df['low'] + 1e-9)
df['sma15']          = df['close'].rolling(15).mean()
df['close_vs_sma15'] = df['close'] / df['sma15'] - 1
df['return_5d']      = np.log(df['close'] / df['close'].shift(5))
df['return_20d']     = np.log(df['close'] / df['close'].shift(20))

hl_sq = np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2
park  = np.sqrt((1 / (4 * np.log(2))) * hl_sq.rolling(20).mean() * 252)
crvol = df['log_return_1d'].rolling(20).std() * np.sqrt(252)
df['rvol_park20'] = park.where(park.notna() & (park > 0), crvol)

ivol   = df['ivol']
ewma20 = ivol.ewm(span=20).mean()
df['ivol_zscore']        = (ivol - ivol.rolling(IVOL_WINDOW).mean()) / (ivol.rolling(IVOL_WINDOW).std() + 1e-9)
df['ivol_ema_ratio']     = ivol / (ewma20 + 1e-9)
df['ivol_pctl']          = ivol.rolling(IVOL_WINDOW).apply(lambda x: (x < x[-1]).sum() / (len(x) - 1), raw=True)
df['ivol_roc5']          = ivol.pct_change(5)
df['rvol_park20_zscore'] = (df['rvol_park20'] - df['rvol_park20'].rolling(IVOL_WINDOW).mean()) / (df['rvol_park20'].rolling(IVOL_WINDOW).std() + 1e-9)
df['vol_spread']         = ivol - df['rvol_park20']

for col in FEATURE_COLS:
    df[col] = df[col].shift(1)

df = df.dropna(subset=FEATURE_COLS + ['close'])
print(f"Features built: {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")


# ─── 3. LOAD CYCLE_000 ────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 3: Load cycle_000 and run day-by-day inference (all OOS days)")
print("=" * 70)

import json
manifest = json.load(open(os.path.join(WEIGHTS_DIR, 'manifest.json')))
cycle_info = manifest['cycles']['cycle_000']
print(f"cycle_000 retrain_date={cycle_info['retrain_date']}  "
      f"train_rows={cycle_info['train_rows']}  oos_days={cycle_info['oos_days']}")

# Load model
def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, len(FEATURE_COLS))),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(0.2),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(0.2),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

model = build_model()
model.load_weights(os.path.join(WEIGHTS_DIR, 'cycle_000.weights.h5'))
scaler = pickle.load(open(os.path.join(WEIGHTS_DIR, 'cycle_000_scaler_X.pkl'), 'rb'))
print("Model and scaler loaded.")

# Scale full dataset with cycle_000 scaler
full_X  = df[FEATURE_COLS].values
full_sc = scaler.transform(full_X)

all_dates = df.index
retrain_date = pd.Timestamp(cycle_info['retrain_date'])

# Find OOS end = start of cycle_001
if 'cycle_001' in manifest['cycles']:
    oos_end_date = pd.Timestamp(manifest['cycles']['cycle_001']['retrain_date'])
    oos_end_idx  = all_dates.searchsorted(oos_end_date)
else:
    oos_end_idx  = len(all_dates)

oos_start_idx = all_dates.searchsorted(retrain_date)
oos_dates     = all_dates[oos_start_idx:oos_end_idx]

print(f"OOS period: {oos_dates[0].date()} → {oos_dates[-1].date()}  ({len(oos_dates)} days)")
print()
print(f"{'Date':<12}  {'d_idx':>6}  {'X_win last row (log_ret, hl_range, ...) [first 3 features]':>50}  pred[d1..d5]")
print("-" * 120)

live_preds = {}
for date in oos_dates:
    d_idx = all_dates.get_loc(date)
    if d_idx < WINDOW_SIZE:
        continue

    X_win = full_sc[d_idx - WINDOW_SIZE : d_idx].reshape(1, WINDOW_SIZE, -1).astype('float32')
    pred  = model(X_win, training=False).numpy()[0, :, 0]  # shape (10,)

    last_row = X_win[0, -1, :3]  # last timestep, first 3 features
    print(f"{date.date()!s:<12}  {d_idx:>6}  "
          f"[{last_row[0]:+.4f}, {last_row[1]:+.4f}, {last_row[2]:+.4f}]  "
          f"pred=[{pred[0]:.6f}, {pred[1]:.6f}, {pred[2]:.6f}, {pred[3]:.6f}, {pred[4]:.6f}]")
    live_preds[date] = pred

live_df = pd.DataFrame.from_dict(live_preds, orient='index', columns=TARGET_COLS)
print()
print("=== Live inference summary ===")
print(f"d1 std  : {live_df['d1'].std():.8f}   (0 = completely flat)")
print(f"d1 min  : {live_df['d1'].min():.6f}")
print(f"d1 max  : {live_df['d1'].max():.6f}")
print(f"d1 mean : {live_df['d1'].mean():.6f}")
print()
print("Are X_win last rows identical across days?")
x_last_rows = np.array([full_sc[all_dates.get_loc(d) - 1] for d in oos_dates if all_dates.get_loc(d) >= WINDOW_SIZE])
print(f"  X last-row std across days (feature 0): {x_last_rows[:, 0].std():.6f}  (0 = features not moving)")
print()
print("Done.")
