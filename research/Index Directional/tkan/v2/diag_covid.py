"""Diagnostic: compare model predictions vs actuals for Feb-Mar 2020 (COVID crash)."""
import os, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tkan import TKAN

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'Data')
WDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
WINDOW_SIZE = 10
DROPOUT_RATE = 0.2
IVOL_EXIT_WINDOW = 126

FEATURE_COLS = [
    'log_return_1d','high_low_range','close_to_high','volume_ratio','close_vs_sma15',
    'ivol_zscore','ivol_ema_ratio','ivol_pctl','ivol_roc5',
    'rvol_park20_zscore','vol_spread','return_5d','return_20d',
]

# Load data & build features (identical to train_models.py)
lvc = pd.read_csv(os.path.join(DATA,'lvc_ohlcv.csv'), parse_dates=['date'], index_col='date').sort_index()
cac = pd.read_csv(os.path.join(DATA,'cac_ohlcv.csv'), parse_dates=['date'], index_col='date').sort_index()
ivol_raw = pd.read_csv(os.path.join(DATA,'cac_ivol.csv'), parse_dates=['date'], index_col='date').sort_index()
common = lvc.index.intersection(ivol_raw.index).intersection(cac.index)
df = pd.DataFrame({
    'open': lvc.loc[common,'open'], 'high': lvc.loc[common,'high'],
    'low': lvc.loc[common,'low'], 'close': lvc.loc[common,'close'],
    'volume': lvc.loc[common,'volume'], 'ivol': ivol_raw.loc[common,'ivol'],
    'cac_high': cac.loc[common,'high'], 'cac_low': cac.loc[common,'low'],
    'cac_close': cac.loc[common,'close'],
}).sort_index()

df['log_return_1d'] = np.log(df['close']/df['close'].shift(1))
df['high_low_range'] = np.log(df['high']/df['low'])
df['close_to_high'] = np.log(df['close']/df['high'])
vol_ma20 = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume']/vol_ma20
sma15 = df['close'].rolling(15, min_periods=1).mean()
df['close_vs_sma15'] = np.log(df['close']/sma15)
ivol_m63 = df['ivol'].rolling(63).mean(); ivol_s63 = df['ivol'].rolling(63).std()
df['ivol_zscore'] = (df['ivol']-ivol_m63)/ivol_s63
ivol_ema20 = df['ivol'].ewm(span=20).mean()
df['ivol_ema_ratio'] = df['ivol']/ivol_ema20
df['ivol_pctl'] = df['ivol'].rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True)
df['ivol_roc5'] = df['ivol'].pct_change(5)
log_hl = np.log(df['cac_high']/df['cac_low'])
park_factor = 1.0/(4.0*np.log(2))
rvol_park20 = np.sqrt(park_factor*(log_hl**2).rolling(20).mean()*252)*100
rvol_m63 = rvol_park20.rolling(63).mean(); rvol_s63 = rvol_park20.rolling(63).std()
df['rvol_park20_zscore'] = (rvol_park20-rvol_m63)/rvol_s63
df['vol_spread'] = df['ivol']-rvol_park20
df['return_5d'] = np.log(df['close']/df['close'].shift(5))
df['return_20d'] = np.log(df['close']/df['close'].shift(20))

lagged = {col: df[col].shift(1) for col in FEATURE_COLS}
X = pd.DataFrame(lagged, index=df.index)
y = df['close']
df.dropna(inplace=True); X = X.loc[df.index]; y = y.loc[df.index]
mask = X.notna().all(axis=1); X = X.loc[mask]; y = y.loc[mask]

dates = X.index

# ── Part 1: What do the TRAINING TARGETS look like? ──
print("=" * 80)
print("PART 1: TRAINING TARGET DISTRIBUTION (raw prices)")
print("=" * 80)
y_vals = y.values
all_targets = []
for i in range(len(y_vals) - 2*WINDOW_SIZE):
    fwd_prices = y_vals[i+WINDOW_SIZE:i+2*WINDOW_SIZE]
    all_targets.append(fwd_prices)
all_targets = np.array(all_targets)
print(f"Total training sequences: {len(all_targets)}")
print(f"Target stats per position (raw EUR price):")
for d in range(WINDOW_SIZE):
    col = all_targets[:, d]
    print(f"  Day {d+1}: mean={col.mean():.2f}  std={col.std():.2f}  "
          f"min={col.min():.2f}  max={col.max():.2f}")

print(f"\nDay 10 percentiles:")
d10 = all_targets[:, 9]
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    v = np.percentile(d10, p)
    print(f"  {p:3d}th pctl: EUR {v:.2f}")

# ── Part 2: Feb-Mar 2020 actual returns ──
print("\n" + "=" * 80)
print("PART 2: ACTUAL LVC PRICES FEB-MAR 2020")
print("=" * 80)
crisis_mask = (dates >= '2020-02-01') & (dates <= '2020-04-15')
crisis_dates = dates[crisis_mask]
crisis_y = y.loc[crisis_mask]
print(f"Period: {crisis_dates[0].date()} to {crisis_dates[-1].date()} ({len(crisis_dates)} days)")
print(f"Price: {crisis_y.iloc[0]:.2f} -> {crisis_y.min():.2f} -> {crisis_y.iloc[-1]:.2f}")
print(f"Max drawdown: {(crisis_y.min()/crisis_y.iloc[0]-1)*100:.1f}%")

# Show day-by-day for first few weeks
print("\nDay-by-day:")
for i in range(min(40, len(crisis_dates))):
    d = crisis_dates[i]
    p = float(crisis_y.iloc[i])
    ret = float(np.log(crisis_y.iloc[i]/crisis_y.iloc[max(0,i-1)])) if i > 0 else 0
    print(f"  {d.date()}  EUR {p:7.2f}  log_ret={ret:+.4f} ({(np.exp(ret)-1)*100:+.1f}%)")

# ── Part 3: Model predictions during Feb-Mar 2020 ──
print("\n" + "=" * 80)
print("PART 3: MODEL PREDICTIONS FOR FEB-MAR 2020")
print("=" * 80)

# Load manifest to find which cycle was active
with open(os.path.join(WDIR, 'manifest.json')) as f:
    manifest = json.load(f)
cycles = sorted(manifest['cycles'].items(), key=lambda x: x[1]['cycle_num'])

# Find active cycle for Feb 2020
crisis_start_idx = dates.get_loc(crisis_dates[0])
active_cycle = None
for ck, info in cycles:
    if info['train_end_idx'] <= crisis_start_idx:
        active_cycle = ck
print(f"Active cycle during crisis: {active_cycle}")

# Build and load model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, 13)),
    TKAN(100, return_sequences=True, use_bias=True),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    TKAN(100, return_sequences=True, use_bias=True),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    TKAN(100, return_sequences=True, use_bias=True),
    tf.keras.layers.Dropout(DROPOUT_RATE),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse')
model.load_weights(os.path.join(WDIR, f'{active_cycle}.weights.h5'))
with open(os.path.join(WDIR, f'{active_cycle}_scaler_X.pkl'), 'rb') as f:
    scaler_X = pickle.load(f)
with open(os.path.join(WDIR, f'{active_cycle}_scaler_Y.pkl'), 'rb') as f:
    scaler_Y = pickle.load(f)

# Predict at several points in crisis
test_dates = ['2020-02-14', '2020-02-21', '2020-02-28', '2020-03-06',
              '2020-03-13', '2020-03-20', '2020-03-27']

for td_str in test_dates:
    td = pd.Timestamp(td_str)
    if td not in dates:
        td = dates[dates.searchsorted(td)]
    idx = dates.get_loc(td)
    if idx < WINDOW_SIZE:
        continue

    anchor = float(y.iloc[idx - 1])
    X_win = X.iloc[idx-WINDOW_SIZE:idx]
    X_sc = scaler_X.transform(X_win).reshape(1, WINDOW_SIZE, -1)

    raw = model.predict(X_sc, verbose=0)
    preds_sc = raw.squeeze(-1)  # (1, 10)
    pred_prices = scaler_Y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()

    print(f"\n--- {dates[idx].date()} (idx={idx}) | anchor={anchor:.2f} EUR ---")
    print(f"  Scaled features:")
    for j, col in enumerate(FEATURE_COLS):
        val = X_sc[0, -1, j]
        print(f"    {col:25s} = {val:+.3f}")

    print(f"  Predicted prices:  {np.array2string(pred_prices, precision=2)}")
    print(f"  Pred move day10:   {(pred_prices[-1]-anchor)/anchor*100:+.2f}%")

    actuals = []
    for d in range(WINDOW_SIZE):
        fi = idx + d
        if fi < len(y):
            actuals.append(float(y.iloc[fi]))
    if actuals:
        actuals = np.array(actuals)
        print(f"  Actual prices:     {np.array2string(actuals, precision=2)}")
        print(f"  Actual move day10: {(actuals[-1]-anchor)/anchor*100:+.2f}%")
        print(f"  Prediction error:  {(pred_prices[-1]-actuals[-1])/actuals[-1]*100:+.2f}%")

# ── Part 4: Feature scaling analysis ──
print("\n" + "=" * 80)
print("PART 4: FEATURE SCALING ANALYSIS")
print("=" * 80)
crisis_X = X.loc[crisis_mask]
crisis_X_sc = scaler_X.transform(crisis_X)
print(f"Feature ranges during crisis (unclipped):")
for j, col in enumerate(FEATURE_COLS):
    sc_vals = crisis_X_sc[:, j]
    print(f"  {col:25s}  scaled=[{sc_vals.min():+.3f}, {sc_vals.max():+.3f}]")

print("\nDiagnostic complete.")
