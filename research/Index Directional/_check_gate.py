import sys, pathlib, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1].parent))
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1].parent / 'sfera-db'))
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1].parent / 'signum'))
import importlib, sfera_db
import signum.engine.chart; importlib.reload(signum.engine.chart)
from signum.engine.chart import Chart as _Chart

df = (sfera_db.query(
    "SELECT trade_date AS date, open_price AS open, close_price AS close "
    "FROM bbgidx.index_total_return WHERE ticker = 'CACT' ORDER BY trade_date")
    .assign(date=lambda d: pd.to_datetime(d['date'])).set_index('date'))
df['log_ret'] = np.log(df['close']/df['close'].shift(1))
df = df.dropna()
mom = np.log(df['close']/df['close'].shift(139))
gate = (mom >= 0.005).astype(float)
cc_ret = np.exp(df['log_ret']) - 1
oc_ret = df['close'] / df['open'] - 1

sl = slice('2022-04-01', '2022-04-30')

# Test all modes with carry_in=True
for exec_mode, lbl, kw in [
    (0, 'exec=0', {}), (1, 'exec=1', {}),
    ("NO", 'exec=NO', {'open_returns': oc_ret}),
    (2, 'exec=2', {}),
]:
    r = _Chart.apply_execution(gate, cc_ret, execution=exec_mode, carry_in=True, **kw)
    r_t = r.loc[sl]
    cc_t = cc_ret.loc[sl]
    # Check first 5 bars where strategy should match B&H (carry-in)
    diffs = (r_t - cc_t).abs()
    first_diff = diffs[diffs > 1e-12].index[0] if (diffs > 1e-12).any() else 'never'
    print(f"{lbl:>10}  first 3 strat_ret: {r_t.values[:3]}  first 3 cc_ret: {cc_t.values[:3]}  first_diverge: {first_diff}")
