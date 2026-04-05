"""Patch all non-archive research notebooks for portability.
Run from btest/ root: python scripts/_patch_notebooks.py
"""
import json, pathlib

ROOT = pathlib.Path(__file__).parent.parent

def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

def save(rel, data):
    (ROOT / rel).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"  saved: {rel}")

def src(lines):
    """Convert a multiline string to notebook source list."""
    parts = lines.split('\n')
    return [l + '\n' for l in parts[:-1]] + [parts[-1]]

# ─── Shared portable header ──────────────────────────────────────────────────
HEADER = """\
import sys, subprocess
import pandas as pd, numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

_REPO = Path(subprocess.check_output(
    'git rev-parse --show-toplevel', shell=True, text=True, cwd='.'
).strip())
DATA = _REPO / 'data'
PROJECT_ROOT = _REPO

try:
    from signum import Chart, Dashboard
except ImportError:
    sys.path.insert(0, str(_REPO.parent / 'signum'))
    from signum import Chart, Dashboard

try:
    import sfera_db
    _SFERA_AVAILABLE = True
except ImportError:
    _SFERA_AVAILABLE = False
    print("sfera_db not available — using CSV snapshots (see data/lev_etf/SETUP.md)")"""

HEADER_NO_SFERA = """\
import sys, subprocess
import pandas as pd, numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

_REPO = Path(subprocess.check_output(
    'git rev-parse --show-toplevel', shell=True, text=True, cwd='.'
).strip())
DATA = _REPO / 'data'
PROJECT_ROOT = _REPO

try:
    from signum import Chart, Dashboard
except ImportError:
    sys.path.insert(0, str(_REPO.parent / 'signum'))
    from signum import Chart, Dashboard"""

SFERA_TRYEXCEPT = """\
try:
    import sfera_db
    _SFERA_AVAILABLE = True
except ImportError:
    _SFERA_AVAILABLE = False
    print("sfera_db not available — using CSV snapshots (see data/lev_etf/SETUP.md)")"""

# ─── 1. ivol_hedge_signal.ipynb ──────────────────────────────────────────────
print("1. ivol_hedge_signal.ipynb")
p = 'research/Index Directional/signals/ivol/ivol_hedge_signal.ipynb'
d = load(p)

d['cells'][1]['source'] = src(HEADER + """
import ipywidgets as widgets
from IPython.display import display""")

d['cells'][2]['source'] = src("""
# ── Load CAC (price, ivol, total return) ──
if _SFERA_AVAILABLE:
    cac_px   = sfera_db.index_prices('CAC')[['close', 'volume']]
    cac_ivol = sfera_db.index_ivol('CAC')[['ivol']]
    try:
        cac_tr_raw = sfera_db.read_table('index_total_return', ticker='CACT')[['close_price']]
        cac_tr_px  = cac_tr_raw.rename(columns={'close_price': 'close'})
        has_tr = True
    except Exception:
        cac_tr_px = None; has_tr = False
else:
    cac_px   = pd.read_csv(DATA / 'c40_ohlcv.csv',          parse_dates=['date'], index_col='date')[['close', 'volume']]
    cac_ivol = pd.read_csv(DATA / 'lev_etf' / 'cac_ivol.csv', parse_dates=['date'], index_col='date')[['ivol']]
    cac_tr_px = pd.read_csv(DATA / 'cactr_ohlcv.csv',        parse_dates=['date'], index_col='date')[['close']]
    has_tr = True

common_idx = cac_px.index.intersection(cac_ivol.index)
if has_tr:
    common_idx = common_idx.intersection(cac_tr_px.index)

cac = pd.DataFrame({'close': cac_px.loc[common_idx, 'close'],
                    'ivol':  cac_ivol.loc[common_idx, 'ivol']}).sort_index()
cac['ret'] = cac['close'].pct_change()
if has_tr:
    cac['close_tr'] = cac_tr_px.loc[common_idx, 'close']
    cac['ret_tr']   = cac['close_tr'].pct_change()
    clean = cac.dropna(subset=['ret', 'ret_tr']).copy()
else:
    cac['ret_tr'] = cac['ret']
    clean = cac.dropna(subset=['ret']).copy()
clean['ivol_ema5']  = clean['ivol'].ewm(span=5).mean()
clean['ivol_ema20'] = clean['ivol'].ewm(span=20).mean()

# ── Load LVC (2x CAC, Amundi) — full history from 2008 ──
lvc_raw = (
    pd.read_csv(DATA / 'lev_etf' / 'lvc_daily.csv', parse_dates=['date'])
    .set_index('date').sort_index()
)
common_lvc = clean.index.intersection(lvc_raw.index)
lvc = pd.DataFrame({
    'close': lvc_raw.loc[common_lvc, 'close'],
    'ivol':  clean.loc[common_lvc, 'ivol'],
    'ret':   lvc_raw.loc[common_lvc, 'close'].pct_change(),
}).dropna()
lvc_bh_eq     = (1 + lvc['ret']).cumprod()
lvc_bh_sharpe = lvc['ret'].mean() / lvc['ret'].std() * np.sqrt(252)
print(f"Data source    : {'sfera_db (live)' if _SFERA_AVAILABLE else 'CSV snapshots'}")
print(f"LVC date range : {lvc.index[0].date()} -> {lvc.index[-1].date()}  ({len(lvc)} days)")
print(f"LVC B&H  Sharpe {lvc_bh_sharpe:.3f}  Total {(lvc_bh_eq.iloc[-1]-1)*100:+.1f}%")""")

save(p, d)

# ─── 2. ivol_parameters_tweak.ipynb ─────────────────────────────────────────
print("2. ivol_parameters_tweak.ipynb")
p = 'research/Index Directional/signals/ivol/ivol_parameters_tweak.ipynb'
d = load(p)

d['cells'][1]['source'] = src(HEADER + """
print(f"Project root: {_REPO}  |  sfera_db: {'live' if _SFERA_AVAILABLE else 'unavailable (CSV fallback)'}")""")

# cell 3 (idx=2): load CAC from sfera_db or CSV
d['cells'][2]['source'] = src("""# ── Load & align data ──
if _SFERA_AVAILABLE:
    cac_px   = sfera_db.index_prices('CAC')[['close', 'volume']]
    cac_ivol = sfera_db.index_ivol('CAC')[['ivol']]
else:
    cac_px   = pd.read_csv(DATA / 'c40_ohlcv.csv',          parse_dates=['date'], index_col='date')[['close', 'volume']]
    cac_ivol = pd.read_csv(DATA / 'lev_etf' / 'cac_ivol.csv', parse_dates=['date'], index_col='date')[['ivol']]

common_idx = cac_px.index.intersection(cac_ivol.index)
cac = pd.DataFrame({
    'close': cac_px.loc[common_idx, 'close'],
    'ivol':  cac_ivol.loc[common_idx, 'ivol'],
}).sort_index()
cac['ret'] = cac['close'].pct_change()

clean = cac.dropna(subset=['ret']).copy()

# EWMA on IVol
clean['ivol_ema5']  = clean['ivol'].ewm(span=5).mean()
clean['ivol_ema20'] = clean['ivol'].ewm(span=20).mean()

bh_eq = (1 + clean['ret']).cumprod()
bh_total = (bh_eq.iloc[-1] - 1) * 100
bh_sharpe = clean['ret'].mean() / clean['ret'].std() * np.sqrt(252)
bh_mdd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min() * 100

print(f"Data: {clean.index[0].date()} -> {clean.index[-1].date()}  ({len(clean)} days)")
print(f"B&H: {bh_total:+.1f}%  Sharpe: {bh_sharpe:.3f}  MaxDD: {bh_mdd:.1f}%")""")

save(p, d)

# ─── 3. levetf_vs_tr.ipynb ───────────────────────────────────────────────────
print("3. levetf_vs_tr.ipynb")
p = 'research/Index Directional/signals/lev_etf/levetf_vs_tr.ipynb'
d = load(p)

d['cells'][1]['source'] = src("""
import sys, subprocess, importlib
import pandas as pd
import numpy as np
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import warnings; warnings.filterwarnings('ignore')

_REPO = Path(subprocess.check_output(
    'git rev-parse --show-toplevel', shell=True, text=True, cwd='.'
).strip())
DATA = _REPO / 'data'
PROJECT_ROOT = _REPO

try:
    from signum import Chart, Dashboard
except ImportError:
    sys.path.insert(0, str(_REPO.parent / 'signum'))
    import signum.engine.chart as _sc; importlib.reload(_sc)
    import signum.engine.dashboard as _sd; importlib.reload(_sd)
    import signum.engine as _se; importlib.reload(_se)
    import signum as _sig; importlib.reload(_sig)
    from signum import Chart, Dashboard

try:
    import sfera_db
    _SFERA_AVAILABLE = True
except ImportError:
    _SFERA_AVAILABLE = False
    print("sfera_db not available — using CSV snapshots (see data/lev_etf/SETUP.md)")

# ── Load CAC price, IVol, total return ──
if _SFERA_AVAILABLE:
    cac_px    = sfera_db.index_prices('CAC')[['close']]
    cac_ivol  = sfera_db.index_ivol('CAC')[['ivol']]
    cac_tr_px = sfera_db.read_table('index_total_return', ticker='CACT')[['close_price']].rename(columns={'close_price': 'close'})
else:
    cac_px    = pd.read_csv(DATA / 'c40_ohlcv.csv',          parse_dates=['date'], index_col='date')[['close']]
    cac_ivol  = pd.read_csv(DATA / 'lev_etf' / 'cac_ivol.csv', parse_dates=['date'], index_col='date')[['ivol']]
    cac_tr_px = pd.read_csv(DATA / 'cactr_ohlcv.csv',         parse_dates=['date'], index_col='date')[['close']]

common_cac = cac_px.index.intersection(cac_ivol.index).intersection(cac_tr_px.index)
cac = pd.DataFrame({
    'close':    cac_px.loc[common_cac, 'close'],
    'close_tr': cac_tr_px.loc[common_cac, 'close'],
    'ivol':     cac_ivol.loc[common_cac, 'ivol'],
}).sort_index()
cac['ret_pr'] = cac['close'].pct_change()
cac['ret_tr'] = cac['close_tr'].pct_change()
cac = cac.dropna(subset=['ret_pr', 'ret_tr'])

# ── Load LVC (2x CAC) — full history from 2008 ──
lvc_raw = (
    pd.read_csv(DATA / 'lev_etf' / 'lvc_daily.csv', parse_dates=['date'])
    .set_index('date').sort_index()
)

common = cac.index.intersection(lvc_raw.index)
df = pd.DataFrame({
    'lvc':    lvc_raw.loc[common, 'close'],
    'cac_pr': cac.loc[common, 'close'],
    'cac_tr': cac.loc[common, 'close_tr'],
    'ivol':   cac.loc[common, 'ivol'],
}).dropna()
df['lvc_ret']    = df['lvc'].pct_change()
df['cac_pr_ret'] = df['cac_pr'].pct_change()
df['cac_tr_ret'] = df['cac_tr'].pct_change()
df = df.dropna()

_rates = pd.read_csv(DATA / 'lev_etf' / 'eur_overnight_rate.csv', parse_dates=['date'], index_col='date')['rate_pct'].reindex(df.index, method='ffill')

IBKR_SPREAD = 0.015   # +1.50% over EUR STR (IBKR Pro EUR, balance < 90K EUR)
df['overnight_pct'] = _rates
df['borrow_daily']  = (_rates / 100 + IBKR_SPREAD) / 252
df['margin_ret']    = 2 * df['cac_tr_ret'] - df['borrow_daily']
df = df.dropna(subset=['margin_ret'])

print(f"Data source : {'sfera_db (live)' if _SFERA_AVAILABLE else 'CSV snapshots'}")
print(f"Date range  : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} days)")
print(f"IBKR borrow : EUR STR + {IBKR_SPREAD*100:.2f}%  current = {(_rates.iloc[-1]/100 + IBKR_SPREAD)*100:.2f}%/yr")""")

save(p, d)

# ─── 4. tkan/versions/v3 — fix sys.path for sfera-db ─────────────────────────
for nb in [
    'research/Index Directional/signals/tkan/versions/v3/feature_analysis.ipynb',
    'research/Index Directional/signals/tkan/versions/v3/TKAN_v3_research.ipynb',
    'research/Index Directional/signals/tkan/versions/v3/TKAN_v3_experiment_maxret.ipynb',
    'research/Index Directional/signals/tkan/versions/v3/Untitled-2.ipynb',
]:
    print(f"4. {nb.split('/')[-1]}")
    d = load(nb)
    for i, cell in enumerate(d['cells']):
        s = ''.join(cell.get('source', []))
        if 'sys.path.insert' in s and 'sfera-db' in s:
            new_src = 'try:\n    import sfera_db\nexcept ImportError:\n    import subprocess\n    from pathlib import Path\n    _REPO = Path(subprocess.check_output(\n        "git rev-parse --show-toplevel", shell=True, text=True, cwd="."\n    ).strip())\n    import sys; sys.path.insert(0, str(_REPO.parent / "sfera-db"))\n    import sfera_db\n'
            # preserve rest of cell after the import sfera_db line
            lines = s.split('\n')
            after = []
            skip_next = False
            for line in lines:
                if 'sys.path.insert' in line and 'sfera-db' in line:
                    skip_next = True
                    continue
                if skip_next and 'import sfera_db' in line:
                    skip_next = False
                    continue
                skip_next = False
                after.append(line)
            rest = '\n'.join(after).strip()
            full = new_src + ('\n' + rest if rest else '')
            d['cells'][i]['source'] = src(full)
            break
    save(nb, d)

# ─── 5. Triple Leveraged ETF — fix NOTEBOOK_DIR + PROJECT_ROOT ───────────────
print("5. triple_leveraged_etf_strategy.ipynb")
p = 'research/Triple Leveraged ETF/triple_leveraged_etf_strategy.ipynb'
d = load(p)

for i, cell in enumerate(d['cells']):
    s = ''.join(cell.get('source', []))
    if 'NOTEBOOK_DIR' in s and 'PROJECT_ROOT' in s:
        new = s.replace(
            'NOTEBOOK_DIR = Path(r"c:\\Personal\\Business & Investments\\Trading portfolio\\Cogilator\\btest\\notebooks")\nPROJECT_ROOT = NOTEBOOK_DIR.parent',
            'import subprocess as _sp\nPROJECT_ROOT = Path(_sp.check_output(\n    "git rev-parse --show-toplevel", shell=True, text=True, cwd="."\n).strip())'
        )
        d['cells'][i]['source'] = src(new)
        break

# also fix cell 46 standalone PROJECT_ROOT
for i, cell in enumerate(d['cells']):
    s = ''.join(cell.get('source', []))
    if 'PROJECT_ROOT = Path(r"c:\\Personal' in s and 'indicies.parquet' in s:
        new = s.replace(
            'PROJECT_ROOT = Path(r"c:\\Personal\\Business & Investments\\Trading portfolio\\Cogilator\\btest")',
            'import subprocess as _sp\nPROJECT_ROOT = Path(_sp.check_output(\n    "git rev-parse --show-toplevel", shell=True, text=True, cwd="."\n).strip())'
        )
        d['cells'][i]['source'] = src(new)
        break

# fix cell 25 standalone project_root
for i, cell in enumerate(d['cells']):
    s = ''.join(cell.get('source', []))
    if 'project_root = r"c:\\Personal' in s:
        new = s.replace(
            'project_root = r"c:\\Personal\\Business & Investments\\Trading portfolio\\Cogilator\\btest"',
            'import subprocess as _sp\nproject_root = _sp.check_output(\n    "git rev-parse --show-toplevel", shell=True, text=True, cwd="."\n).strip()'
        )
        d['cells'][i]['source'] = src(new)
        break

save(p, d)

print("\nAll notebooks patched.")
