# LEV_ETF Research — New Machine Setup & Verification

## TL;DR

**Just want to run the notebook?** → Do **Step 1 only** (clone + install). No Bloomberg, no DB needed.  
**Want live data refresh from your own Bloomberg?** → Follow Option B (Steps 2–5).

---

## Data sources — what requires Bloomberg

| Table / File | Source | Bloomberg required? |
|---|---|---|
| `cac_ivol.csv` | CAC 40 3m 50Δ implied vol | ✅ Yes (field `BVOL3M Index`) |
| `../cactr_ohlcv.csv` | CAC 40 Total Return (CACT) | ✅ Yes (Bloomberg CACT index) |
| `../lvc_ohlcv.csv` | LVC Amundi 2× CAC ETF OHLCV | ⚠️ Bloomberg or Yahoo (`LVC.PA`) |
| `eur_overnight_rate.csv` | EONIA / €STR overnight rate | ❌ No — free from [ECB](https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_short-term_rate/html/index.en.html) |

**Snapshots covering through March 2026 are committed to the repo — Bloomberg is only needed to extend them.**

> For LVC without Bloomberg: Yahoo Finance ticker is `LVC.PA` (Euronext Paris).
> For CACT without Bloomberg: use `^FCHI` (CAC price-only) as an approximation, or Euronext data.

---

## What's in this directory

| File | Description |
|---|---|
| `cac_ivol.csv` | CAC implied vol snapshot |
| `eur_overnight_rate.csv` | ECB overnight rate snapshot |
| *(parent)* `../lvc_ohlcv.csv` | LVC ETF price snapshot |
| *(parent)* `../cactr_ohlcv.csv` | CACT total return snapshot |

---

## Step 1 — Clone and install (required on any machine)

```bash
# 1. Clone btest
git clone https://github.com/olegroshka/btest.git
cd btest

# 2. Install Python dependencies
pip install uv          # fast package manager (optional but recommended)
uv pip install -e .     # installs btest + all deps from pyproject.toml
# or: pip install -e .

# 3. Install signum (charting library)
cd ..
git clone https://github.com/SugoiKitsune/signum.git
pip install -e ./signum

# 4. (Optional) Install sfera-db if you want live DB access
# git clone https://github.com/olegroshka/sfera-db.git  (once public)
# pip install -e ./sfera-db
```

### Verify Step 1 works

Open `btest/research/Index Directional/signals/lev_etf/cac40_tr_leverage.ipynb` and run cells 1–3.

Expected output from cell 3:
```
Data source : local CSV snapshot
Date range  : 2010-01-06 → 2026-03-20  (4126 days)
LVC B&H     Sharpe ...   Total ...%   MaxDD ...%
```

If you see that — **you're done**. Everything below is only for live data refresh.

---

## Option A — Use the committed snapshots (no setup needed)

The CSV files above are committed to the repo and cover data through **March 2026**.
Just clone `btest` and run the notebook — it will use local files automatically.

---

## Option B — Live connection to your own Sfera DB

**Requires:** PostgreSQL installed locally + Bloomberg Terminal (or Yahoo Finance for LVC).

### B1. Install PostgreSQL 15+

Download from https://www.postgresql.org/download/ — use default port 5432.  
Install pgAdmin (bundled) or DataGrip (https://www.jetbrains.com/datagrip/, 30-day free trial).

### B2. Create schema

Connect to `localhost:5432` as `postgres`, run in pgAdmin / DataGrip query console:

```sql
CREATE DATABASE sfera;
\c sfera
CREATE SCHEMA bbgidx;

CREATE TABLE bbgidx.index_implied_vol (
    ticker TEXT NOT NULL, trade_date DATE NOT NULL, ivol NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);
CREATE TABLE bbgidx.index_prices (
    ticker TEXT NOT NULL, trade_date DATE NOT NULL,
    open_price NUMERIC, high_price NUMERIC, low_price NUMERIC,
    close_price NUMERIC, volume BIGINT,
    PRIMARY KEY (ticker, trade_date)
);
CREATE TABLE bbgidx.index_total_return (
    ticker TEXT NOT NULL, trade_date DATE NOT NULL, close_price NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);
```

### B3. Bootstrap without Bloomberg — load committed CSV snapshots

In DataGrip: right-click table → **Import Data** and map columns to the CSV.  
Or in psql (after adding a `ticker` column value manually or via sed):

```sql
-- Easiest: use Python to load
```
```python
import pandas as pd, psycopg, os

conn = psycopg.connect(host='localhost', dbname='sfera', user='postgres', password='...')
cur = conn.cursor()

# CAC IVol
df = pd.read_csv('data/lev_etf/cac_ivol.csv', parse_dates=['date'])
df.insert(0, 'ticker', 'CAC')
cur.executemany(
    "INSERT INTO bbgidx.index_implied_vol VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
    df[['ticker','date','ivol']].itertuples(index=False, name=None)
)

# LVC OHLCV
df = pd.read_csv('data/lvc_ohlcv.csv', parse_dates=['date'])
df.insert(0, 'ticker', 'LVC')
cur.executemany(
    "INSERT INTO bbgidx.index_prices VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
    df.itertuples(index=False, name=None)
)

# CACTR
df = pd.read_csv('data/cactr_ohlcv.csv', parse_dates=['date'])
df.insert(0, 'ticker', 'CACT')
cur.executemany(
    "INSERT INTO bbgidx.index_total_return VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
    df[['ticker','date','close']].itertuples(index=False, name=None)
)
conn.commit()
print("Done")
```

### B4. Configure credentials

Create `~/.sfera/.env` (works on Windows, Mac, Linux — `~` = your home directory):

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sfera
DB_USER=postgres
DB_PASSWORD=your_password_here
```

`sfera_db` finds this file automatically.

### B5. Install sfera-db

```bash
git clone https://github.com/olegroshka/sfera-db.git   # once public
pip install -e ./sfera-db
# or direct: pip install git+https://github.com/olegroshka/sfera-db.git
```

### B6. Verify live connection

```python
import sfera_db
print(sfera_db.tables())                    # lists tables in bbgidx schema
print(sfera_db.index_ivol('CAC').tail())    # should return recent IVol rows
```

### B7. Extending data with Bloomberg

Bloomberg fields used:

| Data | Bloomberg ticker | Field |
|---|---|---|
| CAC implied vol | `BVOL3M Index` | `PX_LAST` |
| CAC Total Return | `CACT Index` | `PX_LAST` |
| LVC ETF OHLCV | `LVC FP Equity` | `PX_OPEN/HIGH/LOW/LAST/VOLUME` |

Pull via BDH in Excel or `blpapi` in Python, shape to match the CSV columns, then:

```python
sfera_db.write_table(df, "index_implied_vol")   # appends new rows
```

**Without Bloomberg:** LVC is available as `LVC.PA` on Yahoo Finance (`yfinance`).  
CACT (total return) has no free substitute — use the snapshot or approximate with `^FCHI`.

---

## Refreshing the CSV snapshots

Once you have a live Sfera DB, run this from a notebook or script to update the committed snapshots:

```python
import sfera_db, pandas as pd
from pathlib import Path

DATA = Path('data')  # btest/data/

# Refresh CAC IVol
ivol = sfera_db.index_ivol('CAC')[['ivol']].reset_index()
ivol.columns = ['date', 'ivol']
ivol.to_csv(DATA / 'lev_etf/cac_ivol.csv', index=False)

# Refresh CACTR
cactr = sfera_db.read_table('index_total_return', ticker='CACT')
cactr = cactr.reset_index().rename(columns={'trade_date':'date','close_price':'close'})
cactr[['date','close']].to_csv(DATA / 'cactr_ohlcv.csv', index=False)

# Refresh LVC
lvc = sfera_db.index_prices('LVC').reset_index()
lvc.columns = ['date','open','high','low','close','volume']
lvc.to_csv(DATA / 'lvc_ohlcv.csv', index=False)

print("Snapshots updated — commit data/ to lock in the new baseline.")
```
