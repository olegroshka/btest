# LEV_ETF Research Data — Setup Guide

## What's in this directory

| File | Description | Source |
|------|-------------|--------|
| `cac_ivol.csv` | CAC 3m 50-delta implied volatility (2006–present) | Sfera DB / Bloomberg |
| `eur_overnight_rate.csv` | EONIA / €STR daily rate (2000–present) | ECB |
| *(parent)* `../lvc_ohlcv.csv` | LVC Amundi 2× CAC OHLCV (2010–present) | Sfera DB / Bloomberg |
| *(parent)* `../cactr_ohlcv.csv` | CAC 40 Total Return index OHLCV | Sfera DB / Bloomberg |

---

## Option A — Use the committed snapshots (no setup needed)

The CSV files above are committed to the repo and cover data through **March 2026**.
Just clone `btest` and run the notebook — it will use local files automatically.

---

## Option B — Live connection to your own Sfera DB

This lets you get the latest data and use `sfera_db` across all research notebooks.

### 1. Install PostgreSQL (15+)

Download from https://www.postgresql.org/download/ and install with default settings.
Note your password for the `postgres` superuser.

### 2. Install DataGrip (recommended SQL client)

Download from https://www.jetbrains.com/datagrip/ (30-day trial / free for students).
Or use pgAdmin (free, comes with PostgreSQL installer).

### 3. Create the Sfera schema

In DataGrip / pgAdmin, connect to `localhost:5432` as `postgres`, then run:

```sql
CREATE DATABASE sfera;
\c sfera

CREATE SCHEMA bbgidx;

-- CAC IVol
CREATE TABLE bbgidx.index_implied_vol (
    ticker      TEXT        NOT NULL,
    trade_date  DATE        NOT NULL,
    ivol        NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);

-- Index OHLCV (CAC, CACT, etc.)
CREATE TABLE bbgidx.index_prices (
    ticker      TEXT        NOT NULL,
    trade_date  DATE        NOT NULL,
    open_price  NUMERIC,
    high_price  NUMERIC,
    low_price   NUMERIC,
    close_price NUMERIC,
    volume      BIGINT,
    PRIMARY KEY (ticker, trade_date)
);

-- Total return index
CREATE TABLE bbgidx.index_total_return (
    ticker      TEXT        NOT NULL,
    trade_date  DATE        NOT NULL,
    close_price NUMERIC,
    PRIMARY KEY (ticker, trade_date)
);
```

### 4. Load data from the CSV snapshots

```sql
-- Load CAC IVol
\COPY bbgidx.index_implied_vol (ticker, trade_date, ivol)
FROM '/path/to/btest/data/lev_etf/cac_ivol.csv'
CSV HEADER;

-- Load LVC prices
\COPY bbgidx.index_prices (ticker, trade_date, open_price, high_price, low_price, close_price, volume)
FROM '/path/to/btest/data/lvc_ohlcv.csv'
CSV HEADER;

-- Load CACT total return
\COPY bbgidx.index_total_return (ticker, trade_date, close_price)
FROM '/path/to/btest/data/cactr_ohlcv.csv'
CSV HEADER;
```

> **DataGrip tip:** You can right-click a table → *Import Data* and use the GUI instead of psql.

### 5. Configure credentials

Create a `.env` file in your home directory at `~/.sfera/.env`:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sfera
DB_USER=postgres
DB_PASSWORD=your_password_here
```

`sfera_db` will find this automatically on startup.

### 6. Install sfera-db and signum

```bash
# From the Python codes workspace root:
pip install -e ./sfera-db
pip install -e ./signum
```

Or install from GitHub (once repos are public):

```bash
pip install git+https://github.com/olegroshka/sfera-db.git
pip install git+https://github.com/olegroshka/signum.git   # adjust URL
```

### 7. Install btest dependencies

```bash
cd btest
pip install uv   # fast package manager (optional)
uv pip install -e .
# or: pip install -e .
```

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
