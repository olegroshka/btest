"""One-shot script: download recent index data from Yahoo Finance and append to indicies.parquet."""
import yfinance as yf
import pandas as pd

TICKER_MAP = {
    'UKX': '^FTSE',
    'CAC': '^FCHI',
    'DAX': '^GDAXI',
    'MIB': 'FTSEMIB.MI',
    'IBEX': '^IBEX',
    'SPX': '^GSPC',
    'CCMP': '^IXIC',
}

PARQUET_PATH = 'equities/indicies.parquet'

existing = pd.read_parquet(PARQUET_PATH)
print('Existing data ends:')
for t in existing['ticker'].unique():
    last = existing[existing['ticker'] == t]['date'].max()
    print(f'  {t}: {last.date()}')

new_frames = []
for name, yf_sym in TICKER_MAP.items():
    print(f'Downloading {name} ({yf_sym})...')
    data = yf.download(yf_sym, start='2025-12-10', end='2026-03-20', progress=False)
    if len(data) == 0:
        print(f'  WARNING: No data for {name}')
        continue
    # Flatten multi-level columns if needed
    if hasattr(data.columns, 'levels'):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    df = pd.DataFrame({
        'date': data.index,
        'ticker': name,
        'open': data['Open'].values,
        'high': data['High'].values,
        'low': data['Low'].values,
        'close': data['Close'].values,
        'volume': data['Volume'].values,
    })
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['volume'] = df['volume'].astype('int64')
    new_frames.append(df)
    last_d = df['date'].max().date()
    first_d = df['date'].min().date()
    print(f'  Got {len(df)} rows: {first_d} to {last_d}')

if new_frames:
    new_data = pd.concat(new_frames, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f'\nCombined: {len(combined)} rows (was {len(existing)})')
    print('New date ranges:')
    for t in combined['ticker'].unique():
        sub = combined[combined['ticker'] == t]
        print(f'  {t}: to {sub["date"].max().date()} ({len(sub)} rows)')

    combined.to_parquet(PARQUET_PATH, index=False)
    print('\nSaved updated parquet!')
else:
    print('No new data downloaded!')
