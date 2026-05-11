import psycopg2, pandas as pd

DB = dict(host="localhost", dbname="sfera", user="postgres", password="lokomotiv")
conn = psycopg2.connect(**DB)
cur = conn.cursor()

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='eodhd' ORDER BY table_name")
print("All eodhd tables:", [r[0] for r in cur.fetchall()])

# Check for ISIN in prices table columns
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='eodhd' AND table_name='prices' ORDER BY ordinal_position")
print("\neodhd.prices columns:", [r[0] for r in cur.fetchall()])

# Check highlights_snapshot for isin
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='eodhd' AND table_name='highlights_snapshot' ORDER BY ordinal_position")
print("\nhighlights_snapshot columns:", [r[0] for r in cur.fetchall()])

# Check general_snapshot if it exists
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='eodhd' AND table_name ILIKE '%general%'")
print("\ngeneral tables:", [r[0] for r in cur.fetchall()])

# Check all column names mentioning isin across all eodhd tables
cur.execute("""
    SELECT table_name, column_name 
    FROM information_schema.columns 
    WHERE table_schema='eodhd' AND column_name ILIKE '%isin%'
""")
print("\nISIN columns across all tables:", cur.fetchall())

# Inspect instruments table
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='eodhd' AND table_name='instruments' ORDER BY ordinal_position")
print("\ninstruments cols:", [r[0] for r in cur.fetchall()])
df_inst = pd.read_sql("SELECT * FROM eodhd.instruments WHERE exchange_code='LSE' LIMIT 5", conn)
print(df_inst.to_string())

# Inspect exchange_tickers table
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='eodhd' AND table_name='exchange_tickers' ORDER BY ordinal_position")
print("\nexchange_tickers cols:", [r[0] for r in cur.fetchall()])
df_et = pd.read_sql("SELECT * FROM eodhd.exchange_tickers WHERE exchange='LSE' LIMIT 5", conn)
print(df_et.to_string())

conn.close()
