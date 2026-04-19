import sys, psycopg
from pathlib import Path
_ENV = Path(__file__).resolve().parents[3] / 'sfera' / '.env'
DB_CONFIG = {"host": "localhost", "port": "5432", "dbname": "financial_data",
             "user": "postgres", "password": ""}
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            _map = {"DB_HOST":"host","DB_PORT":"port","DB_NAME":"dbname",
                    "DB_USER":"user","DB_PASSWORD":"password"}
            if k.strip() in _map:
                DB_CONFIG[_map[k.strip()]] = v.strip()

conn = psycopg.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
    SELECT
        e.title,
        e.description,
        e.resolution_source,
        e.end_date,
        m.question,
        m.description AS market_desc,
        m.resolution_source AS market_res,
        m.last_trade_price,
        m.best_bid,
        m.best_ask,
        m.volume,
        m.volume_24hr
    FROM polymk.markets m
    JOIN polymk.events e ON m.event_id = e.id
    WHERE e.title ILIKE '%before GTA%'
       OR e.title ILIKE '%GTA%'
    ORDER BY m.volume DESC
""")
rows = cur.fetchall()
conn.close()

for r in rows:
    print("=" * 80)
    print(f"EVENT : {r[0]}")
    print(f"END   : {r[3]}")
    print(f"E.DESC: {r[1]}")
    print(f"E.RES : {r[2]}")
    print(f"MARKET: {r[4]}")
    print(f"M.DESC: {r[5]}")
    print(f"M.RES : {r[6]}")
    print(f"PRICE : {r[7]}  Bid/Ask: {r[8]}/{r[9]}  Vol: ${r[10]/1e3:.0f}K  24h: ${r[11]/1e3:.0f}K")
