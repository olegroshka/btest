"""
'What will happen before GTA VI?' — Market Analysis
=====================================================
These are correlated binary markets. All share one hidden variable: GTA 6 release date.

If GTA 6 is delayed → ALL "X before GTA VI" prices go UP (easier to happen first).
If GTA 6 ships Nov 2026 → they all go DOWN (less time for X to happen).

The "postpone again" market (0.32) is the master key.

Trading ideas explored:
  1. Pairs: buy underpriced "X before GTA VI" markets, hedge with "postponed" market
  2. Relative value: which markets are mis-priced vs standalone probability of X?
  3. Correlation structure: do they move together?
"""
import sys, psycopg
from pathlib import Path

# Read DB config from sfera's .env directly (avoid dotenv import)
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
import pandas as pd
import numpy as np

conn = psycopg.connect(**DB_CONFIG)
cur = conn.cursor()

# ── All "before GTA VI" markets ───────────────────────────────────────────────
cur.execute("""
    SELECT
        e.title,
        m.question,
        m.last_trade_price AS price,
        m.best_bid,
        m.best_ask,
        m.volume,
        e.open_interest,
        m.liquidity,
        m.volume_24hr,
        m.condition_id
    FROM polymk.markets m
    JOIN polymk.events e ON m.event_id = e.id
    WHERE e.title ILIKE '%before GTA%'
       OR e.title ILIKE '%GTA%'
    ORDER BY m.volume DESC
""")
rows = cur.fetchall()
cols = ['event','question','price','bid','ask','volume','oi','liquidity','vol_24h','condition_id']
df = pd.DataFrame(rows, columns=cols)
conn.close()

print(f"Found {len(df)} GTA-related markets\n")

# ── Separate the "before GTA VI" correlates from the direct GTA markets ───────
before_gta = df[df['event'].str.contains('before GTA', case=False, na=False)].copy()
direct_gta = df[~df['event'].str.contains('before GTA', case=False, na=False)].copy()

print("=== DIRECT GTA MARKETS ===")
for _, r in direct_gta.iterrows():
    spread = float(r['ask'] or 0) - float(r['bid'] or 0)
    print(f"  {r['question'][:65]}")
    print(f"    Price: {r['price']:.3f}  Spread: {spread:.3f}  Vol: ${r['volume']/1e3:.0f}K  24h: ${r['vol_24h']/1e3:.0f}K")

# ── The correlated "before GTA VI" markets ────────────────────────────────────
POSTPONE_PRICE = 0.32  # "GTA 6 postponed again?" current price

print(f"\n=== 'X BEFORE GTA VI' MARKETS (postpone prob = {POSTPONE_PRICE:.0%}) ===")
print(f"{'Question':<45} {'Price':>6} {'Spread':>7} {'Vol $K':>8} {'Impl X prob':>11} {'Edge?':>7}")
print("-" * 95)

# Key insight:
# P("X before GTA VI") = P(X happens before Nov 2026) given GTA VI timeline uncertainty
# If postpone_prob = p, effective GTA VI date is a mixture:
#   - With prob (1-p): Nov 2026
#   - With prob p: after Nov 2026 (say ~2027)
# So P("X before GTA VI") ≈ (1-p)*P(X by Nov26) + p*P(X by 2027)
# 
# We can back out implied P(X by Nov26) from the market price:
# market_price ≈ (1-p)*P_nov26 + p*P_2027
# Very rough — but useful for relative value

for _, r in before_gta.sort_values('volume', ascending=False).iterrows():
    price = float(r['price'])
    bid   = float(r['bid'] or 0)
    ask   = float(r['ask'] or 0)
    spread = ask - bid
    vol   = float(r['volume'])
    vol24 = float(r['vol_24h'])
    q     = r['question']

    # Implied standalone probability if GTA VI were 100% Nov 2026
    # market_price = (1-postpone)*P_standalone + postpone*(P_standalone + delta_delay)
    # Simplified: P_standalone_approx = market_price / (1 - postpone*0.1)  [rough]
    # Better: if X is a slow-moving event (Jesus, BTC $1M), delay of 6mo barely changes it
    # If X is time-sensitive (ceasefire, election), delay matters a lot
    p_standalone_rough = price / (1 + POSTPONE_PRICE * 0.15)  # rough adjustment

    # Is the spread tradeable?
    edge = "✅" if spread < 0.02 and vol24 > 5000 else ("⚠" if spread < 0.04 else "❌")

    print(f"  {q[:43]:<45} {price:>6.3f} {spread:>7.3f} {vol/1e3:>7.0f}K {p_standalone_rough:>10.1%} {edge:>7}")

# ── Key insight: relative mispricing ─────────────────────────────────────────
print("""
=== RELATIVE VALUE ANALYSIS ===

All these markets share one denominator: GTA 6 timing.
Current structure:
  "GTA 6 postponed again?"     = 0.32  (32% chance of delay beyond Nov 2026)
  "Jesus returns before GTA VI"= 0.49  (effectively: is GTA VI coming before armageddon?)
  "Bitcoin $1M before GTA VI"  = 0.49  (BTC $1M AND GTA VI timing interplay)
  "Ceasefire before GTA VI"    = 0.55
  "GPT-6 before GTA VI"        = 0.69  (GPT-6 in next ~7 months — actually plausible)
  "Rihanna album before GTA VI"= 0.61  (she has had 2-year silence, fans hopeful)

TRADE IDEAS:

1. RELATIVE VALUE PAIR (no GTA VI timing risk):
   - BUY "Rihanna album before GTA VI" (0.61)  vs SELL "Jesus returns" (0.49)
   - Rationale: Rihanna is MORE likely than the 12pp gap suggests.
     She has confirmed music coming, GPT-6 is 69%, Rihanna should be ≥ 65%
   - Both move the same way on GTA VI delay news → GTA VI timing CANCELS OUT
   - Pure bet: Rihanna album comes out before whatever Jesus/BTC event

2. MOMENTUM ON POSTPONEMENT:
   - Watch the "postponed again" market (0.32)
   - If it drops fast (release confirmed) → all "X before GTA VI" prices drop in lockstep
   - BUY "GPT-6 before GTA VI" (0.69) — GPT-6 is likely regardless of GTA VI timing
     As GTA VI release is confirmed sooner, this should hold/increase

3. THE JESUS TRADE (actually rational):
   - "Jesus returns before GTA VI" at 0.49 is literally pricing the probability
     that GTA 6 ships before the second coming of Christ
   - If you believe GTA 6 ships Nov 2026: this should be ~0.02 (near zero)
   - If you believe GTA 6 ships never: this approaches 0.98
   - At 0.49, the market is essentially saying 50% chance GTA 6 ships in a
     reasonable timeframe. The 32% postponement market says something different.
   - ARBITRAGE SIGNAL: these two markets are INCONSISTENT.
     "Postponed again" (0.32) implies ~68% on-schedule → 
     "Jesus before GTA VI" should be closer to 0.30-0.35, not 0.49.
     → BUY "GTA VI before June 2026" NO (0.986 implied) ✅ already maxed
     → SELL "Jesus before GTA VI" (0.49 → fair value ~0.25-0.35) 🔥

CONCLUSION:
  The Jesus market (0.49) is the most mispriced if you believe GTA 6 ships
  in 2026 at all. It's pricing ~50% chance GTA never comes out — inconsistent
  with the postponement market at 0.32 which implies 68% on schedule for Nov 26.
""")
