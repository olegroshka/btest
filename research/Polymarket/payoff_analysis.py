import numpy as np
from datetime import date

today  = date(2026, 4, 16)
expiry = date(2026, 7, 31)
days   = (expiry - today).days
yrs    = days / 365

print(f"Days to July 31 expiry: {days}  ({yrs:.2f} years)\n")

# Resolution structure (3 outcomes):
#   A: X happens before GTA VI ships (both before Jul 31) -> YES = 1.0
#   B: GTA VI ships before X         (both before Jul 31) -> NO  = 1.0
#   C: Neither by July 31                                 -> 50-50 = 0.50
#
# price = P(A)*1.0 + P(B)*0.0 + P(C)*0.50
# P(GTA VI by Jul 31) = 0.014  (from the "before June" market, basically negligible)
# => price ~ P(X) + 0.50 * P(C)
# where P(C) = 1 - P(X) - P(GTA~0) ~ 1 - P(X)
# => price ~ P(X) + 0.50*(1 - P(X)) = 0.50 + 0.50*P(X)
# => P(X by Jul31 before GTA) = 2*(price - 0.50)   [for price > 0.50]
# For price <= 0.50: P(X) = 0 floor, price = 0.50*P(C) i.e. all value = 50-50 payout

P_gta = 0.014   # prob GTA VI ships by July 31

markets = [
    ("Jesus returns",   0.490),
    ("BTC $1M",         0.490),
    ("China/Taiwan",    0.500),
    ("RU/UA ceasefire", 0.550),
    ("Carti album",     0.530),
    ("Rihanna album",   0.610),
    ("GPT-6 released",  0.690),
    ("Trump out",       0.520),
]

hdr = f"{'Market':<22}  {'Price':>5}  {'P(X wins)':>10}  {'P(50-50)':>9}  {'EV of NO':>9}  {'Edge NO':>8}  {'Ann ret NO':>11}"
print(hdr)
print("-" * len(hdr))

for name, price in markets:
    if price > 0.50:
        p_x  = 2 * (price - 0.50)
    else:
        p_x  = 0.0
    p_50 = max(0, 1 - p_x - P_gta)

    # EV of buying NO at cost (1 - price):
    #   GTA ships first (P_gta): receive 1.0
    #   50-50 (P_50):            receive 0.5
    #   X wins (p_x):            receive 0.0
    ev_no   = P_gta * 1.0 + p_50 * 0.5 + p_x * 0.0
    cost_no = 1 - price
    edge_no = ev_no - cost_no
    ann_ret = (ev_no / cost_no) ** (1 / yrs) - 1 if cost_no > 0 else 0

    # EV of buying YES at price:
    #   X wins (p_x):   receive 1.0
    #   50-50 (p_50):   receive 0.5
    #   GTA wins:       receive 0.0
    ev_yes   = p_x * 1.0 + p_50 * 0.5 + P_gta * 0.0
    edge_yes = ev_yes - price

    print(f"{name:<22}  {price:>5.3f}  {p_x:>10.1%}  {p_50:>9.1%}  {ev_no:>9.3f}  {edge_no:>+8.3f}  {ann_ret*100:>+10.0f}%")

print()
print("NOTE: All markets expire Jul 31, 2026. If neither event NOR GTA VI occurs,")
print("      market resolves 50-50. GTA VI expected Nov 2026 >> Jul 31 expiry.")
print()
print("KEY INSIGHT: With P(GTA by Jul31) ~ 1.4%,")
print("  Price <= 0.50 -> pure 50-50 floor, P(X actually wins) ~ 0")
print("  These markets at 0.49 are essentially guaranteed 50-50 payouts.")
print("  Buying them at 0.49 costs 0.49, returns 0.50 (50-50) -> EV = +2% over 3.5mo")
print(f"  Annualized: {((0.50/0.49)**(1/yrs)-1)*100:.0f}% risk-free if GTA stays delayed!")
print()
print("BEST TRADES:")
print("  BUY Jesus / BTC / China-Taiwan at 0.49 -> collect 50-50 in July")
print("  These are essentially 2% cash instruments maturing Jul 31")
print("  Only risk: GTA VI ships before Jul 31 (1.4% probability -> massive tail)")
