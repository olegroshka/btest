"""
Quick interactive analysis helper - loads the backtest result for exploration.
Usage: 
    python interactive_analysis.py
    
Then in the Python REPL you'll have `result` and `strategy` objects ready to use.
"""
from __future__ import annotations

import pickle
import os

# Fast imports
from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest

# Check if we have a cached result
CACHE_FILE = "outputs/lagging_indecies/result_cache.pkl"

if os.path.exists(CACHE_FILE):
    print(f"Loading cached result from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as f:
        result = pickle.load(f)
    print("Cached result loaded!")
else:
    print("Running backtest (this may take a minute)...")
    strategy = build_strategy()
    result = run_backtest(strategy)
    
    # Cache it
    os.makedirs("outputs/lagging_indecies", exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)
    print(f"Result cached to {CACHE_FILE}")

# Now you can explore interactively
strategy = build_strategy()

print("\n" + "="*60)
print("Ready for interactive analysis!")
print("="*60)
print("\nAvailable objects:")
print("  - result: BacktestResult object")
print("  - strategy: Strategy object")
print("\nQuick commands:")
print("  result.summary()")
print("  result.returns.head()")
print("  result.equity.plot()")
print("  result.weights.head()")
print("="*60 + "\n")

# Enter interactive mode
import code
code.interact(local=locals())
