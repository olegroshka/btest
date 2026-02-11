# Debug script for interactive window
# Run each section separately with Shift+Enter

# %% Step 1: Test basic imports (should be fast)
print("Testing basic imports...")
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")

# %% Step 2: Import Universe class (might be slow first time)
print("Importing Universe...")
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice
print("Universe imported successfully!")

# %% Step 3: Create simple Universe (should be instant)
print("Creating Universe...")
universe = Universe(
    name="Indices",
    id_field="ticker",
    filters=[
        HasHistory(min_days=252),
        MinPrice(min_price=5.0),
    ],
)
print(f"Universe created: {universe.name}")
print("SUCCESS!")

# %% Step 4: Check the universe object
print(f"Universe name: {universe.name}")
print(f"ID field: {universe.id_field}")
print(f"Number of filters: {len(universe.filters)}")
