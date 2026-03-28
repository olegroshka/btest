# SMIM Data Remediation Prompts — Phase 2

> For use with Claude Code.  Each section is a self-contained prompt.
> Execute in order: RP1 → RP2 → RP3 → RP4.
> Plan: `docs/smim/DATA_REMEDIATION_PLAN_2.md`

---

## RP1 — Fix BankCreditMapper

We are executing milestone RP1 from `docs/smim/DATA_REMEDIATION_PLAN_2.md`.

**Problem:** `BankCreditMapper.compute()` normalises each bank's time series
independently (per-actor temporal z-score → sigmoid). Cross-sectional rank
across banks is therefore random (ρ = -0.003 for US-LC-FINS). The fix is
identical to what `CorporateCapexMapper` already does: cross-sectional
percentile rank at each quarter.

**Execute these steps in order:**

### Step 1 — Fix `BankCreditMapper` in `intensity_mappers.py`

Read `src/quantdsl_backtest/smim/data/intensity_mappers.py`.

Replace the entire `BankCreditMapper` class body with the following logic:
- `actor_type` property returns `ActorType.BANK` (unchanged)
- `compute(raw_data, actor)`:
  - Raise `KeyError` if `actor.actor_id` not in `raw_data.columns` (unchanged guard)
  - Apply `_cross_section_percentile_rank(raw_data)` to the entire panel
  - Return the column for `actor.actor_id`, renamed to `actor.actor_id`
  - Output is in [0, 1] (inclusive — can reach 0.0 and 1.0, unlike the old sigmoid)
- Update the class docstring:
  - Remove "z-score across time then sigmoid"
  - Write: "cross-sectional percentile rank of YoY asset growth rate"
  - Note output is [0, 1] inclusive
  - Note the `raw_data` panel should contain ALL bank actors in the cross-section
    (not just the target actor), so the rank is meaningful

### Step 2 — Fix `compute_bank_asset_growth()` in `smim_compute_intensities.py`

Read `scripts/smim_compute_intensities.py`.

In `compute_bank_asset_growth()`, change:
```python
growth = assets_panel.pct_change(fill_method=None)
```
to:
```python
growth = assets_panel.pct_change(periods=4, fill_method=None)
```

Rationale: YoY (4-quarter) growth removes seasonal patterns that confound
quarterly bank reporting. Consistent with `return_12m_xsrank` (also 4-period).

### Step 3 — Fix `compute_bank_intensities()` in `smim_compute_intensities.py`

In `compute_bank_intensities()`, change:
```python
intensity = zscore_sigmoid(growth)
return intensity, "asset_growth_zscore_sigmoid"
```
to:
```python
intensity = cross_section_rank(growth)
return intensity, "asset_growth_yoy_xsrank"
```

### Step 4 — Update tests in `tests/unit/smim/data/test_intensity_mappers.py`

The `TestBankCreditMapper` class has tests that assume the old z-score sigmoid
behaviour. Update them for cross-sectional rank:

1. `test_output_strictly_within_unit_interval`:
   Cross-section rank can reach exactly 0.0 and 1.0 (bottom/top rank).
   Change the assertion from `> 0.0` and `< 1.0` (exclusive) to
   `>= 0.0` and `<= 1.0` (inclusive).

2. `test_constant_series_maps_to_half`:
   Old test: a constant raw series with a single actor → 0.5.
   New test: when ALL actors in the cross-section have identical values at
   every date, the rank ties → every actor gets 0.5. Rewrite the test fixture
   to have multiple actors all with the same constant value.
   Keep the assertion `(result.dropna() == 0.5).all()`.

3. `test_monotone_relative_to_z_score`:
   Old test: higher values map to higher output (temporal monotonicity for sigmoid).
   Replace with: **cross-sectional dominance** — an actor that always has higher
   raw values than all peers should always get rank = 1.0.
   New test name: `test_dominant_actor_always_ranks_highest`.
   Fixture: two banks where "a" always has twice the growth of "b"; result_a
   should always be > result_b (and specifically result_a == 1.0 for 2-actor panel).

4. `TestSingleActorDegeneracy.test_zscore_sigmoid_fallback_gives_varying_output`:
   This test verified z-score sigmoid fallback logic in `compute_equity_intensities()`.
   That script-level logic is unchanged. Keep this test (it is NOT testing
   BankCreditMapper directly — it tests the script-level fallback).

Run: `uv run pytest tests/unit/smim/data/test_intensity_mappers.py -q`
All tests must pass before proceeding.

### Step 5 — Recompute all intensity files

Run:
```bash
uv run python scripts/smim_compute_intensities.py
```

This rewrites all `data/smim/intensities/*.parquet` files.

### Step 6 — Verify quality gates

Run:
```python
import pandas as pd, numpy as np, pathlib
from scipy.stats import spearmanr

for f in sorted(pathlib.Path('data/smim/intensities').glob('*.parquet')):
    df = pd.read_parquet(f)
    name = f.stem.replace('_intensities', '')
    panel = df.pivot_table(index='period', columns='actor_id', values='intensity_value')
    rhos = []
    for i in range(len(panel) - 1):
        a = panel.iloc[i].dropna(); b = panel.iloc[i+1].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= 5:
            r, _ = spearmanr(a[common], b[common])
            rhos.append(r)
    rho = np.mean(rhos) if rhos else float('nan')
    status = 'PASS' if rho >= 0.7 else 'WARN'
    print(f'{name}: rho={rho:.3f} ({status}), N={df["actor_id"].nunique()}')
```

Expected: US-LC-FINS ρ ≥ 0.7, US-LC ρ ≥ 0.7. All existing PASS universes must remain PASS.

### Step 7 — Update docs

Update `docs/smim/reports/data_audit.md`:
- In the summary table: remove ⚠️ from US-LC-FINS and US-LC if ρ now passes
- In §9 G-5: change to "✅ RESOLVED 2026-03-28 (RP1)"
- In the gate G1-10 row: update ρ values

Update `docs/smim/reports/data_readiness.md`:
- Update ρ values for US-LC-FINS, US-LC
- Remove "STRUCTURAL" annotations if resolved
- Update "Overall" section

Update `docs/smim/DATA_REMEDIATION_PLAN_2.md` status table:
- RP1 → ✅ Complete

### Step 8 — Commit

```
[SMIM DATA-15] RP1: Fix BankCreditMapper — cross-sectional rank replaces per-actor z-score sigmoid
```

---

## RP2 — Dual Intensity (return_12m_xsrank for all US universes)

We are executing milestone RP2 from `docs/smim/DATA_REMEDIATION_PLAN_2.md`.

**Goal:** Compute `return_12m_xsrank` intensity for all US equity universes
(same approach already used for UK). Store alongside capex files. Measure how
different the two methodologies are to decide whether C4 (cross-geography) is
analytically defensible.

**Execute these steps in order:**

### Step 1 — Add `--method` flag to `smim_compute_intensities.py`

Read the `main()` function in `scripts/smim_compute_intensities.py`.

Add an `argparse` argument `--method` with choices `["capex", "return", "both"]`,
default `"capex"`.

When `--method return` or `--method both`:
- For each universe that has an OHLCV file at `equities/smim/{universe_id}/ohlcv.parquet`:
  - Compute `compute_ohlcv_return_intensities()` regardless of EDGAR availability
  - Write to `data/smim/intensities/{universe_id}_return_intensities.parquet`

When `--method capex` (default): existing behaviour (no change to current output files).

### Step 2 — Run return intensity computation

```bash
uv run python scripts/smim_compute_intensities.py --method return
```

This should produce `_return_intensities.parquet` for:
- US-LC, US-LC-ENERGY, US-LC-TECH, US-LC-FINS, US-LC-HEALTH, US-LC-INDUS, US-MC, US-SC
- (UK-LC, UK-MC already have return_12m_xsrank as their primary method — skip or include)

### Step 3 — Create `scripts/smim_methodology_correlation.py`

Write a new script that:

1. Loads `data/smim/intensities/{universe_id}_intensities.parquet` (capex) and
   `data/smim/intensities/{universe_id}_return_intensities.parquet` (return) for
   each US universe.
2. Joins on `(actor_id, period)`.
3. For each actor with ≥ 8 overlapping quarters, computes Spearman ρ between
   the two intensity series.
4. Aggregates per universe:
   - median ρ, 25th/75th percentile
   - count of actors with ρ > 0.4
   - count of actors with ρ ≤ 0
5. Writes `docs/smim/reports/intensity_methodology_correlation.md` with:
   - A summary table per universe
   - A decision recommendation for C4 (cross-geography experiment):
     - If US-LC median ρ ≥ 0.4: "C4 defensible with disclosure note"
     - If US-LC median ρ < 0.4: "C4 requires homogeneous methodology (run C4b only)"

### Step 4 — Run the correlation script

```bash
uv run python scripts/smim_methodology_correlation.py
```

Review the output and ensure the decision recommendation is explicit.

### Step 5 — Update DATA_REMEDIATION_PLAN_2.md

Mark RP2 → ✅ Complete with the US-LC median ρ value.

### Step 6 — Commit

```
[SMIM DATA-16] RP2: Compute return_12m_xsrank for US universes; methodology correlation report
```

---

## RP3 — Expand MIXED-200 to ~120 Actors

We are executing milestone RP3 from `docs/smim/DATA_REMEDIATION_PLAN_2.md`.

**Goal:** Rebuild `MIXED-200_registry.json` and `experiment_a1_registry.json`
to include actors from multiple sectors and geographies. Target: N ≈ 120 actors
spanning L0–L2, US+UK, 5 sectors.

**Execute these steps in order:**

### Step 1 — Read current registries

Read these files:
- `data/smim/registries/experiment_a1_registry.json` (current A1 registry, 38 actors)
- `data/smim/registries/US-LC-TECH_registry.json`
- `data/smim/registries/US-LC-FINS_registry.json`
- `data/smim/registries/US-LC-HEALTH_registry.json`
- `data/smim/registries/US-LC-INDUS_registry.json`
- `data/smim/registries/UK-LC_registry.json`

Also read `data/smim/processed/edgar_balance_sheet.parquet` to identify which
actors have the best EDGAR CapEx coverage (for US non-energy selection).

### Step 2 — Build `scripts/smim_build_mixed_expanded.py`

Write a script that:

**Selection logic:**

1. Start with all 38 actors from the current `experiment_a1_registry.json`
   (these are the institutional layer + energy equity actors — keep all).

2. From `US-LC-TECH_registry.json`:
   - Select `large_firm` actors only (not sector_leader — already have them via US-LC)
   - Rank by: number of non-null quarters in EDGAR CapEx column (descending)
   - Take top 15

3. From `US-LC-FINS_registry.json`:
   - Select `large_firm` actors (top 10 by EDGAR Assets coverage)
   - Select `bank` actors (top 10 by EDGAR Assets coverage)

4. From `US-LC-HEALTH_registry.json`:
   - Select `large_firm` actors (top 10 by EDGAR CapEx coverage)

5. From `US-LC-INDUS_registry.json`:
   - Select `large_firm` actors (top 10 by EDGAR CapEx coverage)

6. From `UK-LC_registry.json`:
   - Select `large_firm` actors only
   - Rank by OHLCV coverage length (number of non-null quarters in return panel,
     from `data/smim/intensities/UK-LC_return_intensities.parquet` — requires RP2 done)
   - If RP2 not done yet: rank by ticker alphabetically as a placeholder
   - Take top 15

7. Deduplicate by `actor_id` (some actors may appear in multiple sector registries).

8. Verify: 100 ≤ total N ≤ 150.

**Output:**
- `data/smim/registries/experiment_a1_registry.json` — full registry (all actors)
- `data/smim/registries/MIXED-200_registry.json` — equity + shock actors only
  (no central_bank, regulator, intl_org — those are registry-only, not in the universe CSV)
- `data/smim/universes/MIXED-200.csv` — updated equity universe CSV
  (columns: ticker, name, sector, gics_code)
  Only `large_firm` and `bank` and `sector_leader` with geography in {US, UK}

### Step 3 — Run the script

```bash
uv run python scripts/smim_build_mixed_expanded.py
```

Verify:
```bash
python -c "
import json, pathlib
d = json.loads(pathlib.Path('data/smim/registries/experiment_a1_registry.json').read_text())
from collections import Counter
print(f'Total: {len(d[\"actors\"])}')
print(Counter(a['actor_type'] for a in d['actors']))
print(Counter(a.get('geography') for a in d['actors']))
print(Counter(a.get('sector') for a in d['actors']))
"
```

### Step 4 — Recompute intensities

```bash
uv run python scripts/smim_compute_intensities.py
```

### Step 5 — Run quality checks

Verify `experiment_a1_intensities.parquet`:
- Range PASS
- ρ ≥ 0.7
- N ≥ 100

### Step 6 — Update EXPERIMENT_PLAN.md

Update the MIXED-200 row: actual N to the new count; description to "multi-sector,
cross-geography MVP universe".

### Step 7 — Update DATA_REMEDIATION_PLAN_2.md

Mark RP3 → ✅ Complete.

### Step 8 — Commit

```
[SMIM DATA-17] RP3: Expand MIXED-200 to ~120 actors across sectors and geographies
```

---

## RP4 — Trim US-SC and Methodology Robustness Config

We are executing milestone RP4 from `docs/smim/DATA_REMEDIATION_PLAN_2.md`.

**Requires:** RP2 complete (return intensity files must exist).

**Execute these steps in order:**

### Step 1 — Identify US-SC well-covered actors

```python
import pandas as pd, pathlib

df = pd.read_parquet('data/smim/intensities/US-SC_intensities.parquet')
panel = df.pivot_table(index='period', columns='actor_id', values='intensity_value')
missing_frac = panel.isna().mean()
good_actors = missing_frac[missing_frac <= 0.5].index.tolist()
print(f'Well-covered actors: {len(good_actors)}')
```

### Step 2 — Write `US-SC_trimmed_registry.json`

Read `data/smim/registries/US-SC_registry.json`. Filter to `good_actors`.
Write `data/smim/registries/US-SC_trimmed_registry.json`.

### Step 3 — Recompute US-SC trimmed intensities

Modify `smim_compute_intensities.py` to add a `US-SC_trimmed` entry in the
universe processing loop (or run with `--universes US-SC_trimmed`), writing
`data/smim/intensities/US-SC_trimmed_intensities.parquet`.

### Step 4 — Verify: 0 high-missing actors in trimmed version

```python
df = pd.read_parquet('data/smim/intensities/US-SC_trimmed_intensities.parquet')
panel = df.pivot_table(index='period', columns='actor_id', values='intensity_value')
high_missing = (panel.isna().mean() > 0.5).sum()
print(f'High-missing actors: {high_missing}')  # expect 0
```

### Step 5 — Create `docs/smim/METHODOLOGY_ROBUSTNESS_PLAN.md`

Write a document that:

1. Defines the two intensity methodologies:
   - `capex_assets_xsrank`: CapEx/Assets cross-sectionally ranked (US only, EDGAR-dependent)
   - `return_12m_xsrank`: 12-month price return cross-sectionally ranked (all geographies, OHLCV-dependent)

2. For each US universe, reports the median Spearman ρ between the two methods
   (from RP2 correlation report).

3. Defines cross-geography experiment variants:
   - **C4a (primary):** US-LC `return_12m_xsrank` vs UK-LC `return_12m_xsrank`
     — homogeneous methodology, directly comparable
   - **C4b (robustness check):** US-LC `capex_assets_xsrank` vs UK-LC `return_12m_xsrank`
     — heterogeneous methodology, tests sensitivity to intensity choice
   If median ρ(capex vs return) ≥ 0.4 for US-LC, C4a and C4b should produce similar results.
   If divergent, disclose and explain in the paper.

4. Defines cross-cap experiment variants:
   - **C3a:** US-LC `capex_assets_xsrank` vs US-SC_trimmed `capex_assets_xsrank` (primary)
   - **C3b:** US-LC `return_12m_xsrank` vs US-SC_trimmed `return_12m_xsrank` (robustness)

### Step 6 — Update DATA_REMEDIATION_PLAN_2.md

Mark RP4 → ✅ Complete.

### Step 7 — Commit

```
[SMIM DATA-18] RP4: Trim US-SC, methodology robustness plan for C3/C4
```
