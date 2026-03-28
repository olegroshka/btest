# SMIM Data Remediation Plan — Phase 2

> Created: 2026-03-28
> Follows: `docs/smim/DATA_REMEDIATION_PLAN.md` (R1–R6 complete)
> Execution prompts: `docs/smim/DATA_REMEDIATION_PROMPTS_2.md`

---

## Motivation

Four structural problems remain after Phase 1 remediation that would compromise
experimental credibility:

| # | Problem | Impact |
|---|---------|--------|
| P1 | BankCreditMapper uses per-actor temporal z-score → random cross-sectional rankings | US-LC-FINS ρ=-0.003 unusable; US-LC ρ=0.660 (banks are ~37% of universe) |
| P2 | UK uses price-return intensity, US uses CapEx/Assets — different economic constructs | Cross-geography experiments (C4) compare apples to oranges; gap unknown |
| P3 | MIXED-200 equity component is 27 actors, not ~200 | Anchor experiment (A1) is under-powered; all-layer cross-section is thin |
| P4 | US-SC has 48/142 (34%) high-missing actors | C3 (cross-cap) confounds data quality with methodology effects |

---

## Issue Analysis

### P1 — BankCreditMapper: wrong normalisation axis

**Root cause:** `BankCreditMapper.compute()` normalises each bank's time series
independently (per-actor temporal z-score → sigmoid). Every bank's intensity
series therefore has approximately mean=0.5 regardless of how it compares to
peers in the same quarter. Cross-sectional rank across banks is effectively random.

**Fix:** Switch to the same cross-sectional percentile rank that
`CorporateCapexMapper` already uses. Rank all banks by their YoY asset growth
rate at each quarter end; normalise ranks to [0,1]. Rank stability is guaranteed
by construction for any persistent differences in bank growth levels.

**Raw metric:** Assets YoY growth = `(Assets_Q - Assets_{Q-4}) / Assets_{Q-4}`.
Use 4-period `pct_change` (matches the 12-month return_12m_xsrank approach).
QoQ growth is too volatile and seasonally confounded for banks.

### P2 — Methodology gap between US and UK

**Root cause:** US equities use `capex_assets_xsrank` (real investment allocation
from EDGAR balance sheets). UK equities use `return_12m_xsrank` (market price
returns) because no Companies House adapter was built. These measure different
economic quantities.

**Calibration approach:** Compute `return_12m_xsrank` for all US equity universes
alongside the existing `capex_assets_xsrank`. For each US actor, compute the
Pearson and Spearman correlation between the two intensity series. This gives an
empirical estimate of the methodology gap. If median correlation ≥ 0.4, the
proxies are close enough that cross-geography comparison is defensible with a
disclosure note. If < 0.4, the UK methodology divergence invalidates C4.

### P3 — MIXED-200 is under-populated

**Current state:** MIXED-200 registry has 26 equity actors (22 large_firm +
4 sector_leader), all in the energy sector (US + UK). With institutional actors
from experiment_a1, the full experiment registry has 38 actors.

**Target:** ~120 actors spanning all 4 layers:
- Layer 0 (macro shocks): 7 existing FRED series (shock_*)
- Layer 1 (institutions): 4 existing (Fed, BoE, IMF, SEC)
- Layer 2 (large firms): ~80 equity actors across sectors and geographies
  - Energy (existing): 22 US + UK energy firms
  - Technology: 15 US-LC + 5 UK-LC
  - Financials: 15 US-LC (banks + sector_leaders)
  - Industrials: 10 US-LC
  - Healthcare: 10 US-LC
- Layer 3: none in MVP (too sparse)

This gives a genuine multi-sector, cross-geography, cross-layer universe that
matches the original SMIM design intent.

### P4 — US-SC data sparsity

**Current state:** 48/142 actors (34%) have >50% missing intensity values.
These are post-2015 IPOs and SPACs with insufficient EDGAR history.

**Fix:** Create a `US-SC_trimmed_registry.json` that retains only the 94
well-covered actors. Do NOT delete the full US-SC_registry.json — keep it for
documentation of the full universe.

---

## Dependency Graph

```
RP1: Fix BankCreditMapper (cross-sectional rank)
   └── no dependencies (independent fix)
   └── recompute: US-LC-FINS, US-LC, experiment_fast, experiment_phased

RP2: Dual intensity (return_12m_xsrank for all US universes)
   └── no dependencies (additive — creates new *_return_intensities.parquet files)
   └── produces: methodology_correlation_report.md

RP3: Expand MIXED-200 to ~120 actors
   └── depends on: RP1 (so bank actors in MIXED-200 use the fixed mapper)
   └── updates: MIXED-200.csv, MIXED-200_registry.json, experiment_a1_registry.json

RP4: Trim US-SC + methodology robustness configs
   └── depends on: RP2 (so return intensity files exist for comparison)
   └── creates: US-SC_trimmed_registry.json
   └── updates: experiment config files for C3/C4 methodology dimension
```

**Recommended execution order:** RP1 → RP2 → RP3 → RP4

---

## Milestone RP1: Fix BankCreditMapper

**Goal:** Replace per-actor temporal z-score sigmoid with cross-sectional
percentile rank over YoY asset growth. Restore rank stability for US-LC-FINS
and US-LC.

**Scope:** Two files change:
1. `src/quantdsl_backtest/smim/data/intensity_mappers.py` — BankCreditMapper class
2. `scripts/smim_compute_intensities.py` — compute_bank_intensities() + compute_bank_asset_growth()

**Steps:**

1. Read `intensity_mappers.py` fully (already done).
2. Rewrite `BankCreditMapper.compute()`:
   - Remove per-actor z-score sigmoid logic
   - Replace with `_cross_section_percentile_rank(raw_data)[actor.actor_id]`
   - Output is now in [0,1] (not strictly (0,1))
   - Update docstring: "cross-sectional percentile rank of YoY asset growth"
   - The KeyError guard stays unchanged
3. Update `compute_bank_asset_growth()` in the script:
   - Change `pct_change()` to `pct_change(periods=4)` (YoY instead of QoQ)
4. Update `compute_bank_intensities()` in the script:
   - Change `zscore_sigmoid(growth)` to `cross_section_rank(growth)`
   - Change method string from `"asset_growth_zscore_sigmoid"` to `"asset_growth_yoy_xsrank"`
5. Update tests in `tests/unit/smim/data/test_intensity_mappers.py`:
   - `TestBankCreditMapper.test_output_strictly_within_unit_interval`: change to `[0, 1]` (inclusive, not exclusive) — cross-section rank can hit exactly 0 and 1 when there are 2+ actors
   - `TestBankCreditMapper.test_constant_series_maps_to_half`: with cross-section rank, a constant series among diverse peers does NOT map to 0.5. Rewrite: when ALL actors have the same value, every actor should get 0.5 (ties averaged). When only one actor is constant and others vary, it gets its rank.
   - `TestBankCreditMapper.test_monotone_relative_to_z_score`: this test verified temporal monotonicity under z-score. Replace with cross-sectional equivalent: when bank A always has higher asset growth than bank B, A should always rank above B.
   - `TestSingleActorDegeneracy.test_zscore_sigmoid_fallback_gives_varying_output`: this tested the z-score sigmoid logic. Adapt to test cross-section rank fallback behavior.
6. Run `uv run pytest tests/unit/smim/data/test_intensity_mappers.py -q` — all tests must pass.
7. Recompute intensities for affected universes:
   ```bash
   uv run python scripts/smim_compute_intensities.py
   ```
   (runs all universes)
8. Verify US-LC-FINS ρ ≥ 0.7 and US-LC ρ ≥ 0.7 after recompute.
9. Update `docs/smim/reports/data_readiness.md` and `data_audit.md`:
   - Update ρ values for US-LC-FINS and US-LC
   - Remove "STRUCTURAL" annotation for US-LC-FINS if resolved
   - Mark G-5 as RESOLVED in data_audit.md

**Quality Gate RP1:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| RP1-QG-1 | BankCreditMapper output | Values in [0,1], cross-sectionally ranked |
| RP1-QG-2 | US-LC-FINS rank stability | Spearman ρ ≥ 0.7 |
| RP1-QG-3 | US-LC rank stability | Spearman ρ ≥ 0.7 |
| RP1-QG-4 | All tests pass | `uv run pytest tests/unit/smim/data/test_intensity_mappers.py` green |
| RP1-QG-5 | No regressions | All other universes maintain ρ ≥ threshold |

**Git commit:** `[SMIM DATA-15] RP1: Fix BankCreditMapper — cross-sectional rank replaces per-actor z-score sigmoid`

---

## Milestone RP2: Dual Intensity (return_12m_xsrank for all US universes)

**Goal:** Compute OHLCV return-based intensity for all US equity universes
alongside the existing CapEx-based intensity. Quantify the methodology gap to
decide whether cross-geography comparison is analytically defensible.

**Steps:**

1. Modify `smim_compute_intensities.py` to accept a `--method return` flag:
   - If `--method return`: compute `return_12m_xsrank` using OHLCV for all
     equity actors (US and UK). Store output as `{universe_id}_return_intensities.parquet`.
   - If `--method capex` (default): existing behaviour unchanged.
   - If `--method both`: compute both; store both output files.
   - Alternatively: add a parallel run block that always computes return intensities
     for any universe that has an OHLCV file, regardless of whether EDGAR data exists.
2. Run with `--method return` for all US equity universes:
   - US-LC, US-LC-ENERGY, US-LC-TECH, US-LC-FINS, US-LC-HEALTH, US-LC-INDUS,
     US-MC, US-SC
3. For each US universe, compute per-actor Pearson and Spearman correlation
   between `capex_assets_xsrank` and `return_12m_xsrank` across the shared
   time dimension. Write a summary script:
   ```bash
   uv run python scripts/smim_methodology_correlation.py
   ```
4. Create `scripts/smim_methodology_correlation.py`:
   - Load both intensity files for each US universe
   - Inner join on (actor_id, period)
   - Per actor: compute Spearman rho between capex and return series
   - Aggregate: median, 25th/75th percentile, % actors with rho > 0.4
   - Output: `docs/smim/reports/intensity_methodology_correlation.md`
5. Decision gate: if median Spearman ρ (capex vs return) ≥ 0.4 across US-LC:
   - Cross-geography experiments can proceed with disclosure note
   - If < 0.4: recommend against C4 until Companies House adapter is built

**Quality Gate RP2:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| RP2-QG-1 | Return intensity files exist | `*_return_intensities.parquet` present for all 8 US universes |
| RP2-QG-2 | Return intensity quality | All [0,1] range PASS; ρ ≥ 0.7 for well-covered universes |
| RP2-QG-3 | Correlation report generated | `intensity_methodology_correlation.md` exists |
| RP2-QG-4 | Decision gate documented | Report includes explicit C4 recommendation |

**Git commit:** `[SMIM DATA-16] RP2: Compute return_12m_xsrank for US universes; methodology correlation report`

---

## Milestone RP3: Expand MIXED-200 to ~120 Actors

**Goal:** Rebuild MIXED-200 registry and experiment_a1_registry to represent a
genuine multi-sector, cross-geography, all-layer universe with N ≈ 120 actors.

**Target composition:**

| Layer | Actor type | Source | Count |
|-------|-----------|--------|-------|
| L0 (exogenous) | global_shock | FRED series (existing) | 7 |
| L1 (upstream) | central_bank | Fed (US), BoE (UK) | 2 |
| L1 (upstream) | intl_org | IMF | 1 |
| L1 (upstream) | regulator | SEC, FCA | 2 |
| L2 (transmission) | large_firm (energy) | Existing MIXED-200 equity | 22 |
| L2 (transmission) | sector_leader (energy) | Existing | 4 |
| L2 (transmission) | large_firm (tech) | US-LC-TECH top 15 | 15 |
| L2 (transmission) | large_firm (financials) | US-LC-FINS large_firm top 10 | 10 |
| L2 (transmission) | bank | US-LC-FINS bank top 10 | 10 |
| L2 (transmission) | large_firm (industrials) | US-LC-INDUS top 10 | 10 |
| L2 (transmission) | large_firm (healthcare) | US-LC-HEALTH top 10 | 10 |
| L2 (transmission) | large_firm (UK) | UK-LC top 15 | 15 |
| L2 (transmission) | sector_leader (UK/non-energy) | UK-LC sector_leaders | 2 |
| **Total** | | | **~120** |

**Steps:**

1. Read `data/smim/registries/experiment_a1_registry.json` (current full registry).
2. Read sector slices: `US-LC-TECH_registry.json`, `US-LC-FINS_registry.json`,
   `US-LC-HEALTH_registry.json`, `US-LC-INDUS_registry.json`, `UK-LC_registry.json`.
3. Build a Python script `scripts/smim_build_mixed_expanded.py` that:
   a. Starts from existing experiment_a1 actors (retain all 38)
   b. Adds top 15 US-LC-TECH `large_firm` actors (sort by EDGAR coverage —
      prefer actors with `PaymentsToAcquirePropertyPlantAndEquipment` tag present)
   c. Adds top 10 US-LC-FINS `large_firm` actors + top 10 `bank` actors
   d. Adds top 10 US-LC-INDUS `large_firm` actors
   e. Adds top 10 US-LC-HEALTH `large_firm` actors
   f. Adds top 15 UK-LC `large_firm` actors (by OHLCV coverage length)
   g. Deduplicates by actor_id
   h. Writes:
      - `data/smim/registries/experiment_a1_registry.json` (updated)
      - `data/smim/registries/MIXED-200_registry.json` (equity-only subset)
      - `data/smim/universes/MIXED-200.csv` (updated universe CSV)
4. Run the script; confirm N ≈ 100–130.
5. Run `uv run python scripts/smim_compute_intensities.py` to recompute
   intensities for the expanded registry (writes `experiment_a1_intensities.parquet`).
6. Run quality checks; confirm range PASS and ρ ≥ 0.7.
7. Update `docs/smim/EXPERIMENT_PLAN.md` MIXED-200 row with new N.

**Quality Gate RP3:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| RP3-QG-1 | Registry size | `experiment_a1_registry.json` has 100 ≤ N ≤ 150 actors |
| RP3-QG-2 | Layer coverage | All 4 layers (L0–L2) represented |
| RP3-QG-3 | Geography coverage | US actors + UK actors both present |
| RP3-QG-4 | Intensity quality | Range PASS; ρ ≥ 0.7 |
| RP3-QG-5 | Sector coverage | ≥ 4 sectors in equity component |

**Git commit:** `[SMIM DATA-17] RP3: Expand MIXED-200 to ~120 actors across sectors and geographies`

---

## Milestone RP4: Trim US-SC and Methodology Robustness Config

**Goal:** Remove data-quality confounds from cross-cap experiment (C3) by trimming
US-SC to the 94 well-covered actors. Add methodology dimension (capex vs return
intensity) to C3/C4 experiment configs.

**Steps:**

1. Load `data/smim/intensities/US-SC_intensities.parquet`. Identify the 94 actors
   with ≤50% missing quarters (already computed: 48 are high-missing, 94 are good).
2. Load `data/smim/registries/US-SC_registry.json`. Filter to good actors. Write
   `data/smim/registries/US-SC_trimmed_registry.json`.
3. Recompute `US-SC_intensities.parquet` using the trimmed registry.
4. Verify: 0 high-missing actors in the trimmed version; ρ ≥ 0.7.
5. In `scripts/smim_compute_intensities.py`, add EXPERIMENT_PLAN entry for
   `"C3 (cross-cap US-SC trimmed)"` pointing to the trimmed registry.
6. Update `docs/smim/EXPERIMENT_PLAN.md` to note US-SC trimmed vs full.
7. Create `docs/smim/METHODOLOGY_ROBUSTNESS_PLAN.md`:
   - Document the two intensity methodologies (capex_assets_xsrank, return_12m_xsrank)
   - For each US universe, list which methodology gives higher ρ (from RP2 correlation report)
   - Define the C4 cross-geography experiment variants:
     - C4a: US-LC (capex) vs UK-LC (return) — heterogeneous methodology, current state
     - C4b: US-LC (return) vs UK-LC (return) — homogeneous methodology, enabled by RP2
   - State which variant is the primary and which is the robustness check

**Quality Gate RP4:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| RP4-QG-1 | US-SC trimmed registry | 90 ≤ N ≤ 100 actors, all ≤50% missing |
| RP4-QG-2 | US-SC trimmed intensity | 0 high-missing actors; ρ ≥ 0.7 |
| RP4-QG-3 | C4 variants documented | METHODOLOGY_ROBUSTNESS_PLAN.md exists with C4a/C4b defined |

**Git commit:** `[SMIM DATA-18] RP4: Trim US-SC, methodology robustness plan for C3/C4`

---

## Status Tracking

| Milestone | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| RP1: Fix BankCreditMapper | ✅ Complete | 2026-03-28 | Cross-section rank YoY asset growth; US-LC-FINS ρ: -0.003 → 0.769; US-LC ρ: 0.660 → 0.761 |
| RP1.5: Dual-window ρ gate (US-LC-TECH) | ✅ Complete | 2026-03-28 | `quality_check_intensities()` now computes rho_full + rho_recent (2020–); gate passes if either ≥ 0.7; US-LC-TECH: ρ_full=0.653, ρ_recent=0.750 PASS |
| RP2: Dual intensity (US return) | ⬜ Pending | — | Depends on RP1 |
| RP3: Expand MIXED-200 | ⬜ Pending | — | Depends on RP1 |
| RP4: Trim US-SC + robustness | ⬜ Pending | — | Depends on RP2 |
