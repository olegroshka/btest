# SMIM Data Quality Remediation Plan

> Created: 2026-03-28
> Status: Active — all milestones pending
> Reference: `docs/smim/reports/data_audit.md` (source of all issues)
> Execution prompts: `docs/smim/DATA_REMEDIATION_PROMPTS.md`

---

## Issue Registry (from data_audit.md)

| ID | Issue | Severity | Blocking |
|----|-------|----------|---------|
| G-1 | OECD data: 244 rows fetched, ~2,000 expected; most series ≤5 years | 🔴 Critical | FULL/MACRO-ONLY OECD signals unusable |
| G-2 | UK intensities not computed (UK-LC, UK-MC) | 🔴 Critical | E1 experiment blocked |
| G-3 | CPIMEDSL not fetched (healthcare CPI proxy) | 🟡 Medium | US-LC-HEALTH MACRO-ONLY feed degraded |
| G-4 | Companies House adapter never built (UK balance-sheet) | 🟡 Medium | UK balance-sheet 0%; root cause of G-2 |
| G-5 | US-LC-FINS rank stability ρ=0.040 (threshold 0.7) | 🟡 Medium | Financials intensity invalid |
| G-6 | Sector universe sizes diverge from EXPERIMENT_PLAN.md | 🟢 Low | Documentation only |
| G-7 | GDELT raw directory has stale Gen 1 + Gen 2 artifacts | 🟢 Low | Housekeeping only |
| G-8 | Gate G1-5/G1-6 checklist entries inaccurate | 🟢 Low | Documentation only |

**Additional issues discovered during mapper code review:**
| ID | Issue | Severity | Blocking |
|----|-------|----------|---------|
| G-9 | US-LC-FINS sector_leader actors have constant intensity=1.000 (ρ=0.040 root cause) | 🟡 Medium | Same as G-5 |
| G-10 | High-missing actors: experiment_fast/phased 19/179; US-LC-TECH 11/60; US-SC 48/142 | 🟡 Medium | Actor-level intensity gaps; not critical for aggregate |
| G-11 | `CapitalExpenditures` XBRL tag referenced in plan has 0 filings; compute script must use `PaymentsToAcquirePropertyPlantAndEquipment` — verify this is the case | 🟡 Medium | If script uses wrong tag: all equity intensities are NaN-heavy |
| G-12 | Dual Revenue XBRL tags (pre/post ASC 606): `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax` — verify compute script handles both | 🟢 Low | Small post-2018 gaps if unhandled |
| G-13 | OECD raw file naming: DATA_ACQUISITION.md documents versioned names; actual files lack version suffix | 🟢 Low | Documentation inconsistency |
| G-14 | BEA 2024 I/O table PIT pub_date ≈ mid-2026: not visible for experiments run before July 2026 | 🟢 Low | Informational — 2023 is most recent PIT-available year |

---

## Dependency Graph

```
R1: FRED gap fill (CPIMEDSL)
   └── no dependencies (independent)

R2: OECD re-fetch
   └── no dependencies (independent; overwrites existing shard)

R3: Intensity normalisation + recompute
   ├── depends on: R1 (CPIMEDSL in PIT store for macro actor intensity)
   ├── includes: G-5/G-9 root cause investigation + fix
   └── includes: G-10, G-11, G-12 verification + fixes

R4: UK data pipeline
   ├── R4.0: Decision gate (Companies House vs OHLCV-only)
   └── R4.1: Build or bypass Companies House + compute UK intensities
        └── depends on: R4.0 decision

R5: Full validation sweep
   └── depends on: R1, R2, R3, R4 all complete

R6: Housekeeping (GDELT cleanup + documentation)
   └── no dependencies (run anytime, or last)
```

**Critical path:** R1 → R3 → R5 (for Phase A experiments)
**UK path:** R4.0 → R4.1 → R5 (for Phase E / E1 experiment)
**R2 and R6 can run in parallel with the critical path.**

---

## Milestone R1: FRED Gap Fill

**Goal:** Add `CPIMEDSL` (CPI Medical Care) to the FRED PIT store.

**Scope:** Modify `scripts/smim_fetch_fred.py` to include `CPIMEDSL`, re-run for that series only, and update the PIT store. No need to re-run all 27 existing series.

**Steps:**

1. Read `scripts/smim_fetch_fred.py` to understand `MACRO_SERIES` list and how individual series are fetched and ingested.
2. Add `"CPIMEDSL"` to `MACRO_SERIES` in the script. Its layer designation: `Healthcare sector` (same as `DCOILWTICO` is for Energy sector).
3. Run the script in single-series mode (or add a `--series` flag if one exists, else run full script — existing series are idempotent).
4. Verify `data/smim/raw/fred/CPIMEDSL.parquet` exists.
5. Verify `CPIMEDSL` appears in `data/smim/processed/fred_signals.parquet` with:
   - Date range: at minimum 2000-01 to 2026-01
   - Non-null value count: ≥ 200 observations (monthly since 2000)
6. Verify `CPIMEDSL` appears in `data/smim/pit_store/fred.parquet`.
7. Re-run PIT leak detection to confirm 0 violations.
8. Update `data/smim/DATA_ACQUISITION.md` summary table: FRED status changes from "27/29 series" to "28/29 series".

**Quality Gate R1:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| R1-QG-1 | CPIMEDSL raw file | `data/smim/raw/fred/CPIMEDSL.parquet` exists |
| R1-QG-2 | CPIMEDSL in processed | Signal appears in `fred_signals.parquet`, ≥200 rows, no PIT leaks |
| R1-QG-3 | FRED series count | `fred_signals.parquet` distinct signal_id count = 28 |
| R1-QG-4 | PIT leak check | 0 violations after re-run |

**Git commit:** `[SMIM DATA-R1] Add CPIMEDSL to FRED macro signals`

---

## Milestone R2: OECD Full Re-fetch

**Goal:** Fix `scripts/smim_fetch_oecd.py` to retrieve the full monthly history (2000–2026) for CLI indicators and full quarterly history for QNA. Replace the current sparse PIT store shard.

**Root cause:** The script uses the SDMX 3.0 `all` key
(`/data/{agency},{flow},{version}/all?...`) which the OECD API returns with server-side row limits or pagination that the script does not handle. Result: only a subset of the available observations is returned.

**Steps:**

1. Read `scripts/smim_fetch_oecd.py` fully to understand the current fetch approach.
2. Diagnose by inspecting what the `all` key response actually contains: add a debug print of the response structure (number of series, observation counts before filtering).
3. Test a minimal explicit-key alternative for DF_CLI:
   ```
   https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,4.0/USA+GBR.M.LI+BCICP+CCICP.AA.CTGY.ST?format=jsondata&startPeriod=2000-01&endPeriod=2026-03
   ```
   Confirm this returns ≥600 rows for LI alone (2000-01 to 2026-03 = 314 months × 2 countries = 628 rows).
4. If the explicit key works: refactor the script to build explicit dimension keys per dataflow instead of using `all`. If the key structure varies per dataflow, add a `"key_template"` field to `OECD_DATAFLOWS`.
5. If explicit keys return 422 errors due to dimension ordering: try pagination approach — fetch with `all` key and add `?limit=10000&offset=0` parameters, iterating until empty response.
6. After fix: re-run the full script. The script should overwrite (not append) existing raw parquets and PIT store shard.
7. Verify output volumes:
   - DF_CLI (LI, BCICP, CCICP): expect ≥600 rows per indicator × 2 countries = ≥1,200 rows total
   - DF_QNA (B1GQ_POP): expect ≥100 quarters × 2 countries = ≥200 rows
   - Total `oecd_macro.parquet` rows: ≥1,400 (vs 244 before)
8. Verify date ranges: LI should start 2000-01 and continue to at least 2025-12 for both US and GB.
9. Re-run PIT leak detection.
10. Fix raw file name documentation in `docs/smim/DATA_ACQUISITION.md` §6 (remove version suffix from described filenames to match actual files `DSD_STES_DF_CLI.parquet`, `DSD_NAMAIN1_DF_QNA_EXPENDITURE_CAPITA.parquet`).

**Quality Gate R2:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| R2-QG-1 | Total OECD processed rows | `oecd_macro.parquet` row count ≥ 1,400 |
| R2-QG-2 | LI monthly coverage | LI present for US from 2000-01 to at least 2025-12 (≥300 months) |
| R2-QG-3 | LI monthly coverage (GB) | LI present for GB from at least 2003-01 to 2025-12 (≥270 months) |
| R2-QG-4 | No series stops before 2024 | Max event_date per indicator ≥ 2024-01 for both US and GB |
| R2-QG-5 | PIT leak check | 0 violations |

**Git commit:** `[SMIM DATA-R2] Fix OECD SDMX fetch to retrieve full monthly history`

---

## Milestone R3: Intensity Normalisation Fix + Full Recompute

**Goal:** Fix the root cause of US-LC-FINS Spearman ρ=0.040, verify XBRL tag usage, investigate high-missing actors, and recompute all affected intensity files.

**This is the most complex milestone — three sub-tasks:**

### R3a: Diagnose and fix US-LC-FINS sector_leader constant intensity

**Root cause investigation steps:**
1. Read `scripts/smim_compute_intensities.py` fully to understand how equity intensities are computed.
2. Load the US-LC-FINS registry (`data/smim/registries/US-LC-FINS_registry.json`) and list all actors with `actor_type == SECTOR_LEADER`. Count them (expected: 1–3 actors).
3. Load `data/smim/processed/edgar_balance_sheet.parquet`. Filter to the sector_leader actor IDs. Check their CapEx/Assets time series:
   - Is CapEx (PaymentsToAcquirePropertyPlantAndEquipment) non-null for these actors?
   - What is the CapEx/Assets ratio range and variance?
4. Run `compute_edgar_capex_ratio` for US-LC-FINS equity actors and print the resulting ratio matrix. Confirm which actors always have the highest ratio (producing rank=1.0).
5. Identify the fix: options include
   (a) Compute cross-sectional rank separately per actor_type (SECTOR_LEADER vs LARGE_FIRM vs BANK), so sector_leaders rank against each other
   (b) Use a different normalisation for SECTOR_LEADER (time-series z-score sigmoid like BankCreditMapper) when the cross-section has fewer than 5 actors
   (c) Flag SECTOR_LEADER actors with constant output and fall back to 0.5 (no information)
6. Implement chosen fix in `smim_compute_intensities.py`. Update the per-type handling.
7. Add a test in `tests/unit/smim/` that validates: when CorporateCapexMapper is applied to a cross-section with one actor, the result is 0.5 (not 1.0). This prevents regression.

### R3b: Verify XBRL tag correctness

1. Confirm `smim_compute_intensities.py` uses `PaymentsToAcquirePropertyPlantAndEquipment` (not the legacy `CapitalExpenditures` tag) for equity CapEx.
2. Confirm the script handles dual Revenue tags: `Revenues` AND `RevenueFromContractWithCustomerExcludingAssessedTax` (ASC 606, post-2018). The script should use whichever is non-null for each ticker-period.
3. If either tag lookup is wrong, fix it.

### R3c: Investigate high-missing actors

1. For each universe with high-missing actors, identify WHICH actors have >50% NaN quarters:
   - experiment_fast/phased: 19/179 high-missing
   - US-LC-TECH: 11/60
   - US-SC: 48/142
2. For each high-missing actor: determine if the missing data is due to
   (a) Recent IPO / insufficient EDGAR history (expected — already in manifest dropped list)
   (b) Wrong EDGAR CIK mapping
   (c) XBRL tag not filed by this company
   (d) Bug in the intensity compute script
3. Actors in category (a) should already be in the manifest's `dropped` list — if they're in the registry but not dropped, flag them.
4. Actors in categories (b)-(d) need fixes.

### R3d: Recompute all intensity files

After fixes to the compute script:
1. Recompute intensities for: US-LC-FINS, US-LC, US-LC-TECH, experiment_fast, experiment_phased.
2. Optionally recompute all 12 intensity files for consistency.
3. Run quality checks for all universes.
4. Update `docs/smim/reports/data_readiness.md` with new rank stability numbers.

**Quality Gate R3:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| R3-QG-1 | US-LC-FINS rank stability | Spearman ρ ≥ 0.7 after recompute |
| R3-QG-2 | US-LC rank stability | Spearman ρ ≥ 0.7 after recompute |
| R3-QG-3 | US-LC-TECH rank stability | Spearman ρ ≥ 0.7 (or documented root cause if structural) |
| R3-QG-4 | All intensities in [0,1] | No values outside range for any universe |
| R3-QG-5 | XBRL tag verification | Script confirmed using PaymentsToAcquire… and dual Revenue tags |
| R3-QG-6 | Unit test added | Test for single-actor cross-section → 0.5 (not 1.0) passes in existing `tests/unit/smim/data/test_intensity_mappers.py` |
| R3-QG-7 | experiment_a1_intensities recomputed | Rank stability ρ still ≥ 0.7 after fix (confirm the fix didn't break the A1 registry) |

**Git commit:** `[SMIM DATA-R3] Fix intensity normalisation: sector_leader cross-section, XBRL tags`

---

## Milestone R4: UK Data Pipeline

**Goal:** Compute `UK-LC_intensities.parquet` and `UK-MC_intensities.parquet` to unblock E1 experiment.

### R4.0: Decision Gate

Before writing code, evaluate two paths:

**Path A — Build Companies House adapter:**
- Companies House API: https://developer.company-information.service.gov.uk/
- Free, requires registration for API key
- Provides UK company filings (annual accounts, filing history)
- Data format: JSON with XBRL-like financial statements
- Estimated effort: 2–3 sessions (adapter build + bulk fetch + PIT ingest)
- Result: UK balance-sheet data (CapEx, Assets, Revenue) enabling CorporateCapexMapper

**Path B — OHLCV-only intensity for UK equities:**
- Use market return-based intensity: rolling 12-month total return, cross-sectionally ranked
- Simpler and faster (1 session)
- Weaker signal: market returns reflect investment expectations, not actual investment flows
- Must be documented clearly as `methodology = "return_xsrank"` (different from US `capex_assets_xsrank`)
- Does NOT violate A2 (typed comparability) as long as UK equities are normalised among themselves

**Recommendation:** Start with Path B to unblock E1, document the limitation, and plan Path A as a future enhancement. The research paper should disclose the UK intensity methodology difference.

### R4.1: Implement chosen path

**If Path B (recommended for unblocking E1):**

1. Add a `compute_ohlcv_return_intensities()` function to `smim_compute_intensities.py`:
   - Load OHLCV parquet for UK-LC and UK-MC
   - Compute rolling 12-month total return per ticker at each quarter end
   - Apply cross-sectional percentile rank across the universe
   - Output intensity in [0,1] with `methodology = "return_xsrank"`
2. Add UK-LC and UK-MC to the universe list processed by `smim_compute_intensities.py`.
3. Run compute for UK-LC and UK-MC.
4. Verify `data/smim/intensities/UK-LC_intensities.parquet` and `UK-MC_intensities.parquet` exist.
5. Run quality checks: [0,1] range, Spearman ρ, missing actor count.
6. Document methodology difference in `docs/smim/DATA_ACQUISITION.md`.

**If Path A (Companies House adapter):**

1. Register at https://developer.company-information.service.gov.uk/ to obtain API key.
2. Build `src/quantdsl_backtest/smim/data/adapters/companies_house.py` following the FRED adapter pattern (read `ADAPTER_GUIDE.md` first).
3. Create `scripts/smim_fetch_companies_house.py` to bulk-fetch UK equity balance-sheet data.
4. Fetch for UK-LC and UK-MC tickers (with `.L` suffix stripped for Companies House lookup by name).
5. Ingest into PIT store as a new shard (`data/smim/pit_store/companies_house.parquet`).
6. Then compute UK intensities using the same CorporateCapexMapper as US equities.

**Quality Gate R4:**

| Check | Criterion | Pass when |
|-------|-----------|-----------|
| R4-QG-1 | UK-LC intensity file | `data/smim/intensities/UK-LC_intensities.parquet` exists |
| R4-QG-2 | UK-MC intensity file | `data/smim/intensities/UK-MC_intensities.parquet` exists |
| R4-QG-3 | UK-LC range | All values in [0,1] |
| R4-QG-4 | UK-LC rank stability | Spearman ρ ≥ 0.5 (lower threshold acceptable for Path B given return-based method) |
| R4-QG-5 | UK-LC actor count | N ≥ 80 (at least 80 UK-LC actors with non-missing intensity) |
| R4-QG-6 | Methodology documented | `methodology` column in intensity parquet; DATA_ACQUISITION.md updated |

**Git commit:** `[SMIM DATA-R4] Compute UK-LC and UK-MC investment intensities`

---

## Milestone R5: Full Validation Sweep

**Goal:** Re-run all data quality checks and produce updated audit and readiness reports. Confirm all Gate G1 items pass.

**Steps:**

1. Run PIT leak detection across all shards: `uv run python -m quantdsl_backtest.smim.data.quality_checks`
2. Re-run `scripts/smim_data_audit.py` (if it exists) or manually check each source.
3. Run intensity quality checks for all 13 universe files (including UK-LC, UK-MC added in R4).
4. Update `docs/smim/reports/data_audit.md`:
   - Update OECD section with new row counts and date ranges (after R2)
   - Update FRED section: 28/29 series (after R1)
   - Update G1-6 (OECD) to PASS if R2 succeeded
   - Update G1-9 (UK intensities) to PASS if R4 succeeded
   - Update G1-10 (rank stability) with new ρ values (after R3)
5. Update `docs/smim/reports/data_readiness.md`:
   - Update all intensity quality sections with new ρ values
   - Change E1 (UK-LC) intensity from MISS to OK (if R4 succeeded)
   - Update OECD column from WARN to OK (if R2 succeeded)
6. Run a final coherence check: re-read EXPERIMENT_PLAN.md and verify data_readiness matrix covers all planned experiments.

**Quality Gate R5 (Full Gate G1 Recheck):**

| Check | Criterion | Required result |
|-------|-----------|-----------------|
| R5-G1-1 | PIT leak detection | 0 violations |
| R5-G1-2 | FRED series count | 28/29 series (93%) |
| R5-G1-3 | EDGAR US tickers | 765/772 (99%) |
| R5-G1-4 | GDELT continuity | 9 signals, no >4-week gaps |
| R5-G1-5 | IMF country coverage | 7 indicators; 3 cover US+GB+DE+JP; 4 cover US+GB only |
| R5-G1-6 | OECD monthly history | LI ≥300 months for US; all series through at least 2025 |
| R5-G1-7 | BEA sector coverage | 5 sectors, 2010–2023 PIT-available |
| R5-G1-8 | OHLCV coverage | ≥80% Gold/Silver for ≥80% of universes |
| R5-G1-9 | UK intensities | UK-LC and UK-MC intensity files present |
| R5-G1-10 | Rank stability | ρ ≥ 0.7 for US-LC-FINS, US-LC, US-LC-TECH |

**Git commit:** `[SMIM DATA-R5] Post-remediation data audit and readiness reports`

---

## Milestone R6: Housekeeping

**Goal:** Clean up stale GDELT artifacts, fix documentation inconsistencies, update EXPERIMENT_PLAN.md estimates.

**Steps (can be done in any order, or bundled with another milestone):**

1. Archive GDELT Gen 1 and Gen 2 raw files:
   - Move `data/smim/raw/gdelt/gkg_weekly/` (516 files) to `data/smim/raw/gdelt/archive/gkg_weekly/`
   - Move `data/smim/raw/gdelt/docapi_v2/` (119 files) to `data/smim/raw/gdelt/archive/docapi_v2/`
   - Move root-level consolidated files (actor_FED_vol.parquet, etc.) to `data/smim/raw/gdelt/archive/docapi_v2_consolidated/`
   - Note: these files are gitignored — no git tracking needed
2. Update `docs/smim/EXPERIMENT_PLAN.md` universe size table to reflect actual GICS counts:
   - US-LC-ENERGY: 22 (not 40)
   - US-LC-TECH: 68 (not 60)
   - US-LC-FINS: 74 (not 50)
   - US-LC-HEALTH: 60 (not 45)
   - US-LC-INDUS: 78 (not 50)
   - MIXED-200: ~27 equity + institutional (clarify composition)
3. Fix `docs/smim/DATA_ACQUISITION.md` §6 OECD raw file naming (remove version suffixes from documented filenames).
4. Add G-14 (BEA 2024 pub_date visibility) note to `docs/smim/DATA_ACQUISITION.md` §7 BEA section.
5. Mark Remediation items in `docs/smim/DATA_ACQUISITION.md` as ✅ or ❌ based on status after R1-R4.

**Quality Gate R6:**

| Check | Pass when |
|-------|-----------|
| R6-QG-1 | GDELT archive | `data/smim/raw/gdelt/archive/` directory exists; root-level raw/gdelt has only canonical approach artifacts |
| R6-QG-2 | EXPERIMENT_PLAN.md | Universe N values match actual GICS counts |
| R6-QG-3 | DATA_ACQUISITION.md | OECD file names correct; remediation items marked with current status |

**Git commit:** `[SMIM DATA-R6] Housekeeping: GDELT archive, documentation corrections`

---

## Execution Order Summary

```
Week 1:
  Session 1 (R1):  FRED CPIMEDSL — ~1 hour, independent
  Session 2 (R2):  OECD re-fetch — ~2-3 hours (diagnosis + fix + run)
  Session 3 (R6):  Housekeeping — ~30 minutes, independent

Week 2:
  Session 4 (R3):  Intensity fix + recompute — ~3-4 hours
                   Depends on R1 complete for accurate macro intensities

Week 3:
  Session 5 (R4):  UK intensities — ~2-3 hours (Path B) or ~8 hours (Path A)

Week 4:
  Session 6 (R5):  Full validation sweep + updated reports — ~1-2 hours
                   Depends on R1, R2, R3, R4 complete
```

**Minimum viable path for Phase A experiments (MIXED-200 energy):**
- R1 (CPIMEDSL) → R3 (intensity fix) → partial R5 (validate MIXED-200 + experiment_a1 only)
- R2 (OECD) can wait — MIXED-200 energy experiments don't critically depend on OECD signals
- R4 (UK) can wait — E1 is Phase C, not Phase A

---

## Status Tracking

| Milestone | Status | Completed | Notes |
|-----------|--------|-----------|-------|
| R1: FRED CPIMEDSL | ✅ Complete | 2026-03-28 | CPIMEDSL 314 rows; NAPM removed; 28/28 signals in PIT |
| R2: OECD re-fetch | ✅ Complete | 2026-03-28 | 1,922 rows; explicit key METHODOLOGY=H fix; G1-6 passes |
| R3: Intensity fix | ⚠️ Partial | 2026-03-28 | R3a done: z-score sigmoid fallback for constant columns; sector_leader ALL fixed. ρ still -0.003 — structural BankCreditMapper issue |
| R4: UK intensities | ✅ Complete | 2026-03-28 | Path B (OHLCV return_12m_xsrank): UK-LC ρ=0.732 PASS; UK-MC ρ=0.720 PASS; E1 unblocked |
| R5: Full validation | ⬜ Pending | — | Depends on R1–R4 |
| R6: Housekeeping | ⬜ Pending | — | Independent |
