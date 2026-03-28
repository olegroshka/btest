# SMIM Data Readiness Report

> Generated: 2026-03-28 (post R1/R2/R3a remediation)
> Based on: data_audit.md findings; intensities recomputed after R3a fix

---

## Experiment Readiness Matrix

Key:
- `OK` — required data present and fit for use
- `WARN` — data present but quality issues identified; usable with caveats
- `MISS` — data required but absent; experiment blocked or significantly degraded
- `N/A` — this source not required for this experiment

| Experiment | Registry | Intensity | FRED | EDGAR | GDELT | BEA | IMF | OECD |
|------------|----------|-----------|------|-------|-------|-----|-----|------|
| A1 (MIXED-200 energy, full) | OK | OK | OK | OK | OK | OK | OK | OK |
| A2 (US-LC energy sector) | OK | OK | OK | OK | OK | OK | OK | OK |
| B1 (US-LC financials) | OK | ⚠️ WARN | OK | OK | OK | OK | OK | OK |
| C1 (US-LC all sectors) | OK | ⚠️ WARN | OK | OK | OK | OK | OK | OK |
| D1 (US-LC fast run) | OK | ⚠️ WARN | OK | OK | OK | OK | OK | OK |
| E1 (UK-LC) | OK | ❌ MISS | OK | ❌ MISS | OK | N/A | OK | OK |

### Column notes

**OECD OK (all experiments):** Re-fetched 2026-03-28 (R2: explicit dimension key fix). LI/BCICP/CCICP: 289 monthly rows per country, 2000-01 to 2024-01. B1GQ_POP: ~100 quarterly rows per country. OECD signals are now fit for use.

**Intensity WARN (B1, C1, D1):** US-LC-FINS rank stability is ρ=-0.003 (structural: BankCreditMapper uses per-actor temporal z-score sigmoid, producing near-random cross-sectional rankings). US-LC and US-LC-TECH are also below threshold (0.660 and 0.653). See detailed notes in Intensity Quality Summary below.

**E1 EDGAR MISS:** UK equities do not file with SEC EDGAR. Companies House adapter was never built. UK equities have no balance-sheet coverage.

**E1 Intensity MISS:** `UK-LC_intensities.parquet` does not exist. E1 experiment blocked until R4 (UK intensities via OHLCV return-based proxy) is complete.

---

## Intensity Quality Summary

### experiment_a1 (N=23, 1,445 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.844 — PASS (above 0.7 threshold)
- High-missing actors (>50%): 1 (acceptable)

| ActorType | N obs | Mean | Std | Skew | NaN% |
|-----------|-------|------|-----|------|------|
| central_bank | 88 | 0.414 | 0.282 | 0.689 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.106 | 0.203 | 2.866 | 0.00% |
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |
| regulator | 44 | 0.220 | 0.267 | 1.422 | 0.00% |

**Note:** experiment_a1 uses INST-MINIMAL institutional actors (Fed + BoE + IMF) combined with MIXED-200 equity actors. Total N=23 is the clean subset after dropping EXE (insufficient OHLCV history) and 1 high-missing actor. The full MIXED-200 equity list has 27 tickers.

---

### experiment_fast / experiment_phased (N=179, 10,542 obs each)
**Status: READY with rank stability warning** ⚠️
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.711 / 0.712 — PASS (marginally above 0.7)
- High-missing actors (>50%): 19

The high-missing count (19 out of 179) warrants investigation. These are likely actors added to the registry that lack adequate data coverage (see registry validation below).

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| bank | 2,536 | 0.488 | 0.160 | 0.685 |
| central_bank | 44–88 | 0.371–0.414 | 0.267–0.282 | 0.689–0.807 |
| global_shock | 584 | 0.349 | 0.288 | 0.693 |
| intl_org | 44 | 0.106 | 0.203 | 2.866 |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 |

---

### MIXED-200 (N=12, 685 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.759 — PASS
- High-missing actors (>50%): 1

Note: N=12 is the equity-only component (large_firm actors), not the full experiment_a1 registry which adds institutional actors.

---

### US-LC-ENERGY (N=12, 685 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.759 — PASS
- High-missing actors (>50%): 1

---

### US-LC-FINS (N=70, 4,215 obs)
**Status: REQUIRES INVESTIGATION** ❌
- Range [0,1]: PASS
- Rank stability (Spearman ρ): **-0.003 — CRITICAL FAIL** (threshold 0.7)
- High-missing actors (>50%): 2

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| bank | 4,150 | 0.491 | 0.169 | 0.408 |
| sector_leader | 65 | 0.490 | 0.208 | 0.494 |

**R3a fix applied (2026-03-28):** The degenerate constant sector_leader `ALL` (which previously had intensity=1.000, std=0.000 due to being the sole actor in the equity cross-section) now receives z-score sigmoid fallback. Sector_leader `AFL` has 0 rows — no EDGAR data found for Aflac Inc.

**Root cause of remaining ρ failure:** The bank-only rank stability for US-LC-FINS is ρ=-0.007 (nearly zero). `BankCreditMapper` applies z-score sigmoid of per-actor asset growth rate — a temporal (per-actor) normalisation that produces values centered around 0.5 for each bank, but the relative cross-sectional ordering of banks changes randomly each quarter (mean-reverting growth makes high-growth banks switch with low-growth banks). This is a structural incompatibility between per-actor temporal normalisation and cross-sectional rank stability metric.

**Action required:** Computing cross-sectional Spearman ρ across actor types with different normalisation methods (BankCreditMapper vs CorporateCapexMapper) is inherently unstable. Consider either (a) using the same normalisation method for all actor types in US-LC-FINS, or (b) computing rank stability within actor type separately. Do not use US-LC-FINS intensity in experiments until further investigation.

---

### US-LC-HEALTH (N=51, 2,899 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.708 — PASS (marginally)
- High-missing actors (>50%): 6

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| large_firm | 2,782 | 0.514 | 0.290 | -0.001 |
| sector_leader | 117 | 0.438 | 0.249 | -0.278 |

Note: CPIMEDSL (healthcare CPI proxy) — **now present in FRED PIT store** (R1 remediation 2026-03-28). 314 rows, 2000–2026. Healthcare experiments using MACRO-ONLY feeds now have this signal.

---

### US-LC-INDUS (N=59, 3,536 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.733 — PASS
- High-missing actors (>50%): 3

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| large_firm | 3,470 | 0.506 | 0.290 | 0.029 |
| sector_leader | 66 | 0.708 | 0.095 | -0.796 |

---

### US-LC-TECH (N=60, 3,196 obs)
**Status: BORDERLINE — monitor** ⚠️
- Range [0,1]: PASS
- Rank stability (Spearman ρ): **0.653 — WARN** (below 0.7 threshold)
- High-missing actors (>50%): 11

High-missing count (11/60) suggests actors with insufficient EDGAR history. These are likely recent-IPO tech companies. Consider tightening the actor selection to post-2010 EDGAR coverage minimum.

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| large_firm | 3,064 | 0.516 | 0.290 | -0.025 |
| sector_leader | 132 | 0.382 | 0.211 | 0.122 |

---

### US-LC (N=169, 9,826 obs)
**Status: BORDERLINE — monitor** ⚠️
- Range [0,1]: PASS
- Rank stability (Spearman ρ): **0.660 — WARN** (below 0.7 threshold)
- High-missing actors (>50%): 13

The US-LC rank stability issue is inherited from the Financials sector composition within US-LC (which includes US-LC-FINS actors). See US-LC-FINS structural root cause above.

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| bank | 2,536 | 0.488 | 0.160 | 0.685 |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 |

---

### US-MC (N=159, 8,461 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.794 — PASS
- High-missing actors (>50%): 22

22 high-missing actors out of 159 (14%) is elevated. These are likely recent-IPO mid-cap names included in universe but lacking full EDGAR history.

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| bank | 777 | 0.494 | 0.175 | 0.334 |
| large_firm | 3,018 | 0.570 | 0.253 | -0.056 |
| retail_investor | 1,798 | 0.422 | 0.291 | 0.252 |
| sector_leader | 413 | 0.652 | 0.274 | -0.463 |
| sme | 2,455 | 0.459 | 0.303 | 0.163 |

---

### US-SC (N=142, 6,215 obs)
**Status: READY** ✅
- Range [0,1]: PASS
- Rank stability (Spearman ρ): 0.905 — PASS (best among all universes)
- High-missing actors (>50%): 48

48 high-missing actors (34%) is high but expected — US-SC contains many small, recently-listed companies with limited EDGAR history. The 142 active actors (out of 200) provide adequate diversity. Rank stability is strong.

| ActorType | N obs | Mean | Std | Skew |
|-----------|-------|------|-----|------|
| retail_investor | 2,739 | 0.454 | 0.332 | 0.333 |
| sme | 3,476 | 0.546 | 0.242 | -0.227 |

---

### UK-LC (N=—, — obs)
**Status: BLOCKED** ❌

`data/smim/intensities/UK-LC_intensities.parquet` does not exist. Intensity computation was not run for UK-LC. Root causes:
1. No Companies House balance-sheet data available (adapter not built)
2. `smim_compute_intensities.py` was not executed for UK universes (no EDGAR coverage → 0 rows)

Intensity could still be computed using OHLCV-derived metrics (rolling 12-month return cross-section rank) without balance-sheet data — this is a weaker but valid approach. See R4 milestone.

---

### UK-MC (N=—, — obs)
**Status: BLOCKED** ❌

Same as UK-LC. `UK-MC_intensities.parquet` does not exist.

---

## Registry Validation Summary

All 14 registry JSON files exist in `data/smim/registries/`:
- Universe registries (11): MIXED-200, UK-LC, UK-MC, US-LC, US-LC-ENERGY, US-LC-FINS, US-LC-HEALTH, US-LC-INDUS, US-LC-TECH, US-MC, US-SC ✅
- Experiment registries (3): experiment_a1, experiment_fast, experiment_phased ✅

**Known issue:** UK-LC and UK-MC registries exist but have 0% external ID resolution for EDGAR (CIK). The `actor_id` for UK equities will not resolve to PIT store EDGAR records. This is expected but should be documented in the registry metadata.

---

## Pre-Experiment Quality Gate (Post-Remediation)

| Check | Result |
|-------|--------|
| All intensity values in [0,1] | ✅ PASS (all computed universes) |
| Rank stability ρ > 0.7 for all computed universes | ❌ FAIL — US-LC-FINS (-0.003 structural), US-LC-TECH (0.653), US-LC (0.660) |
| UK intensities computed | ❌ FAIL — UK-LC and UK-MC missing (R4 pending) |
| PIT leak detection | ✅ PASS — 0 leaks / 543,768 rows |
| FRED: 28/28 signals | ✅ PASS — CPIMEDSL added (R1 2026-03-28) |
| OECD data fit for use | ✅ PASS — 1,922 rows with full history (R2 2026-03-28) |
| US-LC-FINS sector_leader constant intensity | ✅ FIXED — z-score sigmoid fallback applied (R3a 2026-03-28) |

### Overall: ⚠️ CONDITIONALLY READY — Phase A (MIXED-200 energy) can proceed; Phase B/C/D need rank stability investigation; Phase E (UK) blocked

**Experiments that CAN proceed now:**
- A1 (MIXED-200 energy, full pipeline) — FRED+EDGAR+GDELT+BEA+OECD+IMF all usable
- A2 (US-LC energy sector)

**Experiments that MUST WAIT:**
- E1 (UK-LC) — blocked on UK intensities (R4)
- B1/C1/D1 — flagged WARN due to US-LC-FINS/US-LC/US-LC-TECH rank stability; usable with caveat that cross-sectional ranking across actor types is structurally unstable for banks

**Recommended next actions:**
1. R4: Compute UK intensities (rolling 12-month return xsrank) to unblock E1
2. Investigate BankCreditMapper cross-sectional rank stability structural issue for US-LC-FINS
3. ~~R1: Fetch CPIMEDSL~~ ✅ Done 2026-03-28
4. ~~R2: Fix OECD SDMX fetch~~ ✅ Done 2026-03-28
5. ~~R3a: Fix sector_leader constant intensity~~ ✅ Done 2026-03-28
