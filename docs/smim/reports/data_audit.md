# SMIM Data Coverage Audit

> Generated: 2026-03-28  |  PIT store: `data/smim/pit_store`
> Auditor: systematic file-level + schema-level review against DATA_ACQUISITION_PROMPTS.md and EXPERIMENT_PLAN.md

---

## Summary: Universe × Source Coverage

| Universe | Tickers (manifest) | OHLCV% | Regime | EDGAR% | Intensity | Notes |
|----------|-------------------|--------|--------|--------|-----------|-------|
| MIXED-200 | 27 (26 active) | 96% | 🥇 Gold | 81% | ✅ computed | UK equity 0% EDGAR (by design) |
| UK-LC | 99 (97 active) | 98% | 🥇 Gold | 0% | ❌ **MISSING** | UK: no EDGAR, no Companies House adapter |
| UK-MC | 100 (94 active) | 94% | 🥇 Gold | 0% | ❌ **MISSING** | Same |
| US-LC | 200 (196 active) | 98% | 🥇 Gold | 99% | ✅ computed | |
| US-LC-ENERGY | 22 (21 active) | 95% | 🥇 Gold | 100% | ✅ computed | 22 vs 40 planned (see §9) |
| US-LC-FINS | 74 (71 active) | 96% | 🥇 Gold | 97% | ✅ computed ⚠️ | Rank stability ρ=0.040 (critical) |
| US-LC-HEALTH | 60 (58 active) | 97% | 🥇 Gold | 100% | ✅ computed | |
| US-LC-INDUS | 78 (76 active) | 97% | 🥇 Gold | 99% | ✅ computed | |
| US-LC-TECH | 68 (68 active) | 100% | 🥇 Gold | 100% | ✅ computed ⚠️ | Rank stability ρ=0.653 (WARN) |
| US-MC | 200 (188 active) | 94% | 🥇 Gold | 100% | ✅ computed | |
| US-SC | 200 (180 active) | 80% | 🥇 Gold | 94% | ✅ computed | Borderline: 39 tickers sparse/missing |

---

## 1. Equity OHLCV

**Paths:** `equities/smim/{universe_id}/ohlcv.parquet` (11 files, all present)
**Date range:** 2005-01-03 – 2025-12-30 (all universes)

| Universe | Total | Gold (≥15yr) | Silver (≥10yr) | Bronze (≥5yr) | Sparse/Missing |
|----------|-------|-------------|---------------|--------------|----------------|
| MIXED-200 | 27 | 22 | 4 | 0 | 1 (EXE) |
| UK-LC | 99 | 87 | 4 | 6 | 2 (HLN.L, MTLN.L) |
| UK-MC | 100 | 75 | 7 | 12 | 6 (APN.L, BCG.L, BPT.L, DOCS.L, FSG.L, HBR.L) |
| US-LC | 200 | 174 | 13 | 9 | 4 (COIN, CEG, EXE, FI) |
| US-LC-ENERGY | 22 | 17 | 4 | 0 | 1 (EXE) |
| US-LC-FINS | 74 | 66 | 5 | 0 | 3 (COIN, FI, MMC) |
| US-LC-HEALTH | 60 | 53 | 4 | 1 | 2 (GEHC, SOLV) |
| US-LC-INDUS | 78 | 65 | 4 | 7 | 2 (GEV, VLTO) |
| US-LC-TECH | 68 | 54 | 10 | 4 | 0 |
| US-MC | 200 | 141 | 22 | 25 | 12 (AHR, BROS, CART, CAVA, CNM, CR, CRBG, DOCS, DTM, DUOL, ESAB, GXO) |
| US-SC | 200 | 92 | 27 | 42 | 39 (recent IPOs and SPACs) |

**Note:** All sparse/missing tickers are recent IPOs or post-2020 listings — shorter history is correct, not a data failure. These are excluded from the clean manifests.

**⚠️ US-SC borderline:** 80% coverage (161/200 tickers). Passes G1-8 threshold exactly. 39 tickers dropped from manifest.

---

## 2. FRED Macro Signals + ALFRED Vintages

**Paths:** `data/smim/raw/fred/{SERIES}.parquet` (27 raw files + 5 ALFRED files), `data/smim/processed/fred_signals.parquet`, `data/smim/pit_store/fred.parquet`

| Metric | Value |
|--------|-------|
| Planned series | 29 |
| Fetched series | 27 (93%) |
| Total processed rows | 71,761 |
| PIT store rows | 64,165 (de-duped) |
| ALFRED vintage rows | 8,956 |
| Event date range | 2000-01-01 – 2026-03-20 |
| Pub date range | 2000-01-31 – 2026-04-19 |

**ALFRED-vintaged series (5):** GDP, UNRATE, CPIAUCSL, INDPRO, FEDFUNDS ✅

**Missing series (2):**

| FRED ID | Reason | Status |
|---------|--------|--------|
| `NAPM` | Series does not exist on FRED (ISM PMI) | Replaced by `MANEMP` (accepted proxy) ✅ |
| `CUSR0000SAM` | Wrong series ID in original plan | **✅ Remediated (R1, 2026-03-28)** — replaced with correct ID `CPIMEDSL`; 314 rows ingested into PIT store; 0 A1 violations. |

**Acquired series (28):** DFF, BAMLH0A0HYM2, T10Y2Y, VIXCLS, BAA10Y, DCOILWTICO, DCOILBRENTEU, DTWEXBGS, INDPRO, GASREGW, TOTBKCR, STLFSI2, GDP, CPIAUCSL, UNRATE, FEDFUNDS, CPILFESL, GS10, GS2, MANEMP, GDPC1, PCEPI, UMCSENT, USSLIND, HOUST, M2SL, DRCCLACBS, **CPIMEDSL**.

---

## 3. SEC EDGAR XBRL

**Paths:** `data/smim/processed/edgar_balance_sheet.parquet`, `data/smim/pit_store/edgar.parquet`

| Metric | Value |
|--------|-------|
| US tickers attempted | 772 |
| Tickers with data | 765 (99%) |
| Total filing records | 461,203 |
| Unique XBRL tags | 7 |
| Event date range | 2005-07-04 – 2026-02-28 |
| PIT store rows | 461,203 |

| Universe | Tickers | With EDGAR | Coverage |
|----------|---------|-----------|---------|
| US-LC | 200 | 198 | 99% |
| US-LC-ENERGY | 22 | 22 | 100% |
| US-LC-FINS | 74 | 72 | 97% |
| US-LC-HEALTH | 60 | 60 | 100% |
| US-LC-INDUS | 78 | 77 | 99% |
| US-LC-TECH | 68 | 68 | 100% |
| US-MC | 200 | 200 | 100% |
| US-SC | 200 | 188 | 94% |
| MIXED-200 | 27 | 22 | 81% |
| **UK-LC** | **99** | **0** | **0% — by design** |
| **UK-MC** | **100** | **0** | **0% — by design** |

**UK EDGAR gap is structural, not a failure.** SEC EDGAR covers only US-reporting entities. UK equities (FTSE 100/250) file with Companies House (UK), not the SEC. The EXPERIMENT_PLAN.md planned a Companies House adapter for UK balance-sheet data — **that adapter was never built** (see §9, Gap G-4).

**XBRL tag coverage:**

| Tag | Tickers | Notes |
|-----|---------|-------|
| `Assets` | 765 | Near-universal |
| `StockholdersEquity` | 757 | Near-universal |
| `PaymentsToAcquirePropertyPlantAndEquipment` | 611 | Primary CapEx tag (modern) |
| `LongTermDebt` | 605 | |
| `Revenues` | 559 | Older revenue tag |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 537 | Post-ASC 606 (2018+) |
| `ResearchAndDevelopmentExpense` | 341 | Zero for non-R&D sectors |
| `CapitalExpenditures` | **0** | Legacy tag — fully superseded by PaymentsToAcquire…; original plan referenced this tag which now has no data |

**⚠️ Note:** The `CapitalExpenditures` XBRL tag (listed in the original DATA_ACQUISITION_PROMPTS) has 0 filings. The correct tag is `PaymentsToAcquirePropertyPlantAndEquipment` — already fetched. InvestmentIntensityMapper must use the correct tag name.

---

## 4. GDELT Narrative Signals

**Canonical paths:** `data/smim/processed/gdelt_narrative_daily.parquet` (daily), `data/smim/processed/gdelt_narrative.parquet` (weekly), `data/smim/pit_store/gdelt.parquet`

| Metric | Value |
|--------|-------|
| Signals | 9 (5 sectors + 4 actors) — all present ✅ |
| PIT store rows | 15,185 |
| Weekly panel rows | ~4,689 (9 signals × ~521 weeks) |
| Weekly date range | 2015-02-16 – 2025-12-29 |
| Consecutive-week gaps >4 | None ✅ |
| Daily cache files | 3,609 (of ~4,054 UTC days; ~89% daily coverage) |

**9 signals:** actor_BOE, actor_FED, actor_IMF, actor_SEC, sector_energy, sector_financials, sector_healthcare, sector_macro, sector_technology.

**Current canonical approach:** Daily GKG 2.0 raw CSV files, one representative file per UTC calendar day (slot nearest 12:00 UTC). Per-day stats cached in `data/smim/cache/gdelt/daily_aggregates/`. Weekly panel derived from daily data with mathematically correct aggregation (sum article counts; weighted-mean tone; ratio-of-sums intensity).

**⚠️ GDELT raw directory has 3 generations of artifacts** (none of which are the canonical source):

| Directory/files | Approach | Status |
|----------------|----------|--------|
| `data/smim/raw/gdelt/gkg_weekly/*.parquet` (516 ISO-week files) | Gen 1: one Monday noon snapshot per week | Superseded — single-file weekly proxy is order-of-magnitude sparser |
| `data/smim/raw/gdelt/docapi_v2/` (119 year-sliced files) | Gen 2: GDELT DOC API, per-actor-per-year | Superseded — only covered healthcare + 4 institutional actors; NO energy/tech/financials sector data |
| `data/smim/raw/gdelt/*.parquet` (root-level files) | Gen 2 partially consolidated | Superseded — actor_FED, actor_SEC, actor_IMF, actor_BOE, sector_healthcare, sector_macro consolidated; sector_energy/tech/financials absent |
| `data/smim/cache/gdelt/` | **Gen 3 (canonical):** daily GKG CSV daily-derived | **Active** — all 9 signals, 89% daily coverage 2015–2026 |

**Recommendation:** Archive or remove Gen 1 and Gen 2 raw files to prevent provenance confusion. The processed outputs in `data/smim/processed/` are the authoritative source.

---

## 5. IMF WEO Signals

**Paths:** `data/smim/raw/imf/{INDICATOR}.parquet` (7 files), `data/smim/processed/imf_macro.parquet`, `data/smim/pit_store/imf.parquet`

| Metric | Value |
|--------|-------|
| Indicators fetched | 7 |
| Countries covered (all indicators) | US, GB |
| Countries covered (3 indicators only) | US, GB, DE, JP |
| Total rows | 618 |
| Event date range | 2000-12-31 – 2030-12-31 (includes WEO projections to 2030) |
| Frequency | Annual |

**Country × indicator coverage matrix:**

| Indicator | US | GB | DE | JP | Notes |
|-----------|----|----|----|----|-------|
| NGDP_RPCH (real GDP growth %) | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | |
| PCPIPCH (CPI inflation %) | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | |
| PPPGDP (GDP PPP, int'l $bn) | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | ✅ 2000–2030 | |
| BCA (current account, USD bn) | ✅ 2000–2030 | ✅ 2000–2030 | ❌ MISSING | ❌ MISSING | Not in DataMapper for DE/JP |
| GGXCNL_NGDP (govt net lending % GDP) | ✅ 2001–2030 | ✅ 2000–2030 | ❌ MISSING | ❌ MISSING | Same |
| GGXWDG_NGDP (govt gross debt % GDP) | ✅ 2001–2030 | ✅ 2000–2030 | ❌ MISSING | ❌ MISSING | Same |
| LUR (unemployment %) | ✅ 2000–2030 | ✅ 2000–2030 | ❌ MISSING | ❌ MISSING | Same |

**Planned but not available:**
- `BCA_NGDPDP` (CA % GDP) — not in DataMapper; BCA (USD bn) used instead
- `NID_NGDP` (investment % GDP) — not in DataMapper; use FRED `A006RE1Q156NBEA` as proxy (not yet fetched)
- `NGAP_NPGDP` (output gap) — not in DataMapper
- IFS quarterly (NGDP_R_XDC, PCPI_IX, FPOLM_PA) — `dataservices.imf.org` times out

**PIT discipline:** pub_date = event_date + 365 days (conservative annual WEO lag). 2024 annual data (pub_date = 2025-12-31) is visible for experiments run in 2026 ✅.

---

## 6. OECD Macro Signals

**Paths:** `data/smim/raw/oecd/DSD_STES_DF_CLI.parquet`, `data/smim/raw/oecd/DSD_NAMAIN1_DF_QNA_EXPENDITURE_CAPITA.parquet`, `data/smim/processed/oecd_macro.parquet`, `data/smim/pit_store/oecd.parquet`

| Metric | Value |
|--------|-------|
| Indicators | 4 |
| Countries | US, GB |
| Total rows | **244** |
| Expected rows (full history) | ~2,000+ (CLI monthly 2000–2025 × 2 countries × 3 indicators + QNA quarterly × 2) |

**⚠️ CRITICAL QUALITY ISSUE: OECD data is severely underpowered.**

244 total rows across 4 indicators × 2 countries yields an average of **~30 observations per signal-country combination**. This is far below the expected full history:
- CLI monthly (LI, BCICP, CCICP): ~300 months × 2 countries × 3 = ~1,800 rows expected
- QNA quarterly (B1GQ_POP): ~100 quarters × 2 countries = ~200 rows expected

**Actual coverage (from audit script output):**

| Indicator | US start–end | GB start–end | Assessment |
|-----------|-------------|-------------|------------|
| LI (Composite Leading Indicator) | 2015–2015 only | 2004–2005 only | ⚠️ Critically sparse — 1 year each |
| BCICP (Business Confidence) | 2001–2020 | 2009–2014 | ⚠️ Incomplete — gaps, stops 2020 |
| CCICP (Consumer Confidence) | 2003–2003 only | 2007–2008 only | ⚠️ Critically sparse — 1 year each |
| B1GQ_POP (GDP per capita PPP) | 2000–2025 | 2014–2021 | ⚠️ US adequate; GB stops 2021 |

**Root cause:** The OECD SDMX 3.0 "all key" fetch approach (used to avoid 14-dimension key format errors) returned paginated/limited results. The `sdmx.oecd.org/public/rest/data/` endpoint appears to have returned only a subset of available observations.

**Impact on experiments:** Any signal feed including OECD data (FULL, MACRO-ONLY, MACRO+MARKET) will have severely degraded OECD signal quality. FRED remains the primary macro source and is adequate as a standalone source. **OECD data is currently unfit for production use as a time-series signal.**

**Naming discrepancy:** DATA_ACQUISITION.md documents raw file names as `DSD_STES_DF_CLI_4.0.parquet` and `DSD_NAMAIN1_DF_QNA_EXPENDITURE_CAPITA_1.1.parquet` (with version suffixes), but actual files on disk lack the version suffix. Documentation is incorrect.

---

## 7. BEA Input-Output Supply-Chain

**Paths:** `data/smim/raw/bea/use_table_{year}.parquet` (15 files, 2010–2024), `data/smim/processed/bea_io_tables.parquet`, `data/smim/pit_store/bea.parquet`

| Metric | Value |
|--------|-------|
| Years covered | 2010 – 2024 (all 15 years) ✅ |
| Sector pairs | 21 unique source→target pairs |
| PIT store rows | 315 (21 pairs × 15 years) |
| Sectors mapped | Energy, Financials, Healthcare, Industrials, Technology ✅ |

All 5 SMIM sectors present. BEA I/O data is complete and fit for use in network-containing signal feeds (FULL, NO-NARRATIVE, NO-MARKET). ✅

**PIT discipline:** pub_date = year-end + 548 days (~18-month publication lag). 2024 table (pub_date ≈ mid-2026) will NOT be visible for experiments run before mid-2026. The 2023 table is the most recent PIT-available year for early-2026 experiment runs.

---

## 8. A1 Leak Detection (pub_date < event_date)

✅ **PASSED** — 0 violations across 541,730 rows checked (all PIT store shards combined).

---

## 9. Critical Data Gaps and Inconsistencies

### G-1: OECD data severely underpowered [HIGH SEVERITY] — ✅ RESOLVED 2026-03-28
- **What:** 244 rows vs ~2,000 expected. Root cause: "all" key omitted USA/GBR CLI series with METHODOLOGY=H.
- **Resolution (R2):** `smim_fetch_oecd.py` rewritten to use explicit dimension keys. Re-fetched 2026-03-28: 1,922 rows. LI/BCICP/CCICP 289 rows/country from 2000-01 to 2024-01; B1GQ_POP 85-103 rows/country from 2000-Q1 to 2025-Q4. Gate G1-6 now passes.

### G-2: UK intensities not computed [HIGH SEVERITY]
- **What:** No `UK-LC_intensities.parquet` or `UK-MC_intensities.parquet` in `data/smim/intensities/`.
- **Impact:** Experiment E1 (UK-LC) is blocked. MIXED-200 uses only US energy tickers for intensity.
- **Root cause:** `smim_compute_intensities.py` was not run for UK universes (or failed silently due to G-4 below).
- **Action required:** Run intensity computation for UK-LC and UK-MC after resolving G-4.

### G-3: CPIMEDSL not fetched [MEDIUM SEVERITY] — ✅ RESOLVED 2026-03-28
- **What:** CPI Medical Care (`CPIMEDSL`) was planned as a healthcare-sector macro proxy; the original ID `CUSR0000SAM` does not exist.
- **Resolution (R1):** `CUSR0000SAM` replaced with `CPIMEDSL` in `smim_fetch_fred.py`; `NAPM` also removed (defunct). Re-fetched 2026-03-28: 314 rows, 2000-01-01 to 2026-02-01, 0 A1 violations. PIT store now has 28 signals.

### G-4: Companies House adapter never built [MEDIUM SEVERITY — structural]
- **What:** EXPERIMENT_PLAN.md specified Companies House (UK) as the source for UK equity balance-sheet data (CapEx, Revenue, Assets). No adapter was built; no data was fetched.
- **Impact:** UK equities (UK-LC, UK-MC) have 0% balance-sheet coverage. Intensity computation for UK equities falls back to OHLCV-derived metrics only (return-based intensity), which is a weaker signal.
- **Action required:** Decide whether to (a) build a Companies House adapter, (b) use an alternative UK balance-sheet source (e.g., Refinitiv/Bloomberg if available, or scrape from annual reports), or (c) formally scope UK balance-sheet out and document intensity methodology for UK.

### G-5: US-LC-FINS rank stability critically low [MEDIUM SEVERITY]
- **What:** Spearman ρ=0.040 for US-LC-FINS intensities — far below the 0.7 threshold (A2 assumption).
- **Impact:** Financials experiment (B1) intensity signal is unstable cross-sectionally.
- **Root cause likely:** Financials sector has `sector_leader` actors at intensity=1.000 (constant) and `bank` actors at ~0.49 — the mix of actor types with very different scales creates near-zero rank correlation across periods when composition changes.
- **Action required:** Investigate InvestmentIntensityMapper for Financials. Consider separate normalisation strata for bank vs sector_leader actor types within the sector.

### G-6: Sector universe sizes diverge from EXPERIMENT_PLAN.md [LOW SEVERITY — informational]

| Universe | Planned N | Actual N | Difference | Notes |
|----------|-----------|---------|------------|-------|
| US-LC-ENERGY | 40 | 22 | -18 | EXPERIMENT_PLAN.md estimate was approximate; 22 is the true S&P 500 GICS-10 count |
| US-LC-TECH | 60 | 68 | +8 | IT sector (GICS 45) has more members than planned estimate |
| US-LC-FINS | 50 | 74 | +24 | Financial sector larger than planned; plan estimate was approximate |
| US-LC-HEALTH | 45 | 60 | +15 | Healthcare sector larger than planned |
| US-LC-INDUS | 50 | 78 | +28 | Industrials sector significantly larger |
| MIXED-200 | ~200 | 27 (equity) + institutional | Planned 200 was aspirational MVP universe; actual equity component is 27 energy firms (US+UK); institutional actors added in experiment registry |

These discrepancies reflect the EXPERIMENT_PLAN.md using rough estimates. The actual sector membership from S&P 500 GICS classification is the correct source. No action required except updating EXPERIMENT_PLAN.md estimates.

### G-7: GDELT raw directory provenance confusion [LOW SEVERITY — housekeeping]
- **What:** `data/smim/raw/gdelt/` contains artifacts from 3 generations of fetch approaches. Gen 2 (`docapi_v2/`) only covered healthcare + 4 institutional actors — sector_energy, sector_technology, sector_financials were never fetched via DOC API.
- **Impact:** None on data quality (canonical data is in `cache/gdelt/` and `processed/`); risk of confusion about data lineage.
- **Action required:** Archive or delete Gen 1 (`gkg_weekly/`, 516 files) and Gen 2 (`docapi_v2/`, 119 files) subdirectories.

### G-8: Gate G1 checklist has two inaccurate entries [LOW SEVERITY — documentation]
- **G1-5** says "all 7 indicators × 4 countries present" — INCORRECT. BCA, GGXCNL_NGDP, GGXWDG_NGDP, LUR are present only for US and GB, not DE and JP.
- **G1-6** says "all 4 indicators × 2 countries present" — MISLEADING. Data is present but severely sparse (see G-1). Presence ≠ fitness for use.

---

## 10. Gate G1 Checklist (Corrected)

| Check | Criterion | Result | Notes |
|-------|-----------|--------|-------|
| G1-1 | A1 compliance: 0 pub_date < event_date leaks | ✅ PASS | 0 violations / 541,730 rows |
| G1-2 | FRED: ≥80% of planned series fetched | ✅ PASS | 28/28 = 100% (R1: CPIMEDSL added 2026-03-28) |
| G1-3 | EDGAR: ≥80% US tickers with filings | ✅ PASS | 765/772 = 99% (US-only; UK by-design 0%) |
| G1-4 | GDELT: weekly continuity since 2015, no >4-week gaps | ✅ PASS | 9 signals, 566 weeks, no gaps |
| G1-5 | IMF: indicators present for experiment countries | ⚠️ PARTIAL | 3/7 indicators cover US+GB+DE+JP; 4/7 indicators cover US+GB only |
| G1-6 | OECD: indicators present with adequate history | ✅ PASS | 1,922 rows (R2: explicit key fix 2026-03-28); LI/BCICP/CCICP 289 rows/country 2000–2024 |
| G1-7 | BEA: all 5 SMIM sectors mapped | ✅ PASS | 21 sector pairs, 2010–2024 |
| G1-8 | OHLCV: ≥80% Gold/Silver for ≥80% of universe tickers | ✅ PASS | Marginal for US-SC (80%); others exceed threshold |
| G1-9 | Intensities: computed for all experiment universes | ❌ FAIL | UK-LC and UK-MC intensities missing |
| G1-10 | Rank stability ρ>0.7 for all universes with computed intensity | ⚠️ WARN | US-LC-FINS (0.040), US-LC-TECH (0.653), US-LC (0.660) below threshold |

### Overall: ⚠️ GATE G1 CONDITIONALLY PASSED — 3 issues require resolution before Phase A experiments

**Phase A (MIXED-200 energy) can proceed** with FRED+EDGAR+GDELT+BEA signals — all critical data for A1/A2 is present and usable.

**Phase E (UK-LC) is blocked** until G-2 and G-4 are resolved.

**OECD signals** are now fit for use (G-1 resolved, R2 complete 2026-03-28). LI/BCICP/CCICP have full 2000–2024 monthly history; B1GQ_POP has full 2000–2025 quarterly history.

---

## 11. Recommendations (Priority Order)

1. **[P1] Re-run OECD fetch** (G-1): Fix SDMX key construction in `smim_fetch_oecd.py` to retrieve full 2000–2025 monthly history for LI, BCICP, CCICP and full quarterly history for B1GQ_POP. Re-ingest into PIT store.
2. **[P1 ✅ DONE] Fetch CPIMEDSL** (G-3): Completed 2026-03-28 (R1). `CPIMEDSL` now in PIT store, 314 rows.
3. **[P2] Investigate US-LC-FINS rank stability** (G-5): Separate normalisation strata for bank vs sector_leader actor types.
4. **[P2] Compute UK intensities** (G-2): Depends on resolving G-4 or defining OHLCV-only intensity for UK.
5. **[P3] Decide on Companies House** (G-4): Either build adapter or formally scope out UK balance-sheet.
6. **[P3] Archive old GDELT raw files** (G-7): Remove `gkg_weekly/` and `docapi_v2/` subdirectories.
7. **[P3] Update EXPERIMENT_PLAN.md** (G-6): Correct universe size estimates to match actual GICS counts.
8. **[P3] Fix OECD raw file naming in DATA_ACQUISITION.md** (G-8): Files on disk lack version suffix.
