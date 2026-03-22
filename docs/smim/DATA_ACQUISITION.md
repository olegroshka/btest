# SMIM Data Acquisition Status

Last updated: 2026-03-22 (GDELT re-fetched with corrected actor_SEC alias)

This document tracks every data source required by the experiment plan, what
has been acquired, what failed, and why. Update it after every acquisition run.

All data files are **gitignored** — they live on disk only:
- `data/smim/raw/` — raw downloaded files
- `data/smim/processed/` — normalised parquets
- `data/smim/pit_store/` — PIT store shards (A1-compliant, queryable by `as_of`)

Only `data/smim/universes/*.csv` are committed to git.

---

## Summary

| Source | Script | Status | Records | Event date range |
|--------|--------|--------|---------|-----------------|
| Equity OHLCV (Yahoo Finance) | `smim_build_universes.py` | ✅ Complete | 11 universes | 2005-01-03 – 2025-12-30 |
| FRED macro signals | `smim_fetch_fred.py` | ✅ Complete (27/29 series) | 71,761 rows | 2000-01-01 – 2026-03-20 |
| ALFRED vintages | `smim_fetch_fred.py` | ✅ Complete (5 series) | 8,956 vintage rows (subset above) | 2000-01-01 – 2026-03-20 |
| SEC EDGAR XBRL | `smim_fetch_edgar.py` | ✅ Complete (765/772 tickers) | 461,203 rows | 2005-07-04 – 2026-02-28 |
| GDELT narrative | `smim_fetch_gdelt.py` | ✅ Complete (9/9 signals) | 4,662 rows (518 weeks × 9 signals) | 2015-02-23 – 2025-12-29 |
| IMF SDMX | — | ⬜ Not started | — | — |
| OECD SDMX | — | ⬜ Not started | — | — |
| BEA I/O | — | ⬜ Not started | — | — |
| BIS | — | ⬜ Not started | — | — |

---

## 1. Equity OHLCV (Yahoo Finance via btest)

**Script:** `scripts/smim_build_universes.py`
**Status:** Complete
**Run date:** 2026-03-21

### Universes acquired

| Universe ID | Tickers | OHLCV path |
|-------------|---------|------------|
| `US-LC` | 200 (top S&P 500 by mkt cap) | `equities/smim/US-LC/ohlcv.parquet` |
| `US-LC-ENERGY` | 22 (S&P 500 GICS 10) | `equities/smim/US-LC-ENERGY/ohlcv.parquet` |
| `US-LC-TECH` | 68 (S&P 500 GICS 45) | `equities/smim/US-LC-TECH/ohlcv.parquet` |
| `US-LC-FINS` | 74 (S&P 500 GICS 40) | `equities/smim/US-LC-FINS/ohlcv.parquet` |
| `US-LC-HEALTH` | 60 (S&P 500 GICS 35) | `equities/smim/US-LC-HEALTH/ohlcv.parquet` |
| `US-LC-INDUS` | 78 (S&P 500 GICS 20) | `equities/smim/US-LC-INDUS/ohlcv.parquet` |
| `US-MC` | 200 (S&P 400) | `equities/smim/US-MC/ohlcv.parquet` |
| `US-SC` | 200 (Russell 2000, stratified seed=42) | `equities/smim/US-SC/ohlcv.parquet` |
| `UK-LC` | ~99 (FTSE 100, `.L` suffix) | `equities/smim/UK-LC/ohlcv.parquet` |
| `UK-MC` | ~100 (FTSE 250 ex-100) | `equities/smim/UK-MC/ohlcv.parquet` |
| `MIXED-200` | ~27 (US + UK energy MVP) | `equities/smim/MIXED-200/ohlcv.parquet` |

**Date range:** 2005-01-03 to 2025-12-30. Tickers listed after 2005 start from
their IPO date — shorter history is correct, not a gap.

**Known sparse tickers** (flagged by step-3 QC, all expected post-IPO companies):
`ABNB`, `COIN`, `CRWD`, `MRNA`, and similar recent listings.

**OHLCV parquet schema** (long-form):
```
date       datetime64[ns]
ticker     str
open, high, low, close  float64
volume     float64
sector     str  (from universe CSV; may be NaN for UK tickers)
```

---

## 2. FRED Macro Signals + ALFRED Vintages

**Script:** `scripts/smim_fetch_fred.py`
**Status:** Complete — 27/29 series acquired
**Run date:** 2026-03-22
**API key:** `FRED_API_KEY` environment variable

### Row counts (from `data/smim/processed/fred_signals.parquet`)

| Metric | Value |
|--------|-------|
| Total rows | 71,761 |
| Distinct signals | 27 |
| Event date range | 2000-01-01 – 2026-03-20 |
| Pub date range | 2000-01-31 – 2026-04-19 |
| ALFRED vintage rows | 8,956 (subset of total, for GDP/UNRATE/CPIAUCSL/INDPRO/FEDFUNDS) |

### PIT store (`data/smim/pit_store/fred.parquet`)

| Metric | Value |
|--------|-------|
| Rows | 64,165 (de-duped on natural key vs 71,761 processed; ALFRED revisions keep latest pub_date) |
| actor_id | 1 (always "MACRO") |
| signal_id | 27 |

### Acquired series (27)

| FRED ID | Rows | Frequency | Layer |
|---------|------|-----------|-------|
| `DFF` | 9,575 | Daily | L1 upstream |
| `BAMLH0A0HYM2` | 6,928 | Daily | L1 upstream |
| `T10Y2Y` | 6,840 | Daily | L1 upstream |
| `VIXCLS` | 6,839 | Daily | L0 exogenous |
| `BAA10Y` | 6,839 | Daily | L1 upstream |
| `DCOILWTICO` | 6,836 | Daily | Energy sector |
| `DCOILBRENTEU` | 6,836 | Daily | Energy sector |
| `DTWEXBGS` | 5,270 | Daily | L0 exogenous |
| `INDPRO` | 5,437 | Monthly | L0 exogenous |
| `GASREGW` | 1,368 | Weekly | Energy sector |
| `TOTBKCR` | 1,367 | Weekly | L1 upstream |
| `STLFSI2` | 1,149 | Weekly | L0 exogenous |
| `GDP` | 1,032 | Quarterly | L0 exogenous |
| `CPIAUCSL` | 1,581 | Monthly | L0 exogenous |
| `UNRATE` | 590 | Monthly | L0 exogenous |
| `FEDFUNDS` | 316 | Monthly | L1 upstream |
| `CPILFESL` | 314 | Monthly | L0 exogenous |
| `GS10` | 314 | Daily | L1 upstream |
| `GS2` | 314 | Daily | L1 upstream |
| `MANEMP` | 314 | Monthly | L0 exogenous |
| `GDPC1` | ~300 | Quarterly | L0 exogenous |
| `PCEPI` | ~310 | Monthly | L0 exogenous |
| `UMCSENT` | ~310 | Monthly | L0 exogenous |
| `USSLIND` | ~310 | Monthly | L0 exogenous |
| `HOUST` | ~310 | Monthly | L0 exogenous |
| `M2SL` | ~310 | Monthly | L0 exogenous |
| `DRCCLACBS` | ~85 | Quarterly | Financials sector |

### ALFRED vintage series (5)

Full historical vintage histories (each revision = separate PIT record):

| FRED ID | Why vintaged |
|---------|-------------|
| `GDP` | Heavily revised — initial vs final estimates diverge materially |
| `UNRATE` | Benchmark revisions every 5 years; seasonal adjustments revised annually |
| `CPIAUCSL` | Methodological revisions change long-run levels |
| `INDPRO` | Annual benchmark revisions; real-time data often revised 1–3% |
| `FEDFUNDS` | Monthly average published after month-end; daily `DFF` is the PIT alternative |

### Failed series (2)

| FRED ID | Error | Root cause | Replacement |
|---------|-------|-----------|-------------|
| `NAPM` | `Bad Request. The series does not exist.` | ISM Manufacturing PMI renamed to ISM; never uploaded to FRED as a continuous series. | `MANEMP` (manufacturing employment) — already acquired. Standard academic proxy. |
| `CUSR0000SAM` | `Bad Request. The series does not exist.` | CPI Medical Care subindex ID changed. | Correct ID is `CPIMEDSL`. Not yet re-fetched — see Remediation TODO. |

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/raw/fred/<SERIES>.parquet` | Raw per-series observations from FRED API (27 files) |
| `data/smim/raw/fred/<SERIES>_alfred.parquet` | Raw ALFRED all-releases for 5 vintaged series |
| `data/smim/processed/fred_signals.parquet` | Unified normalised table — columns: `signal_id, event_date, value, pub_date, vintage_id` |
| `data/smim/pit_store/fred.parquet` | PIT store shard — A1-compliant, queryable by `as_of` |

### PIT store schema

```
actor_id   : "MACRO"  (macro series have no single equity actor)
signal_id  : FRED series ID (e.g. "FEDFUNDS")
event_date : observation date (tz-naive)
pub_date   : ALFRED series → realtime_start of the vintage
             non-ALFRED → event_date + 30 days (conservative publication lag)
value      : float64
source     : "fred"
vintage_id : pub_date string for ALFRED series; None otherwise
```

---

## 3. SEC EDGAR XBRL — Complete

**Script:** `scripts/smim_fetch_edgar.py`
**Status:** Complete
**Run date:** 2026-03-22
**API key:** None required — SEC only needs a descriptive `User-Agent` header

### Method

1. Downloads CIK mapping from `https://www.sec.gov/files/company_tickers.json`
2. For each US ticker (US-LC, US-MC, US-SC + 5 sector slices), fetches:
   `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_zero_padded_10}.json`
3. Extracts all 10-K and 10-Q observations for the configured XBRL tags
4. Uses EDGAR `filed` date as `pub_date` — exact date the data became public (A1-compliant)
5. Rate-limited at 0.15 s/request (~6.7 req/s, safely under SEC's 10 req/s limit)

### Coverage (`data/smim/processed/edgar_balance_sheet.parquet`)

| Metric | Value |
|--------|-------|
| Universe tickers | 780 (across 8 US universe files) |
| No CIK mapping | 8 — `DAY`, `FI`, `FRBA`, `MMC`, `MOGA`, `PDLI`, `THRD`, `XTSLA` (de-listed / renamed / non-reporting) |
| Tickers attempted | 772 |
| Tickers with data | 765 |
| No XBRL data | 7 — `BBUC`, `BTDR`, `CMDB`, `GAMB`, `HSHP`, `LZM`, `VTEX` (SPACs / cross-listings / foreign private issuers) |
| Total filing records | 461,203 |
| Event date range | 2005-07-04 – 2026-02-28 |
| Pub date range | 2009-04-15 – 2026-03-20 |
| 10-Q records | 320,332 |
| 10-K records | 140,871 |
| Unique CIKs | 764 (one ticker shares a CIK with another) |

### XBRL tag coverage

| Tag | Tickers | Notes |
|-----|---------|-------|
| `Assets` | 765 | Near-universal |
| `StockholdersEquity` | 757 | Near-universal |
| `PaymentsToAcquirePropertyPlantAndEquipment` | 611 | Modern CapEx tag |
| `LongTermDebt` | 605 | |
| `Revenues` | 559 | Older revenue tag |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 537 | ASC 606 revenue tag (post-2018) |
| `ResearchAndDevelopmentExpense` | 341 | Zero for non-R&D sectors |
| `CapitalExpenditures` | 0 | Legacy tag; superseded by `PaymentsToAcquire…` — no filings use it |

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/processed/edgar_balance_sheet.parquet` | Normalised tidy table — columns: `ticker, cik, event_date, pub_date, tag, value, form_type, period` |
| `data/smim/pit_store/edgar.parquet` | PIT store shard — A1-compliant, queryable by `as_of` |

### PIT store schema

```
actor_id   : ticker (e.g. "AAPL")
signal_id  : XBRL tag (e.g. "Assets")
event_date : period end date (tz-naive)
pub_date   : EDGAR filing date — exact date data became public (A1-compliant)
value      : float64
source     : "edgar"
vintage_id : None (EDGAR filings are point-in-time; not revised in place)
```

### PIT store row counts (`data/smim/pit_store/edgar.parquet`)

| Metric | Value |
|--------|-------|
| Rows | 461,203 |
| Unique actor_id (tickers) | 765 |
| Unique signal_id (XBRL tags) | 7 |

---

## 4. GDELT Narrative Signals — ✅ Complete

**Script:** `scripts/smim_fetch_gdelt.py`
**Adapter:** `smim/data/adapters/gdelt.py` (implemented)
**Data:** Weekly narrative intensity for 5 sector themes + 4 institutional actors
**Source:** GDELT GKG 2.0 raw CSV files (free, no authentication)

### Fetch approach: raw GKG CSV files

One GKG file per ISO week (Monday ~noon UTC slot). Each file is a 15-minute snapshot
of ~200–1,300 articles. Intensity = matched_docs / total_docs_in_snapshot (fractional
share, not absolute count). All parsing done in-process; no rate limits, no cost.

**Why not DOC API or BigQuery:**

| Issue | DOC API | BigQuery | Raw GKG CSV |
|-------|---------|----------|-------------|
| Rate limits | 429 storms at >1 req/5s | None | None |
| Theme code support | Most GKG 2.0 codes return empty | Full | Full |
| Cost | Free | ~$100–250 | Free |
| Auth required | No | Yes (gcloud) | No |

### Theme baskets (actual GKG 2.0 V2EnhancedThemes codes)

| Signal | Codes |
|--------|-------|
| `sector_energy` | `ENV_OIL`, `ECON_OILPRICE`, `FUELPRICES`, `ENV_NATURALGAS`, `WB_507_ENERGY_AND_EXTRACTIVES` |
| `sector_technology` | `WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES`, `SOC_INNOVATION`, `TECH_AUTOMATION`, `CYBER_ATTACK` |
| `sector_financials` | `ECON_STOCKMARKET`, `WB_1920_FINANCIAL_SECTOR_DEVELOPMENT`, `EPU_CATS_FINANCIAL_REGULATION`, `ECON_DEBT` |
| `sector_healthcare` | `GENERAL_HEALTH`, `MEDICAL`, `WB_1350_PHARMACEUTICALS`, `UNGP_HEALTHCARE` |
| `sector_macro` | `ECON_INFLATION`, `WB_442_INFLATION`, `WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT` |

Note: old simple codes (`OIL`, `GAS`, `TECH`, etc.) do not appear in GKG 2.0 V2EnhancedThemes.
These baskets were derived from inspecting actual GKG file top codes.

### Actor alias dictionaries (matched as case-insensitive substrings in V2Organizations)

| Signal | Aliases |
|--------|---------|
| `actor_FED` | "federal reserve", "board of governors of the federal reserve", "fomc" |
| `actor_SEC` | "securities exchange commission", "securities and exchange commission", "u.s. securities", "us securities" |
| `actor_IMF` | "international monetary fund" |
| `actor_BOE` | "bank of england" |

Note: GKG NLP drops "and" from org names — actual form is "securities exchange commission".

### Coverage (`data/smim/processed/gdelt_narrative.parquet`)

| Metric | Value |
|--------|-------|
| Files fetched | 518 (49 weeks skipped — no file found for that Monday) |
| Date range | 2015-02-23 – 2025-12-29 |
| Total rows | 4,662 (518 weeks × 9 signals) |
| Signals | 9 / 9 |
| actor_SEC intensity range | 0.00 – 3.59% (was 0% before alias fix) |
| actor_FED intensity range | 0.00 – 2.90% |
| actor_IMF intensity range | 0.00 – 3.78% |
| actor_BOE intensity range | 0.00 – 0.09% |
| sector_energy intensity range | 0.00 – 16.0% |
| sector_financials intensity range | 0.00 – 20.0% |
| sector_healthcare intensity range | 0.00 – 57.2% |
| sector_macro intensity range | 0.00 – 9.3% |
| sector_technology intensity range | 3.86 – 40.0% |

### PIT store row counts (`data/smim/pit_store/gdelt.parquet`)

| Metric | Value |
|--------|-------|
| Rows | 14,720 |
| Unique actor_id | 9 |
| Unique signal_id | 3 (gdelt_article_count, gdelt_avg_tone, gdelt_intensity) |

### Output schema (`data/smim/processed/gdelt_narrative.parquet`)

```
theme_or_actor : str             — signal ID (e.g. "sector_energy", "actor_FED")
week_start     : datetime64[ns]  — Monday of the ISO week
article_count  : float64         — distinct documents matching signal that week
avg_tone       : float64         — mean primary tone of matching documents
intensity      : float64         — article_count / total_weekly_distinct_docs
```

Weeks before 2015-02-19 are `NaN`, not zero. The GDELT 2.0 true start is 2015-02-19.

### PIT store schema (`data/smim/pit_store/gdelt.parquet`)

```
actor_id   : query ID (e.g. "sector_energy", "actor_FED")
signal_id  : "gdelt_article_count" | "gdelt_avg_tone" | "gdelt_intensity"
event_date : week_start (tz-naive)
pub_date   : week_start + 7 days  (GDELT weekly data complete at end of week)
value      : float64
source     : "gdelt"
vintage_id : None  (GDELT is append-only, no revisions)
```

---

## 5. IMF SDMX — Not Started

**Adapter:** `smim/data/adapters/imf_sdmx.py` (implemented)
**Planned data:** International macro indicators for `INST-INTL` actor set
**Blocker:** None

---

## 6. OECD SDMX — Not Started

**Adapter:** `smim/data/adapters/oecd_sdmx.py` (implemented)
**Planned data:** Structural indicators for international actors
**Blocker:** None

---

## 7. BEA Input-Output — Not Started

**Adapter:** `smim/data/adapters/bea_io.py` (implemented)
**Planned data:** Supply-chain edges (Use and Make tables)
**Blocker:** `BEA_API_KEY` environment variable required (free registration at bea.gov)

---

## 8. BIS — Not Started

**Adapter:** Not yet built
**Planned data:** Global credit and liquidity indicators
**Blocker:** Adapter not implemented; BIS SDMX endpoint needs exploration

---

## Remediation TODO

1. **`CPIMEDSL`** — add to `MACRO_SERIES` in `smim_fetch_fred.py` and re-run; correct ID for CPI Medical Care
2. **ISM PMI** — `MANEMP` is the accepted proxy; actual ISM data requires a subscription or use `DALLASMPMC` (Dallas Fed Manufacturing Activity, available on FRED)
3. **BEA** — register for API key at https://apps.bea.gov/api/signup/ and set `BEA_API_KEY`
4. **GDELT** — ✅ complete. To re-run: `uv run python scripts/smim_fetch_gdelt.py` (uses cached weekly parquets; add `--force-refetch` to re-download all files).
5. **IMF / OECD / BIS** — adapters exist; bulk fetch scripts not yet written
