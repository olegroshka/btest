# SMIM Data Acquisition Status

Last updated: 2026-03-22 (IMF DataMapper + OECD SDMX 3.0 + BEA I/O tables fetched and ingested)

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
| GDELT narrative | `smim_fetch_gdelt.py` | ✅ Complete (9/9 signals, daily-derived weekly) | ~37K daily rows; ~4.7K weekly rows | 2015-02-19 – 2025-12-31 |
| IMF WEO (DataMapper) | `smim_fetch_imf.py` | ✅ Complete (7/7 series) | 618 rows | 2000-12-31 – 2030-12-31 |
| OECD SDMX 3.0 | `smim_fetch_oecd.py` | ✅ Complete (4 signals) | 244 rows | 2000-01-01 – 2025-10-01 |
| BEA I/O | `smim_fetch_bea.py` | ✅ Complete (2010–2024, API) | 26,852 sector rows / 315 PIT pairs | 2010 – 2024 |
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

## 4. GDELT Narrative Signals — ✅ Complete (daily-sampled, weekly-derived)

**Script:** `scripts/smim_fetch_gdelt.py`
**Adapter:** `smim/data/adapters/gdelt.py` (file-based; reads processed parquets)
**Data:** Weekly narrative intensity for 5 sector themes + 4 institutional actors
**Source:** GDELT GKG 2.0 raw CSV files (free, no authentication)

### Fetch approach: daily-sampled raw GKG CSV → daily-derived weekly panel

One representative GKG file per **UTC calendar day** (slot nearest 12:00 UTC).
Per-day stats are cached in `data/smim/cache/gdelt/daily_aggregates/YYYY-MM-DD.parquet`.
The canonical weekly panel is then **derived from daily data** using correct aggregation.

This supersedes the old approach (one file per ISO week, Monday noon snapshot), which
used a single 15-minute file as a proxy for a full week — an order-of-magnitude sparser.

**Why not DOC API or BigQuery:**

| Issue | DOC API | BigQuery | Raw GKG CSV |
|-------|---------|----------|-------------|
| Rate limits | 429 storms at >1 req/5s | None | None |
| Theme code support | Most GKG 2.0 codes return empty | Full | Full |
| Cost | Free | ~$100–250 | Free |
| Auth required | No | Yes (gcloud) | No |

### Daily file selection

For each UTC date in the range:
1. Try the 12:00:00 UTC slot first (exact noon).
2. If not found, try slots in order of proximity to noon (12:15, 11:45, 12:30, 11:30, …).
3. Stop after `MAX_SLOT_TRIES=20` probes (~±2.5 hours). If no file found, day = missing.

Selection types logged per run:
- **exact_noon**: 12:00:00 UTC file found on first try
- **fallback_nearest**: a different slot found within the probe window
- **missing**: no GKG file found for that UTC day

### Weekly aggregation (mathematically correct)

The weekly panel is **not** a naive mean of daily values. Correct formulas:

```
weekly_article_count = sum(daily_article_count)
weekly_avg_tone      = sum(daily_avg_tone × daily_article_count)
                       / sum(daily_article_count)        ← weighted mean
weekly_intensity     = sum(daily_article_count)
                       / sum(daily_total_docs)           ← NOT mean of daily intensities
```

### Theme baskets (actual GKG 2.0 V2EnhancedThemes codes)

| Signal | Codes |
|--------|-------|
| `sector_energy` | `ENV_OIL`, `ECON_OILPRICE`, `FUELPRICES`, `ENV_NATURALGAS`, `WB_507_ENERGY_AND_EXTRACTIVES` |
| `sector_technology` | `WB_133_INFORMATION_AND_COMMUNICATION_TECHNOLOGIES`, `SOC_INNOVATION`, `TECH_AUTOMATION`, `CYBER_ATTACK` |
| `sector_financials` | `ECON_STOCKMARKET`, `WB_1920_FINANCIAL_SECTOR_DEVELOPMENT`, `EPU_CATS_FINANCIAL_REGULATION`, `ECON_DEBT` |
| `sector_healthcare` | `GENERAL_HEALTH`, `MEDICAL`, `WB_1350_PHARMACEUTICALS`, `UNGP_HEALTHCARE` |
| `sector_macro` | `ECON_INFLATION`, `WB_442_INFLATION`, `WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT` |

Note: old simple codes (`OIL`, `GAS`, `TECH`, etc.) do not appear in GKG 2.0 V2EnhancedThemes.
Baskets were derived from inspecting actual GKG file top codes.

### Actor alias dictionaries (matched as case-insensitive substrings in V2Organizations)

| Signal | Aliases |
|--------|---------|
| `actor_FED` | "federal reserve", "board of governors of the federal reserve", "fomc" |
| `actor_SEC` | "securities exchange commission", "securities and exchange commission", "u.s. securities", "us securities" |
| `actor_IMF` | "international monetary fund" |
| `actor_BOE` | "bank of england" |

Note: GKG NLP drops "and" from org names — raw form is "securities exchange commission".

### Output artifacts

| Path | Contents | Rows (est.) |
|------|----------|-------------|
| `data/smim/processed/gdelt_narrative_daily.parquet` | Daily panel | ~37K (3,970 days × 9 signals) |
| `data/smim/processed/gdelt_narrative.parquet` | Weekly panel — **daily-derived, canonical** | ~4.7K (521 weeks × 9 signals) |
| `data/smim/pit_store/gdelt.parquet` | PIT store (weekly) | ~14K |
| `data/smim/cache/gdelt/daily_aggregates/*.parquet` | Per-day stats cache (resumability) | one file per day |
| `data/smim/cache/gdelt/daily_file_index.parquet` | Selection log (date, slot, type) | one row per day |

Before overwriting the canonical files, the previous versions are archived to:
- `data/smim/processed/old/gdelt_narrative_weekly_snapshot_<ts>.parquet`
- `data/smim/pit_store/old/gdelt_weekly_snapshot_<ts>.parquet`

### Daily artifact schema (`gdelt_narrative_daily.parquet`)

```
theme_or_actor : str             — signal ID (e.g. "sector_energy", "actor_FED")
event_date     : datetime64[ns]  — UTC calendar date of the snapshot
article_count  : float64         — distinct documents matching signal in that snapshot
avg_tone       : float64         — mean V2Tone[0] of matching documents
intensity      : float64         — article_count / total_docs_day
total_docs_day : float64         — total distinct docs in that daily snapshot (denominator)
```

### Weekly artifact schema (`gdelt_narrative.parquet`) — canonical

```
theme_or_actor : str             — signal ID
week_start     : datetime64[ns]  — Monday of the ISO week
article_count  : float64         — sum of daily article counts across the week
avg_tone       : float64         — article-count-weighted mean daily tone
intensity      : float64         — sum(daily_matched) / sum(daily_total_docs)
```

Days before 2015-02-19 are absent (not zero). GDELT 2.0 true start is 2015-02-19.

### PIT store schema (`data/smim/pit_store/gdelt.parquet`)

```
actor_id   : theme_or_actor value (e.g. "sector_energy", "actor_FED")
signal_id  : "gdelt_article_count" | "gdelt_avg_tone" | "gdelt_intensity"
event_date : week_start (tz-naive)
pub_date   : week_start + 7 days  (GDELT data for week W complete by Monday W+1)
value      : float64
source     : "gdelt"
vintage_id : None  (GDELT is append-only, no revisions)
```

### Running the script

```bash
# Full fetch (downloads ~3,970 daily files, ~hours first run)
uv run python scripts/smim_fetch_gdelt.py

# Rebuild weekly only from existing daily cache (fast, no downloads)
uv run python scripts/smim_fetch_gdelt.py --weekly-only

# Rebuild all processed outputs from cache (no re-download)
uv run python scripts/smim_fetch_gdelt.py --rebuild

# Validate with yesterday's file
uv run python scripts/smim_fetch_gdelt.py --validate-only

# Specific date range
uv run python scripts/smim_fetch_gdelt.py --start-date 2023-01-01 --end-date 2023-12-31
```

---

## 5. IMF WEO — ✅ Complete (DataMapper API)

**Script:** `scripts/smim_fetch_imf.py`
**Adapter:** `smim/data/adapters/imf_sdmx.py`
**Status:** Complete — 7 WEO indicators, 4 countries
**Run date:** 2026-03-22
**API key:** None required

### Method

Uses the IMF DataMapper API (`https://www.imf.org/external/datamapper/api/v1/`),
which exposes WEO projections + history without authentication.

The IFS SDMX quarterly endpoint (`dataservices.imf.org`) is also attempted at runtime
but currently times out in this environment; the DataMapper fallback is authoritative.

### Coverage

| Indicator | Label | Countries | Rows |
|-----------|-------|-----------|------|
| `NGDP_RPCH` | Real GDP growth (%) | US, GB, DE, JP | 124 |
| `PCPIPCH` | CPI inflation (%) | US, GB, DE, JP | 124 |
| `BCA` | Current account (USD bn) | US, GB | 62 |
| `GGXCNL_NGDP` | Govt net lending (% GDP) | US, GB | 61 |
| `GGXWDG_NGDP` | Govt gross debt (% GDP) | US, GB | 61 |
| `LUR` | Unemployment rate (%) | US, GB | 62 |
| `PPPGDP` | GDP PPP (int'l $bn) | US, GB, DE, JP | 124 |

**Total:** 618 rows, annual frequency, 2000–2030 (includes WEO projections to 2030).

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/raw/imf/<INDICATOR>.parquet` | Per-indicator raw DataMapper response |
| `data/smim/processed/imf_macro.parquet` | Unified tidy table — `actor_id, signal_id, event_date, pub_date, value` |
| `data/smim/pit_store/imf.parquet` | PIT store shard (A1-compliant, pub_date = event_date + 365 days) |

### PIT store schema

```
actor_id   : country alpha-2 code (US, GB, DE, JP)
signal_id  : IMF WEO indicator code (e.g. "NGDP_RPCH")
event_date : December 31 of reference year (tz-naive)
pub_date   : event_date + 365 days (conservative annual WEO lag)
value      : float64
source     : "imf"
vintage_id : None (DataMapper provides single-vintage current estimates)
```

### Not available / partial

| Indicator | Reason |
|-----------|--------|
| `BCA_NGDPDP` (CA % GDP) | Not in DataMapper catalogue; `BCA` (USD bn) used instead |
| `NID_NGDP` (investment % GDP) | Not in DataMapper; use FRED `A006RE1Q156NBEA` as proxy |
| `NGAP_NPGDP` (output gap) | Not in DataMapper catalogue |
| IFS quarterly (NGDP_R_XDC, PCPI_IX, FPOLM_PA) | `dataservices.imf.org` times out in this network environment |

### Running the script

```bash
uv run python scripts/smim_fetch_imf.py
```

---

## 6. OECD SDMX 3.0 — ✅ Complete

**Script:** `scripts/smim_fetch_oecd.py`
**Adapter:** `smim/data/adapters/oecd_sdmx.py`
**Status:** Complete — 4 signals (CLI + QNA), US + UK
**Run date:** 2026-03-22
**API key:** None required

### Method

Uses OECD SDMX 3.0 REST API (`https://sdmx.oecd.org/public/rest/data/`) with the
`all` key approach: fetch the full dataflow (filtered by `startPeriod`/`endPeriod`)
and filter to the desired countries and measures in-memory.

This approach avoids the 14-dimension key format problem of SDMX 3.0 (direct
dimension keys with partial wildcards return 422 errors).

### Coverage

| Dataflow | Signals fetched | Countries | Rows |
|----------|----------------|-----------|------|
| `DSD_STES@DF_CLI,4.0` | LI (Composite Leading Indicator), BCICP (Business Confidence), CCICP (Consumer Confidence) | US, GB | 180 |
| `DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA,1.1` | B1GQ_POP (GDP per capita, USD PPP, level) | US, GB | 64 |

**Total:** 244 rows. LI/BCICP/CCICP are monthly; B1GQ_POP is quarterly.

**Date range:** CLI from 2000-01-01; QNA from earliest available through 2025-Q4.

### OECD country codes

The OECD SDMX 3.0 API uses ISO 3166-1 alpha-3 codes (USA, GBR). The script
normalises these to alpha-2 (US, GB) for consistency with the rest of the PIT store.

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/raw/oecd/DSD_STES_DF_CLI_4.0.parquet` | Raw CLI response (all countries) |
| `data/smim/raw/oecd/DSD_NAMAIN1_DF_QNA_EXPENDITURE_CAPITA_1.1.parquet` | Raw QNA response |
| `data/smim/processed/oecd_macro.parquet` | Unified tidy table |
| `data/smim/pit_store/oecd.parquet` | PIT store shard |

### PIT store schema

```
actor_id   : country alpha-2 code (US, GB)
signal_id  : OECD measure code (e.g. "LI", "BCICP", "B1GQ_POP")
event_date : first day of period (tz-naive)
pub_date   : event_date + 45 days (CLI) or + 75 days (QNA)
value      : float64
source     : "oecd"
vintage_id : None
```

### Running the script

```bash
uv run python scripts/smim_fetch_oecd.py
```

---

## 7. BEA Input-Output — ✅ Complete

**Script:** `scripts/smim_fetch_bea.py`
**Adapter:** `smim/data/adapters/bea_io.py`
**Status:** Complete — Use Table (TableID=259), 2010–2024, all 5 sectors
**Run date:** 2026-03-22
**API key:** `BEA_API_KEY` (free at https://apps.bea.gov/API/signup/ — used if set, else direct Excel download)

### Method

**With API key (primary):** Calls BEA JSON API (`https://apps.bea.gov/api/data`)
with `DataSetName=InputOutput`, `TableID=259` (Use of Commodities by Industries,
Before Redefinitions, Producers' Prices) for years 2010–2024.

**Without API key (fallback):** Downloads published Excel files from
`https://apps.bea.gov/industry/xls/io-annual/` and parses the flow matrix.

Direct-requirements coefficients are computed as:
```
coeff[source→target] = flow[source, target] / total_output[target]
```

### Sector mapping (NAICS prefix → SMIM sector)

| BEA NAICS prefix | SMIM sector |
|-----------------|-------------|
| 211, 213, 324 | Energy |
| 334, 511, 518, 519 | Technology |
| 521–525 | Financials |
| 621–624 | Healthcare |
| 331, 332, 333, 336, 337 | Industrials |

### Coverage

| Metric | Value |
|--------|-------|
| Raw rows (all sector-involved pairs) | 26,852 |
| Year range | 2010 – 2024 |
| Sectors covered | Energy, Financials, Healthcare, Industrials, Technology |
| PIT sector-pair observations | 315 (21 unique source→target pairs × 15 years) |

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/raw/bea/use_table_<year>.parquet` | Raw per-year API response |
| `data/smim/processed/bea_io_tables.parquet` | Sector-mapped coefficient table |
| `data/smim/pit_store/bea.parquet` | PIT store shard |

### Processed table schema (`bea_io_tables.parquet`)

```
source_industry : BEA RowCode (NAICS-based, e.g. "211000")
source_sector   : SMIM sector label (e.g. "Energy") or None
source_desc     : BEA row description
target_industry : BEA ColCode
target_sector   : SMIM sector label or None
target_desc     : BEA column description
coefficient     : direct requirements coefficient (float64)
year            : reference year
pub_date        : year-end + 548 days (A1 conservative lag, ~18 months)
table_id        : "259"
```

### PIT store schema (`bea.parquet`)

```
actor_id   : "SourceSector→TargetSector" (e.g. "Energy→Industrials")
signal_id  : "io_coefficient"
event_date : December 31 of reference year (tz-naive)
pub_date   : event_date + 548 days
value      : mean coefficient for that sector pair (float64)
source     : "bea"
vintage_id : None
```

### Running the script

```bash
# With API key (more complete industry detail):
BEA_API_KEY=<your_key> uv run python scripts/smim_fetch_bea.py

# Without API key (downloads Excel from BEA website):
uv run python scripts/smim_fetch_bea.py
```

---

## 8. BIS — Not Started

**Adapter:** Not yet built
**Planned data:** Global credit and liquidity indicators
**Blocker:** Adapter not implemented; BIS SDMX endpoint needs exploration

---

## Remediation TODO

1. **`CPIMEDSL`** — add to `MACRO_SERIES` in `smim_fetch_fred.py` and re-run; correct ID for CPI Medical Care
2. **ISM PMI** — `MANEMP` is the accepted proxy; actual ISM data requires a subscription or use `DALLASMPMC` (Dallas Fed Manufacturing Activity, available on FRED)
3. **BEA** — ✅ complete (`smim_fetch_bea.py`). API key gives full industry detail; no-key fallback uses published Excel. Re-run anytime to pick up the latest BEA release.
4. **GDELT** — ✅ complete (daily-derived weekly pipeline). To re-run: `uv run python scripts/smim_fetch_gdelt.py` (uses per-day cache; add `--force-refetch` to re-download all files). Use `--weekly-only` to rebuild weekly panel from existing daily cache without any new downloads.
5. **IMF** — ✅ complete (DataMapper API, 7 WEO indicators). IFS SDMX quarterly data (`dataservices.imf.org`) still times out; add a proxy or VPN to unblock. Missing WEO codes: `BCA_NGDPDP`, `NID_NGDP`, `NGAP_NPGDP` — not in DataMapper.
6. **OECD** — ✅ complete (SDMX 3.0, CLI + QNA). Only US + GB currently; add more OECD countries by modifying `filters["REF_AREA"]` in `smim_fetch_oecd.py`.
7. **BIS** — adapters not yet built; bulk fetch script not written
