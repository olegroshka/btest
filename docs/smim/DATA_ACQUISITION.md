# SMIM Data Acquisition Status

Last updated: 2026-03-22

This document tracks every data source required by the experiment plan, what
has been acquired, what failed, and why. Update it after every acquisition run.

---

## Summary

| Source | Script | Status | Observations | Date range |
|--------|--------|--------|-------------|------------|
| Equity OHLCV (Yahoo Finance) | `smim_build_universes.py` | Complete | ~11 universes | 2005–2025 |
| FRED macro signals | `smim_fetch_fred.py` | Complete (27/29 series) | 71,761 | 2000–2026 |
| ALFRED vintages | `smim_fetch_fred.py` | Complete (5 series) | included above | 2000–2026 |
| SEC EDGAR | `smim_fetch_edgar.py` | Complete (765/772 tickers) | 461,203 | 2005–2026 |
| GDELT | — | Not started | — | — |
| IMF SDMX | — | Not started | — | — |
| OECD SDMX | — | Not started | — | — |
| BEA I/O | — | Not started | — | — |
| BIS | — | Not started | — | — |

---

## 1. Equity OHLCV (Yahoo Finance via btest)

**Script:** `scripts/smim_build_universes.py`
**Status:** Complete
**Run date:** 2026-03-21

### Universes acquired

| Universe ID | Tickers | OHLCV path |
|-------------|---------|------------|
| `US-LC` | 200 (top S&P 500 by mkt cap) | `equities/smim/US-LC/ohlcv.parquet` |
| `US-LC-ENERGY` | ~22 (S&P 500 GICS 10) | `equities/smim/US-LC-ENERGY/ohlcv.parquet` |
| `US-LC-TECH` | ~68 (S&P 500 GICS 45) | `equities/smim/US-LC-TECH/ohlcv.parquet` |
| `US-LC-FINS` | ~74 (S&P 500 GICS 40) | `equities/smim/US-LC-FINS/ohlcv.parquet` |
| `US-LC-HEALTH` | ~60 (S&P 500 GICS 35) | `equities/smim/US-LC-HEALTH/ohlcv.parquet` |
| `US-LC-INDUS` | ~78 (S&P 500 GICS 20) | `equities/smim/US-LC-INDUS/ohlcv.parquet` |
| `US-MC` | 200 (S&P 400) | `equities/smim/US-MC/ohlcv.parquet` |
| `US-SC` | 200 (Russell 2000, stratified seed=42) | `equities/smim/US-SC/ohlcv.parquet` |
| `UK-LC` | ~99 (FTSE 100, `.L` suffix) | `equities/smim/UK-LC/ohlcv.parquet` |
| `UK-MC` | ~100 (FTSE 250 ex-100) | `equities/smim/UK-MC/ohlcv.parquet` |
| `MIXED-200` | ~27 (US + UK energy MVP) | `equities/smim/MIXED-200/ohlcv.parquet` |

**Date range:** 2005-01-03 to 2025-12-30. Tickers listed after 2005 start from
their IPO date — shorter history is correct, not a gap.

**Known sparse tickers** (flagged by step-3 QC, all expected post-IPO companies):
`ABNB`, `COIN`, `CRWD`, `MRNA`, and similar recent listings.

---

## 2. FRED Macro Signals

**Script:** `scripts/smim_fetch_fred.py`
**Status:** Complete — 27/29 series acquired
**Run date:** 2026-03-22
**API key:** `FRED_API_KEY` environment variable

### Acquired series (27)

| FRED ID | Description | Frequency | Layer |
|---------|-------------|-----------|-------|
| `GDP` | Nominal GDP | Quarterly | L0 exogenous |
| `GDPC1` | Real GDP (chained) | Quarterly | L0 exogenous |
| `INDPRO` | Industrial Production Index | Monthly | L0 exogenous |
| `MANEMP` | Manufacturing employment (PMI proxy) | Monthly | L0 exogenous |
| `UNRATE` | Unemployment rate | Monthly | L0 exogenous |
| `CPIAUCSL` | CPI All Urban Consumers | Monthly | L0 exogenous |
| `CPILFESL` | CPI Less Food & Energy (core) | Monthly | L0 exogenous |
| `PCEPI` | PCE Price Index | Monthly | L0 exogenous |
| `FEDFUNDS` | Federal Funds Rate (monthly avg) | Monthly | L1 upstream |
| `DFF` | Federal Funds Rate (daily) | Daily | L1 upstream |
| `GS10` | 10-year Treasury yield | Daily | L1 upstream |
| `GS2` | 2-year Treasury yield | Daily | L1 upstream |
| `T10Y2Y` | 10yr–2yr yield spread | Daily | L1 upstream |
| `BAA10Y` | Moody's BAA spread over 10yr | Daily | L1 upstream |
| `BAMLH0A0HYM2` | ICE BofA HY OAS | Daily | L1 upstream |
| `VIXCLS` | CBOE VIX | Daily | L0 exogenous |
| `STLFSI2` | St. Louis Fed Financial Stress Index | Weekly | L0 exogenous |
| `UMCSENT` | U. Michigan Consumer Sentiment | Monthly | L0 exogenous |
| `USSLIND` | US Leading Index | Monthly | L0 exogenous |
| `HOUST` | Housing Starts | Monthly | L0 exogenous |
| `M2SL` | M2 Money Supply | Monthly | L0 exogenous |
| `TOTBKCR` | Total Bank Credit | Weekly | L1 upstream |
| `DTWEXBGS` | Trade-weighted USD (goods, services) | Daily | L0 exogenous |
| `DCOILWTICO` | WTI Crude Oil Price | Daily | Energy sector |
| `DCOILBRENTEU` | Brent Crude Oil Price | Daily | Energy sector |
| `GASREGW` | US Regular Gasoline Price | Weekly | Energy sector |
| `DRCCLACBS` | Credit Card Delinquency Rate | Quarterly | Financials sector |

### ALFRED vintage series (5)

Full historical vintage histories were downloaded for the 5 most important
revision-prone series. Each published revision becomes a separate PIT record.

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
| `NAPM` | `Bad Request. The series does not exist.` | ISM Manufacturing PMI is not available on FRED under this ID. NAPM (National Association of Purchasing Management) was renamed to ISM but never uploaded to FRED as a continuous series. | Use `MANEMP` (manufacturing employment) as the PMI proxy — already acquired. For actual ISM data, ISM charges a subscription fee; free proxy via MANEMP is standard in academic literature. |
| `CUSR0000SAM` | `Bad Request. The series does not exist.` | The CPI Medical Care subindex ID format has changed. FRED uses region-prefixed IDs for CPI sub-components. | Correct ID is `CPIMEDSL` (CPI: Medical Care). Will be added in a follow-up run. |

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/raw/fred/<SERIES>.parquet` | Raw per-series observations from FRED API |
| `data/smim/raw/fred/<SERIES>_alfred.parquet` | Raw ALFRED all-releases for vintaged series |
| `data/smim/processed/fred_signals.parquet` | Unified normalised table (all series, tidy format) |
| `data/smim/pit_store/fred.parquet` | PIT store shard — A1-compliant, queryable by `as_of` |

### PIT store schema

```
actor_id   : "MACRO"  (macro series have no single equity actor)
signal_id  : FRED series ID (e.g. "FEDFUNDS")
event_date : observation date (tz-naive)
pub_date   : for ALFRED series: realtime_start of the vintage
             for others: event_date + 30 days (conservative publication lag)
value      : float64
source     : "fred"
vintage_id : pub_date string for ALFRED series; None otherwise
```

---

## 3. SEC EDGAR — Complete

**Script:** `scripts/smim_fetch_edgar.py`
**Status:** Complete
**Run date:** 2026-03-22

### Method

1. Downloads the CIK mapping from `https://www.sec.gov/files/company_tickers.json`
2. Fetches company-facts XBRL JSON from `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
   for each US ticker across US-LC, US-MC, US-SC, and all sector-slice universes
3. The EDGAR `filed` date is used as `pub_date` — strictly A1-compliant

### Coverage

| Metric | Value |
|--------|-------|
| Tickers attempted | 772 (780 universe tickers, 8 with no CIK mapping) |
| Tickers with data | 765 |
| Total filing records | 461,203 |
| Date range | 2005-07-04 to 2026-02-28 |
| Filing types | 10-K, 10-Q |

### XBRL tag coverage

| Tag | Tickers |
|-----|---------|
| `Assets` | 765 |
| `StockholdersEquity` | 757 |
| `PaymentsToAcquirePropertyPlantAndEquipment` (CapEx) | 611 |
| `LongTermDebt` | 605 |
| `Revenues` | 559 |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 537 |
| `ResearchAndDevelopmentExpense` | 341 |
| `CapitalExpenditures` (legacy tag) | 0 — superseded by PaymentsToAcquire… |

### No-data tickers (7)

`BBUC`, `BTDR`, `CMDB`, `GAMB`, `HSHP`, `LZM`, `VTEX`
— Recent cross-listings, SPACs, or foreign private issuers with no EDGAR XBRL history.

### No-CIK tickers (8)

`DAY`, `FI`, `FRBA`, `MMC`, `MOGA`, `PDLI`, `THRD`, `XTSLA`
— De-listed, renamed, or non-reporting entities not in SEC company_tickers.json.

### Output paths

| Path | Contents |
|------|----------|
| `data/smim/processed/edgar_balance_sheet.parquet` | Normalised tidy table: `ticker, cik, event_date, pub_date, tag, value, form_type, period` |
| `data/smim/pit_store/edgar.parquet` | PIT store shard — A1-compliant, queryable by `as_of` |

### PIT store schema

```
actor_id   : ticker (e.g. "AAPL")
signal_id  : XBRL tag (e.g. "Assets")
event_date : period end date (tz-naive)
pub_date   : EDGAR filing date — the exact date data became public (A1-compliant)
value      : float64
source     : "edgar"
vintage_id : None (EDGAR filings are not revised in place)
```

---

## 4. GDELT — Not Started

**Adapter:** `smim/data/adapters/gdelt.py` (implemented, untested at scale)
**Planned data:** Narrative co-occurrence signals for edge estimation
**Blocker:** None — no API key required

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
**Blocker:** BEA_API_KEY environment variable required (free registration at bea.gov)

---

## 8. BIS — Not Started

**Adapter:** Not yet built
**Planned data:** Global credit and liquidity indicators
**Blocker:** Adapter not implemented; BIS SDMX endpoint needs exploration

---

## Remediation TODO

1. **`CPIMEDSL`** — re-run `smim_fetch_fred.py` after adding the correct ID to `MACRO_SERIES`
2. **ISM PMI** — `MANEMP` is an acceptable proxy; if actual ISM data is needed, source via a subscription or use the Dallas Fed Manufacturing Activity survey (`DALLASMPMC`) which is on FRED
3. **EDGAR** — ✅ Complete (2026-03-22); 765/772 tickers, 461,203 records
4. **BEA** — register for API key at https://apps.bea.gov/api/signup/ and set `BEA_API_KEY`
