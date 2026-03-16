# SMIM Data Source Audit — MVP Energy US + UK

**Experiment**: `experiments/mvp_energy_us_uk.yaml`
**Scope**: 7 primary sources (FRED/ALFRED, EDGAR, GDELT, IMF, OECD, BEA, BIS)
**Critical requirement**: every source must have its publication lag documented and
honoured by the PIT store (Assumption A1).

---

## Quick-Reference Summary

| Source | Protocol | Auth | Coverage start | Frequency | Pub lag | A1 vintage? | Actors covered |
|---|---|---|---|---|---|---|---|
| FRED / ALFRED | REST JSON | API key (free) | 1913+ (varies) | daily/monthly/quarterly | 1–30 d | ✅ ALFRED | Layer 0, 1 |
| SEC EDGAR | REST JSON + XBRL | User-Agent header | 1993+ (XBRL: 2009+) | quarterly/annual | 45–90 d | ❌ point-in-time via index | Layer 2, 3 |
| GDELT 2.0 | Bulk CSV + REST | None | 2013-02-19 | 15-minute | ~15 min | ❌ append-only | Layer 1 (narrative) |
| IMF SDMX | REST JSON/XML | None | 1948+ (varies) | monthly/quarterly/annual | 60–180 d | ❌ revision flags only | Layer 0, 1 |
| OECD SDMX | REST JSON | None | 1960+ (varies) | quarterly/annual | 60–120 d | ❌ revision flags only | Layer 1 |
| BEA I/O | REST JSON | API key (free) | 1997+ | annual | ~18–24 months | ❌ benchmark vs. annual | Layer 2, 3 |
| BIS Statistics | REST SDMX | None | 1970+ (varies) | quarterly | 90–180 d | ❌ revision flags only | Layer 1, 2 |

---

## 1  FRED / ALFRED (Federal Reserve Economic Data)

### Endpoints

| Operation | URL pattern |
|---|---|
| Series observations | `https://api.stlouisfed.org/fred/series/observations?series_id={ID}&api_key={KEY}&file_type=json` |
| Vintage (ALFRED) | Add `realtime_start={YYYY-MM-DD}&realtime_end={YYYY-MM-DD}` to any observations call |
| Series search | `https://api.stlouisfed.org/fred/series/search?search_text={q}&api_key={KEY}` |
| Series categories | `https://api.stlouisfed.org/fred/series/categories?series_id={ID}&api_key={KEY}` |

**Authentication**: free API key via FRED account registration.

**Rate limits**: 120 requests/minute. Burst of up to 1000 requests before throttling.

**Response format**: JSON (`file_type=json`) or XML. JSON preferred.

### Coverage and Frequency

| FRED Series | Description | Earliest obs | Frequency | Pub lag |
|---|---|---|---|---|
| `DCOILBRENTEU` | Brent crude spot price (USD/barrel) | 1987-05-20 | Daily | 1 day |
| `DHHNGSP` | Henry Hub natural gas spot | 1997-01-07 | Daily | 1 day |
| `FEDFUNDS` | Federal Funds Effective Rate | 1954-07-01 | Monthly | 5–7 days after month-end |
| `GPRC_US` | GPR Index — US component | 1985-01-01 | Monthly | ~30 days after month-end |
| `CPIENGSL` | CPI Energy (US) | 1957-01-01 | Monthly | 14–16 days after month-end |
| `INDPRO` | Industrial Production Index | 1919-01-01 | Monthly | 16–18 days after month-end |
| `SLSHERE` | Senior Loan Officer Survey — energy-sector standards | 1990-01-01 | Quarterly | 5–10 days after survey |
| `WPUFD49207` | PPI — Crude petroleum (domestic production) | 1974-01-01 | Monthly | 14–16 days |
| `EURUSD` | EUR/USD exchange rate | 1999-01-04 | Daily | 1 day |
| `GBPUSD` | GBP/USD exchange rate | 1971-01-04 | Daily | 1 day |

**ALFRED vintage support**: Yes. ALFRED (Archival FRED) stores all real-time vintages
via `realtime_start` / `realtime_end` parameters. This is the **only** FRED/ALFRED
source for which full PIT discipline can be verified mechanically without an
external publication-lag table.

Example call (vintage as of 2020-01-01):
```
https://api.stlouisfed.org/fred/series/observations
  ?series_id=FEDFUNDS
  &realtime_start=2005-01-01&realtime_end=2025-01-01
  &observation_start=2004-01-01
  &api_key={KEY}&file_type=json
```

**Actor coverage**: Layer 0 shock series (`GLOBAL_SHOCK`), Layer 1 central bank
indicators (`CENTRAL_BANK`), macro backdrop for all layers.

**A1 compliance**: Full via ALFRED vintage retrieval. For series not in ALFRED,
apply `publication_lag_buffer_days = 5` conservatively.

---

## 2  SEC EDGAR

### Endpoints

| Operation | URL pattern |
|---|---|
| Company facts (all XBRL) | `https://data.sec.gov/api/xbrl/companyfacts/CIK{zero-padded-10}.json` |
| Single concept | `https://data.sec.gov/api/xbrl/companyconcept/CIK{10}/us-gaap/{tag}.json` |
| Company submissions | `https://data.sec.gov/submissions/CIK{10}.json` |
| Full-text search | `https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={}&enddt={}` |
| Bulk facts ZIP | `https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/CY{year}Q{q}I.json` |

**Authentication**: No API key. Mandatory `User-Agent` header:
`User-Agent: SMIM Research research@yourorg.com`

**Rate limits**: 10 requests/second per IP. For bulk retrieval use 0.1 s sleep
between requests or the bulk ZIP downloads (recommended for full-panel pulls).

**Response format**: JSON for all XBRL APIs.

### Coverage

| Dimension | Detail |
|---|---|
| XBRL filers (large accelerated) | 2009-Q1 onwards (SEC mandate effective 2009-06-15) |
| XBRL filers (accelerated) | 2010-Q1 onwards |
| XBRL filers (all public companies) | 2011-Q1 onwards |
| 10-K / 10-Q filing history | 1993+ via EDGAR full-text |
| 20-F (foreign private issuers) | 1993+; XBRL from 2011 |
| Frequency | Quarterly (10-Q), Annual (10-K) |

### Key XBRL Tags for Energy Actors

| Tag (us-gaap namespace) | Description | Period type |
|---|---|---|
| `CapitalExpenditures` | Capital expenditures (cash flow statement) | Duration (quarterly/annual) |
| `PaymentsToAcquirePropertyPlantAndEquipment` | CapEx — alternative tag used by some filers | Duration |
| `Assets` | Total assets | Instant (period-end) |
| `PropertyPlantAndEquipmentNet` | PP&E net (proxy for energy capital stock) | Instant |
| `ResearchAndDevelopmentExpense` | R&D (relevant for energy transition firms) | Duration |
| `LongTermDebtNoncurrent` | Long-term debt (leverage indicator) | Instant |
| `Revenues` | Total revenues (denominator for intensity measures) | Duration |

**Publication lag**: 10-Q must be filed within 40 days of quarter-end (large
accelerated filers) or 45 days (accelerated filers). 10-K within 60 or 75 days
of fiscal year-end. In practice the median lag for energy firms is ~42 days for
10-Q. **Use `period_of_report + 45 days` as the conservative pub date for the PIT
store when no filing-date metadata is available.**

The EDGAR API returns both `filed` (actual filing date) and `accessionNumber`,
so **exact filing dates are available** — use them directly in the PIT store rather
than relying on the conservative buffer.

**Entity resolution**: EDGAR CIK is the canonical identifier. Cross-reference to
LEI via GLEIF bulk file (`https://www.gleif.org/en/lei-data/gleif-concatenated-file`).

**Actor coverage**: Layer 2 large firms (`LARGE_FIRM`), Layer 3 SME aggregates
(`SME` — via SIC-code filtered bulk frames).

**A1 compliance**: Direct — use the `filed` date from the submission index as
`pub_date` in the PIT store. Filing date is immutable once recorded by EDGAR.

---

## 3  GDELT 2.0

### Endpoints

| Operation | URL / method |
|---|---|
| 15-minute export feed | `http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.export.CSV.zip` |
| Mention feed | `http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.mentions.CSV.zip` |
| GKG (Global Knowledge Graph) | `http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.gkg.csv.zip` |
| Master file list | `http://data.gdeltproject.org/gdeltv2/masterfilelist.txt` |
| DOC API v2 | `https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist` |

**Authentication**: None for bulk CSV downloads. Google Cloud credentials needed
for BigQuery access (not required for MVP).

**Rate limits**: Bulk downloads — no stated limit; recommend ≤ 2 concurrent.
DOC API — undocumented; stay under 1 req/s.

**Response format**: CSV (pipe-delimited for export/GKG). Tab-delimited for mentions.
DOC API returns JSON.

### Coverage

| Dimension | Detail |
|---|---|
| GDELT 1.0 (daily) | 1979-01-01 to present |
| GDELT 2.0 (15-minute) | 2013-02-19 to present |
| Languages | 65+ languages; English-language sources most dense |
| GKG themes | ~2,700 predefined CAMEO-derived themes |
| Update frequency | Every 15 minutes (new ZIP file added to master list) |
| Publication lag | ~15 minutes from article publication to GDELT indexing |

### MVP-Relevant Themes

| Theme prefix | Scope | MVP relevance |
|---|---|---|
| `ECON_ENERGY` | Energy economics news | Layer 1 think-tank narrative, Layer 2 firm news |
| `ENV_ENERGY` | Environmental / energy transition | Layer 1 regulator narrative |
| `GOV_ENERGY` | Government energy policy | Layer 1 regulator, IEA/OPEC signals |
| `ECON_` (broader) | General economic commentary | Macro backdrop |

### GKG Fields Used

| Field | Position | Content |
|---|---|---|
| `DATE` | 1 | Timestamp YYYYMMDDHHMMSS |
| `THEMES` | 10 | Semicolon-separated theme codes |
| `TONE` | 15 | 6-element string: `tone,pos,neg,polarity,actref_density,selfref_density` |
| `ORGANIZATIONS` | 11 | Named organizations mentioned |
| `LOCATIONS` | 9 | FIPS country + admin codes |

**Note**: GDELT 1.0 data (1979–2013) uses daily granularity only and has a
different schema. The narrative channel (C4) can only be estimated on 2013–2025
for the MVP experiment (documented limitation in `scope_selection.md`).

**Actor coverage**: Layer 1 think tanks and regulators (`THINK_TANK`, `REGULATOR`,
`INTL_ORG`) via organization mentions + theme filtering.

**A1 compliance**: GDELT is append-only; each 15-minute file is never revised.
The `DATE` field is the event timestamp; `pub_date` in PIT store = file timestamp
(available from master file list). No vintage issue — data is not revised.

---

## 4  IMF SDMX (IFS Dataset)

### Endpoints

| Operation | URL pattern |
|---|---|
| Data query | `http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{database}/{freq}.{country}.{indicator}` |
| Data structure | `http://dataservices.imf.org/REST/SDMX_JSON.svc/DataStructure/{database}` |
| Available databases | `http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow` |
| Code lists | `http://dataservices.imf.org/REST/SDMX_JSON.svc/CodeList/{codelist_id}` |

**Authentication**: None. No API key required.

**Rate limits**: Not officially documented. Empirically: ~10 requests/second before
soft throttling. Recommend 100 ms sleep between requests. The service does not
return `Retry-After` headers — use exponential backoff on HTTP 429.

**Response format**: JSON (SDMX-JSON) or XML. Request JSON via `Accept: application/json`
header or use the `SDMX_JSON.svc` endpoint prefix.

### IFS Dataset (MVP target)

| Dimension | Detail |
|---|---|
| Database code | `IFS` |
| Coverage | 190+ countries, 1948–present for some series |
| Frequency codes | `A` (annual), `Q` (quarterly), `M` (monthly) |
| Key indicator codes (energy-relevant) | |

| Indicator | Code | Frequency | Coverage |
|---|---|---|---|
| GDP at current prices (USD) | `NGDPD_IX` or `NGDP_R_K_IX` | Annual | 1980+ |
| Consumer price index | `PCPI_IX` | Monthly | 1957+ |
| Broad money growth | `FM_BM_USD_CN` | Quarterly | 1975+ |
| Capital account balance | `BCA_BP6_USD` | Quarterly | 1991+ |
| Credit to private sector | `FCR_BP6_USD` | Quarterly | 1995+ |
| Exchange rates (USD/GBP) | `ENDE_A_GBP_USD_RATE` | Daily/Monthly | 1957+ |

**Publication lag**: 2–4 months for quarterly data; 6–12 months for annual.
IFS does not publish vintage archives — revisions are applied silently.
**A1 compliance**: Apply `publication_lag_buffer_days = 90` for quarterly series
and `180` for annual series. Use the IMF DataMapper `lastUpdateDate` metadata
field where available to infer approximate pub date.

**Actor coverage**: Layer 0 macro shock indicators (`GLOBAL_SHOCK`) and Layer 1
international organisation signals (`INTL_ORG`).

---

## 5  OECD SDMX (QNA Dataset)

### Endpoints

| Operation | URL pattern |
|---|---|
| Data query | `https://sdmx.oecd.org/public/rest/data/{agency},{dataset},{version}/{key}?format=jsondata` |
| Legacy endpoint | `https://stats.oecd.org/SDMX-JSON/data/{dataset}/{key}/OECD` |
| Structure query | `https://sdmx.oecd.org/public/rest/datastructure/OECD,{DSD_ID},{version}` |
| Available datasets | `https://sdmx.oecd.org/public/rest/dataflow/OECD` |

**Authentication**: None.

**Rate limits**: Undocumented; 1 request/second is reliable. The new SDMX 2.1
endpoint (`sdmx.oecd.org`) is more reliable than the legacy `stats.oecd.org`.

**Response format**: SDMX-JSON (via `?format=jsondata`) or SDMX-ML XML.

### QNA Dataset (MVP target)

| Dimension | Detail |
|---|---|
| Dataset code | `QNA` |
| Coverage | 38 OECD members + G20, back to 1960 for major economies |
| Frequency | Quarterly |
| Key measure | `B1_GE` (GDP, expenditure approach), `P51G` (Gross fixed capital formation) |

| QNA Series | Key (example for US) | Description |
|---|---|---|
| Gross fixed capital formation | `USA.P51G.GPSA.Q` | Total investment, seasonally adjusted |
| GFCF — energy sector | `USA.P51G_ISIC_D.GPSA.Q` | Energy-sector GFCF (where available) |
| GDP growth | `USA.B1_GE.GPSA.Q` | GDP, quarterly growth |
| Household consumption | `USA.P31S14_S15.GPSA.Q` | Demand-side driver for Layer 3 |

**Publication lag**: 60–90 days after quarter-end for preliminary estimates;
revised 1–2 quarters later. OECD provides a data revision policy document but
does not archive full vintage datasets via SDMX.
**A1 compliance**: Apply `publication_lag_buffer_days = 75` for QNA quarterly data.
Flag QNA-derived signals as "non-vintaged" in PIT store; do not use for
fine-grained PIT calculations.

**Actor coverage**: Layer 1 international macro context (`INTL_ORG`), macro
backdrop for all layer intensities.

---

## 6  BEA Input-Output Tables

### Endpoints

| Operation | URL pattern |
|---|---|
| Data query | `https://apps.bea.gov/api/data?UserID={KEY}&method=GetData&DataSetName={DS}&TableName={T}&Frequency=A&Year={Y}&ResultFormat=JSON` |
| Dataset list | `https://apps.bea.gov/api/data?UserID={KEY}&method=GetDataSetList&ResultFormat=JSON` |
| Table list | `https://apps.bea.gov/api/data?UserID={KEY}&method=GetParameterValues&DataSetName=InputOutput&ParameterName=TableID&ResultFormat=JSON` |

**Authentication**: Free API key via BEA registration (`apps.bea.gov`).

**Rate limits**: 1,000 requests/day per key. 30 requests/minute.

**Response format**: JSON or XML.

### InputOutput Dataset (MVP target)

| Table | ID | Description |
|---|---|---|
| Use of Commodities by Industries (before redefinitions) | 259 | Direct requirements; use for supply-chain edge estimation |
| Make of Commodities by Industries | 258 | Industry production structure |
| Total requirements (industry × industry) | 261 | Leontief total requirement coefficients |
| Detailed Use table (85 × 71 industries) | 2U | Most granular; energy industries: rows 6–9 |

**Coverage**: Annual, 1997–present for detailed tables. Benchmark I/O tables
released every 5 years (1997, 2002, 2007, 2012, 2017); annual tables are
interpolated between benchmarks.

**Energy industries in BEA I/O** (BEA industry codes):
- Code 6: Oil and gas extraction
- Code 7: Mining (except oil and gas)
- Code 31: Petroleum and coal products manufacturing
- Code 32: Chemical manufacturing (energy-adjacent)
- Code 55: Utilities (electric, gas, water)

**Publication lag**: Annual I/O tables published approximately **18–24 months**
after the reference year (e.g., 2022 I/O tables published late 2024). Benchmark
tables take ~3 years.
**A1 compliance**: Apply `publication_lag_buffer_days = 548` (18 months) for I/O
data. Store BEA tables with reference year + 18-month offset as `pub_date`.

**Actor coverage**: Layer 2 supply-chain relationships for `LARGE_FIRM` and
`SECTOR_LEADER`. Layer 3 downstream investment flows for `SME` aggregates.
Used primarily for supply-chain edge estimation (channel C5).

---

## 7  BIS Statistics (CBS + LBS + PP)

### Endpoints

| Operation | URL pattern |
|---|---|
| Data query | `https://data.bis.org/api/v1/data/{datasetId}/{key}?format=csv` |
| Dataset catalogue | `https://data.bis.org/api/v1/dataflow/BIS` |
| Structure | `https://data.bis.org/api/v1/datastructure/BIS/{DSD_ID}` |
| Bulk ZIP downloads | `https://www.bis.org/statistics/full_data_sets.htm` |

**Authentication**: None.

**Rate limits**: Not officially documented. Bulk ZIP downloads are preferred for
full-panel pulls; REST for targeted series.

**Response format**: CSV (with SDMX dimension headers) or JSON-stat.

### Datasets (MVP targets)

| Dataset | Code | Description | Frequency | Coverage |
|---|---|---|---|---|
| Consolidated banking statistics | `CBS` | Cross-border bank claims by counterparty country | Quarterly | 1977-Q1+ |
| Locational banking statistics | `LBS` | Bilateral cross-border positions by currency | Quarterly | 1977-Q1+ |
| Property prices | `PP` | Residential + commercial property price indices | Quarterly | 1970+ (varies) |
| Long-term interest rates | `WEBSTATS` | 10-year sovereign yields | Monthly | 1960+ |
| Credit to private non-financial sector | `TOTAL_CREDIT` | Domestic credit/GDP | Quarterly | 1940+ (varies) |

### Key CBS Dimensions (for bank edge estimation)

| Dimension | Code | MVP use |
|---|---|---|
| Reporting bank country | `RPC` | `US`, `GB` |
| Counterparty country | `CT` | Filter to energy-producing countries |
| Bank type | `BT` | `A` (all banks) |
| Position type | `BS` | `B` (claims by sector), `C` (claims by industry) |
| Currency | `CY` | `TO1` (all currencies, US dollar-equivalent) |

**Publication lag**: 90–150 days after reference quarter. BIS typically publishes
CBS/LBS data in the 3rd month after quarter-end (e.g., Q1 data in late June).
**A1 compliance**: Apply `publication_lag_buffer_days = 105` (3.5 months) for
BIS quarterly datasets. BIS data is revised; vintages are not archived.
Flag as "non-vintaged" in PIT store.

**Actor coverage**: Layer 2 energy-sector banks (`BANK`): cross-border credit
exposure provides financial channel (C2) edge weights. Layer 1 macro backdrop.

---

## Coverage Gap Analysis

### By Signal Family (Proposal §3)

| Signal family | Source | Coverage | Gap |
|---|---|---|---|
| **Macro / policy rates** | FRED/ALFRED | Full (ALFRED vintage) | None — fully vintaged |
| **Firm-level investment** | EDGAR XBRL | 2009+ quarterly | 2005–2008: annual 10-K only; no quarterly CapEx for pre-XBRL period |
| **Narrative / sentiment** | GDELT 2.0 | 2013-02-19+ | 2005–2013: no narrative channel (C4); must exclude from pre-2013 edge estimation |
| **International macro** | IMF IFS | 1948+ (un-vintaged) | No vintage archive; A1 compliance via lag buffer only |
| **OECD macro aggregates** | OECD QNA | 1960+ (un-vintaged) | As IMF — no vintage |
| **Supply-chain linkages** | BEA I/O | 1997+ (annual) | Long pub lag (18+ months); use 2-year lagged I/O for point-in-time supply-chain edges |
| **Cross-border bank flows** | BIS CBS/LBS | 1977+ (quarterly, un-vintaged) | Reporting country breakdown available but industry breakdown limited to broad sectors |
| **UK-specific regulatory** | BoE / OFGEM | Direct downloads only | Not covered by any of the 7 primary sources; requires manual scraping |
| **Energy price volatility (UK)** | FRED | Limited (ICE Brent global) | No UK-specific natural gas spot (NBP) in FRED; needs direct ICE/Refinitiv download |

### By Actor Layer

| Layer | Sources | Coverage rating | Notes |
|---|---|---|---|
| 0 — Exogenous | FRED | ★★★★★ | ALFRED vintage; complete for MVP series |
| 1 — Upstream | FRED + GDELT + IMF + OECD | ★★★☆☆ | Narrative gap 2005–2013; IMF/OECD un-vintaged |
| 2 — Transmission | EDGAR + BIS | ★★★★☆ | EDGAR gap 2005–2008 (annual only); BIS industry breakdown coarse |
| 3 — Downstream | BEA + EDGAR (aggregates) | ★★★☆☆ | BEA 18-month lag; SME XBRL coverage sparse |

### PIT Store: Recommended pub_date Offsets

| Source | Recommended `pub_date` formula | Vintage available? |
|---|---|---|
| FRED/ALFRED | `realtime_start` parameter (exact vintage) | ✅ Yes |
| EDGAR | `filed` date from submission index (exact) | ✅ Yes (immutable) |
| GDELT | File timestamp from master list (exact) | ✅ Yes (append-only) |
| IMF IFS | `reference_date + 90 days` (quarterly) | ❌ Buffer only |
| OECD QNA | `reference_date + 75 days` (quarterly) | ❌ Buffer only |
| BEA I/O | `reference_year_end + 548 days` (18 months) | ❌ Buffer only |
| BIS CBS/LBS | `quarter_end + 105 days` (3.5 months) | ❌ Buffer only |

---

## Implementation Priority for M1.2

Adapters should be built in this order (highest data quality / lowest risk first):

1. **FRED/ALFRED** — cleanest PIT compliance; needed for Layer 0 shocks and Layer 1
   central bank signals; ALFRED vintage retrieval is unique value.
2. **EDGAR** — largest actor coverage (all Layer 2 firms); exact filing dates
   available; XBRL bulk frames allow efficient full-panel pulls.
3. **GDELT** — append-only, no vintage issue; bulk CSV processing; covers
   narrative channel C4 (2013+).
4. **BEA I/O** — supply-chain edge estimation (C5); annual only; simplest
   schema; long lag is handled by conservative PIT buffer.
5. **IMF IFS** — macro backdrop; un-vintaged but low revision risk for
   long-established series; needed for Layer 0/1.
6. **BIS CBS** — financial channel edges (C2); coarser industry breakdown
   limits precision; lower priority than EDGAR for Layer 2.
7. **OECD QNA** — macro context; partially redundant with IMF IFS for MVP;
   lowest priority.
