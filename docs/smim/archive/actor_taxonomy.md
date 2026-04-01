# SMIM Actor Taxonomy — MVP Energy Sector (US + UK)

Experiment scope: `experiments/mvp_energy_us_uk.yaml`
Target actor count: ≤ 200 (budget: ~15 Layer 0–1, ~100 Layer 2, ~80 Layer 3)

Each actor type satisfies the layer-hierarchy constraint from the proposal (Table 2).
At least two actor types appear in each layer. Every type has ≥ 1 named exemplar
and a defined investment-intensity measure $y_{i,t}$.

---

## Layer 0 — Exogenous Environment

Layer 0 actors are not decision-makers; they represent external forcing variables that
enter the model as "global shocks". They have no incoming edges but drive all downstream
layers through channels C1–C7.

| ActorType | `ActorType` enum | Named exemplar | $y_{i,t}$ measure | Primary data source |
|---|---|---|---|---|
| Global shock — energy price | `GLOBAL_SHOCK` | Brent Crude spot index (ICE) | Standardised log-return of commodity price | FRED `DCOILBRENTEU` |
| Global shock — geopolitical | `GLOBAL_SHOCK` | GPR Index (Caldara & Iacoviello) | Percentile rank of GPR index value | FRED `GPRC_US` |

**MVP actor count (Layer 0):** ~5 shock series (energy price, geopolitical, natural-gas, carbon-price, macro-volatility).

---

## Layer 1 — Upstream Institutions

Layer 1 actors shape the rules and financing environment for all downstream actors.
They emit signals through regulatory (C1), fiscal (C3), and narrative (C4) channels.

| ActorType | `ActorType` enum | Named exemplar | $y_{i,t}$ measure | Primary data source |
|---|---|---|---|---|
| Central bank | `CENTRAL_BANK` | Federal Reserve System (US) | Quarterly change in policy rate × energy-sector lending standard index | FRED `FEDFUNDS`, Senior Loan Survey |
| Central bank | `CENTRAL_BANK` | Bank of England (UK) | Quarterly change in Bank Rate × credit conditions index | BoE statistical release |
| Energy regulator | `REGULATOR` | U.S. EPA / DOE joint index | Count of new energy regulations × estimated compliance capex burden / GDP | Federal Register; DOE EIA |
| Energy regulator | `REGULATOR` | OFGEM (UK) | Price control adjustment factor (RIIO-E2) as fraction of RAV | OFGEM annual reports |
| International organisation | `INTL_ORG` | International Energy Agency | IEA member-country investment recommendation compliance score | IEA World Energy Investment |
| Think tank | `THINK_TANK` | Resources for the Future (US) | Publication-weighted citation index on energy-investment topics | GDELT theme `ECON_ENERGY` |

**MVP actor count (Layer 1):** ~15 institutions (6 central banks/regulators per geography + IEA/OPEC/IMF).

---

## Layer 2 — Transmission / Intermediaries

Layer 2 actors translate upstream signals into capital deployment decisions and pass
them downstream via financial (C2) and supply-chain (C5) channels.

| ActorType | `ActorType` enum | Named exemplar | $y_{i,t}$ measure | Primary data source |
|---|---|---|---|---|
| Large energy firm (integrated) | `LARGE_FIRM` | ExxonMobil Corporation | CapEx / Total Assets (quarterly, XBRL tag `CapitalExpenditures` / `Assets`) | SEC EDGAR 10-Q |
| Large energy firm (integrated) | `LARGE_FIRM` | Shell plc (UK-listed) | CapEx / Total Assets (quarterly) | SEC EDGAR 20-F |
| Large energy firm (renewables) | `LARGE_FIRM` | NextEra Energy Inc. | CapEx / Total Assets (quarterly) | SEC EDGAR 10-Q |
| Large energy firm (UK utility) | `LARGE_FIRM` | SSE plc | CapEx / Regulatory Asset Value | LSE annual reports / XBRL |
| Energy-sector bank | `BANK` | JPMorgan Chase & Co. | Energy-sector loan growth (QoQ %, normalised) | FR Y-9C Call Report |
| Energy-sector bank | `BANK` | Barclays plc | Project finance origination in energy (GBP bn, log-normalised) | Barclays annual report |
| Sector leader (index proxy) | `SECTOR_LEADER` | S&P 500 Energy Sector ETF (XLE) | Aggregate CapEx/Assets of top-10 index constituents | EDGAR + Bloomberg |

**MVP actor count (Layer 2):** ~100 firms (top-50 US energy firms by assets + top-50 UK/EU energy firms; ~10 major energy-financing banks).

---

## Layer 3 — Downstream Actors

Layer 3 actors respond to the environment set by Layers 0–2. They receive signals
through imitation (C6) and market-implied (C7) channels. In aggregate they represent
the "investment base" that the framework aims to explain.

| ActorType | `ActorType` enum | Named exemplar | $y_{i,t}$ measure | Primary data source |
|---|---|---|---|---|
| SME energy operator | `SME` | US small energy producers (SIC 1311 aggregate) | Quarterly gross fixed capital formation / total assets (sector aggregate) | BEA Input-Output Tables, EIA survey |
| SME energy operator | `SME` | UK independent oil & gas (AIM-listed aggregate) | CapEx / Assets (aggregate of AIM energy sector) | EDGAR + XBRL UK |
| Municipality / local authority | `MUNICIPALITY` | New York State Energy Research & Development Authority (NYSERDA) | Energy infrastructure budget / total capital budget | NY state CAFR |
| Municipality / local authority | `MUNICIPALITY` | UK local energy authority composite | Local authority renewable energy capex / GVA (ONS) | ONS Local Gov Finance |

**MVP actor count (Layer 3):** ~80 aggregated entities (30 US SME sector aggregates × industry × geography + 30 UK equivalents + 20 local authorities).

---

## Actor Universe Summary

| Layer | Count (MVP) | Actor types used |
|---|---|---|
| 0 | 5 | `GLOBAL_SHOCK` |
| 1 | 15 | `CENTRAL_BANK`, `REGULATOR`, `INTL_ORG`, `THINK_TANK` |
| 2 | 100 | `LARGE_FIRM`, `BANK`, `SECTOR_LEADER` |
| 3 | 80 | `SME`, `MUNICIPALITY` |
| **Total** | **200** | 9 distinct `ActorType` values |

---

## Actor Data (Machine-readable)

The YAML block below is parsed by `ActorRegistry.from_taxonomy()` and contains the
**named exemplar actors** (one per row above). Full production registries are built
from data pipelines, not from this file.

```yaml
actors:
  - actor_id: brent_crude_shock
    name: "Brent Crude Spot Index"
    actor_type: global_shock
    layer: 0
    geography: GLOBAL
    sector: energy

  - actor_id: gpr_geopolitical_shock
    name: "Geopolitical Risk Index (Caldara & Iacoviello)"
    actor_type: global_shock
    layer: 0
    geography: GLOBAL
    sector: energy

  - actor_id: fed_us
    name: "Federal Reserve System"
    actor_type: central_bank
    layer: 1
    geography: US
    sector: energy
    external_ids:
      lei: "549300VL8XB1CLLX8495"

  - actor_id: boe_uk
    name: "Bank of England"
    actor_type: central_bank
    layer: 1
    geography: UK
    sector: energy
    external_ids:
      lei: "YVNBKH52HXQMTJRJ5E76"

  - actor_id: epa_doe_us
    name: "U.S. EPA / DOE Energy Regulation Index"
    actor_type: regulator
    layer: 1
    geography: US
    sector: energy

  - actor_id: ofgem_uk
    name: "OFGEM"
    actor_type: regulator
    layer: 1
    geography: UK
    sector: energy

  - actor_id: iea
    name: "International Energy Agency"
    actor_type: intl_org
    layer: 1
    geography: GLOBAL
    sector: energy

  - actor_id: rff_us
    name: "Resources for the Future"
    actor_type: think_tank
    layer: 1
    geography: US
    sector: energy

  - actor_id: xom_us
    name: "ExxonMobil Corporation"
    actor_type: large_firm
    layer: 2
    geography: US
    sector: energy
    external_ids:
      cik: "0000034088"
      lei: "549300IIBAJFGMNMS349"

  - actor_id: shell_uk
    name: "Shell plc"
    actor_type: large_firm
    layer: 2
    geography: UK
    sector: energy
    external_ids:
      cik: "0001306965"

  - actor_id: nee_us
    name: "NextEra Energy Inc."
    actor_type: large_firm
    layer: 2
    geography: US
    sector: energy
    external_ids:
      cik: "0000753308"

  - actor_id: sse_uk
    name: "SSE plc"
    actor_type: large_firm
    layer: 2
    geography: UK
    sector: energy

  - actor_id: jpm_us
    name: "JPMorgan Chase & Co."
    actor_type: bank
    layer: 2
    geography: US
    sector: energy

  - actor_id: barclays_uk
    name: "Barclays plc"
    actor_type: bank
    layer: 2
    geography: UK
    sector: energy
    external_ids:
      lei: "G5GSEF7VJP5I7OUK5573"

  - actor_id: xle_sector_leader
    name: "S&P 500 Energy Sector (XLE aggregate)"
    actor_type: sector_leader
    layer: 2
    geography: US
    sector: energy

  - actor_id: us_sme_energy_sic1311
    name: "US Small Energy Producers (SIC 1311 aggregate)"
    actor_type: sme
    layer: 3
    geography: US
    sector: energy

  - actor_id: uk_aim_energy_sme
    name: "UK AIM-listed Energy SME Aggregate"
    actor_type: sme
    layer: 3
    geography: UK
    sector: energy

  - actor_id: nyserda_us
    name: "New York State Energy Research & Development Authority"
    actor_type: municipality
    layer: 3
    geography: US
    sector: energy

  - actor_id: uk_local_energy_authority
    name: "UK Local Energy Authority Composite"
    actor_type: municipality
    layer: 3
    geography: UK
    sector: energy
```
