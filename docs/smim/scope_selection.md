# SMIM MVP Scope Selection Report

**Experiment**: `experiments/mvp_energy_us_uk.yaml`
**Actor universe**: ≤ 200 actors | **Date range**: 2005-01-01 – 2025-01-01
**Geographies**: United States + United Kingdom | **Domain**: Energy sector

---

## 1  Why the Energy Sector

### Data availability

Energy is among the most extensively documented sectors in public data.
The following sources provide actor-specific investment data with ≥ 15-year
histories, satisfying the coverage threshold in Gate G1:

| Data family | Source | Coverage |
|---|---|---|
| Corporate CapEx / balance sheet | SEC EDGAR (XBRL) | 2001–present, quarterly |
| Macro investment flows | BEA I/O tables | 2000–2024, annual |
| Energy prices & production | FRED / EIA | 1986–present, daily/monthly |
| Regulatory activity | Federal Register, OFGEM | 1994–present |
| Narrative signals | GDELT themes `ECON_ENERGY`, `ENV_ENERGY` | 2013–present, daily |
| International flows | IEA World Energy Investment | 2005–present, annual |

EDGAR alone provides quarterly CapEx data for > 300 US energy companies
(SIC codes 1311, 1381, 4911, 4931, 4941), giving ample cross-section for
Layer 2 actors with no manual data collection required.

### Clear layer hierarchy

Energy is one of the few sectors where the four-layer hierarchy is empirically
obvious and testable:

- **Layer 0**: Commodity prices (Brent, Henry Hub), geopolitical shock indices.
- **Layer 1**: Federal Reserve (cost of capital), EPA/DOE (regulatory burden),
  IEA (international investment norms), OFGEM (UK price controls).
- **Layer 2**: Integrated majors (ExxonMobil, Shell, BP), independent E&P firms,
  large project-financing banks (JPMorgan Energy, Barclays Project Finance).
- **Layer 3**: SME operators (aggregated by SIC sub-category), UK local energy
  authorities, household-level demand proxies.

The directed influence from Layer 1 → Layer 2 → Layer 3 is supported by
an existing empirical literature on energy investment cascades (IEA 2023,
Dietz & Stern 2015), providing external validity for the estimated edge structure.

### Policy sensitivity

Energy investment responds sharply to identifiable policy and price shocks,
giving the event-study validation (M5.3) five clearly-scoped events:
Fed tightening 2018, COVID-19 2020, Ukraine energy crisis 2022,
Inflation Reduction Act 2022, BoE rate cycle 2022.
These events span both geographies and both Layer 0 and Layer 1 drivers,
stress-testing all four layers of the hierarchy.

---

## 2  Why United States + United Kingdom

### Deep public data

Both jurisdictions mandate quarterly XBRL filings:
- US: SEC 10-Q/10-K, FDIC call reports, FRED macroeconomic series.
- UK: Companies House XBRL (since 2016), Bank of England statistical releases,
  ONS national accounts, OFGEM regulatory asset base data.

Entity resolution between EDGAR CIKs and UK company numbers achieves
> 90% match rate for the top-200 energy firms using LEI cross-reference
(requirement: ≥ 90% at Gate G1).

### Comparable regulatory structures

Both countries have:
- Independent central banks with transparent forward guidance (Fed, BoE).
- Sector-specific energy regulators (EPA/DOE in the US; OFGEM in the UK)
  with public filings.
- Active net-zero legislation (IRA 2022, UK Climate Change Act 2008/2019)
  generating detectable step-changes in investment intensity.

This means the regulatory channel (C1) can be estimated for both
geographies with comparable data quality, enabling a two-data-regime
robustness check within the MVP scope.

### Two data regimes

US and UK differ in:
- Currency and financing conditions (USD vs GBP, differing rate cycles).
- Regulatory intensity (IRA subsidy structure vs UK Carbon Budget mechanism).
- Market structure (US unconventional oil & gas dominance vs UK offshore/renewables).

These differences create natural variation in edge weights and regime
dynamics, making it possible to assess whether the estimated graph
structure is domain-specific or geography-specific — a key question
for the scope-transfer test in M6.3.

---

## 3  Actor Universe Breakdown

Full taxonomy: `docs/smim/actor_taxonomy.md`

| Layer | Actor types | MVP count | Notes |
|---|---|---|---|
| 0 — Exogenous | `GLOBAL_SHOCK` | 5 | Energy price, nat-gas, carbon-price, geopolitical, macro-vol indices |
| 1 — Upstream | `CENTRAL_BANK`, `REGULATOR`, `INTL_ORG`, `THINK_TANK` | 15 | Fed, BoE, EPA, OFGEM, IEA, OPEC+, IMF energy desks, 3 think tanks |
| 2 — Transmission | `LARGE_FIRM`, `BANK`, `SECTOR_LEADER` | 100 | Top-50 US energy firms + top-50 UK/EU energy firms + 10 energy banks |
| 3 — Downstream | `SME`, `MUNICIPALITY` | 80 | 30 US SME aggregates + 30 UK equivalents + 20 local authorities |
| **Total** | **9 types** | **200** | Exactly at the `actor_universe_max` cap |

The 200-actor budget is conservative relative to the `actor_universe_max = 500`
allowed by `ScopeConfig`, ensuring that all matrix operations ($N \times N$ graph,
$N \times K^*$ modal frame) remain computationally tractable on a single machine
with < 16 GB RAM for the Phase I experiments.

---

## 4  Date Range Justification (2005–2025)

The 20-year window was chosen to satisfy all model-selection requirements:

| Requirement | Satisfied by |
|---|---|
| ≥ 15 years macro coverage (Gate G1) | 2005–2025 = 20 years |
| ≥ 2 complete economic cycles | GFC (2008–2009), COVID (2020), recovery/inflation (2021–2023) |
| ≥ 5 identifiable event-study shocks | All 5 MVP events fall within 2018–2022 |
| Rolling 5-year OOS holdout (ValidationConfig) | Requires ≥ 15-year in-sample; 2005–2020 in-sample, 2020–2025 holdout |
| Kim filter A5 (regime duration > 8 quarters) | 20 years = 80 quarters; supports M = 4 regimes with mean duration ≥ 20Q |
| DMD reconstruction baseline | ≥ 80 quarters needed for stable Koopman eigenvalues |

**Start: 2005-01-01** — avoids pre-XBRL data gaps for EDGAR; post-Sarbanes-Oxley
reporting consistency; IEA World Energy Investment series starts 2005.

**End: 2025-01-01** — most recent complete calendar year for which annual filings
and BEA I/O tables are available; avoids partial-year data artefacts.

---

## 5  Known Limitations

### Coverage gaps

- **Layer 0 actors** have no EDGAR filings; shock indices must be sourced
  entirely from FRED, which has no publication-lag vintaging for most series.
  Mitigation: apply a conservative `publication_lag_buffer_days = 5` globally
  and use ALFRED vintage retrieval for FRED series where available.

- **UK SME actors** lack EDGAR-quality quarterly XBRL filings before 2016.
  Mitigation: use annual Companies House filings with linear interpolation;
  flag interpolated quarters in the PIT store.

- **GDELT narrative signals** begin only in 2013, leaving an 8-year narrative-
  data gap (2005–2013) for Layer 1 think-tank actors. The narrative channel (C4)
  will be estimated on 2013–2025 only; the 2005–2013 period uses channels C1–C3
  and C5–C7 only.

### Scope constraints

- The 200-actor universe under-represents the full energy investment system
  (the real global system has > 10,000 relevant actors). The framework is
  designed to be scaled, but Phase I results are not intended to be
  extrapolated beyond the MVP scope.

- Using NAICS/SIC aggregate proxies for Layer 3 SME actors means the
  model measures *sector-level* rather than *firm-level* gaps for downstream
  actors. This is an acceptable approximation for Phase I but limits
  firm-level policy targeting.

### Methodological

- Both geographies use USD-denominated normalisation where cross-currency
  comparisons are needed (via PPP-adjusted GDP deflator). Exchange-rate
  movements may introduce artificial variation in Layer 2 actor intensities.
  Robustness check: re-run with local-currency normalisation (M2.5 ablation).

- The structural benchmark (§2 in `benchmark_specs.md`) is approximated in
  Phase I; the exact structural decomposition is deferred to WP6 (M6.1).
  Phase I structural gaps should be interpreted with this caveat.
