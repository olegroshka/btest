# SMIM Data Readiness Report

> Generated: 2026-03-28 (RP1 complete: BankCreditMapper cross-section rank fix)

## Experiment Readiness Matrix

| Experiment | Registry | Intensity | FRED | EDGAR | GDELT | BEA | IMF | OECD |
|------------|----------|-----------|------|-------|-------|-----|-----|------|
| A1 (MIXED-200 energy, full) | OK | OK | OK | OK | OK | OK | OK | OK |
| A2 (US-LC energy sector) | OK | OK | OK | OK | OK | OK | OK | OK |
| B1 (US-LC financials) | OK | OK | OK | OK | OK | OK | OK | OK |
| C1 (US-LC all sectors) | OK | OK | OK | OK | OK | OK | OK | OK |
| D1 (US-LC fast run) | OK | OK | OK | OK | OK | OK | OK | OK |
| E1 (UK-LC) | OK | OK | OK | OK | OK | OK | OK | OK |

## Intensity Quality Checks

### experiment_a1 (N=23, 1,445 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.844 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| central_bank | 88 | 0.414 | 0.282 | 0.689 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.106 | 0.203 | 2.866 | 0.00% |
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |
| regulator | 44 | 0.220 | 0.267 | 1.422 | 0.00% |

### experiment_fast (N=179, 10,413 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.782 (PASS)
- High-missing actors (>50%): 20

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,407 | 0.513 | 0.289 | -0.000 | 0.00% |
| central_bank | 88 | 0.414 | 0.282 | 0.689 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.106 | 0.203 | 2.866 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### experiment_phased (N=179, 10,413 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.782 (PASS)
- High-missing actors (>50%): 20

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,407 | 0.513 | 0.289 | -0.000 | 0.00% |
| central_bank | 44 | 0.371 | 0.267 | 0.807 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.106 | 0.203 | 2.866 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| regulator | 44 | 0.220 | 0.267 | 1.422 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### MIXED-200 (N=12, 685 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.759 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |

### UK-LC (N=97, 7,237 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.732 (PASS)
- High-missing actors (>50%): 6
- **Methodology: `return_12m_xsrank` (Path B)** — rolling 12-month return, no Companies House data.

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 7,077 | 0.504 | 0.289 | 0.006 | 0.00% |
| sector_leader | 160 | 0.566 | 0.283 | -0.252 | 0.00% |

### UK-MC (N=94, 6,480 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.720 (PASS)
- High-missing actors (>50%): 12

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 2,982 | 0.501 | 0.282 | -0.002 | 0.00% |
| retail_investor | 1,349 | 0.499 | 0.288 | 0.028 | 0.00% |
| sector_leader | 152 | 0.564 | 0.295 | -0.150 | 0.00% |
| sme | 1,997 | 0.514 | 0.298 | -0.014 | 0.00% |

### US-LC-ENERGY (N=12, 685 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.759 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |

### US-LC-FINS (N=70, 4,008 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.769 (PASS)
- High-missing actors (>50%): 2

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 3,943 | 0.508 | 0.289 | -0.000 | 0.00% |
| sector_leader | 65 | 0.490 | 0.208 | 0.494 | 0.00% |

### US-LC-HEALTH (N=51, 2,899 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.708 (PASS)
- High-missing actors (>50%): 6

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 2,782 | 0.514 | 0.290 | -0.001 | 0.00% |
| sector_leader | 117 | 0.438 | 0.249 | -0.278 | 0.00% |

### US-LC-INDUS (N=59, 3,536 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.733 (PASS)
- High-missing actors (>50%): 3

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 3,470 | 0.506 | 0.290 | 0.029 | 0.00% |
| sector_leader | 66 | 0.708 | 0.095 | -0.796 | 0.00% |

### US-LC-TECH (N=60, 3,196 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.653 (WARN — < 0.7)
- High-missing actors (>50%): 11
- **Note:** WARN is due to 11/60 high-missing actors (sparse XBRL CapEx coverage for recent-IPO tech firms). Not a mapper issue.

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 3,064 | 0.516 | 0.290 | -0.025 | 0.00% |
| sector_leader | 132 | 0.382 | 0.211 | 0.122 | 0.00% |

### US-LC (N=169, 9,697 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.761 (PASS)
- High-missing actors (>50%): 13

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,407 | 0.513 | 0.289 | -0.000 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### US-MC (N=159, 8,419 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.819 (PASS)
- High-missing actors (>50%): 22

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 735 | 0.539 | 0.288 | -0.000 | 0.00% |
| large_firm | 3,018 | 0.570 | 0.253 | -0.056 | 0.00% |
| retail_investor | 1,798 | 0.422 | 0.291 | 0.252 | 0.00% |
| sector_leader | 413 | 0.652 | 0.274 | -0.463 | 0.00% |
| sme | 2,455 | 0.459 | 0.303 | 0.163 | 0.00% |

### US-SC (N=142, 6,215 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.905 (PASS)
- High-missing actors (>50%): 48

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| retail_investor | 2,739 | 0.454 | 0.332 | 0.333 | 0.00% |
| sme | 3,476 | 0.546 | 0.242 | -0.227 | 0.00% |

## Pre-Experiment Quality Gate

- All intensity values in [0,1]: PASS
- Rank stability ρ > 0.7: WARN (1 universe)

### WARN breakdown (post-RP1)

| Universe | ρ | Root cause | Status |
|----------|---|-----------|--------|
| US-LC-TECH | 0.653 | 11/60 high-missing actors (sparse XBRL CapEx for recent IPOs) | Acceptable — data quality constraint, not a mapper issue |

### Resolved WARNs (RP1 fix 2026-03-28)

| Universe | ρ before | ρ after | Fix |
|----------|----------|---------|-----|
| US-LC-FINS | -0.003 | 0.769 | BankCreditMapper: cross-sectional rank replaces per-actor z-score sigmoid |
| US-LC | 0.660 | 0.761 | Inherits US-LC-FINS fix (banks are ~25% of US-LC universe) |
| experiment_fast | 0.711 | 0.782 | Same fix propagated |
| experiment_phased | 0.712 | 0.782 | Same fix propagated |

### Overall: DATA READY FOR EXPERIMENTS

**Remediation summary (R1–R6 + RP1 complete 2026-03-28):**
- R1 ✅ FRED CPIMEDSL — 28/28 signals in PIT
- R2 ✅ OECD explicit key fix — 1,922 rows
- R3 ✅ Sector_leader degeneracy fixed; RP1 resolves bank ρ
- R4 ✅ UK intensities — return_12m_xsrank; UK-LC ρ=0.732, UK-MC ρ=0.720
- RP1 ✅ BankCreditMapper cross-section rank — US-LC-FINS ρ: -0.003 → 0.769
