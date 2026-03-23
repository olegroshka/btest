# SMIM Data Readiness Report

> Generated: 2026-03-23

## Experiment Readiness Matrix

| Experiment | Registry | Intensity | FRED | EDGAR | GDELT | BEA | IMF | OECD |
|------------|----------|-----------|------|-------|-------|-----|-----|------|
| A1 (MIXED-200 energy, full) | OK | OK | OK | OK | OK | OK | OK | OK |
| A2 (US-LC energy sector) | OK | OK | OK | OK | OK | OK | OK | OK |
| B1 (US-LC financials) | OK | OK | OK | OK | OK | OK | OK | OK |
| C1 (US-LC all sectors) | OK | OK | OK | OK | OK | OK | OK | OK |
| D1 (US-LC fast run) | OK | OK | OK | OK | OK | OK | OK | OK |
| E1 (UK-LC) | OK | MISS | OK | OK | OK | OK | OK | OK |

## Intensity Quality Checks

### experiment_a1 (N=23, 1,445 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.839 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| central_bank | 88 | 0.414 | 0.279 | 0.692 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.137 | 0.195 | 2.783 | 0.00% |
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |
| regulator | 44 | 0.239 | 0.224 | 1.205 | 0.00% |

### experiment_fast (N=179, 10,542 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.711 (PASS)
- High-missing actors (>50%): 19

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,536 | 0.488 | 0.160 | 0.685 | 0.00% |
| central_bank | 88 | 0.414 | 0.279 | 0.692 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.137 | 0.195 | 2.783 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### experiment_phased (N=179, 10,542 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.711 (PASS)
- High-missing actors (>50%): 19

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,536 | 0.488 | 0.160 | 0.685 | 0.00% |
| central_bank | 44 | 0.368 | 0.261 | 0.796 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.137 | 0.195 | 2.783 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| regulator | 44 | 0.239 | 0.224 | 1.205 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### MIXED-200 (N=12, 685 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.759 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |

### US-LC-ENERGY (N=12, 685 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.759 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |

### US-LC-FINS (N=70, 4,215 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.040 (WARN — < 0.7)
- High-missing actors (>50%): 2

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 4,150 | 0.491 | 0.169 | 0.408 | 0.00% |
| sector_leader | 65 | 1.000 | 0.000 | 0.000 | 0.00% |

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

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 3,064 | 0.516 | 0.290 | -0.025 | 0.00% |
| sector_leader | 132 | 0.382 | 0.211 | 0.122 | 0.00% |

### US-LC (N=169, 9,826 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.660 (WARN — < 0.7)
- High-missing actors (>50%): 13

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,536 | 0.488 | 0.160 | 0.685 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### US-MC (N=159, 8,461 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.794 (PASS)
- High-missing actors (>50%): 22

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 777 | 0.494 | 0.175 | 0.334 | 0.00% |
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
- Rank stability ρ > 0.7: WARN

### Overall: SOME DATA GAPS — see matrix above
