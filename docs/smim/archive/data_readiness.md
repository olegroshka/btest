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
| E1 (UK-LC) | OK | OK | OK | OK | OK | OK | OK | OK |

## Intensity Quality Checks

### experiment_a1 (N=93, 6,332 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.774 | recent (2020–): 0.796 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 620 | 0.550 | 0.287 | 0.000 | 0.00% |
| central_bank | 44 | 0.371 | 0.267 | 0.807 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| intl_org | 44 | 0.106 | 0.203 | 2.866 | 0.00% |
| large_firm | 4,792 | 0.517 | 0.288 | -0.008 | 0.00% |
| regulator | 88 | 0.339 | 0.303 | 0.817 | 0.00% |
| sector_leader | 160 | 0.442 | 0.291 | 0.273 | 0.00% |

### experiment_fast (N=179, 10,413 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.782 | recent (2020–): 0.806 (PASS)
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
- Rank stability (mean Spearman rho): 0.782 | recent (2020–): 0.807 (PASS)
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

### MIXED-200 (N=89, 6,156 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.771 | recent (2020–): 0.787 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 620 | 0.550 | 0.287 | 0.000 | 0.00% |
| global_shock | 584 | 0.349 | 0.288 | 0.693 | 0.00% |
| large_firm | 4,792 | 0.517 | 0.288 | -0.008 | 0.00% |
| sector_leader | 160 | 0.442 | 0.291 | 0.273 | 0.00% |

### UK-LC (N=97, 7,237 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.732 | recent (2020–): 0.722 (PASS)
- High-missing actors (>50%): 6

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 7,077 | 0.504 | 0.289 | 0.006 | 0.00% |
| sector_leader | 160 | 0.566 | 0.283 | -0.252 | 0.00% |

### UK-MC (N=94, 6,480 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.720 | recent (2020–): 0.722 (PASS)
- High-missing actors (>50%): 12

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 2,982 | 0.501 | 0.282 | -0.002 | 0.00% |
| retail_investor | 1,349 | 0.499 | 0.288 | 0.028 | 0.00% |
| sector_leader | 152 | 0.564 | 0.295 | -0.150 | 0.00% |
| sme | 1,997 | 0.514 | 0.298 | -0.014 | 0.00% |

### US-LC-ENERGY (N=12, 685 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.759 | recent (2020–): 0.722 (PASS)
- High-missing actors (>50%): 1

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 685 | 0.548 | 0.288 | -0.001 | 0.00% |

### US-LC-FINS (N=70, 4,008 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.769 | recent (2020–): 0.773 (PASS)
- High-missing actors (>50%): 2

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 3,943 | 0.508 | 0.289 | -0.000 | 0.00% |
| sector_leader | 65 | 0.490 | 0.208 | 0.494 | 0.00% |

### US-LC-HEALTH (N=51, 2,899 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.708 | recent (2020–): 0.720 (PASS)
- High-missing actors (>50%): 6

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 2,782 | 0.514 | 0.290 | -0.001 | 0.00% |
| sector_leader | 117 | 0.438 | 0.249 | -0.278 | 0.00% |

### US-LC-INDUS (N=59, 3,536 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.733 | recent (2020–): 0.792 (PASS)
- High-missing actors (>50%): 3

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 3,470 | 0.506 | 0.290 | 0.029 | 0.00% |
| sector_leader | 66 | 0.708 | 0.095 | -0.796 | 0.00% |

### US-LC-TECH (N=60, 3,196 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.653 | recent (2020–): 0.750 (PASS (recent))
- High-missing actors (>50%): 11

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| large_firm | 3,064 | 0.516 | 0.290 | -0.025 | 0.00% |
| sector_leader | 132 | 0.382 | 0.211 | 0.122 | 0.00% |

### US-LC (N=169, 9,697 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.761 | recent (2020–): 0.794 (PASS)
- High-missing actors (>50%): 13

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| bank | 2,407 | 0.513 | 0.289 | -0.000 | 0.00% |
| large_firm | 6,769 | 0.509 | 0.286 | -0.012 | 0.00% |
| sector_leader | 521 | 0.440 | 0.309 | 0.218 | 0.00% |

### US-MC (N=159, 8,419 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.819 | recent (2020–): 0.830 (PASS)
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
- Rank stability (mean Spearman rho): 0.905 | recent (2020–): 0.916 (PASS)
- High-missing actors (>50%): 48

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| retail_investor | 2,739 | 0.454 | 0.332 | 0.333 | 0.00% |
| sme | 3,476 | 0.546 | 0.242 | -0.227 | 0.00% |

### US-SC_trimmed (N=94, 5,120 obs)
- Range [0,1]: PASS
- Rank stability (mean Spearman rho): 0.907 | recent (2020–): 0.920 (PASS)
- High-missing actors (>50%): 0

| ActorType | N | Mean | Std | Skew | NaN% |
|-----------|---|------|-----|------|------|
| retail_investor | 2,284 | 0.452 | 0.331 | 0.359 | 0.00% |
| sme | 2,836 | 0.551 | 0.241 | -0.258 | 0.00% |

## Pre-Experiment Quality Gate

- All intensity values in [0,1]: PASS
- Rank stability ρ > 0.7 (full or recent): PASS

### Overall: DATA READY FOR EXPERIMENTS
