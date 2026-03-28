# SMIM Intensity Methodology Correlation Report

> Generated: 2026-03-28
> Methodology A: `capex_assets_xsrank` (CapEx/Assets, EDGAR, US only)
> Methodology B: `return_12m_xsrank` (12-month price return, OHLCV)
> Per-actor Spearman ρ across shared quarterly periods (min 8 quarters)

## Per-Universe Summary

| Universe | N actors | Median ρ | P25–P75 | ρ > 0.4 | ρ ≤ 0 |
|----------|----------|----------|---------|---------|-------|
| US-LC | 168 | -0.003 | [-0.125, 0.087] | 2/168 (1%) | 85 |
| US-LC-ENERGY | 12 | -0.009 | [-0.079, 0.058] | 0/12 (0%) | 7 |
| US-LC-TECH | 59 | -0.041 | [-0.158, 0.118] | 1/59 (2%) | 32 |
| US-LC-FINS | 69 | 0.069 | [-0.078, 0.242] | 6/69 (9%) | 25 |
| US-LC-HEALTH | 50 | -0.037 | [-0.111, 0.095] | 2/50 (4%) | 29 |
| US-LC-INDUS | 59 | -0.029 | [-0.144, 0.123] | 1/59 (2%) | 33 |
| US-MC | 159 | -0.032 | [-0.175, 0.100] | 6/159 (4%) | 91 |
| US-SC | 138 | -0.017 | [-0.170, 0.114] | 10/138 (7%) | 75 |

## Interpretation

- **ρ > 0.4**: the two methodologies agree on relative actor ranking — cross-geography comparison is defensible.
- **ρ ≤ 0**: methodologies are uncorrelated or inversely correlated — inclusion in cross-geography experiments would be misleading.
- The decision gate is based on **US-LC median ρ** (largest, most diverse universe).

## C4 Decision Gate

**C4 REQUIRES HOMOGENEOUS METHODOLOGY** — US-LC median ρ(capex, return) = -0.003 < 0.4. CapEx and return intensities diverge too much for cross-geography comparison. Run C4a only (return vs return) until a Companies House CapEx adapter is built for UK equities.

## Experiment Variants

| Variant | US intensity | UK intensity | Type |
|---------|-------------|-------------|------|
| C4a (primary) | `return_12m_xsrank` | `return_12m_xsrank` | Homogeneous |
| C4b (robustness) | `capex_assets_xsrank` | `return_12m_xsrank` | Heterogeneous |

| Variant | US universe | US intensity | SC universe | SC intensity | Type |
|---------|------------|-------------|------------|-------------|------|
| C3a (primary) | US-LC | `capex_assets_xsrank` | US-SC_trimmed | `capex_assets_xsrank` | Homogeneous |
| C3b (robustness) | US-LC | `return_12m_xsrank` | US-SC_trimmed | `return_12m_xsrank` | Homogeneous |
