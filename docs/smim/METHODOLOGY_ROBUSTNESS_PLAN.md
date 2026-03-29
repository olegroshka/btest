# SMIM Methodology Robustness Plan

> Created: 2026-03-29
> Status: Active — defines experiment variants for C3 and C4 families
> Depends on: RP2 (methodology correlation report), RP4 (US-SC trimming)

---

## 1. The Two Investment Intensity Methodologies

| ID | Name | Formula | Source | Universes | Notes |
|----|------|---------|--------|-----------|-------|
| **M-A** | `capex_assets_xsrank` | CapEx / Total Assets, cross-sectionally ranked | SEC EDGAR XBRL (`PaymentsToAcquirePropertyPlantAndEquipment` / `Assets`) | US-LC, US-LC-*, US-MC, US-SC | Primary US methodology; balance-sheet investment allocation |
| **M-B** | `return_12m_xsrank` | 12-month price return, cross-sectionally ranked | Yahoo Finance OHLCV | All universes (US and UK) | Primary UK methodology; market-based proxy; also available for US |

### Are M-A and M-B interchangeable?

**No.** Per-actor Spearman correlation between M-A and M-B intensity series (RP2, 2026-03-29):

| Universe | N actors | Median rho(M-A, M-B) | rho > 0.4 |
|----------|----------|---------------------|-----------|
| US-LC | 168 | **-0.003** | 1% |
| US-LC-ENERGY | 12 | -0.009 | 0% |
| US-LC-TECH | 59 | -0.041 | 2% |
| US-LC-FINS | 69 | 0.069 | 9% |
| US-LC-HEALTH | 50 | -0.037 | 4% |
| US-LC-INDUS | 59 | -0.029 | 2% |
| US-MC | 159 | -0.032 | 4% |
| US-SC | 138 | -0.017 | 7% |

**Interpretation:** M-A (CapEx/Assets) and M-B (12-month return) are orthogonal constructs
at the per-actor level. CapEx/Assets measures *balance-sheet investment allocation* — how
much of total assets is deployed into physical capital this quarter. Rolling 12-month return
measures *risk-adjusted market expectations* — how the market has valued the actor relative
to peers over the past year. These are economically distinct quantities.

**Consequence:** Mixing M-A and M-B in a cross-universe comparison (e.g. US capex vs UK
return) does not test methodology robustness — it tests a confounded combination of
(a) investment allocation differences and (b) market expectation differences. C4b is
therefore **dropped** as a named experiment variant. Only homogeneous comparisons are valid.

---

## 2. US-SC Data Quality: Trimming Decision

**Problem:** US-SC_registry.json contains 142 equity actors, of which 48 (34%) have > 50%
missing quarters in the intensity panel. These are predominantly recent IPOs (2018–2023)
with fewer than 10 quarters of CapEx data. Their inclusion confounds C3 results: any
difference between US-LC and US-SC could reflect data sparsity rather than genuine
cross-capitalisation investment behaviour.

**Resolution (RP4, 2026-03-29):** Created `US-SC_trimmed_registry.json` with the 94
well-covered actors (missing fraction <= 50%).

| Registry | N actors | High-missing | rho_full | rho_recent | Status |
|----------|----------|-------------|----------|------------|--------|
| US-SC (full) | 142 | 48 (34%) | 0.905 | 0.916 | Baseline |
| US-SC_trimmed | 94 | 0 (0%) | 0.907 | 0.920 | **Primary for C3** |

The trimmed ρ is marginally higher (0.907 vs 0.905) because sparse actors add noise to the
cross-sectional rank calculation. The trimmed universe is the canonical C3 dataset.

---

## 3. Experiment Variant Definitions

### C3: Cross-Capitalisation Experiments (US-LC vs US-SC)

| Variant | Universe A | Intensity A | Universe B | Intensity B | Type | Status |
|---------|-----------|-------------|-----------|-------------|------|--------|
| **C3a** (primary) | US-LC | M-A (`capex_assets_xsrank`) | US-SC_trimmed | M-A (`capex_assets_xsrank`) | Homogeneous | **Active** |
| C3b (robustness) | US-LC | M-B (`return_12m_xsrank`) | US-SC_trimmed | M-B (`return_12m_xsrank`) | Homogeneous | Active |

**C3a** is the primary experiment: both universes use balance-sheet investment intensity,
ensuring that any observed cross-cap differences reflect genuine investment-behaviour
differences rather than methodology artefacts.

**C3b** uses market-return intensity for both universes. If C3a and C3b agree on
the direction and significance of cross-cap gaps, the result is robust to the intensity
methodology choice. If they diverge, the paper must investigate why.

**Files:**
- C3a: `US-LC_intensities.parquet` vs `US-SC_trimmed_intensities.parquet`
- C3b: `US-LC_return_intensities.parquet` vs `US-SC_trimmed_return_intensities.parquet`

---

### C4: Cross-Geography Experiments (US-LC vs UK-LC)

| Variant | Universe A | Intensity A | Universe B | Intensity B | Type | Status |
|---------|-----------|-------------|-----------|-------------|------|--------|
| **C4a** (primary) | US-LC | M-B (`return_12m_xsrank`) | UK-LC | M-B (`return_12m_xsrank`) | Homogeneous | **Active** |
| ~~C4b~~ (dropped) | US-LC | M-A (`capex_assets_xsrank`) | UK-LC | M-B (`return_12m_xsrank`) | Heterogeneous | **Dropped** |

**C4a** is the only valid cross-geography variant. Both universes use M-B, so any observed
US/UK intensity differences reflect genuine cross-geography investment-behaviour differences.

**C4b is dropped** because M-A and M-B are orthogonal (median rho=-0.003); mixing them
does not test methodology sensitivity — it tests a fundamentally different quantity.

**C4 primary intensity files:**
- US-LC: `US-LC_return_intensities.parquet`
- UK-LC: `UK-LC_return_intensities.parquet` (already M-B primary)

**Disclosure note for paper:** US-LC's native methodology is M-A (EDGAR CapEx/Assets).
For the cross-geography experiment, M-B (OHLCV return) is used for both US-LC and UK-LC
to ensure comparability. A supplementary table should report US-LC gap estimates under
both M-A and M-B to quantify the sensitivity.

---

## 4. Future Path: Companies House Adapter (Path A for UK)

If a Companies House XBRL adapter is built, UK equities will have:
- `PaymentsToAcquirePropertyPlantAndEquipment` (CapEx)
- `Assets` (Total Assets)
- Then: `capex_assets_xsrank` computable for UK-LC and UK-MC

This would enable a true homogeneous C4 using M-A for both geographies, and would
allow C4b (M-A vs M-B) as a genuine methodological robustness check. Until then,
C4a (M-B vs M-B) remains the only valid cross-geography experiment.

---

## 5. Intensity File Index

| File | Universe | Method | Rows | rho_full | rho_recent |
|------|----------|--------|------|----------|------------|
| `US-LC_intensities.parquet` | US-LC | M-A | 9,697 | 0.761 | 0.794 |
| `US-LC_return_intensities.parquet` | US-LC | M-B | 14,585 | 0.699 | 0.727 |
| `US-LC-ENERGY_intensities.parquet` | US-LC-ENERGY | M-A | 685 | 0.759 | 0.722 |
| `US-LC-FINS_intensities.parquet` | US-LC-FINS | M-A | 4,008 | 0.769 | 0.773 |
| `US-LC-HEALTH_intensities.parquet` | US-LC-HEALTH | M-A | 2,899 | 0.708 | 0.720 |
| `US-LC-INDUS_intensities.parquet` | US-LC-INDUS | M-A | 3,536 | 0.733 | 0.792 |
| `US-LC-TECH_intensities.parquet` | US-LC-TECH | M-A | 3,196 | 0.653 | 0.750 |
| `US-MC_intensities.parquet` | US-MC | M-A | 8,419 | 0.819 | 0.830 |
| `US-SC_intensities.parquet` | US-SC (full) | M-A | 6,215 | 0.905 | 0.916 |
| `US-SC_trimmed_intensities.parquet` | US-SC (94 actors) | M-A | 5,120 | 0.907 | 0.920 |
| `US-SC_trimmed_return_intensities.parquet` | US-SC (94 actors) | M-B | 6,269 | — | — |
| `UK-LC_intensities.parquet` | UK-LC | M-B | 7,237 | 0.732 | 0.722 |
| `UK-MC_intensities.parquet` | UK-MC | M-B | 6,480 | 0.720 | 0.722 |
| `MIXED-200_intensities.parquet` | MIXED-200 | M-A+M-B | 4,316 | 0.789 | 0.811 |
| `experiment_a1_intensities.parquet` | experiment_a1 (103 in registry; 70 actors with computed intensities) | M-A+M-B | 4,492 | 0.792 | 0.820 |
