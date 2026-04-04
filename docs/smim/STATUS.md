# SMIM Project Status

> Last updated: 2026-04-04 (Iteration 5.2 complete: nested CV delta +0.057)
> This is the single source of truth for current project status.
> For detailed experiment findings: EXPERIMENT_RESULTS.md
> For drill-down methodology and results: DRILLDOWN_PLAN.md, DRILLDOWN_2_PLAN.md
> For iteration 2 plan and findings: ITERATION_2_PLAN.md
> For iteration 3 (Paper 2) plan and findings: ITERATION_3_PLAN.md
> For paper draft: paper/smim_paper.tex

---

## 1. Best Model Configuration (DIAMOND -- rolling basis)

| Parameter | Value | Source |
|-----------|-------|--------|
| Decomposition | Dynamic Mode Decomposition (DMD) | DD-3, B2 |
| Modes | K = 8 | DD-1, unrealised items |
| Demeaning | Exponentially-weighted, halflife = 8Q | DD-5 |
| Training window | T = 5 years (20 quarters), **rolling 1Q** | **V2-1** |
| Basis update | **Recompute DMD each quarter with latest data** | **V2-1** |
| Transition matrix | F = 0.99 * I (regularised, not EM) | E2-5b |
| Initial state noise | Q = 0.5 * I (larger initial Q) | E2-6 |
| Observation covariance | Spherical: R = (tr(R_hat)/N) * I | DD-9 |
| State noise adaptation | Online Q, lambda = 0.3 | DD remaining items |
| Benchmark | Modal (alpha_filtered, not alpha_predicted) | A1 |

**Performance: R² = 0.691** (mean across 10 FULL-ROLL windows, 2015-2024)
- vs PLATINUM (frozen basis): +0.143, wins 10/10 windows
- vs AR(1) T=10yr: +0.266, wins 10/10 windows
- vs Random walk: +0.386
- Peak window: W2020 R² = 0.764

**Key insight (Drilldown 2)**: The spectral basis rotates 26 deg/quarter continuously.
A frozen basis misses this rotation. Quarterly basis reestimation (rolling DMD) captures
the structural evolution and provides +14.3pp improvement. Ablation confirms this is
from the basis update, not from state reset (which hurts by -3pp).

**Structural finding**: The cross-sectional investment structure is consistently
8-dimensional across 60 quarterly windows (2010-2024) but the spectral directions
continuously rotate. No mode births or deaths occur. High-rotation quarters align with
macro events: Euro crisis (2012-Q3, 41 deg), tariff war (2018-Q2, 37 deg), Fed
tightening (2022-Q3, 38 deg).

### Static (frozen) configuration (PLATINUM)

For reference, the static configuration with frozen basis:

**R² = 0.543** (mean across 10 FULL-ROLL windows)
- vs AR(1) T=10yr: +0.118

## 2. Performance Ladder (how we got to 0.691)

| Step | R² | Delta | Innovation |
|------|-----|-------|-----------|
| Original (T=10yr, K=3, Schur, full demean) | 0.339 | -- | Baseline |
| + EWM demeaning (tau=8Q) | 0.381 | +4.2pp | Adapts to non-stationary levels |
| + Shorter T=5yr + K=5 | 0.392 | +1.1pp | Current regime, more modes |
| + Spherical R Kalman | 0.434 | +4.2pp | Eliminates N^2 overparameterisation |
| + DMD basis | 0.467 | +3.3pp | Temporal dynamics > static correlation |
| + Online Q + K=8 | 0.524 | +5.7pp | Regime-adaptive state dynamics |
| + F regularisation (F=0.99*I, no EM) | 0.538 | +1.4pp | Eliminates K^2 overparameterisation |
| + Q=0.5*I (higher initial Q) | 0.543 | +0.5pp | More room for online Q adaptation |
| **+ Rolling basis (recompute DMD each Q)** | **0.691** | **+14.8pp** | **Tracks 26 deg/Q basis rotation** |

## 3. Key Findings by Phase

### Phase A (Anchor)
- A1: Framework works, R²=0.524 (after iterating to GOLD+)
- A2: AR(1) strongest baseline (0.425), random walk = 0.305
- A3: Pipeline wiring confirmed at N=50
- A4: Sub-linear scaling, NaN data bug fixed

### Phase B (Ablation)
- B1: Graph-factor OLS (L1) > Kalman (L2) before regularisation
- B2: DMD wins spectral comparison (+0.020 vs 5 static methods, all tied)
- B3: External signals dispensable at L1 (intensity correlation sufficient)
- B5: M=2 >> M=1 at Kalman depth; emergence negative
- B6: Graceful N degradation (R²=0.82 at N=20, 0.70 at N=125)
- B7: L1 stable across T; Kalman erratic
- B8: 95% noise retention at sigma=1.0
- B9: 100% edge corruption retention (symmetric operator)

### Phase C (Transfer)
- C1-C3: EM overwrites initialisation (transfer protocol flaw)
- C4: UK R²=0.058 (return method weak, not geography)
- C5: Era transfer R²=0.50-0.59 (strong across structural breaks)
- C6: Noise regularises EM (silver > gold)

### Phase D (Economic Validation)
- D1: Gap half-life 1.7Q aggregate (4.7Q for shocks)
- D2: Gaps predict CapEx revision (t=-6.95 after level control) **KEY RESULT**
- D3: L0->L2 direct transmission (r=0.86), L1->L2 insignificant
- D4: 0/8 event alignment (rank normalisation absorbs spikes)
- D5: Pred-modal diverge in 4/10 windows (mean 0.016)
- D6: Gap dispersion leads VIX by 1-4Q (suggestive, small sample)

### Drill-down
- Spherical R is the core technical contribution
- DMD captures temporal dynamics no static operator can
- AR(1) residual analysis: spectral basis explains 11.6% of AR(1) errors
- True zero-shot: frozen F/Q retains 103-106% of full retrain
- D2 control test: gap survives level control (genuine structure)

### Iteration 2 (Emergence & Directed Operators)
- **F regularisation (+1.9pp)**: EM estimation of F overfits. F=0.99*I + Q=0.5*I
  with online Q adaptation gives R²=0.543, beating GOLD+ (0.524) in all 10 windows.
  This extends the spherical-R regularisation insight to the transition matrix.
- **TE operator IS asymmetric** (mean 1.17) but TE-derived spectral basis is too noisy
  for Kalman filtering. K=8 diverges; K=3 gives R²=0.36 vs GOLD+ 0.52.
- **Granger on intensity** produces directed edges (asymmetry 1.3, density 12%)
  but same Kalman instability at K>=5. Blended operator reaches 0.50.
- **Economic emergence features** (dispersion, rotation, concentration) are redundant
  with the DMD-Kalman pipeline. Best delta=-0.003. BRONZE criterion FAILS.
- **Actor-specific loadings** hurt (-0.5pp). DMD provides sufficient loading structure.
- **Kim filter K-means init** gives +1.4pp but from F=0.9*I effect, not regime switching.

### Iteration 3 (Daily Frequency & Emergence — Paper 2)

Framed as standalone Paper 2: "Does emergence exist at daily frequency?"
Two panels tested: Panel A (140 US-LC stocks, daily momentum rank) and
Panel B (28 sector/macro actors, genuine daily variation).

**Panel A: Definitive negative.**
- SMIM R²=0.162 vs AR(1) R²=0.870 on raw momentum (0/10 wins)
- Pre-whitened (AR1 + spectral): R²=0.852 vs AR(1)=0.870, delta=-0.018 (0/10)
- K sweep (K=1..8): monotonic degradation. Even K=1 hurts (-0.006)
- Root cause: daily momentum is 87% per-actor persistence. Cross-sectional
  spectral structure of AR(1) residuals is pure noise at daily frequency.

**Panel B: Spectral framework has moderate R² but loses to AR(1).**
- Best: SMIM K=8 R²=0.578 vs AR(1)=0.780 (delta=-0.20, 0/5 wins)

**H_NL (nonlinear mode coupling): NOT SUPPORTED.**
- Polynomial AR on Panel B alpha: poly wins 2/5, mean MSE ratio 1.015

**H_MI (MI operator): Structurally different but predictively worse.**
- MI and correlation bases are 68 degrees apart (nearly orthogonal)
- Correlation(|corr|, MI) = 0.17 — genuinely different dependence structure
- BUT: MI basis R²=0.035 vs correlation basis R²=0.217 (static), 0/5 wins
- MI rolling R²=0.383 vs DMD rolling R²=0.507, delta=-0.124, 0/5 wins
- MI-weighted DMD ≈ standard DMD (no improvement)
- **Interpretation: nonlinear dependence exists but is orthogonal to
  predictively useful structure. The linear correlation/DMD basis captures
  the dynamics that matter for forecasting.**

**Paper 2 verdict: comprehensive negative.** SMIM's value is specific to
quarterly CapEx intensity. Daily equity/sector dynamics lack the cross-sectional
spectral structure the framework exploits. This is a publishable negative result
that definitively closes Paper 1's "future work" on emergence.

## 4. What Does NOT Work

| Component | Finding | Root Cause |
|-----------|---------|-----------|
| PID synergy (emergence) | CV selects weight=0 | T=20 too short for reliable PID at K=8 |
| TDA complexity | No meaningful contribution | Same data volume limitation |
| **Economic emergence** | **Delta=-0.003** | **Redundant with DMD-Kalman (E2-4)** |
| Event alignment | 0/8 events | Rank normalisation absorbs event spikes |
| Return-based intensity | R²=-0.15 | No cross-sectional dynamics to exploit |
| **TE operator as basis** | **R²=0.36 at K=3** | **KSG TE too noisy at T=40 for spectral basis (E2-1)** |
| **Granger intensity basis** | **Diverges at K>=5** | **Same noisy-basis Kalman instability (E2-2)** |
| **Actor-specific loadings** | **Delta=-0.005** | **DMD basis already optimal (E2-3)** |
| **EM estimation of F** | **Overfits by -1.9pp** | **F near-identity + online Q is strictly better (E2-5b)** |
| Kim filter M>1 | M=2=M=3 | Symmetric EM initialisation |
| Financials sector | R²=0.06 | asset_growth method too persistent (rho=0.70) |
| **Daily momentum SMIM** | **R²=0.16 vs AR1=0.87** | **87% persistence, no spectral structure in residuals (E3-1)** |
| **Pre-whitened daily SMIM** | **Delta=-0.018 all K** | **K=1..8 monotonic degradation; residuals are noise (E3-1e/f)** |
| **MI operator basis** | **R²=0.04 vs corr=0.22** | **MI captures nonlinear dep but orthogonal to predictive structure (MI-1)** |
| **Poly AR on daily alpha** | **MSE ratio 1.015** | **No nonlinear mode coupling at daily frequency (E3-H_NL)** |
| **Sector/macro SMIM** | **R²=0.58 vs AR1=0.78** | **AR(1) dominates even on genuine daily panel (E3-1b)** |

## 5. Data Locations

```
results/
  metrics/
    level1_A1-MVP-FULL.parquet          10 win x 3 benchmarks
    level1_A2-BASELINES.parquet         8 models x 10 win
    level1_B1-COMPONENT-ABLATION.parquet 5 depths x 10 win
    level1_B2-SPECTRAL-METHODS.parquet  7 methods x 10 win
    level1_B6-N-SWEEP.parquet           4 N values
    level1_B7-T-SWEEP.parquet           5 T values
    level1_B8-NOISE.parquet             6 noise levels x 10 win
    level1_B9-EDGE-DEGRADE.parquet      6 corruption levels x 10 win
    level1_C1C2-SECTOR-TRANSFER.parquet cross-sector
    level1_C3-CAP-TRANSFER.parquet      cross-cap
    level1_C5-PERIOD-TRANSFER.parquet   era transitions
    level1_C6-DATA-REGIME.parquet       data quality
    level4_D1-PERSISTENCE.parquet       per-actor half-life
    level4_D2-CORRECTION.parquet        gap prediction
    level4_D3-DIFFUSION.parquet         layer transmission
    level4_D4-EVENTS.parquet            event alignment
    level4_D6-EMERGENCE-TIMING.parquet  VIX lead-lag
    drilldown_DD-{1..9}.parquet         drill-down experiments
    drilldown_GOLD_PP.parquet           noise augmentation
    drilldown_TRUE_ZERO_SHOT.parquet    dynamics portability
    iter2_E2-{1..6}.parquet             iteration 2 experiments
    iter2_E2-4b.parquet                 dispersion weighting
    iter2_E2-5b.parquet                 F regularisation sweep
    iter3_validation_gates.csv          daily signal validation (V1-V5)
    iter3_phase1_results.parquet        daily linear SMIM (E3-1a/b/c)
    iter3_e3_1e_prewhitened.parquet     pre-whitened SMIM results
    iter3_panel_b_poly_ar.parquet       polynomial AR on Panel B
    iter3_mi_drilldown.parquet          MI vs DMD vs correlation comparison
    iter5_2_phase_a.parquet             dual-reg sweep (F×Q₀×λ, 120 configs)
    iter5_2_phase_b.parquet             K=1 vs K=2 test (8 configs)
    iter5_2_phase_c.parquet             alternative intensities (4 panels)
    iter5_2_phase_d.parquet             rolling DMD window sweep (5 configs)
    iter5_2_phase_e.parquet             fine EWM grid (9 configs)
    iter5_2_phase_f.parquet             interaction effects (8 configs)
    iter5_2_nested_cv.parquet           nested CV with 5.2 config space
  configs/
    *.yaml                              per-experiment configs
```

## 6. Paper Status

### Paper 1: "Regularised Spectral State-Space Models..."
- **Status**: UPDATED (2026-04-04). Iteration 5.2: multi-signal table, K≈1 finding, EWM sensitivity, fixed-config correction.
- Draft: `docs/smim/paper/smim_paper.tex` (~900 lines, 12 figures, 7 tables)
- **Headline (CapEx/Revenue, 146 US firms, predictive alpha):**
  - Nested CV: SMIM R²=0.705 vs AR(1)=0.648, delta=+0.057, 8/8 wins, perm p=0.003
  - Holdout: SMIM R²=0.837 vs AR(1)=0.780, delta=+0.056
  - Fixed config (K=2,EWM=8,T=2yr): SMIM R²=0.737 vs AR(1)=0.671, delta=+0.066, 10/10 wins
  - Previous (5.1v2, K=2,EWM=12,T=3yr): nested CV delta=+0.042, 8/8 wins
- **Original panel (93 actors, CapEx/Assets) corrected:** predictive R²=0.489 (loses to AR(1)=0.610)
  - Modal R²=0.691 valid as spectral reconstruction quality
  - Structural findings preserved: ablation, rotation (26°/Q), D2 regression

### Paper 2: "Multi-Frequency Spectral Investment Models"
- **Status**: Iteration 3 complete (definitive negatives), Iteration 4 planned
- Iteration 3: daily data has no independent spectral structure; MI ≠ correlation but orthogonal to prediction
- Iteration 3: daily data has NO independent spectral structure (definitive negative)
- Iteration 4: MF Kalman (Q+daily) FAILS — daily updates hurt (R2=-0.015 vs Q-only=0.519)
- Rotation prediction from daily features: LOO R2=-0.15 (no leading information)
- **Iteration 5/5.1: CapEx/Revenue with K=3, EWM=12, T=3yr + operator learning**
  → Fixed config: SMIM R²=0.733 vs AR(1)=0.699, delta=+3.4pp, 9/10 wins
  → Nested CV: SMIM R²=0.698 vs AR(1)=0.685, delta=+1.2pp, 6/8 wins, DM p=0.012
  → All genuine predictive alpha (confirmed 2026-04-04)
- **Iteration 5.1v2: K=2, EWM=12, T=3yr, NO operator learning**
  → Nested CV: SMIM R²=0.711 vs AR(1)=0.669, delta=+4.2pp, 8/8 wins, DM p<0.001
  → At K=2 with rolling basis, Kalman is functionally redundant (+0.3pp only)
- **Iteration 5.2: Parameter space exploration (2026-04-04)**
  → Phase A: dual-reg constants (F, Q₀, λ) are optimal at baseline — noise at K=2
  → Phase B: K=1 nearly matches K=2 (Δ=+0.066 vs +0.066) — signal is "nearly 1-D"
  → Phase C: GOLD — Revenue/Assets (+1.0pp, 8/10) and Multi-ratio (+2.8pp, 9/10) also beat AR(1)
  → Phase D: DMD window W=12 slightly better than all (+0.4pp)
  → Phase E: EWM=8 is true optimum (+5.9pp over EWM=12)
  → Phase F: nested CV delta=+5.7pp, 8/8 wins, perm p=0.003 (was +4.2pp)
  → Holdout delta=+5.6pp (consistent with CV estimate)
  → **New best config: K=2, EWM=8, T=2yr, no OpLearn, DMD_W=12**
- Plan + results: ITERATION_3_PLAN.md, ITERATION_4_PLAN.md, ITERATION_5_PLAN.md, ITERATION_5_2_PLAN.md

## 7. Script Inventory (`scripts/smim/`)

All SMIM scripts live under `scripts/smim/`.

### Data Acquisition
| Script | Description |
|--------|-------------|
| `smim_build_universes.py` | Build universe CSVs and download OHLCV data |
| `smim_fetch_fred.py` | Fetch FRED macro signals + ALFRED vintages |
| `smim_fetch_edgar.py` | Fetch SEC EDGAR XBRL balance sheet data |
| `smim_fetch_gdelt.py` | Fetch GDELT narrative signals (incremental, daily cache) |
| `smim_fetch_imf.py` | Fetch IMF WEO + IFS macro indicators |
| `smim_fetch_oecd.py` | Fetch OECD economic indicators via SDMX |
| `smim_fetch_bea.py` | Fetch BEA input-output tables |
| `smim_build_registries.py` | Build actor registries from universe CSVs |
| `smim_build_mixed_expanded.py` | Build expanded MIXED universe |
| `smim_compute_intensities.py` | Compute investment intensity panels |
| `smim_data_audit.py` | Audit data coverage and quality |

### Experiment Runners
| Script | Phase | Description |
|--------|-------|-------------|
| `run_smim_a1.py` | A1 | MVP full pipeline (GOLD+ config) |
| `run_smim_a2.py` | A2 | Naive baselines (8 models) |
| `run_smim_a3.py` | A3 | Measurement stack validation (N=50) |
| `run_smim_a4.py` | A4 | Computational scaling profile |
| `run_smim_b1.py` | B1 | Component layer ablation |
| `run_smim_b2.py` | B2 | Spectral method comparison |
| `run_smim_b3.py` | B3 | Signal family leave-one-out |
| `run_smim_b5.py` | B5 | Signal x component interaction |
| `run_smim_b6.py` | B6 | Actor count sweep |
| `run_smim_b7.py` | B7 | Training window sweep |
| `run_smim_b8_b9.py` | B8-B9 | Noise and edge corruption robustness |
| `run_smim_phase_c.py` | C1-C6 | Transfer experiments (sector, cap, geo, era, data regime) |
| `run_smim_d1.py` | D1 | Gap persistence analysis |
| `run_smim_d2_d3_d5_d6.py` | D2-D6 | Economic validation (correction, diffusion, divergence, emergence timing) |
| `run_smim_d4.py` | D4 | Event alignment |
| `run_smim_drilldown.py` | DD | Drill-down experiments (DD-1 to DD-9) |
| `run_smim_drilldown_p2.py` | DD-P2 | Drill-down phase 2 (noise augmentation, zero-shot) |
| `run_smim_drilldown_c.py` | DD-C | Drill-down phase C |
| `run_smim_iter2.py` | E2 | Iteration 2 experiments (emergence, directed operators) |
| `run_smim_dd2_sprint1.py` | DD2-S1 | Drilldown 2 sprint 1 (rolling basis) |
| `run_smim_dd2_v2.py` | DD2-V2 | Drilldown 2 V2 (DIAMOND config) |
| `run_smim_iter5_1_sweep.py` | I5.1 | 210-config K×EWM×T sweep |
| `run_smim_iter5_1_cv2.py` | I5.1v2 | Nested CV (K=2, no OpLearn) |
| `run_smim_iter5_2.py` | I5.2 | Parameter space exploration (6 phases) |

### Paper & Analysis (Workstream)
| Script | Description |
|--------|-------------|
| `run_smim_ws_a_nested_cv.py` | WS-A: Nested cross-validation |
| `run_smim_ws_b_inference.py` | WS-B: Statistical inference |
| `run_smim_ws_c_ablation.py` | WS-C: Paper ablation table |
| `run_smim_ws_d_rotation_bootstrap.py` | WS-D: Basis rotation bootstrap |
| `run_smim_ws_e_capex_regression.py` | WS-E: CapEx regression |
| `paper_figures.py` | Generate paper figures (v1) |
| `paper_figures_v2.py` | Generate paper figures (v2) |
| `smim_medium_figures.py` | Generate Medium article figures |
| `smim_methodology_correlation.py` | Methodology correlation analysis |

### Testing & Benchmarks
| Script | Description |
|--------|-------------|
| `run_smim_acceptance.py` | Run 130 acceptance tests with gate report |
| `gpu_speedup_report.py` | GPU vs CPU speedup report from benchmark results |

## 8. Document Index

### Active (current, maintained)
| File | Purpose |
|------|---------|
| STATUS.md | This file -- single source of truth |
| EXPERIMENT_RESULTS.md | Detailed per-experiment findings |
| DRILLDOWN_PLAN.md | Drill-down methodology and results |
| ITERATION_2_PLAN.md | Iteration 2 plan and findings |
| CLAUDE.md | SMIM-specific development context |
| DECISIONS.md | Architectural decision log |
| paper/smim_paper.tex | Research paper draft |
| paper/paper_plan.md | Paper structure and narrative plan |

### Reference (read when needed, not maintained)
| File | Purpose |
|------|---------|
| EXPERIMENT_PLAN.md | Original experiment programme design |
| EXPERIMENT_OBJECTIVES.md | Research questions and hypotheses |
| DATA_STATUS.md | Data coverage and quality assessment |
| METHODOLOGY_ROBUSTNESS_PLAN.md | Intensity method variants |
| notation.md | Mathematical notation register |

### Archived (build-phase, no longer actively needed)
| File | Purpose | Status |
|------|---------|--------|
| EXPERIMENT_SESSION_PROMPTS.md | Per-experiment prompts and outcomes | All experiments complete |
| NEXT_SESSION_PROMPT.md | Iteration 2 session prompt | Iteration 2 complete |
| IMPLEMENTATION_PLAN.md | Milestone/gate plan | All gates passed |
| TASK_REGISTRY.md | Per-task status | All WP0-WP6 complete |
| GPU_ACCELERATION_PLAN.md | GPU design | GPU-0 through GPU-4 complete |
| GPU_ACCELERATION_TARGETS.md | GPU benchmarks | Complete |
| ACCEPTANCE_TESTS.md | AT catalogue | 130/130 pass |
| ACCEPTANCE_TEST_REVIEW.md | AT review | Complete |
| ADAPTER_GUIDE.md | Data adapter howto | Reference only |
| PROPOSAL_SUMMARY.md | Condensed proposal | Superseded by paper |
| actor_taxonomy.md | Actor universe design | Superseded by experiment data |
| benchmark_specs.md | Benchmark definitions | Implemented in code |
| scope_selection.md | MVP scope justification | Historical |
| data_source_audit.md | Data source inventory | Superseded by DATA_STATUS |
| reports/data_readiness.md | Per-universe readiness | Superseded |
