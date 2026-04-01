# SMIM Project Status

> Last updated: 2026-04-01 (Iteration 2 complete)
> This is the single source of truth for current project status.
> For detailed experiment findings: EXPERIMENT_RESULTS.md
> For drill-down methodology and results: DRILLDOWN_PLAN.md
> For iteration 2 plan and findings: ITERATION_2_PLAN.md
> For paper draft: paper/smim_paper.tex

---

## 1. Best Model Configuration (PLATINUM)

| Parameter | Value | Source |
|-----------|-------|--------|
| Decomposition | Dynamic Mode Decomposition (DMD) | DD-3, B2 |
| Modes | K = 8 | DD-1, unrealised items |
| Demeaning | Exponentially-weighted, halflife = 8Q | DD-5 |
| Training window | T = 5 years (20 quarters) | DD-2 |
| State estimation | Kalman filter, **no EM for F/Q** | **E2-5b** |
| Transition matrix | **F = 0.99 * I (regularised, not EM)** | **E2-5b** |
| Initial state noise | **Q = 0.5 * I (larger initial Q)** | **E2-6** |
| Observation covariance | Spherical: R = (tr(R_hat)/N) * I | DD-9 |
| State noise adaptation | Online Q, lambda = 0.3 | DD remaining items |
| Benchmark | Modal (alpha_filtered, not alpha_predicted) | A1 |

**Performance: R² = 0.543** (mean across 10 FULL-ROLL windows, 2015-2024)
- vs AR(1) T=10yr: +0.118, wins 10/10 windows
- vs GOLD+ (EM F): +0.019, wins 10/10 windows
- vs Random walk: +0.238
- Peak window: W2020 R² = 0.659

**Key insight (Iteration 2)**: EM estimation of F is counterproductive. F near-identity
(0.99*I) with higher initial Q (0.5*I) outperforms EM-estimated F by +1.9pp across all
10 windows. The online Q adaptation does all the temporal adaptation work; EM F estimation
overfits to training noise. This simplifies the pipeline (no EM for F/Q needed) AND
improves performance.

**Variance decomposition**: 52% of R²=0.543 comes from per-actor mean (captured by
EWM demeaning); 48% from spectral dynamics (DMD-Kalman). The spectral component
(0.262) exceeds what AR(1) persistence adds beyond the mean (0.144).

**Training window insight**: SMIM benefits from SHORT T (current regime structure),
while AR(1) benefits from LONG T (more data per actor). At T=5yr: SMIM=0.543 vs
AR(1)=0.209. At T=10yr: SMIM=0.339 vs AR(1)=0.425. Each model at its optimal T:
SMIM wins by +0.118.

## 2. Performance Ladder (how we got to 0.543)

| Step | R² | Delta | Innovation |
|------|-----|-------|-----------|
| Original (T=10yr, K=3, Schur, full demean) | 0.339 | -- | Baseline |
| + EWM demeaning (tau=8Q) | 0.381 | +4.2pp | Adapts to non-stationary levels |
| + Shorter T=5yr + K=5 | 0.392 | +1.1pp | Current regime, more modes |
| + Spherical R Kalman | 0.434 | +4.2pp | Eliminates N^2 overparameterisation |
| + DMD basis | 0.467 | +3.3pp | Temporal dynamics > static correlation |
| + Online Q + K=8 | 0.524 | +5.7pp | Regime-adaptive state dynamics |
| **+ F regularisation (F=0.99*I, no EM)** | **0.538** | **+1.4pp** | **Eliminates K^2 overparameterisation** |
| **+ Q=0.5*I (higher initial Q)** | **0.543** | **+0.5pp** | **More room for online Q adaptation** |

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
  configs/
    *.yaml                              per-experiment configs
```

## 6. Paper Status

- Draft: `docs/smim/paper/smim_paper.tex` (~660 lines, 9 figures, 6 tables)
- Figures: `docs/smim/paper/img/` (11 PDFs + PNGs, generated by `scripts/paper_figures.py`)
- Plan: `docs/smim/paper/paper_plan.md`

## 7. Document Index

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
| EXPERIMENT_SESSION_PROMPTS.md | Per-experiment prompts and outcomes |
| DATA_STATUS.md | Data coverage and quality assessment |
| METHODOLOGY_ROBUSTNESS_PLAN.md | Intensity method variants |
| notation.md | Mathematical notation register |

### Archived (build-phase, no longer actively needed)
| File | Purpose | Status |
|------|---------|--------|
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
