[smim_CLAUDE.md](../../../../Documents/Research/SMIM/implementaion/Claude%20Code/smim_CLAUDE.md)# SMIM Implementation Plan — Operational Reference

This is the machine-readable operational reference for milestones, quality gates, and
acceptance criteria. It distils the full LaTeX implementation plan and incorporates
adjustments for the btest platform integration.

For task-level decomposition, see TASK_REGISTRY.md.
For mathematical context, see PROPOSAL_SUMMARY.md.

---

## Guiding Principles

1. **Gate discipline**: no downstream milestone starts until its upstream gate passes.
2. **Point-in-time integrity (A1)**: every artefact respects the information set at time t.
3. **Benchmark labelling**: every reported Δ_{i,t} must specify its BenchmarkClass.
4. **Falsification as milestones**: negative controls are scheduled tasks, not afterthoughts.
5. **Emergence integral from WP4**: PID, TE, criticality, TDA computed in WP4, not deferred to WP6.

## btest Integration Principle

SMIM lives as `src/quantdsl_backtest/smim/` — a subpackage that shares btest's data
infrastructure, ArcticDB caching, and backtest engine, but owns its own pipeline.
New data adapters follow btest's existing adapter pattern. WP5 backtesting reuses
btest's engine via bridge signals in `smim/signals/`. Do not rebuild what btest already provides.

---

## WP0: Formal Scoping and Target-Variable Design

**Duration**: 4 weeks | **Trajectories**: T1

### Milestones

**M0.1 — Data dictionary and notation sheet**
- Deliverable: `docs/smim/notation.md` defining every symbol with cross-references
- Acceptance: compiles; every symbol from proposal has unique definition; peer-reviewed
- Duration: 2 weeks | Depends on: —

**M0.2 — Actor taxonomy**
- Deliverable: spreadsheet with ≥2 actor types per layer (ℓ=0..3), ≥8 types total
- Acceptance: each type has ≥1 named exemplar; layer assignments match proposal Table 2
- Duration: 2 weeks | Depends on: —

**M0.3 — Investment-intensity definitions**
- Deliverable: InvestmentIntensityMapper implementations per actor type in `smim/data/`
- Acceptance: every type from M0.2 has a mapper; normalisation to [0,1] explicit; ≥1 alternative documented
- Duration: 2 weeks | Depends on: M0.2

**M0.4 — Benchmark candidate specifications**
- Deliverable: formal definitions for predictive and structural benchmarks
- Acceptance: computable formula for each; benchmark class labels explicit
- Duration: 1 week | Depends on: M0.1, M0.3

**M0.5 — Scope selection report**
- Deliverable: 2–3 page report + experiment YAML in `experiments/`
- Acceptance: domain, geography, actor universe (N≤500) selected and justified
- Duration: 1 week | Depends on: M0.2

### Gate G0: Target-Variable Coherence

- **Required artefacts**: M0.1–M0.5 complete
- **Thresholds**: zero symbol conflicts; ≥2 types per layer; every type has computable y_{i,t}
- **Assumption check**: A2 (typed comparability) plausibly satisfied
- **If fails**: iterate M0.2–M0.3 (2 extra weeks); if still fails, narrow scope
- **Owner**: PI

---

## WP1: Public-Data Inventory and Ingestion Layer

**Duration**: 7 weeks | **Trajectories**: T1

### Milestones

**M1.1 — Data source audit**
- Deliverable: table mapping each signal family to APIs, endpoints, coverage, frequency, lags
- Acceptance: ≥1 source per family; macro coverage ≥15yr; firm-level ≥10yr; lags documented
- Duration: 2 weeks | Depends on: G0

**M1.2 — Ingestion adapters**
- Deliverable: new adapters in `smim/data/adapters/` following btest's existing FRED adapter pattern
- Acceptance: ≥4 adapters working (FRED/ALFRED, EDGAR, GDELT, one international); unit tests pass; ALFRED vintage retrieval confirmed
- Duration: 4 weeks | Depends on: M1.1
- **btest integration**: extend existing adapter framework, don't rebuild

**M1.3 — Point-in-time store**
- Deliverable: `smim/data/pit_store.py` — dual-timestamp Parquet store (event_date + pub_date)
- Acceptance: vintages stored for ALFRED series; pub-date tagging for non-vintaged sources; leak-detection passes on 50 random sample checks with 0 failures
- Duration: 3 weeks | Depends on: M1.2

**M1.4 — Coverage and quality report**
- Deliverable: automated quality checks + generated report
- Acceptance: missingness <30% for ≥80% of signals; frequency harmonisation documented; entity resolution ≥90% match rate
- Duration: 2 weeks | Depends on: M1.3

### Gate G1: Reproducible Data Inventory

- **Required artefacts**: M1.1–M1.4; pipeline runs from clean environment
- **Thresholds**: ≥80% signal coverage; PIT leak-detection 0/50 failures; entity match ≥90%
- **Assumption check**: A1 enforced by store architecture
- **If fails**: coverage <60% → add sources (3 weeks) or narrow universe; PIT leakage → halt and fix, never proceed with leaky store
- **Owner**: Data Engineer lead

---

## WP2: Influence Graph Construction

**Duration**: 8 weeks | **Trajectories**: T2

### Milestones

**M2.1 — Edge estimation module**
- Deliverable: `smim/graph/edges/` with ≥3 EdgeEstimator implementations (financial, narrative, supply-chain)
- Acceptance: each produces A_t^{(r)} sparse matrix (N×N); Granger uses BIC lag; narrative uses cosine similarity; supply-chain uses BEA I/O
- Duration: 5 weeks | Depends on: G1

**M2.2 — Sparsification and null-model comparison**
- Deliverable: L1 sparsification + ≥100 degree-preserving rewirings per channel
- Acceptance: sparsified density <20%; empirical p<0.05 for ≥1 channel with ≥100 null instances
- Duration: 3 weeks | Depends on: M2.1

**M2.3 — Graph-free baseline**
- Deliverable: dynamic factor model (Stock-Watson) + VAR without graph structure
- Acceptance: DFM with K=3,5,8; VAR with BIC lag; OOS R² and RMSE on same holdout
- Duration: 2 weeks | Depends on: G1 (parallel with M2.1)

**M2.4 — Aggregate operator**
- Deliverable: A_t = Σ ω_r A_t^{(r)} with uniform initial weights, sparse storage
- Acceptance: dimensions N×N; stored in sparse format
- Duration: 1 week | Depends on: M2.1, M2.2

**M2.5 — Sensitivity report**
- Deliverable: comparison across edge methods, sparsification thresholds, channel ablations
- Acceptance: ≥3 ablation experiments; ranked recommendation; includes M2.3 baselines
- Duration: 2 weeks | Depends on: M2.2, M2.3, M2.4

### Gate G2: Stable Interpretable Graph Family

- **Thresholds**: ≥1 graph family outperforms graph-free baselines on ≥1 of: spectral gap stability (CV<0.5), OOS R², or interpretability (3 high-centrality actors recognisable); null-model p<0.05 for ≥1 channel
- **Assumption check**: A3 verified — sparsified operator retains >80% spectral energy
- **If fails**: try alternative edge methods (4 weeks); if still fails, **pivot to graph-free DFM** — this is a terminal risk for the graph approach
- **Owner**: PI + T2 lead

---

## WP3: Modal Representation Experiments

**Duration**: 9 weeks | **Trajectories**: T3

### Milestones

**M3.1 — Spectral decomposition module**
- Deliverable: `smim/spectral/` with ≥4 SpectralDecomposer implementations (Schur, DV, polar, Hermitian dilation)
- Acceptance: unit tests verify algebraic properties (Schur T upper-triangular, polar U orthogonal and P sym PSD, dilation eigenvalues ±σ_k pairs)
- Duration: 4 weeks | Depends on: G2

**M3.2 — DMD/Koopman baseline**
- Deliverable: exact DMD + extended DMD implementations
- Acceptance: reconstruction error decreases with mode count; eigenvalues interpretable
- Duration: 3 weeks | Depends on: G2 (parallel with M3.1)

**M3.3 — Mode selection**
- Deliverable: three-criteria selector (MDL, LZ compressibility, RG relevance)
- Acceptance: K* between 3 and 30; per-mode scores reported; total description length computed
- Duration: 3 weeks | Depends on: M3.1, M3.2

**M3.4 — Comparative evaluation report**
- Deliverable: ranked methods on interpretability, compression, OOS stability, compute cost
- Acceptance: ≥1 method achieves OOS R²>0 vs naïve baselines
- Duration: 3 weeks | Depends on: M3.3

### Gate G3: Stable Compressed Representation OOS

- **Thresholds**: OOS R²>0 vs random walk and historical mean; rolling-window modal autocorrelation >0.3; K*≤0.1N
- **Assumption check**: A4 — eigenmode rank correlation >0.5 across ≥80% of rolling windows
- **If fails**: revisit graph (WP2); try time-vertex joint modes; fall back to DMD-only
- **Owner**: PI + T3 lead

---

## WP4: Regime-Switching Benchmark Model and Emergence Infrastructure

**Duration**: 13 weeks | **Trajectories**: T3, T4

### Milestones

**M4.1 — No-regime state-space model**
- Deliverable: Kalman filter + EM in `smim/dynamics/`
- Acceptance: EM log-likelihood monotonically non-decreasing; filtered states finite and bounded; OOS one-step error computed
- Duration: 3 weeks | Depends on: G3

**M4.2 — Regime-switching extension (Kim filter)**
- Deliverable: Kim filter with M=2,3,4 regimes; EM estimation; MDL/BIC selection
- Acceptance: no numerical overflow; smoothed regime probs non-degenerate; M*≥2 (or M*=1 documented as not justified)
- Duration: 4 weeks | Depends on: M4.1

**M4.3 — Observability diagnostics**
- Deliverable: observability matrix O_z computation per regime, singular values, condition number
- Acceptance: κ(O_z) < 10^6 for all regimes; if exceeded, reduce K*
- Duration: 2 weeks | Depends on: M4.2

**M4.4 — Predictive benchmark gaps**
- Deliverable: Δ^{pred}_{i,t} and Δ^{mode}_{i,t} for all actors
- Acceptance: gaps labelled with BenchmarkClass; distribution summaries per type and regime
- Duration: 2 weeks | Depends on: M4.2

**M4.5 — Phase-transition calibration**
- Deliverable: GL landscape fit, order parameter ψ_t, criticality index C_t
- Acceptance: coefficients estimated for d=1,2,3; C_t spikes compared against known crisis dates
- Duration: 3 weeks | Depends on: M4.2 (parallel with M4.6, M4.7)

**M4.6 — PID and transfer entropy**
- Deliverable: synergy matrix S, TE profiles across layers
- Acceptance: PID for all (j,k) pairs; bootstrap 95% CIs; ≥1 pair with S>0 significant; TE for all directed layer pairs; anomalous direct TE flagged
- Duration: 4 weeks | Depends on: M4.4 (parallel with M4.5, M4.7)

**M4.7 — Persistent homology**
- Deliverable: persistence diagrams, topological complexity T_t, Wasserstein distances
- Acceptance: ≥50 windows; T_t time series produced
- Duration: 3 weeks | Depends on: M4.2 (parallel with M4.5, M4.6)

**M4.8 — Emergence-aware benchmark**
- Deliverable: Δ^{em}_{i,t} incorporating synergistic corrections, C_t, T_t
- Acceptance: labelled emergence_aware; incremental R² vs Δ^{pred} computed
- Duration: 2 weeks | Depends on: M4.5, M4.6, M4.7

### Gate G4: Plausible Regimes; Emergence Adds Value

- **Thresholds**: (a) M* justified by ΔBIC>10; (b) Diebold-Mariano rejects equal accuracy vs random walk at p<0.10 for ≥30% of actors; (c) Δ^{em} achieves ≥0.5pp higher OOS R² than Δ^{pred}, or C_t provides ≥1Q lead time
- **Assumption checks**: A4 re-verified per regime; A5 — average regime duration >8Q
- **If fails**: (a) regimes not justified → single-regime ok; (b) no OOS value → revisit WP2/WP3; (c) emergence adds nothing → proceed without corrections, document negative finding
- **Owner**: PI + T4 lead + T5 lead jointly

---

## WP5: Backtesting and Falsification

**Duration**: 10 weeks | **Trajectories**: T5

### Milestones

**M5.1 — Rolling OOS backtesting engine**
- Deliverable: `smim/validation/rolling_oos.py` + bridge signals in `smim/signals/`
- Acceptance: ≥3 holdout strategies; computes R², RMSE, Spearman ρ, NDCG; PIT discipline verified
- Duration: 3 weeks | Depends on: G4
- **btest integration**: bridge signals wrap Δ_{i,t} as btest-compatible signals; OOS evaluation can optionally run through btest's existing backtest engine

**M5.2 — Falsification test suite (all 7 tests)**
- Deliverable: `smim/validation/falsification.py` with 7 FalsificationTest implementations
- Acceptance: each generates B≥100 null instances; empirical p-values computed; results in structured table
- Duration: 4 weeks | Depends on: M5.1 (parallel with M5.3)

**M5.3 — Event-study validation**
- Deliverable: ≥5 historical policy shocks tested
- Acceptance: ≥3/5 events show significant |Δ| increase (p<0.10) in 2Q post-shock window
- Duration: 3 weeks | Depends on: M4.4, M5.1

**M5.4 — Persistence and correction analysis**
- Deliverable: mean-reversion analysis of large gaps
- Acceptance: half-life estimated; ≥1 of: top-decile gaps predict correction at t+h (p<0.10), gaps predict disclosure changes, or gaps diffuse to neighbours
- Duration: 3 weeks | Depends on: M5.1

**M5.5 — Model comparison report**
- Deliverable: head-to-head comparison vs 4 baselines (historical mean, DFM, graph-free, symmetric Laplacian)
- Acceptance: ≥3 metrics compared; Diebold-Mariano or MCS tests applied
- Duration: 2 weeks | Depends on: M5.1, M5.2

**M5.6 — Evidence synthesis report**
- Deliverable: ≥15 page report integrating M5.1–M5.5
- Acceptance: all gaps labelled; go/no-go recommendation for WP6
- Duration: 2 weeks | Depends on: M5.2–M5.5

### Gate G5: Predictive Evidence

- **Thresholds**: Δ predicts ≥1 downstream outcome with p<0.05; survives ≥4/7 falsification tests (p<0.05 vs null)
- **MVE check**: all 11 Appendix J items satisfied
- **If fails**: survives falsification but no prediction → descriptive/diagnostic value only; fails falsification → return to WP2/WP3 (8 weeks) or terminate programme
- **Owner**: PI with full team consensus

---

## WP6: Structured Extensions

**Duration**: 10 weeks | **Trajectories**: T3, T4

### Milestones

**M6.1 — Structural benchmark** (Phase II)
- Acceptance: ≥2 channels identified as stable, ≥1 as distortionary; Δ^{str} differs from Δ^{pred} by >0.1σ
- Duration: 5 weeks | Depends on: G5

**M6.2 — Actor-level state-space extension**
- Acceptance: estimated for N'≤50 actors; comparison shows whether residual variance is explained
- Duration: 5 weeks | Depends on: G5

**M6.3 — Cross-sectional/geography transfer**
- Acceptance: transfer test on second sector/geography; which parameters transfer documented
- Duration: 4 weeks | Depends on: G5

**M6.4 — Extension evaluation report**
- Acceptance: each extension's incremental value with effect sizes; only p<0.10 improvements retained
- Duration: 2 weeks | Depends on: M6.1–M6.3

### Gate G6: Justified Extensions

- **Threshold**: ≥1 extension provides p<0.10 improvement on ≥1 WP5 metric
- **If fails**: Phase I framework is the final product — valid and publishable
- **Owner**: PI

---

## Overall Timeline

| WP | Duration | Cumulative | Key risk |
|----|----------|------------|----------|
| 0 | 4 wk | Month 1 | Scope too broad |
| 1 | 7 wk | Month 2–3 | Coverage gaps |
| 2 | 8 wk | Month 3–5 | Graph adds no value (terminal) |
| 3 | 9 wk | Month 5–7 | No stable modes |
| 4 | 13 wk | Month 7–10 | Regime overfitting; spurious emergence |
| 5 | 10 wk | Month 10–13 | No predictive value (terminal) |
| 6 | 10 wk | Month 13–16 | Extensions add nothing (acceptable) |
| Buffer | 8 wk | Month 16–18 | — |

**Total: ~18 months including buffer.**

---

## Termination Criteria

Terminate (publish negative result) if:
1. G2 terminal: no graph family outperforms graph-free after remediation
2. G3+G4 joint: no spectral method achieves R²>0 and no regime model improves on baseline
3. G5 falsification: framework indistinguishable from ≥5/7 null models after remediation
4. Resource exhaustion: WP4 not complete within 15 months
