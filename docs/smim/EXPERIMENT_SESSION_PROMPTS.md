  # SMIM Experiment Session Prompts

> Created: 2026-03-29
> Status: Active — one prompt per experiment session; update status column as experiments run
> Context files: EXPERIMENT_PLAN.md · EXPERIMENT_OBJECTIVES.md · DATA_STATUS.md · METHODOLOGY_ROBUSTNESS_PLAN.md · CLAUDE.md · docs/smim/CLAUDE.md

Each session prompt is self-contained: it carries enough context to start a new Claude Code
conversation without needing to re-read the full plan. After each session, record the key
result in the **Outcome** row so the next session can reference it.

---

## How to Use This File

1. Pick the next experiment by priority (see matrix in EXPERIMENT_PLAN.md §Execution Matrix)
2. Copy the full prompt block into a new Claude Code session
3. After the session ends, fill in the **Outcome** row and push the update

---

## Phase A — Anchor Experiments

### Session A3 — Measurement Stack Validation *(run FIRST)*

> **Why A3 before A1**: A3 is the sanity check. It runs the full pipeline on 50 actors,
> RECENT period, minimal institutions — fast and cheap. If it fails, stop and fix before
> committing to the 680-run programme.

**Status:** `[x] COMPLETE — 2026-03-29 — PASS (5/5)`
**Outcome:** All checks pass. OOS R²=-1.65 (negative, expected — T=24 quarters < N=50 actors,
MDL selects K*=1, model underdetermined). Pipeline wiring confirmed. Runtime 0.55s total at N=50.
Granger edges dominate (36%), data load (33%), Kalman EM (14%). No bugs found.
Full findings: `docs/smim/EXPERIMENT_RESULTS.md §A3`.
Outputs: `results/metrics/level1_A3-STACK-VALIDATION.parquet`, `results/configs/A3-STACK-VALIDATION.yaml`.
Runner: `scripts/run_smim_a3.py`. **A3 gate passed — A1 unblocked.**

**Session A3 notes** *(filled)*:
- [x] check 1: PASS — OOS R²=-1.6459, DM stat=3.922, DM p=0.000, coverage=1.000 (all finite)
- [x] check 2: PASS — |sum(-1.5478) - total(-1.5478)| = 0.0000 (exact 2-component decomposition)
- [x] check 3: PASS — B100 LagDestroyedPlacebo completed, p_value=0.000
- [x] check 4: PASS — 11/11 pipeline components timed
- [x] check 5: PASS — 17 required columns present in parquet

**Prompt:**
```
Context files to read at session start:
  docs/smim/CLAUDE.md                    ← standing assumptions, notation, known deviations
  docs/smim/DATA_STATUS.md               ← data coverage, intensity files, Gate G1 status
  docs/smim/EXPERIMENT_PLAN.md §Phase-A  ← A3 spec (lines ~146–167)

Experiment: A3-STACK-VALIDATION
Goal: verify the full measurement pipeline runs end-to-end and produces
      finite, sensible outputs. This is a wiring check, not a science experiment.

Config:
  universe:       US-LC subset, N=50 (pick first 50 by market cap from US-LC registry)
  intensity file: data/smim/intensities/US-LC_intensities.parquet  (M-A, capex_assets_xsrank)
  institutions:   INST-MINIMAL (Fed + BoE + IMF only)
  signals:        MACRO+MARKET  (FRED macro + Yahoo Finance OHLCV price/return signals only)
                  NOTE: "no balance sheet signals" means EDGAR Revenues/LongTermDebt/etc. are
                  excluded as INPUT signals. The intensity file (y_{i,t} = CapEx/Assets) is the
                  TARGET being modelled — it is always from EDGAR regardless of signal config.
  period:         RECENT (train 2018–2023, test 2024–2025)
  pipeline depth: full (all layers active)

Required checks (all must PASS before calling A3 complete):
  1. All L1 metrics (OOS R², DM stat, coverage) are finite — no NaN, Inf
  2. Component ΔR² values are present and sum approximately to total OOS R²
     (within ±5pp tolerance; not required to be positive)
  3. Falsification B100 completes without error
  4. Runtime profiler captures timing for each pipeline component
  5. Results parquet schema matches results/experiments.parquet spec in EXPERIMENT_PLAN.md

If any check fails: investigate root cause, fix, re-run. Document fix in DECISIONS.md.
If all checks pass: record runtime per component and note any ΔR² anomalies.

Output to produce:
  - results/metrics/level1_A3-STACK-VALIDATION.parquet
  - results/configs/A3-STACK-VALIDATION.yaml
  - Short summary: [PASS/FAIL per check, runtime breakdown, any warnings]

Do NOT proceed to A1 if A3 fails.
```

---

### Session A4 — Computational Scaling Profile *(can run in parallel with A1)*

**Status:** `[x] COMPLETE (v2) -- 2026-03-29 -- gate PASS, all OOS R2 finite`
**Outcome:** Decision gate PASS (all components alpha <= 2.5). Total pipeline alpha=0.41.
v1 showed OOS R2=nan at N>50 and inflated Kalman EM alpha=2.19. Root cause: one actor had
all-NaN intensity in training period; fillna(col_means) left NaN intact; NaN propagated
through pipeline. Fix: filter actors with no training-period data before pipeline. After fix:
OOS R2 finite at all N (-2.46 to -1.66, all negative as expected for T=24 << N). Kalman EM
actual alpha=0.81 (sub-linear). Spectral decomp is now the scaling bottleneck (alpha=2.30).
B-series very feasible at N~93 (~1s/run). Full findings: `EXPERIMENT_RESULTS.md` section A4.
Outputs: `results/metrics/level5_A4-SCALING.parquet`, `results/configs/A4-SCALING.yaml`.
Runner: `scripts/run_smim_a4.py`. No Kalman EM regularisation needed.

**Prompt:**
```
Context files to read at session start:
  docs/smim/CLAUDE.md
  docs/smim/EXPERIMENT_PLAN.md §Phase-A  ← A4 spec (lines ~168–183)

Experiment: A4-SCALING
Goal: measure runtime and memory as a function of actor universe size N,
      to confirm Phase B/C experiments are computationally feasible.

Config:
  institutions: INST-MINIMAL
  signals:      MACRO+MARKET
  period:       RECENT (single OOS window, 2024–2025)
  pipeline:     full

Iterations (run all 5):
  1. US-LC subset N=20
  2. US-LC subset N=50  (reuse A3 results if available — same config)
  3. US-LC subset N=100
  4. US-LC subset N=200 (full US-LC)
  5. US-LC + US-MC combined N≈400

For each iteration record:
  - Wall time (seconds) per pipeline component
  - Peak memory (MB)
  - OOS R² (sanity check — should be consistent across N)

Fit scaling exponent: t ~ N^α for each component separately.
Expected: graph construction O(N²), spectral O(N³), Kim filter O(N·K·M·T), emergence O(K²·T).

Output:
  - results/metrics/level5_A4-SCALING.parquet  (N, component, time_s, mem_mb)
  - Scaling table: component × {α_exponent, t_N200_s, feasible_N500}
  - Flag any component where N=400 run exceeds 30 minutes

Decision gate: if any Critical-path component scales worse than O(N²·⁵),
raise with user before launching B-series (those run N=200+ repeatedly).
```

---

### Session A1 — MVP Full Pipeline *(after A3 passes)*

**Status:** `[x] COMPLETE (v3) -- 2026-03-29 -- PASS gate (pred R2=0.305, modal R2=0.327)`
**Outcome:** v1 STOP (R2=-2.44): missing demeaning. v2 PASS (R2=0.283): demeaning fix. v3 iterated
with 5 pipeline improvements + 3 operator learning approaches (multi-scale conv, synergy-guided,
end-to-end optimisation). Final: pred R2=0.305 (matches random walk baseline), modal R2=0.327
(+7.2% above RW, best window 0.427 matches AR1). K_min=3. End-to-end operator optimisation
(Approach C) produces best modal results. Emergence via PID synergy is negligible at T=40/K=3
(CV weight = 0); modal filtering IS the emergence signal. B-series unblocked.
Runner: `scripts/run_smim_a1.py`. Full findings: `EXPERIMENT_RESULTS.md` section A1.

**Prompt:**
```
Context files to read at session start:
  docs/smim/CLAUDE.md
  docs/smim/DATA_STATUS.md               ← confirm experiment_a1 data state
  docs/smim/EXPERIMENT_PLAN.md §Phase-A  ← A1 spec (lines ~93–120)
  docs/smim/reports/data_readiness.md    ← experiment_a1: N=93, ρ=0.774

Prerequisite: A3 must have PASSED all 5 checks.

Experiment: A1-MVP-FULL
Goal: establish the reference benchmark — the "full score" the framework achieves
      on the MVP domain under ideal conditions.

Config:
  universe:       experiment_a1 registry (103 actors in registry, 93 have intensity)
  intensity file: data/smim/intensities/experiment_a1_intensities.parquet
                  (M-A+M-B mix: US equity=capex_assets_xsrank, UK equity=return_12m_xsrank,
                   institutional actors via GDELT; 9 US energy actors are signal-only — no gap)
  institutions:   INST-US + INST-UK
  signals:        FULL  ← NOTE: BIS cross-border edges absent (G-11); BEA I/O domestic
                          supply-chain edges present. Disclose in result commentary.
  period:         FULL-ROLL (rolling 10yr train, 10 non-overlapping 1yr test windows)
  regimes:        compare [1, 2, 3]  ← run all three; select best by OOS R² for B-series
  emergence:      true
  benchmarks:     [predictive, modal, emergence_aware]
  falsification:  all_7_tests
  measurements:   [L1, L2, L3, L4, L5]

Key data notes:
  - 10 actors in registry have no intensity (9 US energy with no CapEx EDGAR tag = G-13,
    1 institutional edge case). These are signal-only graph nodes — they contribute to
    edge estimation and spectral structure but have no gap estimate. Do NOT impute or
    fill their gaps with zeros.
  - inst_fca_uk uses actor_BOE as GDELT proxy (proxy, not direct measurement).
    Flag this in the results commentary.
  - FULL signal feed is partial on cross-border financial channels (G-11).

Decision gate (CRITICAL):
  If A1 OOS R² ≤ 0 against random walk across all 10 windows: STOP.
  Do not proceed to B-series. Diagnose first (check: is random walk truly random?
  Is the test set correctly held out? Are results leaking through the intensity panel?).

Output:
  - results/metrics/level1_A1-MVP-FULL.parquet   ← 10 windows × metrics
  - results/metrics/level2_A1-MVP-FULL.parquet   ← component ΔR²
  - results/metrics/level3_A1-MVP-FULL.parquet   ← robustness
  - results/metrics/level4_A1-MVP-FULL.parquet   ← economic validity
  - results/metrics/level5_A1-MVP-FULL.parquet   ← runtime profile
  - results/configs/A1-MVP-FULL.yaml
  - Record: best regime count M*, best spectral method, OOS R², DM stat vs random walk
```

---

### Session A2 — Naïve Baselines *(can run in parallel with A1)*

**Status:** `[x] COMPLETE -- 2026-03-29`
**Outcome:** 8 baselines on same A1 universe/windows. AR(1) strongest (R2=0.425). Random walk
primary denominator (R2=0.305). SMIM predictive matches RW; SMIM modal beats it (mean 0.327,
best 0.427). Symmetric Laplacian < SMIM (H2a evidence). DFM-K10 overfits. Sector mean weakest.
Runner: `scripts/run_smim_a2.py`. Full findings: `EXPERIMENT_RESULTS.md` section A2.

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-A  ← A2 spec (lines ~122–142)

Experiment: A2-BASELINES
Goal: produce the baseline performance numbers that ALL subsequent ΔR² comparisons
      use. These are the denominators of the value story.

Config:
  universe:   experiment_a1 (same registry and intensity file as A1 — required for
              direct ΔR² comparison; the EXPERIMENT_PLAN.md labels this universe
              "MIXED-200" but the actual data file is experiment_a1_intensities.parquet)
  intensity:  data/smim/intensities/experiment_a1_intensities.parquet  (N=93 actors)
  signals:    FULL (same as A1)
  period:     FULL-ROLL (same 10 windows as A1)
  measurements: [L1]

Models to run (all 8):
  1. historical_mean          ← per-actor trailing mean
  2. random_walk              ← last observation carried forward
  3. sector_mean              ← cross-sectional sector average
  4. ar1_per_actor            ← per-actor AR(1)
  5. dynamic_factor_model_k5  ← DFM with K=5 factors
  6. dynamic_factor_model_k10 ← DFM with K=10 factors
  7. var_bic                  ← VAR with BIC-selected lag
  8. symmetric_laplacian_spectral ← spectral with symmetric Laplacian (A3a baseline)

For each model × window: record OOS R², MAE, coverage.
The random_walk R² is the primary denominator for all future ΔR² claims.

Output:
  - results/metrics/level1_A2-BASELINES.parquet  ← 8 models × 10 windows
  - results/configs/A2-BASELINES.yaml
  - Summary table: model × {mean_OOS_R², std_OOS_R², rank}

Note: model 8 (symmetric_laplacian_spectral) tests H2a directly —
if SMIM (directed) > symmetric, that is the first evidence for H2a.
```

---

## Phase B — Ablation Experiments

> Run Phase B only after A1 OOS R² > 0 is confirmed. B1 is the most critical —
> it directly answers Q1 (component value). Run B1 and B6 before others in this phase.

### Session B1 — Component Layer Ablation *(most critical B experiment)*

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/CLAUDE.md                     ← layer definitions (Q1 in EXPERIMENT_OBJECTIVES.md)
  docs/smim/EXPERIMENT_PLAN.md §Phase-B   ← B1 spec (lines ~196–228)
  results/metrics/level1_A1-MVP-FULL.parquet   ← A1 reference (needed for comparison)
  results/metrics/level1_A2-BASELINES.parquet  ← baselines (needed for ΔR² denominator)

Prerequisite: A1 complete and OOS R² > 0.

Experiment: B1-COMPONENT-ABLATION
Goal: decompose the A1 result into per-layer contributions. Answer Q1: does each
      layer of pipeline complexity earn its keep?

Config:
  universe:     MIXED-200 / experiment_a1 (same as A1)
  institutions: INST-US + INST-UK
  signals:      FULL
  period:       FULL-ROLL (same 10 windows as A1)

Iterations (5 depths):
  L1: graph factors only (no state-space, no regimes, no emergence)
  L2: spectral factors + single regime (no regime switching, no emergence)
  L3: spectral + regime switching (best M* from A1, no emergence)
  L4: + emergence diagnostics (PID, TE, criticality — no phase transition)
  L5: full pipeline (= A1 reference; reuse A1 results, do not re-run)

For each depth: OOS R², ΔR² vs L(N-1), DM test p-value vs L(N-1).

Decision rule per layer:
  ΔR² ≥ 0.5pp AND DM p ≤ 0.10 → component ADDS VALUE for this condition
  ΔR² ≥ 0.5pp AND DM p > 0.10 → component ADDS VALUE but not statistically significant
  ΔR² < 0.5pp                  → component MARGINAL — flag for further investigation

Output:
  - results/metrics/level1_B1-COMPONENT-ABLATION.parquet
  - results/metrics/level2_B1-COMPONENT-ABLATION.parquet
  - Component value table (as in EXPERIMENT_PLAN.md §B1 Expected Output)
  - Complexity map entry: MIXED-200/Gold/full → recommended depth

The component value table from B1 is the centrepiece of the paper's results section.
```

---

### Session B2 — Spectral Method Comparison

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B   ← B2 spec (lines ~232–253)
  results/metrics/level1_A1-MVP-FULL.parquet  ← reference baseline

Prerequisite: A1 complete.

Experiment: B2-SPECTRAL-METHODS
Goal: determine which spectral decomposition method performs best on the SMIM
      operator. Tests H2a (directed > symmetric) and H2b (polar/dilation > Schur
      on ill-conditioned graphs).

Config:
  universe: MIXED-200 / experiment_a1
  institutions: INST-US + INST-UK
  signals:  FULL
  period:   FULL-ROLL
  measurements: [L1, L2, L5]

Methods to compare (7):
  1. schur              ← SMIM primary (Schur decomposition of non-normal operator)
  2. polar              ← polar decomposition, rotation part
  3. hermitian_dilation ← converts non-Hermitian to Hermitian, standard EVD
  4. directed_variation ← directed graph variation approach
  5. dmd                ← Dynamic Mode Decomposition
  6. extended_dmd       ← kernel DMD with delay embedding
  7. pca                ← symmetric baseline (PCA of symmetrised operator)

For each method × window: OOS R², K* (retained modes), condition number of operator.

Primary hypotheses:
  H2a: any directed method (1–6) > pca (7) on OOS R²
  H2b: polar or hermitian_dilation > schur when operator condition number is high

Output:
  - results/metrics/level1_B2-SPECTRAL-METHODS.parquet   ← 7 methods × 10 windows
  - results/metrics/level2_B2-SPECTRAL-METHODS.parquet   ← method attribution
  - Method ranking table + winner recommendation for B3–B10

The winning method from B2 becomes the default for all subsequent experiments.
```

---

### Session B3 — Signal Family Leave-One-Out

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B   ← B3 spec (lines ~258–281)
  docs/smim/DATA_STATUS.md §2             ← signal family → source mapping
  results/metrics/level1_B1-COMPONENT-ABLATION.parquet  ← recommended depth for this universe

Prerequisite: A1 complete, B1 recommended depth known.

Experiment: B3-SIGNAL-LOO
Goal: which signal families are dispensable? Remove one at a time, measure R² loss.
      Answers Q4: "which data sources drive performance?"

Config:
  universe: MIXED-200 / experiment_a1
  institutions: INST-US + INST-UK
  period: FULL-ROLL
  pipeline_depth: best depth from B1 (not necessarily full)

Iterations (6):
  reference:       FULL  (= A1 result, reuse)
  drop narrative:  NO-NARRATIVE  (remove GDELT signals)
  drop market:     NO-MARKET     (remove Yahoo Finance OHLCV)
  drop network:    NO-NETWORK    (remove BEA I/O supply-chain edges)
  drop policy:     FULL minus policy  (remove FRED policy signals: DFF, FEDFUNDS, T10Y2Y,
                                      BAA10Y, BAMLH0A0HYM2, STLFSI2 — the "financial
                                      conditions" cluster; keep macro quantity signals)
  drop balance:    FULL minus balance (remove EDGAR signals: Revenues, LongTermDebt,
                                       StockholdersEquity, R&D — balance-sheet inputs only;
                                       the intensity y_{i,t} is always retained as the target)
  drop macro:      FULL minus macro   (remove FRED macro: GDP, INDPRO, CPIAUCSL, etc.)

NOTE: "drop network" already excludes BIS cross-border edges (G-11). The NO-NETWORK
condition tests value of BEA I/O domestic edges, not BIS edges. Disclose this.

For each: ΔR² vs reference, DM p-value. Negative ΔR² = that family was helping.

Output:
  - results/metrics/level1_B3-SIGNAL-LOO.parquet
  - Dispensability table: family × {ΔR²_loss, DM_p, verdict}
  - Signal attribution entry for reports/signal_attribution.md
```

---

### Session B4 — Signal Family Leave-One-In

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B   ← B4 spec (lines ~283–305)
  results/metrics/level1_B3-SIGNAL-LOO.parquet  ← dispensability context (run B3 first)

Prerequisite: B3 complete (marginal value is interpreted relative to dispensability).

Experiment: B4-SIGNAL-LOI
Goal: what is the marginal value of each signal family when added to macro-only?
      Complements B3. Combined B3+B4 reveals the full signal value picture.

Config:
  universe: MIXED-200 / experiment_a1
  period: FULL-ROLL
  pipeline_depth: best from B1

Iterations (6):
  MACRO-ONLY                          ← minimal baseline
  MACRO+MARKET                        ← add market pricing
  MACRO+NARRATIVE                     ← add GDELT narrative
  BALANCE+MACRO                       ← add EDGAR balance sheet
  MACRO-ONLY + network position only  ← add BEA I/O supply-chain only
  MACRO-ONLY + policy only            ← add Fed/policy signals only

For each: OOS R², ΔR² vs MACRO-ONLY baseline.

Output:
  - results/metrics/level1_B4-SIGNAL-LOI.parquet
  - Marginal value table: family × {ΔR²_gain, rank}

Cross-reference with B3: if family X is dispensable (B3 ΔR²_loss ≈ 0) but has
positive marginal value (B4 ΔR²_gain > 0), it is substitutable (other families
compensate when it is present). Note any such substitution patterns.
```

---

### Session B5 — Signal × Component Interaction

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B5 spec (lines ~307–329)
  results/metrics/level1_B3-SIGNAL-LOO.parquet
  results/metrics/level1_B1-COMPONENT-ABLATION.parquet

Prerequisite: B1 and B3 complete.

Experiment: B5-SIGNAL-COMPONENT-INTERACTION
Goal: do certain signals only add value when certain pipeline components are active?
      Tests H1c (narrative × regime interaction) and H4a (emergence × signal interaction).

Config:
  universe: MIXED-200 / experiment_a1
  period: FULL-ROLL
  measurements: [L1, L2]

2×2×2 factorial (8 runs):
  FULL / emergence=true  / regimes=best  ← A1 result; reuse
  FULL / emergence=true  / regimes=1
  FULL / emergence=false / regimes=best
  FULL / emergence=false / regimes=1
  NO-NARRATIVE / emergence=true  / regimes=best
  NO-NARRATIVE / emergence=true  / regimes=1
  NO-NARRATIVE / emergence=false / regimes=best
  NO-NARRATIVE / emergence=false / regimes=1

Interaction test: fit 2×2×2 ANOVA on OOS R² across the 8 cells.
  Main effects:    narrative (±), emergence (±), regimes (±)
  Interaction:     narrative × regime  (H1c: expect positive)
                   narrative × emergence (H4a: expect positive)
                   regime × emergence (expect positive)
  Three-way:       narrative × regime × emergence

If compute allows, repeat factorial for NO-MARKET instead of NO-NARRATIVE.

Output:
  - results/metrics/level1_B5-SIGNAL-COMPONENT-INTERACTION.parquet  ← 8 cells
  - ANOVA table (effects, p-values, η²)
  - Verdict for H1c and H4a
```

---

### Session B6 — Robustness: Actor Count Sweep

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B6 spec (lines ~334–355)
  results/metrics/level5_A4-SCALING.parquet  ← scaling profile (use if A4 complete)

Prerequisite: A3 passed. (Does not require A1 to be complete.)

Experiment: B6-N-SWEEP
Goal: find the performance cliff as N shrinks. Tests Q5: where does each component
      stop adding value?

Config:
  institutions: INST-MINIMAL
  signals:      MACRO+MARKET
  period:       RECENT (single window)
  pipeline:     full

Iterations (5):
  N=20:  random sample from US-LC
  N=50:  same 50 as A3 (enables direct comparison)
  N=100: random sample from US-LC
  N=200: full US-LC
  N=400: US-LC + US-MC combined

For each N: OOS R², ΔR² per layer (need B1 depth per layer at each N),
            spectral stability (eigenvalue rank correlation across windows),
            runtime (wall time, memory).

Primary question: at what N does regime switching stop adding value?
(Expected: M > 1 requires sufficient cross-sectional variation, likely N > 50.)

Output:
  - results/metrics/level1_B6-N-SWEEP.parquet  ← 5 N values × metrics
  - results/metrics/level3_B6-N-SWEEP.parquet  ← robustness
  - results/metrics/level5_B6-N-SWEEP.parquet  ← compute
  - Performance cliff table: component × min_N_for_value
```

---

### Session B7 — Robustness: Time Series Length Sweep

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B7 spec (lines ~357–378)

Prerequisite: A1 complete.

Experiment: B7-T-SWEEP
Goal: find minimum training window for each component. Tests H3b: regime switching
      should add nothing with short training windows (insufficient regime transitions).

Config:
  universe: US-LC (full N=200)
  institutions: INST-US
  signals:      FULL
  pipeline:     full

All iterations use the same test period: 2024–2025 (single OOS window).
Training windows (5):
  T=5yr:  2019–2023 (train), 2024–2025 (test)
  T=8yr:  2016–2023
  T=10yr: 2014–2023
  T=15yr: 2009–2023
  T=20yr: 2004–2023

For each T: OOS R² per layer, M* selected by MDL, regime duration (quarters).

Expected finding: regime switching needs T ≥ 10yr to see multiple regime transitions.
Kim filter convergence should improve with T.

Output:
  - results/metrics/level1_B7-T-SWEEP.parquet
  - results/metrics/level3_B7-T-SWEEP.parquet
  - Minimum T table: component × {min_T_for_value, reason}
```

---

### Session B8 — Robustness: Noise Injection

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B8 spec (lines ~380–403)

Prerequisite: A1 complete.

Experiment: B8-NOISE
Goal: characterise degradation curve under signal noise. Graceful or cliff?

Config:
  universe: US-LC (N=200)
  institutions: INST-US
  signals:      FULL (with noise applied before pipeline entry)
  period:       FULL-ROLL
  pipeline:     full

Noise is independent Gaussian, added to all signals simultaneously.
Noise levels: σ_noise = {0.0, 0.1, 0.2, 0.3, 0.5, 1.0} × σ_signal (per signal)
Level 0.0 = clean = A1-equivalent result for US-LC.

For each level: OOS R², ΔR² per layer.
Fit degradation curve: R²(σ) = R²₀ · exp(-α·σ²).

Expected: graph construction degrades first (edge weights are most sensitive to noise).
          Spectral compression may be robust (modes aggregate noise away).

Output:
  - results/metrics/level1_B8-NOISE.parquet
  - results/metrics/level3_B8-NOISE.parquet
  - Degradation curve per component (α exponent)
  - Cliff identification: σ level where ΔR² < 0 for each component
```

---

### Session B9 — Robustness: Edge Degradation

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B9 spec (lines ~405–429)

Prerequisite: A1 complete.

Experiment: B9-EDGE-DEGRADE
Goal: how sensitive is the framework to graph misspecification? Answers Q2 for
      the network signal family specifically.

Config:
  universe: MIXED-200 / experiment_a1
  institutions: INST-US + INST-UK
  signals:      FULL
  period:       FULL-ROLL
  measurements: [L1, L2, L3]

Edge corruption = randomly flip edge direction for a fraction of edges
before running the spectral decomposition. Applied independently per window.
Levels: {0.0, 0.1, 0.2, 0.3, 0.5, 1.0} (fraction of edges randomised)
Level 1.0 = completely random graph (Erdős–Rényi with same edge density).

For each level: OOS R², spectral gap (λ₁ - λ₂), rank correlation of leading modes
between corrupted and clean graph.

Robustness criterion: performance at 0.3 corruption ≥ 70% of clean performance.
If met, the framework is robust to reasonable graph estimation error.
If not met, graph estimation quality is critical (raises stake for Granger/TE estimation).

Output:
  - results/metrics/level1_B9-EDGE-DEGRADE.parquet
  - results/metrics/level2_B9-EDGE-DEGRADE.parquet
  - results/metrics/level3_B9-EDGE-DEGRADE.parquet
  - Sensitivity curve and cliff threshold
```

---

### Session B10 — Regime Count Sensitivity

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-B  ← B10 spec (lines ~431–453)
  docs/smim/CLAUDE.md §KimFilter-Limitations  ← symmetric init caveat

Prerequisite: A1 complete (M* from A1 is the reference).

Experiment: B10-REGIME-SWEEP
Goal: validate MDL regime selection. Is the A1 M* genuinely optimal?
      Tests H3a: regime switching adds value in volatile periods.

Config:
  universe: MIXED-200 / experiment_a1
  signals:  FULL
  period:   FULL-ROLL
  measurements: [L1, L2]

Regime counts: M ∈ {1, 2, 3, 4, 5, 6, auto_mdl}
auto_mdl = MDL-selected (this is the A1 configuration; reuse A1 results for M=auto).

IMPORTANT KimFilter note: symmetric initialisation means EM cannot break symmetry
from a symmetric start. When testing M > 1, ensure asymmetric initialisation is used
(different F matrices per regime, or random perturbation from a fitted M=1 solution).
See docs/smim/CLAUDE.md §KimFilter-Limitations for details.

For each M: OOS R², BIC, OOS R² volatility across windows.

Expected: OOS R² peaks at M* (MDL-selected), then plateaus or degrades (overfitting).
BIC should agree with MDL selection.

If MDL selects M=1: regime switching adds no value on this universe. Flag.

Output:
  - results/metrics/level1_B10-REGIME-SWEEP.parquet
  - Regime curve: M → OOS R², BIC
  - Verdict for H3a/H3b
```

---

## Phase C — Transfer Experiments

> Prerequisite for all C experiments: A1 PASSED (OOS R² > 0) AND B1 component
> value table available (to know which components to freeze vs re-estimate).

### Session C1+C2 — Cross-Sector Transfer (Zero-Shot + Fine-Tuned)

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-C        ← C1 (lines ~461–483), C2 (lines ~485–498)
  docs/smim/DATA_STATUS.md §1                  ← per-sector universe coverage
  results/metrics/level1_A1-MVP-FULL.parquet   ← A1 reference
  results/metrics/level2_B1-COMPONENT-ABLATION.parquet  ← which components to freeze

Prerequisites: A1 complete, B1 component value table available.

Experiments: C1-SECTOR-TRANSFER-ZERO and C2-SECTOR-TRANSFER-FINETUNE
Goal: test H5a — do model parameters learned on energy generalise to other sectors?

Training setup:
  train_universe:   US-LC-ENERGY
  train_intensity:  data/smim/intensities/US-LC-ENERGY_intensities.parquet (M-A, N=12)
  train_period:     2010–2022
  train_signals:    FULL
  train_institutions: INST-US

NOTE on N=12: US-LC-ENERGY has only 12 actors with intensity (9 energy actors missing
CapEx EDGAR tag = G-13; 12 have it). The energy spectral operator is 12×12.
Cross-sectional diversity is limited — K* (retained modes) will be small (expect K*≤5).
This is a stress test: can a small-N trained model transfer?

C1 (zero-shot) — apply energy-trained model without ANY re-estimation:
  test_universe: US-LC-TECH,   test_period: 2023–2025, intensity: US-LC-TECH_intensities.parquet
  test_universe: US-LC-FINS,   test_period: 2023–2025, intensity: US-LC-FINS_intensities.parquet
  test_universe: US-LC-HEALTH, test_period: 2023–2025, intensity: US-LC-HEALTH_intensities.parquet
  test_universe: US-LC-INDUS,  test_period: 2023–2025, intensity: US-LC-INDUS_intensities.parquet

C2 (fine-tune) — same sectors, but re-estimate edge_weights and channel_weights
while keeping modal_structure, K*, M*, regime_dynamics frozen from energy training.

For each sector × {zero_shot, fine_tune}:
  OOS R², ΔR² vs sector-trained baseline (train on sector itself 2010–2022, test 2023–2025)
  Transfer retention = OOS_R²_transfer / OOS_R²_sector_trained

Transfer verdict (thresholds below are interpretive — EXPERIMENT_PLAN.md states
"tests H5a" without explicit retention thresholds; calibrate against A1 reference):
  ≥80%: strong transfer
  50–80%: moderate — sector-specific fine-tuning recommended
  <50%: poor — modal structure doesn't generalise

Output:
  - results/metrics/level1_C1-SECTOR-TRANSFER-ZERO.parquet
  - results/metrics/level1_C2-SECTOR-TRANSFER-FINETUNE.parquet
  - Transfer scorecard (sector × method × retention%)
  - "Transfer recipe": which components require re-estimation per sector
```

---

### Session C3 — Cross-Cap Transfer

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-C        ← C3 spec (lines ~503–542)
  docs/smim/METHODOLOGY_ROBUSTNESS_PLAN.md §3   ← C3a/C3b variant definitions
  docs/smim/DATA_STATUS.md §4.3                 ← intensity variant assignments

Prerequisite: A1 complete, B1 available.

Experiment: C3-CAP-TRANSFER
Goal: test H5b (US-MC ≥70% retention) and H5c (US-SC <50% due to data degradation).
      C3a and C3b agreement validates robustness across intensity methodology.

Training setup:
  train_universe: US-LC
  train_period:   2010–2022

C3a (PRIMARY — homogeneous M-A):
  train_intensity: data/smim/intensities/US-LC_intensities.parquet
  test_pairs:
    - US-MC         (M-A): US-MC_intensities.parquet,              zero_shot + fine_tune
    - US-SC_trimmed (M-A): US-SC_trimmed_intensities.parquet,      zero_shot + fine_tune

C3b (ROBUSTNESS — homogeneous M-B):
  train_intensity: data/smim/intensities/US-LC_return_intensities.parquet
  test_pairs:
    - US-MC         (M-B): US-MC_return_intensities.parquet               zero_shot + fine_tune
    - US-SC_trimmed (M-B): US-SC_trimmed_return_intensities.parquet       zero_shot + fine_tune

  DATA GAP WARNING: As of 2026-03-29, US-MC_return_intensities.parquet has NOT been
  generated (not in intensity file index in METHODOLOGY_ROBUSTNESS_PLAN.md §5).
  US-SC_trimmed_return_intensities.parquet row count is "—" (status uncertain).
  Before running C3b, generate missing files:
    uv run python scripts/smim_compute_intensities.py --method return --universe US-MC
    uv run python scripts/smim_compute_intensities.py --method return --universe US-SC_trimmed
  Verify ρ_full > 0.7 before proceeding. If generation fails, C3b is BLOCKED and
  only C3a (M-A) can be run. Document in DECISIONS.md.

CRITICAL: use US-SC_trimmed (N=94), NOT US-SC (N=142 full, 48 high-missing).
Rationale: 48 high-missing actors would confound C3 with data sparsity (see G-10 / RP4).

Interpretation guide:
  C3a and C3b agree on direction+significance → result robust to intensity methodology
  C3a and C3b diverge → investigate (see METHODOLOGY_ROBUSTNESS_PLAN.md §3)

Output:
  - results/metrics/level1_C3-CAP-TRANSFER.parquet  ← C3a + C3b combined
  - Transfer scorecard: cap_tier × variant × {zero_shot_retention, finetune_retention}
  - Verdict for H5b and H5c
  - If C3a ≠ C3b: create METHODOLOGY_DIVERGENCE note in DECISIONS.md
```

---

### Session C4 — Cross-Geography Transfer

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-C            ← C4 spec (lines ~543–576)
  docs/smim/METHODOLOGY_ROBUSTNESS_PLAN.md §§3–4    ← C4a only; C4b dropped
  docs/smim/DATA_STATUS.md §4.2                     ← M-A vs M-B orthogonality (ρ=-0.003)

Prerequisite: A1 complete, B1 available.

Experiment: C4-GEO-TRANSFER
Goal: which components are geography-specific? Expected: macro regime dynamics
      transfer, edge weights require re-specification for UK institutional actors.

CRITICAL METHODOLOGY NOTE:
  C4 MUST use homogeneous M-B intensity for BOTH geographies.
  M-A (capex_assets_xsrank) is EDGAR-based and unavailable for UK equities.
  M-A and M-B are orthogonal per-actor constructs (median ρ = -0.003; RP2 2026-03-29).
  C4b (US M-A vs UK M-B) was DROPPED — it would test confounded differences, not
  geography. Only C4a is valid. See METHODOLOGY_ROBUSTNESS_PLAN.md for full rationale.

Training setup:
  train_universe:    US-LC
  train_intensity:   data/smim/intensities/US-LC_return_intensities.parquet  (M-B)
  train_period:      2010–2022
  train_institutions: INST-US

C4a (PRIMARY and ONLY valid variant):
  test_universe:    UK-LC
  test_intensity:   data/smim/intensities/UK-LC_intensities.parquet  (M-B, ρ=0.732)
  test_institutions: INST-UK
  zero_shot:  no re-estimation
  fine_tune:  re-estimate edge_weights + re-specify institutional actors (INST-UK)

UK-LC data notes:
  - intensity = return_12m_xsrank (M-B; no EDGAR for UK — Companies House adapter not built)
  - 6 high-missing actors (of 97); acceptable (ρ_recent = 0.722 PASS)
  - BIS cross-border edges absent (G-11); this matters most for C4 (cross-geography)
    Disclose in results: "cross-border financial-channel edges absent from FULL feed"

Supplementary table required by paper:
  Report US-LC gap estimates under both M-A and M-B for the same period.
  This quantifies the sensitivity to intensity methodology choice for US-LC actors.
  Source data: US-LC_intensities.parquet (M-A) vs US-LC_return_intensities.parquet (M-B).

Output:
  - results/metrics/level1_C4-GEO-TRANSFER.parquet
  - Transfer scorecard: geo × method × retention%
  - Note: which components required full re-specification for UK
  - Supplementary table: US-LC gap sensitivity to M-A vs M-B
```

---

### Session C5 — Cross-Period Transfer (Structural Break)

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-C  ← C5 spec (lines ~577–598)

Prerequisite: A1 complete, B10 regime sweep available (to understand M* per era).

Experiment: C5-PERIOD-TRANSFER
Goal: which era transitions cause model collapse? Hardest transfer test in the programme.

Config:
  universe:      US-LC (N=200)
  institutions:  INST-US
  signals:       FULL
  transfer_mode: zero_shot (no re-estimation)
  measurements:  [L1, L3]

Era transitions (4):
  PRE-GFC→POST-GFC:    train 2005–2007, test 2008–2009   (pre-crisis → crisis)
  POST-GFC→PRE-COVID:  train 2010–2014, test 2015–2019   (expansion → late expansion)
  PRE-COVID→POST-COVID: train 2015–2019, test 2020–2021  (stable → pandemic)
  POST-COVID→RECENT:   train 2020–2022, test 2023–2025   (pandemic → normalisation)

For each transition: OOS R², component-level ΔR² change vs in-sample.

Expected failure pattern:
  Regime dynamics: should fail across structural breaks (M* changes, transition probs wrong)
  Modal structure: may survive if graph topology stable across eras
  Edge weights: likely fail when economic relationships reorganised (GFC, COVID)

Output:
  - results/metrics/level1_C5-PERIOD-TRANSFER.parquet
  - results/metrics/level3_C5-PERIOD-TRANSFER.parquet  ← stability
  - Transfer table: era_transition × component × survival (%)
  - Most fragile component identification
```

---

### Session C6 — Data Regime Degradation

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-C       ← C6 spec (lines ~600–627)
  docs/smim/EXPERIMENT_OBJECTIVES.md §Q2      ← data regime definitions (Gold/Silver/Bronze/Sparse)

Prerequisite: A1 complete, B1 available.

Experiment: C6-DATA-REGIME
Goal: determine the data regime boundary for each component. Tests Q2.

Config:
  universe: US-LC (N=200)
  institutions: INST-US
  period: FULL-ROLL
  pipeline: full

Regime simulation (4 levels):
  gold:   FULL signals, daily+ frequency  (= A1 equivalent for US-LC; reuse if available)
  silver: remove intraday/daily market, keep quarterly+ only  (FULL minus high-freq OHLCV)
  bronze: MACRO+MARKET (quarterly frequency only, no EDGAR detail)
  sparse: MACRO-ONLY, annual frequency only

For each regime × component depth (L1–L5): OOS R².
Fit: R²(regime) as ordinal regression or threshold model.

Expected: emergence diagnostics require Gold. Graph adds value even in Bronze.
          Regime switching requires Silver or better (needs temporal resolution).

Output:
  - results/metrics/level1_C6-DATA-REGIME.parquet
  - results/metrics/level2_C6-DATA-REGIME.parquet
  - results/metrics/level3_C6-DATA-REGIME.parquet
  - Data regime boundary table: component × min_regime_for_value
  - Complexity map entry for each regime
```

---

## Phase D — Economic Validation

> Phase D is analysis-only (no new pipeline runs). It consumes A1, B1, C-series results.
> D1–D3 are most critical for the paper's economic validity claims.

### Session D1 — Gap Persistence Analysis

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D1 spec (lines ~635–657)
  docs/smim/EXPERIMENT_OBJECTIVES.md §Q6 ← economic validity criteria
  results/metrics/level4_A1-MVP-FULL.parquet  ← gap series output from A1

Prerequisite: A1 complete (gap series y*_{i,t} and Δ_{i,t} must exist in results).

Experiment: D1-PERSISTENCE
Goal: do gaps mean-revert? What is the half-life? Tests H6a (expected: 4–12 quarter half-life).

Analysis:
  1. Per-actor AR(1) on gap series Δ_{i,t}: estimate ρ, half-life = log(0.5)/log(ρ)
  2. Panel AR(1) with actor fixed effects: pooled half-life estimate
  3. Top decile tracking: actors in top 10% |Δ| at time t — track over next 8Q
  4. Quintile transition matrix: gap quintile persistence (does decile 5 stay in decile 5?)

Benchmark classes to analyse separately:
  predictive benchmark → Δ^pred
  modal benchmark      → Δ^modal
  emergence_aware      → Δ^em

If half-life < 2Q: gaps are noise (market efficient, or measurement too noisy)
If half-life 4–12Q: gaps are persistent enough to be economically meaningful (H6a)
If half-life > 16Q: gaps are structural constants, not time-varying signals

Stratify by ActorType (large_firm vs bank vs sector_leader) and by data regime (Gold vs Silver).

Output:
  - results/metrics/level4_D1-PERSISTENCE.parquet
  - Half-life table: benchmark_class × actor_type × {ρ_ar1, half_life_Q, CI_95}
  - Quintile transition matrix (5×5, for each benchmark class)
  - Verdict for H6a
```

---

### Session D2 — Correction Prediction

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D2 spec (lines ~659–686)
  results/metrics/level4_A1-MVP-FULL.parquet
  results/metrics/level4_D1-PERSISTENCE.parquet  ← D1 must run first

Prerequisite: A1 and D1 complete.

Experiment: D2-CORRECTION
Goal: do large gaps predict subsequent repricing or CapEx revision?
      Tests H6b: top gap quintile should predict worse subsequent outcomes.

Outcome variables (all 4 required):
  next_4q_capex_revision:   change in CapEx/Assets over next 4Q (from EDGAR)
  next_4q_total_return:     equity total return (from OHLCV)
  next_4q_credit_spread:    CDS or BAMLH0A0HYM2 proxy change (from FRED)
  next_4q_analyst_revision: consensus EPS revision (external data required — flag if unavailable)

Analysis (3 methods per outcome variable):
  1. Portfolio sort: gap quintile 5 vs quintile 1 average outcome
  2. Panel regression: outcome_{i,t+1→t+4} = β₀ + β₁·Δ_{i,t} + sector_FE + year_FE
     Key: β₁ controls for momentum (last_4q_return) and value (BM ratio)
  3. Logistic: P(correction > cross_sectional_median | |Δ|>75th_pct) vs base rate

The key claim: the gap adds information BEYOND simple momentum and value signals.
Run the regression both with and without momentum/value controls.

Output:
  - results/metrics/level4_D2-CORRECTION.parquet
  - Outcome prediction table: outcome × method × {β₁, t_stat, R²_incremental}
  - Verdict for H6b
  - Note: if analyst_revision data unavailable, use EDGAR next-4Q CapEx as substitute
```

---

### Session D3 — Graph Diffusion Prediction

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D3 spec (lines ~688–711)

Prerequisite: A1 complete (full MIXED-200 run needed for multilayer graph structure).

Experiment: D3-DIFFUSION
Goal: do upstream (Layer 1) gaps predict downstream (Layer 2) gaps along graph edges?
      Tests the transmission channel hypothesis.

Analysis universe: MIXED-200 / experiment_a1 (all layers present: L0, L1, L2)

Layer structure:
  Layer 0: exogenous shocks (global_shock actors)
  Layer 1: upstream institutions (central_bank, regulator, intl_org, bank)
  Layer 2: transmission (large_firm, sector_leader)

Analysis:
  1. Layer lead-lag: does mean |Δ_{L1,t}| Granger-cause mean |Δ_{L2,t}| at lag 1Q?
     Use Granger causality with BIC lag selection, 1Q to 4Q lags.
  2. Neighbour contagion: for each edge (i→j) in graph, does Δ_{i,t} predict Δ_{j,t+1}?
     Panel regression with edge-pair fixed effects.
  3. Channel-specific diffusion: repeat #2 separately for each edge channel
     (macro, credit, supply_chain, narrative) — which channel carries gap diffusion?
  4. Diffusion speed: estimate lag at which predictability peaks (1Q, 2Q, 3Q, 4Q).

Output:
  - results/metrics/level4_D3-DIFFUSION.parquet
  - Lead-lag table: layer_pair × {Granger_F, p, max_lag}
  - Channel diffusion table: channel × {β, t_stat, peak_lag_Q}
  - Transmission diagram: annotated graph with significant channels highlighted
```

---

### Session D4 — Historical Event Alignment

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D4 spec (lines ~713–779)

Prerequisite: A1 complete (US-LC full run needed for sector-level gaps).

Experiment: D4-EVENTS
Goal: do gaps spike around known misallocation episodes?
      If ≥6/8 events show alignment, the framework captures known history.

Events to test (8):
  1. Energy over-investment 2012–2014:  2013-Q2, energy, expected sign = POSITIVE
  2. Pre-GFC bank leverage:             2007-Q2, financials, POSITIVE
  3. Post-GFC tech under-investment:    2010-Q1, technology, NEGATIVE
  4. COVID capital reallocation:        2020-Q2, all sectors, MIXED
  5. 2022 energy super-profits:         2022-Q2, energy, POSITIVE
  6. Fed tightening 2022:               2022-Q3, all sectors, MIXED
  7. AI investment surge 2023:          2023-Q3, technology, POSITIVE
  8. Regional bank stress 2023:         2023-Q1, financials, NEGATIVE

Analysis per event:
  a. gap_level_around_event: compare mean |Δ_sector| in [event±2Q] vs [-8Q, -3Q] control
     Use t-test or Wilcoxon signed-rank.
  b. gap_sign_alignment: does sign(mean Δ_sector) match expected direction?
  c. emergence_spike: do C_t or T_t (TDA) exceed 2× rolling median in event window?
  d. modal_attribution: which spectral mode (k=1,2,...,K*) activates? What does it represent?

Scoring: event ALIGNED if (a) p < 0.10 AND (b) sign correct.
Target: ≥6/8 events aligned.

Output:
  - results/metrics/level4_D4-EVENTS.parquet
  - Event alignment table: event × {gap_magnitude, sign_match, p_value, aligned?}
  - Modal attribution table (which modes activate per event)
  - Hit rate: N/8 events aligned
```

---

### Session D5 — Benchmark Divergence

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D5 spec (lines ~784–807)
  docs/smim/CLAUDE.md §KimFilter-Limitations  ← alpha_pred approximation note

Prerequisite: A1 complete (all three benchmark classes must be present in results).

Experiment: D5-BENCHMARK-DIVERGENCE
Goal: do the three benchmark classes (predictive, modal, emergence_aware) produce
      meaningfully different gaps? Tests H6c.

IMPORTANT KimFilter note: `alpha_pred[t] ≈ alpha_filt[t]` (line 172 of kim_filter.py).
This means predictive and modal benchmarks may be nearly identical by construction.
This is a known approximation — report it explicitly in results.

Analysis:
  1. Pairwise correlation: ρ(Δ^pred, Δ^modal), ρ(Δ^pred, Δ^em), ρ(Δ^modal, Δ^em)
     Compute per actor, report median and P5–P95 across actors.
  2. Mean absolute divergence: mean |Δ^pred - Δ^modal| per actor_type (normalised by σ_Δ)
  3. Divergence by regime: does |Δ^pred - Δ^em| increase during high-criticality periods?
     Correlate divergence with C_t (criticality) and T_t (TDA).
  4. Sign disagreement: fraction of (i,t) where sign(Δ^pred) ≠ sign(Δ^em).

H6c target: mean divergence > 0.3σ at least between some pair of benchmarks.
If all three correlate >0.95: benchmark distinction is academic (only predictive needed).
If Δ^em diverges from Δ^pred during crises: structural benchmark captures real distortions.

Output:
  - results/metrics/level4_D5-BENCHMARK-DIVERGENCE.parquet
  - Divergence table: benchmark_pair × {median_ρ, mean_abs_divergence, sign_disagreement%}
  - Regime-conditioned divergence (crisis vs tranquil periods)
  - Verdict for H6c + note on KimFilter approximation impact
```

---

### Session D6 — Emergence Timing

**Status:** `[ ] not started`
**Outcome:** —

**Prompt:**
```
Context files to read at session start:
  docs/smim/EXPERIMENT_PLAN.md §Phase-D  ← D6 spec (lines ~809–844)
  docs/smim/EXPERIMENT_OBJECTIVES.md     ← H3c, H4a, H4b, H4c hypotheses

Prerequisite: A1 complete (full MIXED-200 run with emergence diagnostics: C_t, T_t, S matrix).

Experiment: D6-EMERGENCE-TIMING
Goal: do emergence diagnostics provide genuine early warning beyond standard risk measures?
      Tests H3c, H4a, H4b, H4c. The hardest test in the programme.

Emergence diagnostic variables (from A1 results):
  C_t:        criticality (phase transition proximity)
  T_t:        TDA topological invariant (Wasserstein distance between persistence diagrams)
  S_total_t:  total synergy (sum of PID synergy matrix S)

Standard comparators:
  VIX:        VIXCLS from FRED (already fetched)
  Realised vol: rolling 3M realised vol from OHLCV
  Correlation: average pairwise correlation of intensity series

Known regime transitions to use as events:
  GFC peak:      2008-Q3
  GFC recovery:  2009-Q2
  Euro crisis:   2011-Q3
  Taper tantrum: 2013-Q2
  COVID shock:   2020-Q1
  Fed pivot:     2022-Q1

For each transition event:
  1. Criticality lead time: when does C_t first exceed 2× rolling median?
     Compare to: when does VIX first exceed 2× rolling median?
     Lead time = VIX_signal_date - C_t_signal_date (positive = C_t leads VIX)
  2. Synergy clustering: compare S_total in [-8Q,-1Q] before vs [-16Q,-9Q] baseline
  3. TDA structural break: does T_t spike vs CUSUM/Bai-Perron test?
  4. Head-to-head: C_t vs VIX, T_t vs realised vol, S_total vs correlation
     Does emergence provide information beyond the standard measure (Granger test)?

Target (H4b): C_t leads VIX by ≥1Q on average across events.
Target (H4c): T_t provides information beyond realised vol (Granger p < 0.05).

Output:
  - results/metrics/level4_D6-EMERGENCE-TIMING.parquet
  - Lead time table: event × diagnostic × {signal_date, lead_time_Q, vs_standard}
  - Granger causality table: emergence_var → standard_var (p-values)
  - Verdict for H3c, H4a, H4b, H4c
```

---

## Execution Priority Order

```
CRITICAL PATH (run in this order, each gates the next):
  1. A3  ← stack validation gate (must pass before anything else)
  2. A1  ← reference baseline (all ΔR² comparisons anchor here)
  3. A2  ← baseline denominator (run in parallel with A1)
  4. B1  ← component value table (required for B3–B10 and all C experiments)

HIGH PRIORITY (run after critical path):
  5.  A4        ← scaling profile (can run in parallel with A1)
  6.  B2        ← best spectral method (sets default for all subsequent)
  7.  B3 + B4   ← signal attribution (run together)
  8.  B6        ← N robustness (independent of B2–B5)
  9.  C1 + C2   ← sector transfer (uses energy training, independent of B5)
  10. C5        ← period transfer (independent, stress test)
  11. C6        ← data regime (independent, bounds-setting)

MEDIUM PRIORITY:
  12. B5        ← signal×component interaction (needs B1 + B3)
  13. B7        ← T-sweep (independent)
  14. B8        ← noise injection (independent)
  15. B9        ← edge degradation (needs A1 MIXED-200)
  16. B10       ← regime sweep (needs A1 for M* reference)
  17. C3        ← cross-cap (needs B1)
  18. C4        ← cross-geography (needs B1; C4a only, M-B homogeneous)

ANALYSIS PHASE (run after critical path + high priority):
  19. D1        ← persistence (needs A1 gap series)
  20. D2        ← correction prediction (needs D1)
  21. D3        ← diffusion (needs A1 MIXED-200)
  22. D4        ← event alignment (needs A1 US-LC)
  23. D5        ← benchmark divergence (needs A1)
  24. D6        ← emergence timing (needs A1; hardest test, save for last)
```

---

## Cross-Cutting Notes for All Sessions

**Data integrity checks (run at start of every session):**
- Confirm intensity file row counts match DATA_STATUS.md §1
- Confirm no pub_date > backtest_date in any signal used (A1 standing assumption)
- For MIXED-200 / experiment_a1: confirm N_intensity=93, not 103 (10 actors are signal-only)
- G-13 rule: 27+ US equity actors (inc. 9 in US-LC-ENERGY) have no CapEx EDGAR tag.
  These are SIGNAL-ONLY graph nodes. Do NOT impute gaps, fill with zeros, or apply
  OHLCV fallback for US actors. They contribute to edge estimation and spectral structure
  but have no gap estimate Δ_{i,t}. Treat them as missing in any gap-level output.
  Full list in DATA_STATUS.md §4.4.

**Standing assumptions (CLAUDE.md):**
- A1 (Point-in-time): never use data with pub_date > backtest_date
- A2 (Typed comparability): normalisation is per-ActorType, not pooled
- A3 (Sparse propagation): operator retains >80% spectral energy after sparsification
- A4 (Stable modes): eigenmode rank correlation >0.5 across ≥80% of rolling windows
- A5 (Regime persistence): average regime duration >8 quarters — check after every M>1 run

**Known implementation deviations (do not revert):**
- I-MB-1: attribution sum = gap_modal − gap_pred (not raw gap)
- P-2: BIC may select M>1 even for pure noise; OOS R² is definitive
- R-TE-1: 50% tolerance on TE vs IDTxl (Kraskov Alg-1 vs Frenzel-Pompe divergence)
- I-TDA-1: d_B < 2ε (not ε; VR stability theorem)
- KimFilter: symmetric init → EM cannot break symmetry from symmetric start;
  always use asymmetric initialisation for M>1 tests

**Results schema:** all outputs follow results/ structure in EXPERIMENT_PLAN.md §Results-Storage.
**DECISIONS.md:** append any architectural decision made during a session.
**DATA_STATUS.md:** update §1 if any new intensity files are generated.
