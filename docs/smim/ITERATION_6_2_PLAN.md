# Iteration 6.2: Does DMD Earn Its Complexity?

> Date: 2026-04-06
> Status: PROPOSED (revised)
> Predecessor: Iteration 6.1 (augmentation works; Ridge/PCA match DMD)
> Core question: After residualisation, does DMD-specific structure beat
>   simpler regularised residual models in any pre-specified regime?

---

## 0. Situation

Iteration 6.1 established:
- Two-stage residual augmentation works (+0.03-0.04 vs AR(1), all panels)
- PCA + diagonal AR(1) ≈ DMD/Kalman diag(Ã) at matched complexity
- Ridge ≈ DMD/Kalman full Ã at unmatched complexity
- DMD provides interpretability (sector-rotation modes) but zero
  demonstrated forecasting edge

The open question: is the equivalence intrinsic, or does it break under
matched complexity, stress conditions, or alternative evaluation?

---

## 1. Design Principles

**Sequential gates, not parallel exploration.** Three gates run in order.
Each has an explicit primary endpoint and kill criterion. Later gates run
only if earlier gates justify them.

**93-actor panel only until a win.** Do not validate on other panels until
a DMD-specific advantage is demonstrated. Cross-panel validation is Gate C.

**Symmetric fairness.** If DMD gets eigenvalue-based F^h for multi-step
forecasts, Ridge gets both direct and iterated h-step. If DMD gets Hankel
embedding, Ridge gets lag-augmented features. No asymmetric advantages.

**Complexity accounting.** Report three things for every model:
1. Raw OOS predictive R²
2. Complexity class: tiny (K), medium (K²), large (regularised N×N)
3. Effective degrees of freedom where estimable (trace of hat matrix
   for Ridge; K or K² for spectral models where the basis is fixed from
   training data)

Do not compute "R² per parameter" as a headline metric. Instead report
the Pareto frontier: R² vs complexity class, letting the reader judge.

---

## 2. Gate A — h=1 Decisive Test (93-actor panel only)

### Goal

Settle the DMD-vs-alternatives question under default conditions (h=1,
T=5yr) before branching to stress conditions.

### Model Table

All models operate on pooled+FE residuals. Grouped by complexity.

**Complexity class: TINY (~K parameters)**

| # | Model | Basis | Dynamics | Params |
|---|-------|-------|----------|--------|
| 1 | PCA/SVD + diag AR | PCA K=8 | per-mode φ_k | 8 |
| 2 | DMD + diag(Ã) | DMD K=8 | diag of Ã | 8 |
| 3 | PCA/SVD reduced-state (no Kalman) | PCA K=8 | diag AR, direct | 8 |
| 4 | DMD reduced-state (no Kalman) | DMD K=8 | diag(Ã), direct | 8 |

**Complexity class: MEDIUM (~K² parameters)**

| # | Model | Basis | Dynamics | Params |
|---|-------|-------|----------|--------|
| 5 | PCA/SVD + full reduced op | PCA K=8 | K×K OLS | 64 |
| 6 | DMD + full Ã | DMD K=8 | full Ã | 64 |
| 7 | PCA + ridge VAR | PCA K=8 | ridge K×K | 64 (shrunk) |
| 8 | Reduced-rank Ridge K=8 | SVD of Ĉ | rank-K approx | ~NK+K² |

**Complexity class: LARGE (regularised N×N)**

| # | Model | Basis | Dynamics | Params |
|---|-------|-------|----------|--------|
| 9 | Ridge on raw residuals | none | ridge β | N²/α (shrunk) |

**Reference (not second-stage)**

| # | Model | R² (6.1) |
|---|-------|----------|
| 0a | Pooled+FE only | 0.591 |
| 0b | Per-actor AR(1) | 0.594 |
| 0c | + residual per-actor AR(1) | 0.605 |

### Three Contrast Blocks

**Block 1 — Basis contrast (dynamics = diagonal AR, fixed):**

Compare 1 vs 2: PCA+diag vs DMD+diag. Same K, same dynamics family,
only basis differs.
- If 2 > 1 with paired-window CI excluding zero → DMD basis adds value
- If 2 ≈ 1 → basis choice is irrelevant at tiny complexity

**Block 2 — Dynamics contrast (basis = DMD, fixed):**

Compare: 4 (projection) → 2 (diag Ã) → 6 (full Ã).

Additionally: construct DMD+ridge VAR (DMD basis, but ridge-penalised
dynamics instead of Ã). This hybrid isolates whether the value is in
Koopman eigenvalues or in generic shrinkage of transition parameters.

- Projection → diag: measures value of per-mode dynamics
- Diag → full: measures value of cross-mode coupling
- Full Ã vs ridge VAR: measures whether Koopman structure in the
  dynamics outperforms generic shrinkage

**Block 3 — Kalman contribution (basis + dynamics fixed):**

Compare 2 vs 4 (DMD+diag, with vs without Kalman).
Compare 1 vs 3 (PCA+diag, with vs without Kalman).

"Without Kalman" means: project r_t onto basis, apply diagonal F
directly to the modal amplitudes, project back. No Kalman gain, no P
matrix, no Q adaptation. Pure reduced-state linear prediction.

- If Kalman adds ≥+0.005 → the filtering machinery (gain, covariance
  tracking, Q adaptation) contributes beyond simple state dynamics
- If Kalman ≈ no Kalman → Kalman is overhead with no forecasting return

### Information Combination Test

For models 2 (DMD/diag), 9 (Ridge), and 1 (PCA/diag):

**Pairwise combinations:**
- DMD + Ridge (equal-weight and train-only CV weight)
- DMD + PCA (equal-weight and train-only CV weight)
- PCA + Ridge (equal-weight and train-only CV weight)

**Triple combination:**
- DMD + PCA + Ridge (train-only OLS weights)

**Report for each combination:**
- Combined R²
- ΔR² vs better input model
- Paired-window loss differential with DM-style t-statistic and p-value
- Forecast-error correlation ρ_pred between component models

**Interpretation:**
- ρ_pred > 0.98 → predictions functionally identical; no unique content
- ρ_pred < 0.95 AND combination DM-significant vs max input → unique
  content exists
- Combination ≈ max input regardless of weights → strict information subset

### Quality Gates
- QG1: AR(1) reproduces 0.594 ±0.002
- QG2: Pooled+FE reproduces 0.591 ±0.002
- QG3: DMD+diag(Ã) reproduces 0.615 ±0.003
- QG4: Ridge reproduces 0.632 ±0.003
- QG5: No NaN/Inf in any predictions

### Kill Rule A

DMD does not beat PCA/SVD at matched complexity (tiny or medium) on the
93-actor panel with paired-window CI excluding zero, AND the combination
test shows DMD has no unique content (ρ_pred > 0.98 with Ridge,
combination does not DM-significantly beat max input).

**If killed at Gate A:** Generic h=1 DMD-specific claim is dead.
Proceed to Gate B regardless — DMD may still win under stress conditions.

**Time:** 6h.

---

## 3. Gate B — Regime-Specific Tests (93-actor panel only)

**Run after Gate A regardless of outcome.** Even if DMD loses at
h=1/T=5yr, it may win under stress conditions.

### Test B1. Training-Window Sweep

Run top models at T ∈ {2, 3, 5, 8} years:
- Per-actor AR(1)
- PCA + diag AR (tiny)
- DMD + diag(Ã) (tiny)
- Ridge (large)
- DMD + full Ã (medium)

**Hypothesis:** At short T, Ridge overfits even with regularisation.
DMD's rank constraint is natural regularisation. DMD should gain
relative to both Ridge and PCA as T shrinks.

**Primary comparison:** DMD+diag vs PCA+diag at each T (matched
complexity, only T varies).

### Test B2. Multi-Horizon Sweep

Run at h ∈ {1, 2, 4} quarters. Symmetric model formulations:

| Model | h-step method |
|-------|--------------|
| DMD+diag(Ã) | F^h · α (eigenvalue extrapolation) |
| PCA+diag | φ^h · α (persistence extrapolation) |
| Ridge iterated | β^h · r (iterated one-step) |
| Ridge direct h | β_h · r (separate model per h) |
| Lag-augmented Ridge | β · [r_t; r_{t-1}] (multi-lag, direct h-step) |

**Symmetry rule enforced:** if later DMD gets Hankel embedding, Ridge
gets lag-augmented features in the same test.

**Diagnostic:** Per-mode predictive R² by horizon. Which modes forecast
well at h=1 vs h=4? If slow modes (high |λ|) gain at longer h, the
eigenvalue structure is earning its keep.

### Test B3. Refit-Frequency Robustness

Run top models under three schedules:
- Quarterly refit (current)
- Annual refit (retrain at start of test year)
- No refit (fixed from initial training window)

**Metric:** R² degradation = (quarterly R²) − (annual R²). Report per
model. DMD should degrade less if its compact representation is stable.

### Forecast-Error Correlation by Regime

Across all B tests, report ρ_pred between DMD, PCA, and Ridge at each
{T, h, refit} condition. If error correlations change (e.g., DMD and
Ridge become less correlated at short T), this reveals where DMD
captures genuinely different dynamics.

### Kill Rule B

DMD has no CI-supported advantage over PCA/SVD at matched complexity in
ANY of {T=2yr, T=3yr, h=2, h=4, annual refit, no refit}:

**The DMD-specific forecasting claim is dead across all tested regimes.**

**Time:** 8h.

---

## 4. Gate C — Cross-Panel Validation (only surviving wins)

**Only if Gates A or B identify a DMD edge.**

Take ONLY the winning configuration (specific T, h, complexity, model)
and validate on:
- 146-firm CapEx/Revenue panel
- 270-actor multi-ratio panel

Run the winning DMD model AND its matched-complexity comparator on each
panel under the winning conditions.

**Replication criterion:** CI excluding zero on ≥2 of 3 panels.

**Time:** 3h (conditional).

---

## 5. Gate D — DMD Variants (only if Gates A-C identify a replicable edge)

| Variant | When justified | Time |
|---------|---------------|------|
| Hankel-DMD (d=2,3) | Multi-horizon edge in B2 | 5h |
| OptDMD / BOP-DMD | Short-T edge in B1 | 4h |
| Extended DMD on modes | Combination shows unique nonlinear content | 4h |
| Subspace / fbDMD | Eigenvalue noise identified as limiting | 3h |

**Closed topics (from 6.1, do not reopen):**
- Kim filter / regime switching
- Spectral Q / structured R
- Raw-panel DMDc or standalone SMIM
- New panel construction

**Time:** 0-16h (conditional).

---

## 6. Success Criteria

### BRONZE
Gate A complete. Clean model table, three contrast blocks, combination
test, Pareto frontier. Publishable methods comparison regardless of winner.

### SILVER
DMD shows a measurable edge in at least one Gate B regime:
- Short-T advantage (DMD > PCA at T≤3yr, matched K, CI excl. zero)
- Multi-horizon advantage (DMD > PCA/Ridge at h≥2)
- Refit robustness (DMD degrades less under annual refit)
- Combination gain (DMD+Ridge DM-significant vs Ridge)

### GOLD
DMD advantage replicates in Gate C across ≥2 panels.

### HONEST COMPLETION
DMD never exceeds PCA/Ridge under any condition. Paper contribution:

1. Two-stage residual-dynamics architecture (+0.03-0.04, all panels)
2. DMD, PCA, Ridge achieve equivalent R² at comparable effective complexity
3. DMD's unique value is structural: sector-rotation modes, per-mode
   dynamics, basis rotation tracking
4. Diagnostic arc: why standalone fails, why augmentation works

---

## 7. Execution Timeline

| Phase | Time | Cumulative | Decision |
|-------|------|------------|----------|
| Gate A | 6h | 6h | h=1 DMD edge? Unique content? |
| Gate B | 8h | 14h | Any regime-specific edge? |
| **Final decision** | — | 14h | **DMD-specific claim alive or dead?** |
| Gate C | 3h | 17h | Cross-panel validation (conditional) |
| Gate D | 0-16h | 17-33h | Variant amplification (conditional) |

**Most likely path:** Gates A+B (14h) → HONEST COMPLETION → paper.

---

## 8. Falsification Commitment

Gate A's combination test is the fastest decisive test. If ρ_pred(DMD,
Ridge) > 0.98, DMD is a strict information subset. No variant can change
this at h=1.

Gate B's multi-horizon test is the last theoretical hope. If DMD
eigenvalues don't outperform at h≥2, no remaining theoretical basis
for DMD-specific forecasting value exists.

If Gates A and B both null: stop. The paper is writable under HONEST
COMPLETION and the iteration history (6.0→6.1→6.2) is a strength.

---

## 9. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_2_gate_a.py` | Gate A: benchmark + contrasts + combination |
| `scripts/smim/run_iter6_2_gate_b.py` | Gate B: T-sweep, h-sweep, refit robustness |
| `scripts/smim/run_iter6_2_gate_c.py` | Gate C: cross-panel validation (conditional) |
| `results/metrics/iter6_2_*.parquet` | Per-gate results |
| `docs/smim/ITERATION_6_2_PLAN.md` | This file |
| `docs/smim/ITERATION_6_2_DECISION.md` | Decision memo (after execution) |