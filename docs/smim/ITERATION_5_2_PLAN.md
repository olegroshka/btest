# Iteration 5.2: Parameter Space Exploration

> Created: 2026-04-04
> Status: PLAN
> Predecessor: Iteration 5.1 (K=2 discovery, cross-sectional pooling finding)
> Best known config: K=2, EWM=12Q, T=3yr, no operator learning
> Nested CV headline: R²=0.711, delta=+0.042, 8/8 wins, DM p<0.001

---

## 1. Motivation

Iteration 5.1 swept K×EWM×T (210 configs) and found K=2 dominates. But we only
tested one intensity construction (CapEx/Revenue) and one set of dual-reg constants
(F=0.99, Q₀=0.5, λ=0.3). These constants were inherited from the 93-actor structural
panel and never re-optimised for the 146-firm predictive panel. There may be
additional gains from:

1. **Dual-reg constants** — F, Q₀, λ might have different optima at K=2
2. **EWM halflife fine-grid** — the 5.1 sweep used {4,6,8,10,12,16,20}; there may be
   a sharper optimum between 6 and 12
3. **Alternative intensity constructions** — Revenue/Assets (rho=0.09) was a near-miss
   in Iteration 5; with K=2 and no operator learning it might cross over
4. **Multi-ratio panel** — stacking CapEx/Rev + Rev/Assets for the same firms creates
   virtual heterogeneity that SMIM can exploit
5. **Rolling window length** — the basis is recomputed each quarter using ALL training
   data; what if we use only the most recent W quarters for DMD?
6. **Shrinkage parameter** — instead of F=0.99 (fixed), try F∈{0.90,0.95,0.97,0.99,1.0}
7. **K=1** — if K=2 beats K=3, does K=1 beat K=2?

All experiments use predictive alpha. All run on GPU where available.

---

## 2. Experiments

### Phase A: Dual-Regularisation Sweep (~5 min)

Fix K=2, EWM=12, T=3yr. Sweep:
- F ∈ {0.90, 0.93, 0.95, 0.97, 0.99, 1.00}
- Q₀ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- λ_Q ∈ {0.1, 0.2, 0.3, 0.5}

Grid: 6×5×4 = 120 configs × 10 windows × ~0.1s = ~2 min

**Question**: Are F=0.99, Q₀=0.5, λ=0.3 actually optimal for the predictive panel?

### Phase B: K=1 Test (~1 min)

Run K=1 with EWM∈{8,12}, T∈{2,3} (4 configs).

**Question**: Is the signal truly 2-dimensional, or does a single mode suffice?

### Phase C: Alternative Intensities (~5 min)

Run the K=2 pipeline (best config from Phase A) on:
- Revenue/Assets (rho=0.09, ~259 firms)
- CapEx/Assets (rho=0.47, ~442 firms, the original signal)
- Multi-ratio: CapEx/Rev + Rev/Assets for overlapping firms (~134 × 2 = 268 cols)

**Question**: Does K=2 without operator learning work on other signals?

### Phase D: Rolling Window for DMD (~3 min)

Instead of using ALL training data for DMD, use only the most recent W quarters:
- W ∈ {8, 12, 16, 20, all} (5 configs)
- Fix K=2, EWM=12, T=3yr

**Question**: Does a shorter DMD window capture more recent structure better?

### Phase E: Fine EWM Grid (~2 min)

EWM ∈ {6, 7, 8, 9, 10, 11, 12, 13, 14} with K=2, T=3yr.

**Question**: Where exactly is the EWM optimum?

### Phase F: Interaction Effects (~10 min)

Take the top findings from A-E and test interactions:
- Best F × best Q₀ × best EWM × K∈{1,2} × T∈{2,3}
- Run nested CV on the best config to get a confirmatory number

**Question**: Can we push the nested CV delta above +0.05?

---

## 3. Success Criteria

| Level | Criterion |
|-------|----------|
| BRONZE | Find a config with fixed-config delta > +0.040 (current: +0.035) |
| SILVER | Nested CV delta > +0.050 (current: +0.042) |
| GOLD | Revenue/Assets or multi-ratio panel also beats AR(1) with K=2 |
| PLATINUM | K=1 works (the signal is truly 1-dimensional) |

---

## 4. Key Scripts

| Script | Purpose |
|--------|---------|
| `run_smim_iter5_1_sweep.py` | Template for parameter sweeps |
| `run_smim_iter5_1_cv2.py` | Nested CV runner (adapt for new configs) |
| `run_smim_iter5_1_ablation.py` | Ablation runner (adapt for new signals) |

---

## 5. What NOT to Do

- Do NOT add operator learning back — K=2 doesn't need it
- Do NOT change the evaluation protocol (nested CV with 2yr inner validation)
- Do NOT run on daily data (closed by Iterations 3-4)
- Do NOT over-optimise — if Phase A finds F=0.97 beats F=0.99 by 0.1pp, that's noise
- Do NOT change the headline numbers unless the improvement is substantial AND
  passes nested CV

---

## 6. Expected Runtime

All phases combined: ~25 min on CPU, ~5 min on GPU.
Total configs: ~150-200 across all phases.

---

## 7. Session Prompt

Copy the block below into the next session to execute this plan.

---

# SMIM Iteration 5.2: Parameter Space Exploration

## Context (read these docs first)

Read `docs/smim/ITERATION_5_2_PLAN.md` for the full plan, then `docs/smim/STATUS.md`
for current project status.

**Current best result** (from Iteration 5.1, all verified):
- Panel: 146-firm US CapEx/Revenue, quarterly, cross-sectional rank ∈ [0,1]
- Config: K=2 modes, EWM halflife=12Q, T=3yr training, no operator learning
- Pipeline: DMD + Kalman (spherical R, F=0.99I, Q₀=0.5I, online Q λ=0.3) + rolling basis
- Nested CV: SMIM R²=0.711 vs AR(1) R²=0.669, delta=+0.042, 8/8 wins, DM p<0.001
- T-sweep: delta grows as T shrinks (+5.7pp at T=2yr), cross-sectional pooling effect
- At K=2 with rolling basis, Kalman is functionally redundant (adds only +0.3pp)
- Paper: `docs/smim/paper/smim_paper.tex` (~870 lines, ready for arXiv)

**Key scripts to reuse:**
- `scripts/smim/run_smim_iter5_1_sweep.py` — 210-config sweep template (K×EWM×T)
- `scripts/smim/run_smim_iter5_1_cv2.py` — nested CV runner (K=2, no OpLearn)
- `scripts/smim/run_smim_iter5_1_ablation.py` — ablation ladder
- `scripts/smim/run_smim_a1i5.py` — CapEx/Revenue panel builder + operator library

**Data:**
- CapEx/Revenue panel: built from `data/smim/processed/edgar_balance_sheet.parquet`
- 93-actor structural panel: `data/smim/intensities/experiment_a1_intensities.parquet`

## Your task

Execute the plan in `docs/smim/ITERATION_5_2_PLAN.md`. Six phases:

**Phase A (~5 min): Dual-regularisation sweep.**
Fix K=2, EWM=12, T=3yr. Sweep F ∈ {0.90,0.93,0.95,0.97,0.99,1.00},
Q₀ ∈ {0.1,0.3,0.5,0.7,1.0}, λ_Q ∈ {0.1,0.2,0.3,0.5}. Report top 10 by
mean delta vs AR(1). Question: are the inherited constants optimal?

**Phase B (~1 min): K=1 test.**
Run K=1 with EWM∈{8,12}, T∈{2,3}. If K=1 beats K=2, that's a major finding.

**Phase C (~5 min): Alternative intensities.**
Run K=2 best config on Revenue/Assets (~259 firms), CapEx/Assets (~442 firms),
and multi-ratio panel (CapEx/Rev + Rev/Assets, ~268 cols). Build the panels from
EDGAR data using the same `build_capex_revenue_panel()` pattern.

**Phase D (~3 min): Rolling DMD window.**
Instead of using ALL training data for DMD, use only the most recent W quarters:
W ∈ {8,12,16,20,all}. Fix K=2, EWM=12, T=3yr.

**Phase E (~2 min): Fine EWM grid.**
EWM ∈ {6,7,8,9,10,11,12,13,14} with K=2, T=3yr.

**Phase F (~10 min): Interactions + nested CV.**
Take top findings from A–E. If any config beats current best (+0.042 nested CV delta),
run full nested CV to confirm.

## Rules

- ALL R² must be PREDICTIVE (alpha_{t|t-1}), never modal
- Use `PYTHONIOENCODING=utf-8` on Windows
- PROJECT_ROOT = `Path(__file__).resolve().parent.parent.parent`
- Save results to `results/metrics/iter5_2_*.parquet`
- Do NOT add operator learning back
- Do NOT change the headline paper numbers unless improvement is substantial
  AND passes nested CV
- If Phase A finds F=0.97 beats F=0.99 by <0.5pp, treat as noise — don't update
- If K=1 works, that IS worth updating the paper for

## Success criteria

| Level | Criterion |
|-------|----------|
| BRONZE | Fixed-config delta > +0.040 (current: +0.035) |
| SILVER | Nested CV delta > +0.050 (current: +0.042) |
| GOLD | Alternative intensity also beats AR(1) with K=2 |
| PLATINUM | K=1 works |

## After experiments

1. Update `docs/smim/STATUS.md` with findings
2. If any result is paper-worthy, update `docs/smim/paper/smim_paper.tex`
3. Regenerate affected figures if numbers change
4. Commit everything
