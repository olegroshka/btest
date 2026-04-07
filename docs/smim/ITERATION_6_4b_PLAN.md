# Iteration 6.4b: Heterogeneity-Aware Local Decomposition

> Date: 2026-04-06
> Status: PROPOSED (revised with selective-off baseline + economic pre-specification)
> Predecessor: Iteration 6.4 (local decomposition helps for specific blocks)
> Scope: ONE focused end-to-end test. Not a search tree.

---

## 0. The One Question

Iteration 6.4 showed that 2–3 actor blocks are actively harmed by the
global spectral basis (diversified: R² drops from 0.471 pooled to 0.392
augmented) and substantially helped by local models (diversified: +0.088,
macro_inst: +0.073, both CI excluding zero).

**The one untested claim:** Can these local gains be converted into a
full-panel improvement over global always-on augmentation?

**But there is a prior question:** Is the gain from ADDING local models,
or merely from REMOVING the harmful global augmentation? In 6.4C, the
diversified block's local Ridge (0.480) barely exceeds local pooled-only
(0.471). Most of the +0.088 gain over global comes from turning OFF a
harmful second stage (+0.079), not from turning ON a better one (+0.009).

If the gain is mostly harm-removal → the architectural fix is simpler
("don't augment blocks the global basis misserves") and more robust.
If the gain is genuinely from local modelling → the fix is richer but
requires more parameters and smaller training samples.

Either way, this is the last experiment on this dataset.

---

## 1. Pre-Specification of Block Assignments (Economic Rationale)

Block assignments are justified by economic structure, NOT by 6.4C R²
numbers. This avoids look-ahead bias.

| Block | Actors | N | Economic rationale |
|-------|--------|---|-------------------|
| SEC_diversified | Diversified-sector firms | ~23 | Most heterogeneous sector by definition: actors span unrelated sub-industries. A global basis dominated by sector-rotation modes structurally misrepresents this block. |
| LAYER_macro_inst | Layer 0 + Layer 1 | ~11 | Fundamentally different data types: FRED macro series (min-max normalised) + institutional indicators. Pooling with firm-level cross-sectional ranks is conceptually wrong. |
| MERGED_tech_health | Technology + Healthcare firms | ~25 | Highest within-block loading similarity from 6.1 mode interpretation (modes 1–2 load here). Coherent sub-panel. |
| REMAINDER | All other actors | ~34 | Energy, industrials, financials — sectors where the global basis performs well (6.4C: global beats local for all three). |

These assignments are FIXED before execution. No optimisation of block
membership. No actor can move between blocks based on results.

---

## 2. Architecture Variants

### Stage 1 (unchanged for all variants)
Pooled AR(1) + FE → ŷ^{pool}

### Five Stage 2 variants to compare

| # | Architecture | Stage 2 for local blocks | Stage 2 for REMAINDER | What it tests |
|---|-------------|-------------------------|----------------------|---------------|
| G0 | **Pooled-only** | None | None | Baseline: no augmentation |
| G1 | **Global always-on** | Global C1 full Ã | Global C1 full Ã | Current best (R²≈0.630) |
| S1 | **Selective-off** | None (pooled-only) | Global C1 full Ã | Removing harm only |
| M1 | **Mixture (local Ridge)** | Local Ridge per block | Global C1 full Ã | Adding local models |
| M2 | **Mixture (local PCA+ridge)** | Local PCA+ridge per block | Global C1 full Ã | Alternative local method |

**The critical comparison is S1 vs M1:**
- If M1 ≈ S1 → gain is from harm-removal, not local modelling
- If M1 > S1 → local Ridge genuinely adds value beyond pooled-only
- If S1 > G1 but M1 ≈ G1 → local models add noise that offsets the
  harm-removal benefit

### For the REMAINDER block

All architectures (S1, M1, M2) use the SAME global C1 augmentation for
REMAINDER actors, estimated on the FULL 93-actor panel. This ensures
REMAINDER predictions are identical across S1/M1/M2, so any difference
comes entirely from the treatment of the local blocks.

### Local Model Details

**Local Ridge per block b (N_b actors):**
1. Extract pooled+FE residuals for actors in block b
2. EWM-demean within block (hl=12)
3. Ridge regression: r^{(b)}_{t+1} = C_b · r^{(b)}_t
4. Ridge α ∈ {0.1, 1.0, 10.0} × N_b, selected by LOO within training
5. Predict: ŷ^{local}_{i,t+1} = ŷ^{pool}_{i,t+1} + Ĉ_b · (r^{(b)}_{i,t} − μ̄_b)

**Local PCA+ridge per block b:**
1. Extract residuals, EWM-demean
2. PCA: K_b = min(4, N_b // 5) principal components
3. Ridge VAR on PCA factors: f_{t+1} = A_b · f_t
4. Reconstruct: ŷ^{local} = ŷ^{pool} + U_b · Â_b · (U_b^T · r_t)

**Do NOT use local DMD** — 6.4C confirmed method equivalence at local level.

### Parameter Accounting

| Architecture | Stage 2 params | Notes |
|-------------|---------------|-------|
| G1 (global) | K²=64 (full Ã) | One basis for all 93 actors |
| S1 (selective-off) | K²=64 for 34 actors | Same global, fewer actors augmented |
| M1 (mixture Ridge) | 64 (global) + Σ N_b² (local) | More params but better-conditioned |
| M2 (mixture PCA+ridge) | 64 (global) + Σ K_b² (local) | Fewer local params than Ridge |

---

## 3. Evaluation

### Primary endpoint

Mean OOS predictive R² on the FULL 93-actor panel across 10 rolling
windows.

### Comparison table (pre-registered)

| # | Architecture | R² | ΔR² vs G1 | CI | W/10 |
|---|-------------|-----|-----------|-----|------|
| G0 | Pooled-only | 0.591 | | | |
| G1 | Global always-on | 0.630 | — | — | — |
| S1 | Selective-off | ? | ? | ? | ? |
| M1 | Mixture (Ridge) | ? | ? | ? | ? |
| M2 | Mixture (PCA+ridge) | ? | ? | ? | ? |

### Win condition

**Primary:** M1 or M2 beats G1 by ≥+0.005 with CI excluding zero.
**Secondary:** S1 beats G1 (even without local models, harm-removal helps).
**Diagnostic:** M1 vs S1 (does local modelling add value beyond
harm-removal?).

### Per-block decomposition

Report per-block R² for ALL five architectures. Confirm:
- Local blocks improve under S1 / M1 / M2 vs G1
- REMAINDER block is unchanged (same global predictions)
- The improvement is not offset by REMAINDER degradation

### Quality gates

- QG1: G1 reproduces 0.630 ±0.003
- QG2: G0 reproduces 0.591 ±0.003
- QG3: No NaN/Inf in any predictions
- QG4: Strict causality — block assignments fixed before execution;
  local models use only training data; Ridge α selected by inner LOO

### Cross-panel robustness (conditional)

Only if M1 or M2 beats G1 on the 93-actor panel with CI excluding zero.
Run the winning architecture on:
- 146-firm CapEx/Revenue (homogeneous — mixture may not help)
- 270-actor multi-ratio (moderate heterogeneity)

---

## 4. Outcomes

### If M1 > G1 AND M1 > S1 (+0.005, CI excluding zero for both)

**Strongest result.** Local models genuinely add forecasting value beyond
harm-removal. The paper's recommended architecture upgrades to:

    "pooled+FE → block-specific regularised residual dynamics,
     with blocks pre-specified by economic sector/type structure"

### If S1 > G1 but M1 ≈ S1 (CI for M1−S1 includes zero)

**Harm-removal result.** The global basis is misaligned for heterogeneous
blocks. The fix is simpler: don't augment them. The paper says:

    "The global spectral basis is harmful for heterogeneous blocks
     (diversified sector, macro/institutional actors). Selective
     deactivation of augmentation for these blocks improves full-panel
     R² by [Δ]. Local models do not add value beyond deactivation."

### If M1 ≈ G1 (ΔR² < 0.005 or CI includes zero)

Local gains from 6.4C don't aggregate into full-panel improvement.
Possible reasons:
- Small-block models are noisier (N_block << N_global)
- Cross-block correlations lost by local decomposition
- Well-served REMAINDER blocks don't benefit

### If M1 < G1

Local models overfit. Cross-block pooling in the global basis provides
implicit regularisation that outweighs misalignment cost. Report this.

---

## 5. Implementation Notes

### Ridge α cross-validation

For each block b in each training window:
- α grid: {0.1, 1.0, 10.0} × N_b
- Selection: LOO within training window
- Report: selected α per block per window (verify small blocks get
  stronger regularisation)

### Handling REMAINDER global predictions

The global C1 model for REMAINDER is estimated on the FULL 93-actor
panel (including the local-block actors). This is intentional: it
matches the G1 baseline exactly for REMAINDER actors, ensuring any
full-panel R² difference comes ONLY from the local-block treatment.

### Minimum block size

N ≥ 8 required. All pre-specified blocks meet this (diversified: 23,
macro_inst: 11, tech_health: 25, remainder: ~34).

---

## 6. Timeline

| Step | Time | What |
|------|------|------|
| Implement 5 architectures | 2h | G0, G1, S1, M1, M2 |
| Run on 93-actor panel | 1h | 10 windows, all 5 architectures |
| Per-block diagnostics | 0.5h | Confirm gains/losses by block |
| Cross-panel (conditional) | 1h | Only if M1 or M2 positive |
| Decision memo | 0.5h | ITERATION_6_4b_RESULTS.md |

**Total: 3–5h**

---

## 7. This Is the Last Experiment

Regardless of outcome, this is the final experiment on the current
quarterly investment-intensity panels. The falsification programme
(6.0→6.1→6.2→6.3→6.4→6.4b) has tested:

- Standalone spectral methods (6.0)
- Transition repair + augmentation architecture (6.1)
- Method equivalence at matched complexity (6.2)
- Global geometric forecasting (6.3)
- Target formulation, local coherence, conditional gating,
  algorithmic complexity (6.4)
- Heterogeneity-aware local decomposition (6.4b)

After 6.4b, the paper is written with whatever the results show. No
further iterations on this dataset.

---

## 8. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_4b.py` | End-to-end 5-architecture comparison |
| `results/metrics/iter6_4b.parquet` | Per-window, per-block, per-architecture results |
| `docs/smim/ITERATION_6_4b_RESULTS.md` | Final results + decision memo |
