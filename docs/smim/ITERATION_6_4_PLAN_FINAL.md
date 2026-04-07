# Iteration 6.4: Is the Ceiling Fundamental, or Locally Breakable?

> Date: 2026-04-06
> Status: PROPOSED (final)
> Predecessor: Iteration 6.3 (global rotation real but unpredictable)
> Core questions:
>   1. Is the ~0.630 R² ceiling information-theoretically fundamental?
>   2. Are there LOCAL subspaces whose rotation is forecastable even
>      though the GLOBAL rotation is not?
>   3. Can complexity and conditional complexity proxies gate when and
>      where augmentation helps?

---

## 0. Why This Iteration Exists

Iterations 6.0–6.3 exhausted global method-level improvements:

| Iter | Question | Finding |
|------|----------|---------|
| 6.0 | Standalone spectral vs baselines? | Fails — F=0.99I destroys dynamics |
| 6.1 | Transition repair + augmentation? | +0.036 R² — architecture works |
| 6.2 | DMD vs PCA vs Ridge? | All equivalent at matched complexity |
| 6.3 | Predict the rotating geometry? | Global rotation is estimation noise |

**6.3's null has a critical caveat:** The 49° global rotation mixes
macro, institutions, and firms across all sectors. This may be a mixture
of a smooth 15–20° local oscillation within a coherent sector block plus
cross-block noise. 6.3 kills GLOBAL geometry. It does NOT kill LOCAL
geometry within coherent sub-panels, especially when conditioned on the
right complexity regime.

**Iteration 6.4 has four layers:**
1. **Complexity and conditional complexity closure** — is the residual
   fundamentally incompressible? Does knowing r_t help compress r_{t+1}?
2. **Conditional local geometry** — do coherent actor blocks have
   predictable local rotation?
3. **Complexity-gated prediction** — can NCD and algorithmic similarity
   measures detect WHEN augmentation is trustworthy?
4. **Target and scoring audit** — are we measuring the wrong thing?

---

## 1. Design Principles

**Complexity closure first.** Gate C runs before everything else. It now
includes BOTH single-object complexity (is the residual compressible?)
and relational/conditional complexity (does knowing the current
cross-section help compress the next one?). These are fundamentally
different questions.

**Pre-specified partitions only.** Local geometry uses ONLY pre-declared
blocks. No subset fishing. Allowed partitions:
- GICS sector labels (pre-existing, economic)
- Layer labels: macro / institution / firm (pre-existing)
- 2-cluster from training-only k-means on actor loadings
- Regime bins: median-split on training-window NCD or LZ

**Statistical honesty.** 10 windows, limited per-block samples. Univariate
diagnostics only. Report sample sizes alongside all claims.

**93-actor panel only until a positive finding.**

---

## 2. Phase 0 / Gate C — Complexity and Conditional Complexity Closure

### Goal

Determine three things:
1. Is the post-augmentation residual compressible? (single-object)
2. Does knowing r_t help compress r_{t+1}? (conditional complexity)
3. Does complexity vary across actor blocks and windows? (spatial/temporal
   structure of complexity itself)

If the residual is incompressible AND the conditional complexity equals
the marginal, no method at any granularity can improve. If either shows
structure, we know WHERE to look in Gates A/B/E.

### C1. Single-Object Complexity (from previous plan, unchanged)

Compute on three nested objects:

| Object | Definition |
|--------|-----------|
| Raw modal amplitudes α_k(t) | K=8 series, T≈52 |
| Pooled+FE residuals r_t | N=93 × T≈52 |
| Post-augmentation residuals e_t | y_t − ŷ^{aug}_t |

Proxies: normalised LZ, gzip compression ratio, permutation entropy,
sample entropy.

### C2. Conditional Entropy Rate (from previous plan, unchanged)

Compute h(e_{t+1} | e_t) vs H(e_t) via KSG estimator. h ≈ H → memoryless.
h < H → conditional structure exists.

### C3. Normalised Compression Distance — Temporal (NEW)

For each consecutive pair of cross-sectional residual vectors, compute:

    NCD(r_t, r_{t+1}) = (C(r_t · r_{t+1}) − min(C(r_t), C(r_{t+1}))) / max(C(r_t), C(r_{t+1}))

where C is compressed size (gzip or zstd) and · is concatenation.

**Representation:** Serialize each cross-sectional vector as a byte
sequence. Options (compute all, compare):
- Raw float bytes (preserves magnitude)
- Sign-coded binary (+ → 1, − → 0, N bits per snapshot)
- Tercile-coded (3 symbols per actor, base-3 encoding)
- Rank-coded (ordinal encoding of actor positions)

**Temporal NCD series:** Produces T−1 ≈ 51 values of NCD(r_t, r_{t+1}).

**Null comparison:** Compute NCD(r_t, r_π(t)) for random permutations π.
If NCD(r_t, r_{t+1}) << NCD(r_t, r_shuffle), consecutive cross-sections
share algorithmic structure beyond chance.

**Interpretation table:**

| NCD(r_t, r_{t+1}) vs NCD(r_t, r_shuffle) | Meaning |
|------------------------------------------|---------|
| NCD_temporal ≈ NCD_shuffle | Consecutive snapshots are algorithmically independent |
| NCD_temporal < NCD_shuffle by >10% | Temporal algorithmic structure exists |
| NCD_temporal decreasing over time | System becoming more predictable |

### C4. Conditional Algorithmic Complexity (NEW)

Approximate K(r_{t+1} | r_t) ≈ C(r_t · r_{t+1}) − C(r_t):

    novelty_t = K(r_{t+1} | r_t) / K(r_{t+1})

This is the fraction of the next snapshot that is genuinely NEW — not
predictable from the current one via any computable function.

- novelty ≈ 1.0 → r_{t+1} is algorithmically independent of r_t
- novelty ≈ 0.5 → half of r_{t+1} is predictable from r_t
- novelty ≈ 0.0 → r_{t+1} is almost fully determined by r_t

Compare against the model's achieved R² ≈ 0.630:
- If novelty ≈ 0.37 (i.e., 63% predictable), the model is near ceiling
- If novelty ≈ 0.50 (50% predictable), ~13% of headroom remains
- If novelty ≈ 0.90 (10% predictable), the model is already heroic

This is THE most fundamental diagnostic: it tells us the gap between what
we achieve (R²=0.63) and what is information-theoretically achievable
(1 − novelty_mean).

### C5. NCD Matrix — Cross-Temporal Regime Detection (NEW)

Compute the full T×T NCD matrix: NCD(r_t, r_s) for all pairs.

**Structural analysis:**
- Diagonal dominance → temporal locality (nearby quarters similar)
- Block structure → regimes (periods share algorithmic structure)
- Off-diagonal islands → recurrence (2015 pattern reappears in 2022)

**Spectral clustering on the NCD matrix** (K=2 or K=3 clusters) provides
model-free regime classification. Compare NCD-derived regimes to:
- Known economic periods (expansion, tightening, crisis)
- Realised rotation magnitude (from 6.3)
- Augmentation gain (from 6.1)

If NCD regimes align with economic periods AND gain concentrates in
specific regimes → complexity-based regime detection is informative.

### C6. Cross-Block NCD (NEW, feeds Gate E)

For each pair of pre-specified actor blocks (b1, b2), compute:

    NCD(r^{b1}_t, r^{b2}_t) for each quarter t

This measures how algorithmically similar two blocks' residual cross-
sections are at each point in time.

| NCD(tech, fin) | Meaning |
|---------------|---------|
| Low (~0.3) | Blocks share compressible structure — global basis OK |
| High (~0.9) | Blocks algorithmically independent — local bases needed |
| Time-varying | Block coherence is regime-dependent |

**Direct input to Gate E:** If cross-block NCD is consistently high,
mixing blocks in a global basis is wrong and local decomposition is
motivated. If cross-block NCD is low, blocks move together and local
decomposition is unnecessary.

### C7. Per-Window and Per-Block Complexity Table

Compute ALL of the above per window (10 values) and per block (6 blocks):

| Window | LZ_global | LZ_tech | LZ_fin | NCD_temporal | novelty | NCD(tech,fin) |
|--------|----------|---------|--------|-------------|---------|--------------|
| 2015 | | | | | | |
| ... | | | | | | |
| 2024 | | | | | | |

This table is the foundation for ALL conditioning in Gates B and E.

### Compressibility Ladder (REVISED)

| Object | LZ | gzip | Perm.E | h/H | NCD_temp | novelty |
|--------|-----|------|--------|-----|----------|---------|
| Raw α_k(t) | | | | | | |
| Pooled resid r_t | | | | | | |
| Post-aug resid e_t | | | | | | |
| Shuffled e_t (null) | | | | | 1.00 | 1.00 |

### Kill Rule C

**Hard kill:** Post-aug residuals within 5% of shuffled null on LZ AND
h/H > 0.95 AND NCD_temporal ≈ NCD_shuffle AND novelty > 0.95:
**The ceiling is information-theoretically fundamental at ALL levels.**
No method, no granularity, no conditioning can help. Report and stop
the entire programme.

**Soft proceed:** If ANY of {LZ compressible, h/H < 0.90, NCD_temporal
< NCD_shuffle by >10%, novelty < 0.85, per-block complexity varies}:
**Exploitable structure exists.** The specific pattern tells us where:
- Low NCD_temporal → temporal structure → Gates A, B
- High cross-block NCD → blocks are independent → Gate E
- Per-block complexity variation → local structure → Gate E
- Low novelty → headroom exists → all gates motivated

### Time: 4h (expanded from 3h to accommodate NCD computations)

---

## 3. Gate A — Target Audit

### Goal

Test whether the R² ceiling is an artifact of the target formulation.

### A1. Raw Ratios vs Ranks

Run pooled+FE and augmented on percentile ranks vs z-scored raw ratios
vs minmax raw ratios. Compare augmentation GAIN.

### A2. Changes vs Levels

Run on level target y_{i,t+1} vs change target Δy_{i,t+1}. Compare gain.

### A3. Low-Frequency vs High-Frequency Component

Decompose target into 4Q moving average + quarterly residual. Run
augmentation on each separately.

### A4. Split-Half Reliability

Split actors into random halves, compute cross-half prediction
correlation ρ_split. Noise-corrected ceiling = R² × ρ_split.

### Kill Rule A

No variant changes augmentation gain by more than ±0.005 → ceiling is
not target-specific.

### Time: 5h

---

## 4. Gate E — Conditional Local Geometry (THE PIVOTAL GATE)

### Goal

Test whether coherent actor blocks have more predictable local rotation
than the global panel, especially when conditioned on complexity regime.

### E0. Pre-Specified Actor Blocks (declared before execution)

| Block ID | Definition | N_actors | Rationale |
|----------|-----------|----------|-----------|
| SECTOR-TECH | GICS IT + Healthcare | ~20 | Modes 1-2 load here (6.1) |
| SECTOR-FIN | GICS Financials | ~12 | Opposite pole (6.1) |
| SECTOR-IND | GICS Industrials + Energy | ~15 | Cyclical |
| LAYER-MACRO | Layer 0 | 7 | Macro shocks |
| LAYER-FIRM | Layer 2 US firms | ~49 | Homogeneous |
| CLUSTER-1/2 | K-means (K=2) on train loadings | ~46/47 | Data-driven |

### E1. Local Rotation Diagnostics

For each block, re-estimate local DMD basis with K_b = min(4, N_b/5).

Compare to global (6.3):

| Diagnostic | Global | Block b |
|-----------|--------|---------|
| Mean geodesic distance | 49.2° | ? |
| ACF(1) of d_t | −0.07 | ? |
| Axis stability (cosine) | 0.047 | ? |
| NCD(r^{b}_t, r^{b}_{t+1}) from C6 | global NCD | local NCD |

**The NCD diagnostic is new here:** If local temporal NCD is lower than
global temporal NCD, the block has more algorithmic temporal structure
than the full panel — supporting local geometry.

### E2. Local Subspace Prediction

For blocks with ACF > 0.15 or axis cosine > 0.2 or local NCD < global
NCD − 0.05, run P0–P3 prediction models:

P0 (persistence), P1 (last-rotation), P2 (mean-rotation), P3 (EWM).

Kill per block: no P1–P3 beats P0 with CI excluding zero.

### E3. Complexity-Conditioned Local Geometry

Split 52 transitions into two regimes using Gate C diagnostics:

- **LOW-COMPLEXITY:** below-median NCD_temporal in training window
- **HIGH-COMPLEXITY:** above-median NCD_temporal

**Why NCD is the right conditioning variable here (not LZ):** NCD measures
the temporal SIMILARITY between consecutive snapshots — directly testing
"is the next cross-section algorithmically predictable from the current
one?" LZ measures the compressibility of a SINGLE snapshot. A snapshot
can be internally structured (low LZ) but temporally novel (high NCD).
The relevant question for prediction gating is temporal, not structural.

For each block × regime, recompute:
- Local rotation magnitude, ACF, axis stability
- Local projector prediction quality (P0 vs P1)

**Hypothesis:** In low-NCD quarters (high temporal similarity), local
rotation is smoother and more predictable. In high-NCD quarters
(algorithmic novelty), rotations are noise.

### E4. Local Geometry-Aware Actor Reconstruction

If E2 finds a block with predictable rotation, test actor-level R²:

    ŷ^{local-geo} = μ̂_i + Û^{(b)}_{t+1|t} · α̂^{(b)}_{t+1|t}

Compare vs stale local frame, global augmentation, pooled+FE only.

### E5. Mixture-of-Subspaces Architecture

If E4 shows gains for ≥1 block:

1. Pooled AR(1)+FE for all actors (Stage 1)
2. Per-block local DMD + local Ã on residuals (Stage 2)
3. Optionally NCD-gate: only apply Stage 2 in low-NCD quarters

Parameters: 3 blocks × K_b=4 × diagonal = 12 (comparable to global K=8).

### E6. Phase-Locking Within Local Blocks

If E1 finds blocks with stable axes:
- Hilbert transform of local modal amplitudes
- Phase coherence C_{ij} within each block
- Phase coherence between blocks (cross-block lag)

C_{ij} > 0.5 within a block = coupled oscillatory sector dynamics.

### Kill Rule E

E-partial: no block has ACF > 0.15 or axis cosine > 0.2 or local NCD
advantage → skip E2–E6.

E-full: no P1–P3 beats P0 for any block → skip E4–E6.

E-reconstruction: local geo ≤ global augmentation for all blocks →
forecastable but not useful.

### Time: 6h

---

## 5. Gate B — Conditional Predictability (Window-Level)

### Goal

Test whether augmentation gain is conditional on window-level diagnostics,
now including NCD-based measures from Gate C.

### B1. Univariate Gain Diagnostics

Spearman ρ between gain_w and each diagnostic:

| Diagnostic | Source | Hypothesis |
|-----------|--------|-----------|
| Cross-sectional dispersion | Training panel | More dispersion → more to predict |
| Effective rank | Residual covariance | Higher rank → richer structure |
| Eigenvalue concentration | Residual SVD | Concentrated → exploitable |
| Residual persistence | Training residuals | Higher → more signal |
| Rotation magnitude d_t | 6.3 | Large rotation → stale-frame risk |
| Global LZ complexity | Gate C | Compressible → exploitable |
| **Temporal NCD** | **Gate C3** | **Low NCD → more predictable** |
| **Novelty ratio** | **Gate C4** | **Low novelty → more headroom** |
| **Cross-block NCD** | **Gate C6** | **High → blocks independent** |

The three NCD diagnostics are the main addition. They provide model-free,
nonparametric conditioning variables that measure algorithmic predictability
directly rather than through statistical proxies like autocorrelation.

With N=10, need |ρ| > 0.65 for p < 0.05. Report all.

### B2. NCD-Gated Augmentation (NEW)

Construct a causal gate using temporal NCD:

    if NCD(r_{t-1}, r_t) < median(past NCD values):
        use augmented model    (low NCD → temporally similar → predictable)
    else:
        use pooled-only        (high NCD → novel → don't trust second stage)

This is the most theoretically motivated gate: NCD directly measures
"is the current cross-section algorithmically similar to the previous one?"
If yes, the second stage (which extrapolates cross-sectional dynamics)
is more likely to be correct.

Compare: pooled-only vs always-on vs NCD-gated vs best-diagnostic-gated
(from B1).

### B3. Actor-Level Diagnostic (no gating)

Per-actor gain by layer, sector, volatility, modal loading. Diagnostic only.

### Kill Rule B

Best |ρ| < 0.4 AND no gate beats always-on → augmentation is not
meaningfully conditional.

### Time: 3h

---

## 6. Gate D — Ranking and Distributional Metrics

### D1. Ranking Metrics

Spearman ρ, NDCG@20, top/bottom quintile hit rate, directional accuracy.

### D2. Distributional Prediction

Next-quarter dispersion, top-minus-bottom spread.

### D3. Per-Block Ranking (connects to Gate E)

If Gate E identifies coherent blocks, ranking metrics within each block.

### D4. NCD as a Model Quality Metric (NEW)

Compute NCD(ŷ_t, y_t) — how algorithmically similar are predictions to
actuals? Compare across models:

| Model | R² | NCD(ŷ, y) |
|-------|-----|----------|
| Pooled+FE | 0.591 | ? |
| Augmented | 0.630 | ? |
| Ridge | 0.632 | ? |

If NCD tells a different story than R² (e.g., one model has lower NCD
despite similar R²), the models capture different kinds of structure.

### Kill Rule D

Advantage proportional across all metrics → ceiling is metric-invariant.

### Time: 3h

---

## 7. Execution Order

| Phase | Gate | Time | Cumul. | Core question |
|-------|------|------|--------|--------------|
| 0 | **C** (complexity + NCD closure) | 4h | 4h | Residual incompressible? NCD = shuffle? |
| 1 | **A** (target audit) | 5h | 9h | Wrong target? |
| 2 | **E** (local geometry + NCD conditioning) | 6h | 15h | Local rotation forecastable? |
| 3 | **B** (NCD-gated conditioning) | 3h | 18h | When does augmentation help? |
| 4 | **D** (ranking + NCD quality) | 3h | 21h | Wrong scoring? |

**Total: 21h**

### Decision Gates

**After Phase 0 (Gate C):**

| Outcome | Action |
|---------|--------|
| All measures ≈ shuffled null, NCD_temp ≈ NCD_shuffle, novelty > 0.95 | **HARD KILL.** Ceiling fundamental. Run A/B/D/E as characterisation only. |
| NCD_temp < NCD_shuffle (temporal structure exists) | Promising. NCD becomes gating variable for B and E. |
| Cross-block NCD high (blocks independent) | Local decomposition motivated → Gate E critical. |
| Per-block complexity varies | Local structure exists → Gate E critical. |
| Novelty < 0.85 (>15% headroom) | Substantial room to improve. All gates motivated. |

**After Phase 2 (Gate E) — THE PIVOTAL DECISION:**

| Outcome | Action |
|---------|--------|
| ≥1 block has ACF > 0.15 AND lower NCD AND P1 beats P0 AND E4 gains | **POSITIVE.** Build mixture architecture. Validate on other panels. |
| Blocks smoother but P1 ≤ P0 everywhere | Structural finding only. |
| Phase coherence C_{ij} > 0.5 within a block | **STRUCTURAL WIN** even without R² gain. |
| No block improves over global | Local geometry dead at all scales. |

---

## 8. Success Criteria

### BRONZE
Gate C complete with full compressibility ladder including NCD temporal,
conditional complexity, novelty ratio, cross-block NCD, and NCD matrix.
Gate E diagnostics for all pre-specified blocks. Publishable structural
findings regardless of forecasting value.

### SILVER
Gate E: ≥1 block with locally predictable rotation + actor-level R² gain.
OR Gate B: NCD-gated augmentation beats always-on.
OR Gate C: novelty ratio < 0.85 (substantial headroom confirmed,
motivating future work at different frequency/features).
OR Gate A: raw ratios show materially larger augmentation gain.
OR phase coherence C_{ij} > 0.5 within a block.

### GOLD
Mixture-of-subspaces (E5) beats global augmentation on 93-actor panel
with CI excluding zero, optionally NCD-gated.

### PLATINUM
Gold replicates across ≥2 panels AND NCD gating concentrates improvement
in low-complexity regimes.

### HONEST COMPLETION (likely)

All gates confirm ceiling:
- Post-aug residuals near-incompressible globally and locally (Gate C)
- NCD_temporal ≈ NCD_shuffle (no algorithmic temporal structure)
- Local rotation also unpredictable within coherent blocks (Gate E)
- No target variant changes augmentation gain (Gate A)
- Augmentation not meaningfully conditional (Gate B)
- Ranking advantage proportional to R² advantage (Gate D)

**Conclusion:** The ~0.630 R² ceiling is a fundamental property of
quarterly US investment-intensity data — confirmed at the global level,
the local block level, the information-theoretic level, AND the
algorithmic complexity level. The two-stage architecture reaches this
ceiling with any regularised second stage. Further gains require
different data, not different methods, targets, or decompositions.

---

## 9. Why NCD Adds Value Beyond Standard Complexity Measures

Standard complexity (LZ, entropy) answers: "Is this sequence structured?"
NCD answers: "Are these TWO sequences algorithmically related?"

These are different questions with different implications:

| Measure | Question | Implication for forecasting |
|---------|----------|---------------------------|
| LZ(e_t) | Is the residual internally patterned? | If yes → hidden periodicity |
| H(e_t) | How much information per observation? | Upper bound on predictability |
| h(e_{t+1}\|e_t) | Does the past constrain the future? | Temporal exploitability |
| NCD(r_t, r_{t+1}) | Are consecutive snapshots algorithmically similar? | Model-free temporal predictability test |
| K(r_{t+1}\|r_t)/K(r_{t+1}) | How much of the next snapshot is novel? | Achievable R² upper bound |
| NCD(r^{b1}, r^{b2}) | Are two blocks algorithmically similar? | Should they share a basis? |

NCD and conditional K are the ONLY measures that directly estimate the
information-theoretic ceiling without assuming a model class. LZ and
entropy characterise individual sequences. h(e|·) requires an embedding
and estimator choice. NCD uses the compressor as a universal model —
any pattern that ANY computable function could exploit will show up as
lower-than-shuffle NCD.

If NCD_temporal ≈ NCD_shuffle, the conclusion is airtight: no method,
regardless of sophistication, can predict the next cross-section from
the current one better than we already do. This is a much stronger
statement than "our linear methods hit a ceiling."

---

## 10. Falsification Commitment

**Gate C's conditional complexity (NCD + novelty) is the most fundamental
test in the entire iteration history.** It directly estimates the gap
between achieved R² and achievable R². If the gap is <5% (novelty > 0.95),
the programme is definitively complete.

**Gate E with NCD conditioning is the last structural test.** If local
blocks have lower temporal NCD than the global panel but still can't
produce predictable rotation, the geometric approach is dead at all scales.

**If all gates null:** five iterations of systematic falsification.
6.0→6.1→6.2→6.3→6.4 is comprehensive and few papers can match it.

---

## 11. What NOT To Do

- Do not reopen global method horse races (6.2)
- Do not reopen global geometric forecasting (6.3)
- Do not reopen standalone SMIM (6.0)
- Do not fit multivariate gain models on 10 windows
- Do not use complexity as a forecasting ENGINE — only as diagnostic,
  ceiling estimator, and gating variable
- Do not use deep learning, GNNs, or high-capacity models
- Do not create new panels or datasets
- Do not use actor blocks not pre-specified before execution
- Do not claim local geometry value without E4 actor-level demonstration
- Do not open Hankel-DMD, OptDMD, or other DMD variants (6.2 definitive)

---

## 12. Connection to the Proposal

**MDL/compressibility** (Definition 3): Gate C closes the loop with the
proposal's information-theoretic pillar. NCD extends this from single-
object compressibility to RELATIONAL compressibility between consecutive
observations — the operational form of the proposal's vision.

**Multilayer hierarchy** (Section 4): Gate E's block structure and
cross-block NCD directly test whether per-layer dynamics are more
coherent than pooled — the proposal's core architectural bet.

**Phase-transition dynamics** (Section 5.6): NCD-based regime detection
(C5) provides model-free phase classification that connects to the
proposal's Ginzburg-Landau vision without requiring parametric
specification.

**Emergence** (Section 6): Phase coherence within blocks (E6) tests for
cross-actor synchronisation. Cross-block NCD variation tests whether
blocks that become algorithmically similar also become more predictable
— a complexity-theoretic signature of emergent cross-sector coupling.

---

## 13. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_4_gate_c.py` | Complexity + NCD closure (run first) |
| `scripts/smim/run_iter6_4_gate_a.py` | Target audit |
| `scripts/smim/run_iter6_4_gate_e.py` | Conditional local geometry + NCD conditioning |
| `scripts/smim/run_iter6_4_gate_b.py` | NCD-gated window conditioning |
| `scripts/smim/run_iter6_4_gate_d.py` | Ranking / distributional + NCD quality |
| `src/.../smim/complexity/ncd.py` | NCD computation (temporal, cross-block, matrix) |
| `src/.../smim/complexity/conditional_complexity.py` | K(x\|y), novelty ratio |
| `src/.../smim/complexity/compressibility.py` | LZ, gzip, permutation entropy |
| `src/.../smim/complexity/conditional_entropy.py` | KSG conditional entropy rate |
| `src/.../smim/geometry/local_grassmannian.py` | Per-block subspace operations |
| `src/.../smim/geometry/phase_coherence.py` | Hilbert transform + phase-locking |
| `results/metrics/iter6_4_*.parquet` | Per-gate results |
| `docs/smim/ITERATION_6_4_PLAN.md` | This file |
| `docs/smim/ITERATION_6_4_DECISION.md` | Decision memo (after execution) |
