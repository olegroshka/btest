# Iteration 6.4 (Revised): Is the Ceiling Fundamental, Locally Breakable, or State-Dependent?

> Date: 2026-04-06  
> Status: PROPOSED (revised after 6.3 review)  
> Predecessor: Iteration 6.3 (global rotation real but unpredictable)  
> Core question: Is the ~0.63 predictive ceiling a global data property only, or can it be improved by targeting **lower-noise targets**, **locally coherent actor subspaces**, and **state-dependent activation** of the second stage?

---

## 0. Why This Revision Exists

The current 6.4 draft has the right ambition, but two structural weaknesses:

1. It puts the **most fragile diagnostics** (compression / NCD / novelty) first and treats them as quasi-formal ceilings.
2. It still risks repeating the 6.2 mistake: finding a local/geometric pattern without checking whether it survives against **simple matched local baselines**.

This revision keeps the good ideas — local geometry, conditional complexity, target audit — but makes the programme:
- **more sequential**,
- **more falsifiable**,
- **less dependent on fragile complexity interpretations**, and
- **more likely to produce a publishable interesting result even if the forecasting result stays null**.

The key reframing is:

> Compression-based and complexity-based quantities are **diagnostic signals and gating variables**, not formal upper bounds on achievable \(R^2\).

And:

> Any positive local/geometric result must beat not just the global model, but also **local PCA / local ridge** analogues.

---

## 1. Big-Picture Hypotheses

### H1. The ceiling is partly a **target / measurement ceiling**.
The quarterly rank/intensity target may inject enough jitter that no method can recover additional one-step mean forecastability.

### H2. The global panel is too coarse.
The 93-actor panel may mix several local oscillations. Global geometry can be unforecastable even if **local coherent blocks** are forecastable.

### H3. Predictability is **state-dependent**.
The augmentation gain may concentrate in windows or blocks with low novelty, low temporal complexity, or strong covariance concentration.

### H4. Complexity helps most as a **diagnostic and gate**, not as a forecasting engine.
Conditional complexity proxies may tell us **when** and **where** to trust the second stage, even if they do not improve prediction directly.

### H5. Any local/geometric win must survive against **simple local baselines**.
If a local DMD/geometric model is matched by local PCA or local ridge, then the true finding is local decomposition / conditioning, not spectral superiority.

---

## 2. Design Principles

1. **93-actor panel only until a real positive result appears.**  
   The decisive panel stays decisive. Other panels are for validation only after a positive local/conditional result.

2. **Train-only construction everywhere.**  
   All blocks, clusters, gates, thresholds, and complexity states are derived from training data inside each rolling fold.

3. **Compression metrics are diagnostics, not proofs.**  
   LZ, gzip, NCD, entropy, and novelty are informative but representation-dependent. They can motivate or kill follow-on work, but they do not by themselves prove a hard information-theoretic ceiling.

4. **One primary forecasting endpoint.**  
   The main test is whether a pre-registered local/conditional architecture beats the current global always-on augmentation on the 93-actor panel.

5. **Every promising local/geometric result must face local PCA and local ridge.**

6. **Multiple-testing control by hierarchy, not by brute force.**  
   We proceed gate by gate and do not fan out to many variants unless an upstream gate passes.

---

## 3. Primary Endpoints and Global Kill Rules

### Primary endpoint
On the 93-actor panel, does any **pre-registered local / conditional second-stage architecture** improve predictive \(R^2\) over the current global always-on augmentation by at least **+0.01** with paired-window CI excluding zero?

### Secondary endpoints
- improvement over global always-on augmentation on ranking metrics,
- improvement in identified low-complexity / low-novelty states,
- local subspace/projector predictability beyond persistence,
- economically interpretable local oscillation / phase-coherence findings.

### Global kill rule
If all of the following are null:
- target/noise audit,
- local coherence scan,
- local model horse race,
- conditional gating,
then the ceiling is practically complete on this panel at this frequency. Complexity results are reported only as supporting diagnostics.

---

## 4. Gate M — Measurement / Target / Noise Audit (Run First)

### Goal
Before searching for more structure, determine whether the current target construction itself is capping forecastability.

### M1. Target-construction sensitivity
Compare the current target with alternatives built from the same data:
- percentile ranks (current),
- z-scored raw ratios,
- winsorised z-scores,
- min-max scaled raw ratios,
- sector-relative ranks,
- two-quarter moving-average target,
- four-quarter moving-average target.

### M2. Change-vs-level formulation
Compare:
- next-level prediction \(y_{t+1}\),
- next-change prediction \(\Delta y_{t+1}\),
- smoothed change prediction.

### M3. Rank-jitter / perturbation audit
For each training fold, perturb the raw construction slightly:
- denominator perturbation,
- winsorisation threshold variation,
- leave-one-actor-out rank recalculation,
- small additive noise in raw ratios before ranking.

Measure how much the target moves. This gives a direct estimate of target fragility.

### M4. Reliability / bootstrap ceiling proxy
Use repeated perturb-and-rebuild or repeated half-sample target reconstructions to estimate how much target noise alone limits \(R^2\).

### Metrics
- augmentation gain under each target variant,
- target stability under perturbation,
- reliability-adjusted approximate ceiling.

### Kill rule M
If no target variant changes augmentation gain by more than ±0.01 and target perturbation shows substantial instability, then the ceiling is likely target/noise-driven and later gates should be interpreted accordingly.

### Time
4–5h.

---

## 5. Gate L — Local Coherence Discovery (Before Any Local Forecasting)

### Goal
Identify whether any **pre-registered local blocks or train-only local neighborhoods** show smoother dynamics than the global panel.

### Pre-registered block families
These are allowed because they are economically or algorithmically defined **before** test evaluation:

1. **Economic blocks**
- all GICS sectors with sufficient size,
- macro / institution / firm layer blocks,
- broad cyclical vs defensive grouping if already encoded.

2. **Train-only cluster blocks**
- K=2 and K=3 k-means on train-only actor loadings or train-only residual covariance embeddings,
- spectral clustering on train-only similarity graph.

3. **Train-only neighborhood blocks**
- nearest-neighbor actor neighborhoods by train-only residual correlation or loading similarity,
- fixed-size neighborhoods (e.g. 8–15 actors) around each actor or sector centroid.

The key addition versus the previous draft is **local neighborhoods**, not only hard global blocks. This is the best way to capture “sub actor spaces that are similar” without post-hoc fishing.

### Stable-actor requirement
For each block/neighborhood, use a stable actor set over the relevant train/test period. Do not let actor count variation silently create dimension mismatch or incomparable projectors.

### Local coherence diagnostics
For each block, compute:
- local explained variance concentration / effective rank,
- local residual persistence,
- local temporal NCD / complexity proxies,
- local projector persistence error,
- local geodesic distance,
- local principal-angle dominance,
- axis stability / direction persistence,
- local phase coherence if a conjugate pair is identifiable.

### Selection rule to proceed
A block proceeds to local forecasting only if it meets at least one of:
- lower temporal complexity than the global panel,
- higher local persistence or smoother projector dynamics,
- stronger covariance concentration / lower effective rank,
- projector persistence that is materially better than global persistence.

### Deliverable
A ranked table of candidate local blocks/neighborhoods with diagnostics and sample sizes.

### Kill rule L
If no block/neighborhood looks smoother or more predictable than the global panel, skip all local-geometric forecasting and move straight to Gate G (conditional gating only).

### Time
4–5h.

---

## 6. Gate E — Local Model Horse Race (The Pivotal Gate)

### Goal
For promising local blocks from Gate L, test whether local modelling actually improves actor-level prediction.

### Crucial fairness rule
Every local DMD / local geometry model must be compared against **matched local baselines**:
- local persistence,
- local PCA + diagonal dynamics,
- local ridge,
- optional local reduced-rank ridge,
- and local DMD if still warranted.

This is the main correction to the previous draft.

### Model classes within each selected block
1. local pooled residual baseline,
2. local PCA + diag AR,
3. local PCA + ridge VAR,
4. local DMD + diag operator,
5. local DMD + full operator,
6. local geometric predictor only if local projector predictability beats persistence,
7. local geometry + amplitude model only if above passes.

### Actor-level architectures to compare
- current global always-on augmentation,
- local-only augmentation for actors in block,
- hybrid global + local residual correction,
- mixture-of-subspaces across several disjoint blocks,
- overlapping neighborhood correction if train-only neighborhood discovery survives.

### Primary comparison
Best local/conditional architecture vs current global always-on augmentation.

### Secondary comparison
Best local DMD/geometric architecture vs best local PCA/ridge architecture.

### Kill rule E
If local blocks beat the global model but local DMD does not beat local PCA/ridge, then the finding is **local decomposition matters**, not spectral/geometric superiority.

If no local architecture beats the global model, local geometry is descriptively interesting but not forecast-useful.

### Time
5–6h.

---

## 7. Gate G — Conditional / Gated Activation

### Goal
Test whether the second stage should be activated only in certain states.

### Important improvement over prior draft
The previous draft used only 10 windows for gain correlations. That is too underpowered. This revised gate uses both:
- **window-level diagnostics** for coarse interpretation,
- **quarter-level causal diagnostics** inside rolling evaluation for actual gating.

### G1. Candidate gating variables (train-only)
- temporal NCD or complexity proxies,
- local block complexity from Gate L,
- residual covariance concentration / effective rank,
- cross-sectional dispersion,
- sector dispersion,
- block-level predictability diagnostics,
- simple train-only local forecastability score.

### G2. Baseline gates to beat
Complexity should not get a free pass. Compare against cheap gates:
- no gate (always-on augmentation),
- pooled-only,
- dispersion gate,
- effective-rank gate,
- persistence gate,
- complexity gate,
- combined simple rule-based gate.

### G3. Quarter-level gating
For each forecast quarter, using only available training information, decide whether to apply:
- pooled-only,
- global augmentation,
- local augmentation for selected block(s).

### G4. Actor-level gating
If supported by Gate L/E, allow block-specific or actor-neighborhood-specific activation.

### Primary endpoint
Best causal gated architecture vs global always-on augmentation.

### Kill rule G
If no causal gate beats always-on by more than +0.005 with CI excluding zero, then predictability is not meaningfully state-dependent in a usable way.

### Time
3–4h.

---

## 8. Gate C — Complexity / Compression Coda (Diagnostic, Not Gatekeeping)

### Goal
Characterise what remains after the best model, and whether complexity proxies line up with where the model helps.

### Objects
- raw modal amplitudes,
- pooled residuals,
- post-best-model residuals,
- selected local-block residuals,
- shuffled nulls.

### Proxies
- Lempel–Ziv / compression ratio,
- permutation entropy,
- sample entropy,
- temporal NCD between consecutive snapshots,
- cross-block NCD,
- optional conditional entropy rate if estimation is stable.

### Strong caution
Do **not** interpret:
- novelty ratio as a formal achievable \(R^2\) bound,
- compressor-based NCD as a literal Kolmogorov estimator.

Instead interpret them as:
- relative diagnostics of compressibility,
- model-free indicators of temporal similarity,
- and candidate gating covariates.

### Representation rule
Do not use raw float bytes as the main representation. Use symbolized forms:
- sign patterns,
- tercile symbols,
- rank-coded snapshots,
- optionally quantized magnitudes.

### Deliverables
- compressibility ladder,
- temporal NCD vs shuffled null,
- cross-block NCD map,
- alignment between complexity states and augmentation gain.

### Time
3–4h.

---

## 9. Gate R — Ranking / Distributional Value

### Goal
Check whether the remaining value shows up better in ranking or spread metrics than in mean \(R^2\).

### Metrics
- Spearman rank correlation,
- NDCG / top-k overlap,
- top/bottom quintile hit rate,
- directional accuracy,
- next-quarter cross-sectional dispersion,
- top-minus-bottom spread prediction.

### Important addition
Evaluate these both:
- globally,
- and within any successful local blocks from Gate E.

### Kill rule R
If all advantages are proportional to the mean-\(R^2\) pattern, then the ceiling is metric-invariant. If local/conditional models help more on ranking/spread than on mean \(R^2\), that is a publishable secondary result.

### Time
2–3h.

---

## 10. Execution Order (Revised)

### Phase 1
**Gate M (measurement / target / noise)**

Why first: if the target is too fragile, all later nulls become much easier to interpret.

### Phase 2
**Gate L (local coherence discovery)**

Why second: before building local models, establish whether any local patches are smoother.

### Phase 3
**Gate E (local model horse race)**

Why third: this is the first place where a meaningful forecasting improvement can still emerge.

### Phase 4
**Gate G (conditional gating)**

Why fourth: gating should be built on actual local or diagnostic signals, not on speculation.

### Phase 5
**Gate C (complexity coda)**

Why fifth: complexity now interprets and supports the findings rather than prematurely killing them.

### Phase 6
**Gate R (ranking / distributional metrics)**

Why last: only after we know whether the mean predictor moved.

---

## 11. Decision Tree

### After Gate M
- If target audit strongly changes gains: focus the rest of 6.4 on the better target.
- If not: proceed, interpreting the ceiling as not mainly target-specific.

### After Gate L
- If no local block/neighborhood is smoother than global: skip local geometry and go directly to Gate G/C/R.
- If some are smoother: proceed to Gate E.

### After Gate E
- If a local architecture beats global always-on augmentation: this is the main positive path.
- If local DMD/geometric models do not beat local PCA/ridge: the result is local decomposition / conditioning, not spectral superiority.
- If nothing beats global: local structure is descriptive only.

### After Gate G
- If a causal gated policy beats always-on: this becomes the architectural refinement.
- If not: the current always-on architecture is already near-optimal.

### After Gate C/R
- Use complexity and ranking results to contextualize the ceiling, not to reopen a dead modelling branch.

---

## 12. Success Criteria

### BRONZE
Target/noise audit complete, local coherence map complete, compressibility diagnostics complete. Publishable structural/diagnostic additions even without forecast gain.

### SILVER
One of:
- a local architecture beats the current global always-on model,
- a causal gate beats always-on,
- a target variant reveals materially more headroom,
- ranking/distributional value improves meaningfully even if mean \(R^2\) does not.

### GOLD
A pre-registered local/conditional architecture improves mean predictive \(R^2\) over the global always-on model by at least +0.01 with CI excluding zero, and the mechanism is interpretable.

### HONEST COMPLETION
No target, local, gated, or complexity-conditioned refinement improves the current architecture. Then the paper can say:
- the remaining ceiling survives measurement audit,
- survives local block decomposition,
- survives conditional gating,
- and is consistent with complexity diagnostics.

That is much stronger than saying “we tried more methods.”

---

## 13. What This Means for the Paper

This revised 6.4 is useful under both outcomes:

### If positive
The paper becomes:
- two-stage augmentation works globally,
- and can be improved further by **local / conditional activation**.

### If null
The paper becomes stronger in a different way:
- the ceiling is not just method-invariant,
- it also survives target audit, local decomposition, conditional gating, and complexity diagnostics.

That is a very serious falsification arc.

---

## 14. What Not To Do

- Do not treat compression proxies as formal proofs of Kolmogorov limits.
- Do not let local DMD win by default because local PCA/ridge were not tested.
- Do not use only 10-window gain correlations as the main conditional evidence.
- Do not allow post-hoc block discovery on test data.
- Do not reopen global geometry or global method horse races.
- Do not add deep models, graph nets, or high-capacity nonlinear models.

---

## 15. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_4_gate_m.py` | target / measurement / noise audit |
| `scripts/smim/run_iter6_4_gate_l.py` | local coherence discovery |
| `scripts/smim/run_iter6_4_gate_e.py` | local model horse race |
| `scripts/smim/run_iter6_4_gate_g.py` | conditional gating |
| `scripts/smim/run_iter6_4_gate_c.py` | complexity / compression coda |
| `scripts/smim/run_iter6_4_gate_r.py` | ranking / distributional metrics |
| `results/metrics/iter6_4_*.parquet` | per-gate results |
| `docs/smim/ITERATION_6_4_PLAN_REVISED.md` | this file |
| `docs/smim/ITERATION_6_4_DECISION.md` | decision memo after execution |

