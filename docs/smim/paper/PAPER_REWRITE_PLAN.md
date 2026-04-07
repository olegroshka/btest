# Paper Rewrite Prompt — Final Integrated Version

## Role

You are rewriting a quantitative finance methods paper FROM SCRATCH. The
existing paper (`smim_paper.tex`) is framed around "spectral augmentation
with DMD." That framing is dead. The new paper is about **heterogeneity-
aware two-stage forecasting for mixed financial panels**, with DMD as one
of several interchangeable second-stage engines and as interpretive
support for the structural analysis.

Read the rewrite plan (`PAPER_REWRITE_PLAN.md`) as the structural
blueprint. This prompt provides critical corrections, tone guidance,
ordering instructions, and a claim audit.

---

## 1. The New Centre of Gravity

### The paper's central claim (one sentence)

> Shared persistence is global, but residual predictability is local in
> heterogeneous mixed-type panels. A global first stage plus block-specific
> local second stage materially improves prediction over the best global
> augmentation.

### The three-layer conceptual spine

| Layer | Claim | Evidence |
|-------|-------|---------|
| L1 | Global first-stage pooling is beneficial for estimating shared persistence | Two-stage augmentation +0.036 R² on 3 panels |
| L2 | Global second-stage pooling can be harmful in heterogeneous panels because residual structure is block-specific | Diversified sector: R² drops from 0.471 to 0.392; geodesic 49° global → 15° local |
| L3 | Global Stage 1 + local Stage 2 is the right architecture | Mixture +0.047 over global, z=7.56 placebo, 10/10 windows |

Every section of the paper should advance one of these three layers. If
a paragraph does not serve L1, L2, or L3, it belongs in the appendix or
should be cut.

### What this is NOT about

- It is NOT about DMD being uniquely valuable for forecasting
- It is NOT about spectral methods beating alternatives
- It is NOT about Koopman operator theory
- It is NOT about "SMIM" as a branded framework

DMD appears as one of three interchangeable engines (alongside PCA and
Ridge), and as the basis for structural spectral analysis in the appendix.
The acronym "SMIM" must not appear anywhere in the new paper.

---

## 2. Title

**Primary recommendation:**

> When Global Factors Hurt: Heterogeneity-Aware Forecasting of
> Cross-Sectional Investment Dynamics

**Alternatives (choose one if the primary feels too provocative):**

> Heterogeneity-Aware Two-Stage Forecasting for Mixed Financial Panels

> Global Persistence, Local Residual Structure: Forecasting Heterogeneous
> Investment Panels

The title must foreground heterogeneity, not spectral methods.

---

## 3. Results Ordering (CRITICAL)

The empirical sections must follow this sequence. Do NOT present per-block
mechanism before the full-panel headline.

1. **Two-stage augmentation works** (L1): +0.036 R² vs AR(1), 3 panels,
   all CIs excluding zero. Table 2.

2. **Method equivalence**: DMD ≈ PCA ≈ Ridge at matched complexity.
   ρ_pred > 0.98. The engine doesn't matter. Table 3.

3. **Full-panel mixture result** (L3): +0.047 over global augmentation,
   CI [+0.036, +0.058], 10/10 windows. Table 5. THIS IS THE HEADLINE.
   Present it BEFORE explaining why it works.

4. **Placebo validation**: z = 7.56 vs 50 random partitions. Random
   blocks HURT (mean Δ = −0.006). Figure 3.

5. **Cross-panel scope condition**: gain on 93-actor heterogeneous panel
   only. 146-firm: Δ = −0.003. 270-actor: Δ = +0.001. Table 8.

6. **Per-block mechanism** (L2): AFTER the full-panel result is
   established. Diversified harmed, tech/health boosted, macro/inst
   improved. Selective-off is WORSE than global → local models add
   genuine value, not just harm-removal. Table 6.

7. **Geodesic mechanism**: 49° global rotation decomposes into ~15° local
   + cross-block interference. Table 4, Figure 4.

8. **What doesn't work**: compressed falsification arc. Each negative
   result rules out an alternative explanation.

---

## 4. Section-by-Section Instructions

### Abstract (~200 words, ONE paragraph)

Lead with the surprise, not the method:

"Global factor augmentation of cross-sectional forecasts can actively
degrade prediction for actor subgroups whose dynamics are misrepresented
by the shared basis. [Quantify: −8pp for diversified sector.] We propose
a heterogeneity-aware two-stage architecture: [describe briefly]. On a
93-actor multilayer panel, the mixture improves R² from 0.630 to 0.677
(Δ = +0.047, CI [...], z = 7.56 vs placebo). [Method invariance sentence.]
[Scope condition: heterogeneous panels only.] [Structural finding: basis
rotation, geodesic decomposition.]"

Do NOT mention DMD, Kalman, or Koopman in the abstract. The abstract is
about the architecture and the heterogeneity finding.

### Section 1: Introduction (~1,500 words)

**Para 1-2 (Hook):** Factor models are ubiquitous. The standard practice:
global basis → project all actors → model dynamics. Nobody asks whether
the global basis makes things WORSE for some actors.

**Para 3-4 (Problem):** On a multilayer investment panel, global spectral
augmentation improves average prediction but reduces R² for the
diversified sector by 8pp below pooled-only. The global basis captures
cross-sector rotation that is noise for within-sector prediction.

**Para 5-6 (Solution):** Heterogeneity-aware architecture: global
persistence + block-specific residual dynamics. +0.047 R², z=7.56
placebo, strict train-only evaluation.

**Para 7-8 (What we rule out):** Systematic falsification: not the method
(DMD ≈ PCA ≈ Ridge), not the geometry (rotation is estimation noise), not
the target (ceiling is target-invariant), not conditional gating
(augmentation is unconditionally beneficial). Only source: block structure.

**Para 9 (Broader relevance):** Applies to multi-asset factor models,
macro-financial panels, multilayer networks. Provide a diagnostic
(geodesic decomposition, cross-block NCD) for detecting when global
models are suboptimal.

**Para 10 (Outline):** Brief roadmap.

Do NOT discuss DMD mathematics, Koopman theory, graph signal processing,
or spectral operators in the introduction. These are technical details
for §3 or appendix.

### Section 2: Data and Setting (~800 words)

**2.1 The 93-actor multilayer panel**

Report: three layers (7 macro, 5 institutional, 81 firms), seven sector
blocks, quarterly 2005Q1–2024Q4.

**CRITICAL:** State the normalisation method per layer explicitly:
- Layer 0 (macro): FRED min-max normalisation (trending, ρ ≈ 0.88)
- Layer 2 (firms): cross-sectional percentile rank (mean-reverting, ρ ≈ 0.60)

The mixed normalisation reflects fundamentally different data types —
trending macro indices and cross-sectionally ranked firm ratios — which
drives the heterogeneity effect. The root cause is the difference in
data-generating processes (trending vs mean-reverting), not the
normalisation choice per se. The reader must see this in the data
section, not discover it in §4.

**2.2 Validation panels** (146-firm, 270-actor)

**2.3 Evaluation protocol** (10 windows, 5yr training, quarterly refit,
predictive R², bootstrap CI, permutation tests)

**Table 1:** Panel descriptive statistics.

### Section 3: The Two-Stage Architecture (~1,200 words)

**3.1 Stage 1: Global pooled AR(1)+FE**
- Why global: persistence benefits from pooling
- Equation: ŷ^{pool} = μ̂ + ρ̂(y_{t-1} − μ̂)

**3.2 Stage 2: Residual dynamics**
- Three interchangeable engines: PCA+diag AR, DMD+Ã, Ridge
- Combined forecast: ŷ = ŷ^{pool} + ŷ^{resid}

**3.3 Why two stages?**
- Standalone spectral fails (R²=0.486 vs AR(1) 0.594) — one paragraph,
  not a full section. Brief reference to appendix for the transition
  diagnostic.

**3.4 Method equivalence**
- 13 models across 3 complexity classes (Table 3)
- At matched complexity: PCA ≈ DMD ≈ Ridge
- **Include forecast-error correlations:** ρ(DMD, PCA) = 0.990,
  ρ(DMD, Ridge) = 0.980. Predictions are functionally identical, not
  just equally accurate on average.
- Forecasting ceiling ≈ 0.630 regardless of method

**Table 2:** Two-stage augmentation vs baselines (3 panels)

**BASELINE CONSISTENCY WARNING:** The 6.1 evaluation protocol used fixed
AR(1) (ρ estimated once per window, R²=0.594 on 93-actor). The 6.4b
protocol uses rolling AR(1) (re-estimated quarterly, R²≈0.610). A
referee will notice if Table 2 says "AR(1)=0.594" and Table 5 implies
"AR(1)≈0.610." Resolution: report GAINS (Δ = augmented − AR(1)) as
the primary comparison in Table 2, not absolute R². Alternatively,
re-run Table 2 under the 6.4b rolling protocol for all three panels.
The gain (+0.036) is robust to the AR(1) specification; the absolute
levels are not.

**Table 3:** Method comparison (13 models, grouped by complexity class)

**3.5 Heterogeneity-aware extension (the mixture architecture)**

Present the mixture architecture as the natural next step:
- Pre-specified blocks by sector/layer
- Per-block local PCA+ridge on residuals
- Global augmentation for well-served remainder blocks
- Algorithm box showing the full mixture pipeline (NOT the global-only
  pipeline from the current paper)

### Section 4: When Global Models Hurt (~1,500 words)

**THIS IS THE HEART OF THE PAPER.**

**4.1 Full-panel result (present FIRST)**

The five-architecture comparison on the full 93-actor panel:

| Architecture | R² | Δ vs G1 | CI | Windows |
|-------------|-----|---------|-----|---------|
| G0 Pooled-only | 0.591 | −0.039 | | 0/10 |
| G1 Global always-on | 0.630 | — | | — |
| S1 Selective-off | 0.599 | −0.031 | | 0/10 |
| M1 Mixture (Ridge) | 0.669 | +0.039 | [...] | 10/10 |
| M2 Mixture (PCA+ridge) | 0.677 | +0.047 | [...] | 10/10 |

**Emphasise:** S1 < G1 → the gain is NOT harm-removal. The local models
genuinely ADD predictive value (M1 − S1 = +0.070, t = 16.12).

**4.2 Per-block decomposition (present AFTER 4.1)**

| Block | N | Pooled | Global | Mixture |
|-------|---|--------|--------|---------|
| SEC_diversified | 23 | 0.415 | 0.392 | 0.449–0.461 |
| LAYER_macro_inst | 11 | 0.600 | 0.649 | 0.689–0.692 |
| MERGED_tech_health | 25 | 0.554 | 0.681 | 0.764–0.808 |
| REMAINDER | 34 | 0.622 | 0.646 | 0.646 |

Key narrative: global augmentation REDUCES R² for the diversified sector
below pooled-only (0.392 < 0.415). Local models recover this and more.
MERGED_tech_health is the star: +0.127 local vs global.

**4.3 The geodesic mechanism**

Why global hurts: the global basis spans 93 actors with incompatible
dynamics. Different blocks have different co-movement structures. The
global basis rotates fast (~49°/Q) because it chases cross-block
interference. Within coherent blocks, rotation is slow (~15°/Q) and the
basis is stable.

**Table 4:** Local coherence diagnostics (geodesic, persistence, NCD,
effective rank per block)
**Figure 4:** Geodesic distance bar chart: global vs per-block

**4.4 The synergy: global Stage 1 + local Stage 2**

The full-panel mixture gain (+0.047) exceeds the weighted sum of 6.4C
per-block gains (~+0.025). This is CONSISTENT WITH an architectural
synergy (ADR-015): global Stage 1 estimates ρ from all 93 actors (more
data), producing cleaner residuals for the local Stage 2. However, this
comparison is INDIRECT — the 6.4C and 6.4b results used different
scripts and protocols. Frame as "consistent with synergy" rather than
"demonstrates synergy." Do NOT add a sixth architecture row to test
this formally — it risks diluting the clean 5-architecture table.

### Section 5: Validation (~1,000 words)

**5.1 Placebo test**
- 50 random partitions, same block sizes
- Real: +0.047. Placebo mean: −0.006. Placebo max: +0.012. z = 7.56.
- State explicitly: "Each placebo uses the same 10 windows, same block
  sizes, and same local model specification. The only difference is
  actor assignment."
- Random blocks HURT on average → local estimation noise dominates when
  blocks are arbitrary.

**Figure 3:** Placebo distribution histogram with vertical line at +0.047.

**5.2 Cross-panel validation**
- 146-firm: Δ = −0.003 (no benefit)
- 270-actor: Δ = +0.001 (no benefit)
- The gain requires DATA-TYPE heterogeneity (macro indices + firm ratios)
- This is a scope condition, not a failure

**Table 8:** Cross-panel results

**5.3 Economic validation (brief)**
- Gap-revision regression: both pooled and augmented gaps predict
  revision (β < 0, |t| > 10), both survive actor FE
- Augmented gaps are less predictable → signal absorption, not weaker
  content
- ONE paragraph, ONE table
- **NOTE:** This validation uses the global augmentation model (6.1 C1),
  not the mixture (M2). State: "The economic validation in this section
  uses the global augmentation model. The mixture architecture's gap
  properties are a natural extension but were not separately tested."

**5.4 Train-only causality audit (brief)**
- Block assignments: static economic metadata
- All models: re-estimated each quarter from training data only
- No look-ahead bias

### Section 6: What Doesn't Work and Why (~1,200 words)

**Frame as "ruling out alternative explanations for the heterogeneity
finding." NOT as a chronological research diary.**

Each subsection answers: "Could the full-panel gain be from X instead
of block decomposition? No, because [evidence]."

**Include a falsification summary table:**

| Alternative explanation | Test | Result | Implication |
|------------------------|------|--------|-------------|
| Better spectral method | 6.2: 13 models, 3 complexity classes | DMD ≈ PCA ≈ Ridge (ρ > 0.98) | Architecture, not method |
| Forecastable global geometry | 6.3: 10 geometric models | All worse than persistence | Rotation is estimation noise |
| Target formulation | 6.4A: 7 variants | Max |ΔGain| = 0.011 | Ceiling is target-invariant |
| Conditional gating | 6.4D: 5 policies | All worse than always-on | Augmentation is unconditional |
| Residual compressibility | 6.4E: NCD ratio | 0.976 vs shuffled | Near-incompressible |

**Main-text vs appendix triage (DECIDE EXPLICITLY):**

| Negative result | Main text? | Appendix? | Rationale |
|----------------|-----------|----------|-----------|
| Method equivalence (6.2) | **Main §3.4** | Full 13-model table | Directly supports L1 (architecture is the contribution) |
| Standalone spectral failure | **1 paragraph §3.3** | Transition diagnostic in appendix | Motivates two-stage architecture |
| Global geometry failure (6.3) | **1 paragraph §6.2** | Rotation diagnostics in appendix | Supports L2 (global basis is noisy) |
| Target audit (6.4A) | **2 sentences §6.3** | Full variant table in appendix | Rules out target explanation |
| Gating failure (6.4D) | **2 sentences §6.4** | Full policy table in appendix | Rules out conditional explanation |
| NCD/complexity (6.4E) | **2 sentences §6.5** | Compressibility ladder in appendix | Weak evidence; do not overclaim |
| Kim filter, spectral Q/R | **1 sentence §6.1** | Not even appendix | Too technical, zero result |
| Emergence, TDA, PID, TE | **1 sentence §6.1** | Not even appendix | Never produced results |

### Section 7: Discussion (~800 words)

**7.1 "Pool globally, decompose locally" as a design principle**
- Applicable beyond this specific panel
- Decision rule: if cross-block NCD is high AND local geodesic is
  substantially below global → local decomposition warranted

**7.2 Connection to heterogeneous factor model literature**
- Ando & Bai (2017): grouped panel data with latent group structure
- Su, Shi & Phillips (2016): identifying latent structures
- Bonhomme & Manresa (2015): grouped patterns of heterogeneity
- Ke, Fan & Wu (2015): homogeneity pursuit
- Our approach: pre-specified economic blocks rather than data-driven
  grouping. Future work: automated block discovery.

**7.3 Limitations** (state prominently, not hidden)
1. 10 OOS windows is small. Placebo mitigates but does not eliminate.
2. Block assignment informed by 6.4 Gate C results — blocks are economic
   classifications but SELECTION of which get local treatment was informed
   by within-sample diagnostics.
3. Gain does not replicate on homogeneous panels.
4. Quarterly frequency only.
5. Improvement is +0.047 R² — statistically robust but modest in
   absolute terms.
6. **Block boundary sensitivity untested.** The robustness of the result
   to marginal changes in block membership (e.g., reclassifying 3-4
   borderline actors between diversified and REMAINDER) is not tested.
   The placebo test addresses random FULL partitions but not small
   perturbations of the real partition.

**7.4 Extensions**
- Automated block discovery via cross-validation
- Higher-frequency data where rotation may be trackable
- Change-based targets (augmentation gain 2.5× larger on first-differences)

### Section 8: Conclusion (~400 words)

Restate the three-layer spine (L1, L2, L3). End with the design
principle: "pool globally for what is shared, decompose locally for
what is heterogeneous." Note the scope condition: the gain requires
panel heterogeneity.

---

## 5. Figure Hierarchy (revised)

| Priority | Figure | Content | Section |
|----------|--------|---------|---------|
| 1 | Architecture diagram (DUAL) | Side-by-side: LEFT = global pipeline (global S1 → global S2 → ŷ), RIGHT = mixture pipeline (global S1 → block assignment → per-block S2 → ŷ). Same input, different Stage 2 routing. Makes the contribution visually immediate. | §3 |
| 2 | Per-window R² comparison | G1 vs M2 (with AR(1) and pooled+FE as reference lines), showing M2 > G1 in all 10 windows. Include the per-window numerical values as a supplementary panel or inset table — seeing all 10 individual windows is more convincing than "10/10 positive" | §4.1 |
| 3 | Placebo distribution | Histogram of 50 placebo Δ values + vertical red line at +0.047, annotated z=7.56 | §5.1 |
| 4 | Geodesic bar chart | Global vs per-block geodesic distance. Visual of 49° → 15° reduction | §4.3 |
| 5 | Per-block R² by architecture | Grouped bar: 4 blocks × 3-5 architectures | §4.2 |
| 6 (optional) | Cross-block NCD heatmap | Block-pair NCD matrix | §4.3 |

**Figures from current paper to MOVE TO APPENDIX:**
- Basis rotation time series (current Figure 8) → Appendix
- Structural performance ladder (current Figure 4) → Appendix
- Variance decomposition pie charts (current Figure 9) → Appendix

**Figures from current paper to CUT FROM MAIN TEXT:**
- Spectral method comparison bar chart (current Figure 5) → Appendix
- Regularisation path hockey-stick (current Figure 6) → Appendix
- Per-window rolling vs static (current Figure 7) → Appendix

---

## 6. Tone and Claim Discipline

### 6.1 Vocabulary rules

| Context | USE | AVOID |
|---------|-----|-------|
| Describing the global-basis problem | "cross-block interference," "misalignment of shared residual basis," "suboptimal pooling" | "harm" (sparingly OK in title/abstract hook only) |
| Describing the result | "the mixture architecture improves R² by +0.047" | "our novel architecture dramatically outperforms" |
| Describing method equivalence | "DMD, PCA, and Ridge achieve statistically indistinguishable R² at matched complexity" | "DMD is as good as Ridge" (implies DMD is the reference) |
| Describing the ceiling | "consistent with limited remaining exploitable structure" | "proves the ceiling," "information-theoretically impossible" |
| Describing the falsification arc | "systematic testing of alternative explanations" | "our journey," "we discovered through trial and error" |

### 6.2 Numbers discipline

- EVERY R² claim includes CI and window count in the same sentence
- EVERY comparison includes paired-window significance (t-stat or p-value)
- EVERY "does not improve" includes CI showing zero is included
- The placebo z-score appears within 2 sentences of the +0.047 claim

### 6.3 No residual DMD/spectral branding

- Do NOT use "SMIM" anywhere
- Do NOT call the architecture "spectral augmentation" in the title
- Do NOT lead any section with "the DMD basis" as if DMD is the key
- When describing the second stage, list all three engines: "PCA+ridge,
  DMD+Ã, or Ridge regression (§3.4 demonstrates these are interchangeable)"

---

## 7. References to Add

**Heterogeneous panel / grouped factor models:**
- Ando, T. and Bai, J. (2017). Clustering huge number of financial time
  series: A panel data approach. JASA, 112(519):1182-1198.
- Su, L., Shi, Z., and Phillips, P.C.B. (2016). Identifying latent
  structures in panel data. Econometrica, 84(6):2215-2264.
- Bonhomme, S. and Manresa, E. (2015). Grouped patterns of heterogeneity
  in panel models. Econometrica, 83(3):1147-1184.
- Ke, Z.T., Fan, J., and Wu, Y. (2015). Homogeneity pursuit. JASA,
  110(511):175-194.

**Factor model mis-specification / heterogeneous loadings:**
- Gagliardini, P., Ossola, E., and Scaillet, O. (2016). Time-varying
  risk premium in large cross-sectional equity datasets. RFS, 29(3):714-742.

All existing references (Bates & Granger, Timmermann, Reinsel & Velu,
etc.) are retained.

---

## 8. Claim Audit Checklist

### Claims we can make confidently

| Claim | Evidence | Strength |
|-------|---------|----------|
| Two-stage augmentation improves R² by +0.02-0.04 on 3 panels | Tables 2, 8; all CIs excl. zero | Strong |
| DMD ≈ PCA ≈ Ridge at matched complexity | Table 3; ρ_pred > 0.98 | Strong |
| Global augmentation reduces R² for diversified sector | Per-block table; 0.471 → 0.392 | Strong |
| Mixture improves full-panel R² by +0.047 | CI [+0.036, +0.058]; 10/10; z=7.56 | Strong |
| Selective-off is worse than global → local models add value | S1 vs G1: −0.031; M1 vs S1: +0.070 | Strong |
| Gain requires panel heterogeneity | Cross-panel null on 146-firm and 270-actor | Strong |
| Local blocks have lower geodesic rotation than global | 15° vs 49°; 9/10 blocks smoother | Strong |
| Augmentation is unconditionally beneficial (no useful gate) | 5 policies all worse than always-on | Strong |

### Claims we should make cautiously

| Claim | Qualification |
|-------|-------------|
| "The ceiling is ~0.630" | At this frequency, on this panel, with these methods. Not a formal bound. |
| "Residuals are near-incompressible" | Under tercile symbolisation. Compression estimate (0.559) < achieved R² (0.630) — symbolisation is lossy. |
| "The geodesic mechanism explains the harm" | Correlational: local geodesic is lower where local models help. Not a causal proof. |
| Block assignment is purely pre-specified | Blocks are economic classifications but the selection of which get local treatment was informed by 6.4C. |

### Claims we should NOT make

| Claim | Why not |
|-------|--------|
| "DMD is uniquely valuable for forecasting" | 6.2 definitively falsified |
| "Koopman eigenvalues provide multi-step advantage" | 6.2 B2: Ridge matches at h=2 |
| "We prove an information-theoretic ceiling" | NCD ratio is a diagnostic, not a theorem |
| "The finding generalises to all panels" | Cross-panel null on homogeneous panels |
| "Spectral methods outperform" | They match, at best |
| "Macro→firm propagation drives the gain" | A4 showed diagonal ≈ full Ã; no propagation evidence |

---

## 9. Appendix Structure

| Appendix | Content | Source |
|----------|---------|--------|
| A | DMD mathematics (algorithm, Koopman connection, spectral radius clipping) | Current §2.3-2.5, compressed |
| B | Standalone transition diagnostic (the F=0.99I bottleneck, shrinkage sweep, A4 diagonal result) | Iterations 6.0-6.1 |
| C | Structural spectral analysis (performance ladder, basis rotation, variance decomposition, regularisation path) | Current §4.5-4.10, compressed |
| D | Target sensitivity details | 6.4 Gate A full table |
| E | Gating policy details | 6.4 Gate D full table |
| F | Rotation diagnostics and predictability tests | 6.3 Gate A |
| G | Compressibility ladder and NCD details | 6.4 Gate E |

---

## 10. Deliverable

Produce a COMPLETE rewritten paper as a `.tex` file with:

1. New title, abstract, and introduction per the instructions above
2. Restructured sections 2-8 following the ordering in §3 of this prompt
3. All main-text tables (Tables 1-8 + falsification summary table)
4. Figure placeholders with descriptive captions
5. Updated bibliography including heterogeneous panel references
6. Appendices A-G
7. Acknowledgements and disclaimer (keep from current paper)

Target length: ~30 pages main text + ~8 pages appendix (single-column,
11pt).

Do NOT reproduce content from the current paper verbatim. This is a
complete rewrite with different framing, structure, and emphasis. The
only preserved elements are numerical results and bibliography entries.

Write in neutral, precise scientific prose. Every claim is accompanied
by its quantitative evidence. The demanding reader should finish the
paper thinking: "The positive result is real, the mechanism is clear,
the scope condition is honest, and the long falsification programme
serves the final architecture rather than being a research diary."