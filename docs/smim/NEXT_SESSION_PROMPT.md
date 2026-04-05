# Paper rewrite based on the Iteration 6.1 Results

Read `docs/smim/ITERATION_6_1_RESULTS.md` first. It contains the full rationale, 
hypotheses and the results.

---

The paper (`smim_paper.tex`) now needs a full Iteration 6.1 integration rewrite. Please revise it as a complete, internally consistent paper, preserving the existing structural-analysis sections but changing the forecasting narrative from “standalone SMIM beats AR(1)” to the correct 6.1 conclusion:

**Uniform-(F) standalone SMIM was mis-specified; repairing (F) helps but standalone still loses to simple baselines; the robust positive result is a two-stage spectral augmentation architecture.**

This rewrite must be careful, numerically exact, and referee-facing. Do not invent results. Use the Iteration 6.1 results exactly as established.

## Core framing — non-negotiable

The revised paper must communicate this nuanced answer clearly:

1. **Diagnostic negative result:** standalone spectral state-space forecasting on raw panel levels fails because the standard (F=0.99I) predict step destroys mode-specific dynamics.
2. **Repair result:** replacing (F) with DMD-informed dynamics materially improves standalone performance, but even the repaired standalone model remains below simple baselines.
3. **Architectural positive result:** the correct use of the spectral machinery is as a **second-stage residual model** on top of pooled persistence, not as a standalone replacement.
4. **Parsimony result:** the essential gain is from **per-mode dynamics in the working reduced coordinates**, not from rich cross-mode propagation.
5. **Complexity result:** extra Kalman/filter complexity beyond the corrected transition does not help.

Do **not** simplify this to “spectral methods were fine all along” or “the failure was purely implementation-specific.” The truth is more precise and more interesting.

## Exact architecture to present

Use this distinction consistently throughout:

* **Recommended default architecture:**
  Stage 1 pooled AR(1)+FE, then residual DMD/Kalman with (F=\mathrm{diag}(\tilde A_r)).
  This is the parsimonious default and should be presented as the recommended pipeline.

* **Maximum-performance architecture:**
  Same two-stage design, but residual-stage (F=\tilde A_r) (full reduced operator).
  This is the best-performing 93-actor variant, but only marginally better than the default and not uniformly best across panels.

Be careful:

* On the **93-actor panel**, full (\tilde A_r) gives the headline max performance (R^2=0.630), while diag((\tilde A_r)) gives (0.619).
* On the **146-firm panel**, the default diag variant is better than the full-(\tilde A_r) variant.
* Therefore the paper must separate **default architecture** from **max-performance variant** and avoid implying that full (\tilde A_r) is universally the preferred deployed model.

## Numbers that must be reflected exactly

### Standalone diagnostic arc (93-actor panel)

* SMIM baseline (F=0.99I): (R^2 = 0.415)
* SMIM + diag((\tilde A)): (R^2 = 0.483)
* SMIM + full (\tilde A): (R^2 = 0.486)
* Per-actor AR(1): (R^2 = 0.594)
* Pooled+FE: (R^2 = 0.591)
* Layer-specific pooled+FE: (R^2 = 0.598)

Interpretation:

* Repairing (F) closes part of the gap, but standalone SMIM remains about 0.11 below AR(1).
* A4 established that the gain is mainly **per-mode self-dynamics in the working reduced coordinates**, not cross-mode propagation.

### Residual-stage ablation ladder (93-actor panel)

* Pooled+FE only: (0.591)
* * residual AR(1): (0.605)
* * residual PCA projection: (0.404)
* * residual PCA+VAR / DFM: (0.577)
* * residual DMD projection: (0.469)
* * residual DMD/Kalman with (F=0.99I): (0.471)
* * residual DMD/Kalman with diag((\tilde A_r)): (0.619)
* * residual DMD/Kalman with full (\tilde A_r): (0.630)

Interpretation:

* This is the key evidence table.
* Projection-only models actively destroy the result.
* Residual DFM does not help.
* Residual AR(1) is a fair but weaker second-stage baseline.
* The gain is specifically from DMD/Kalman with learned residual dynamics.
* (F=0.99I) is catastrophic even on residuals, confirming that transition mis-specification was the bottleneck.

### Strong baselines on the 93-actor panel

* Per-actor AR(1): (0.594)
* Pooled+FE: (0.591), CI ([-0.015, +0.011]), p=0.650
* Layer-specific pooled+FE: (0.598), CI ([-0.009, +0.021]), p=0.349
* DFM (K=8): (0.568), CI ([-0.040, -0.011]), p=0.994
* C1 combined full-(\tilde A_r): (0.630), (\Delta) vs AR(1) (= +0.036), CI ([+0.021, +0.054]), p=0.001, 10/10 wins
* C1 vs layer-specific pooled+FE: (\Delta = +0.032), CI ([+0.022, +0.042]), 10/10 wins

### Portability across panels

Report both the **default diag architecture** and the **max-performance variant where different**:

* **146-firm CapEx/Revenue**

  * AR(1): (0.728)
  * Pooled: (0.745)
  * C1 diag: (0.749)
  * C1 full: (0.745)

* **270-actor multi-ratio**

  * AR(1): (0.728)
  * Pooled: (0.738)
  * C1 diag: (0.745)
  * C1 full: (0.753)

* **93-actor multilayer**

  * AR(1): (0.594)
  * Pooled: (0.591)
  * C1 diag: (0.619)
  * C1 full: (0.630)

Use the established CI/win-count results from the robustness table. Phrase this as:

* augmentation improves over AR(1) across all three panels,
* gains are largest on the most heterogeneous panel,
* but the preferred deployed architecture is the parsimonious default unless the full variant’s improvement is clearly worth the extra parameters.

### diag((\tilde A_r)) vs full (\tilde A_r) on 93-actor residual stage

* Mean (\Delta = +0.011)
* t(9)=2.47
* p=0.036
* CI ([+0.003, +0.019])
* full wins 7/10

Interpretation:

* full (\tilde A_r) is a marginally significant max-performance variant,
* diag((\tilde A_r)) remains the recommended default for parsimony and credibility,
* the paper should present both honestly.

### D1 / D2 / A5 nulls

* Spectralising (Q) and (R): no gain
* State persistence across basis updates: no gain
* Kim switching: dropped / not warranted

Interpretation:

* the bottleneck was in (F), not the rest of the filter,
* once the transition is corrected, extra spectral filter complexity does not help.

### Economic validation

Use both pooled and C1 gaps, with and without actor FE:

* Pooled+FE gaps:

  * No FE: (\beta=-0.589), t=-27.8, (R^2=0.173)
  * Actor FE: (\beta=-0.630), t=-10.3, (R^2=0.194)

* C1 gaps:

  * No FE: (\beta=-0.530), t=-23.1, (R^2=0.127)
  * Actor FE: (\beta=-0.566), t=-12.0, (R^2=0.142)

Gap-strength decomposition:

* pooled gap (\sigma = 0.189), (\rho = 0.139)
* C1 gap (\sigma = 0.179), (\rho = 0.054)

Interpretation:

* both gap types survive actor FE with negative coefficients, so both have within-actor as well as cross-sectional content
* C1 gaps are **less** predictable because the better model has absorbed some systematic structure into the forecast
* do **not** frame weaker C1 gap predictability as a worse benchmark
* explicitly state that stronger gap predictability can reflect a weaker model leaving more structure in the residual

Also: remove any outdated claim in the current paper that the gap result flips sign under actor FE. That is no longer true in the revised architecture.

## Revision tasks — section by section

### 1) Title

Propose **three** title options at the top of the deliverable, with one recommended. At least one option should foreground augmentation.

Examples of the right direction:

* “Spectral Augmentation of Panel Forecasts with Dynamic Mode Decomposition”
* “Dynamic Mode Decomposition as Residual Spectral Augmentation for Cross-Sectional Investment Forecasts”
* “Cross-Sectional Spectral Augmentation: Residual Dynamic Mode Decomposition for Investment Panels”

Do not hard-code one title without offering options.

### 2) Abstract

Rewrite the abstract completely.

It must say, in this order:

1. We study cross-sectional investment forecasting with spectral state-space methods.
2. Standalone spectral prediction on raw levels is informative but fails as a replacement for simple baselines because uniform transition regularisation destroys mode-specific dynamics.
3. A corrected transition partially repairs standalone SMIM but remains below AR(1)/pooled baselines.
4. The recommended solution is a two-stage architecture: pooled AR(1)+FE for shared persistence, then residual DMD/Kalman for cross-sectional rotational structure.
5. The augmentation result on the 93-actor panel is the headline positive finding.
6. The result generalises across all three panels.
7. The ablation ladder shows the gain is specific to DMD/Kalman with learned residual dynamics, not generic second-stage models.
8. Structural spectral findings remain valid.

Be accurate about default vs max-performance transition variants. Do not imply that the diagonal and full variants are the same thing.

### 3) Introduction

Rewrite the preview-of-results list so the new headline is **spectral augmentation**, not standalone SMIM.

Include explicit bullets for:

* standalone diagnostic negative result
* transition repair result
* augmentation positive result
* residual-stage ablation ladder
* robustness across panels
* structural findings that remain intact
* economic-content interpretation of the new gaps

Also update the Introduction’s framing sentence so the paper no longer claims that spectral structure is “descriptive rather than forecastable” in general. The revised claim is:

* on raw levels, spectral structure is largely descriptive and standalone spectral prediction is inferior;
* after removing shared persistence, there remains forecastable cross-sectional residual structure that spectral augmentation can exploit.

### 4) Contribution / scope / caveats sections

Revise the contribution section accordingly:

* Diagnostic contribution
* Methodological augmentation contribution
* Empirical cross-panel contribution

Update scope/caveats:

* only three panels
* US data / sample window caveat
* gains are modest but stable
* full-(\tilde A_r) advantage over diag((\tilde A_r)) is statistically marginal and not uniform across panels
* the economic-validation regressions are predictive associations, not causal identification

### 5) Framework section

Add a new subsection for the **two-stage spectral augmentation architecture** and make it the recommended pipeline.

It should include:

* Stage 1 pooled AR(1)+FE forecast
* residual construction
* Stage 2 DMD on residuals
* residual Kalman filter with DMD-informed transition
* combined forecast equation

Clarify that:

* diag((\tilde A_r)) is the default residual transition,
* full (\tilde A_r) is the max-performance residual transition,
* (F=0.99I) is now diagnostic context, not the recommended transition.

Update Algorithm 1 so the recommended algorithm is the two-stage pipeline. If useful, keep the standalone SMIM algorithm as a diagnostic/legacy algorithm or move it to an appendix or subsection.

### 6) Results section

Reorganise to include the following new/rewritten subsections:

#### 6a) Standalone transition diagnostic

Use Table A.
Purpose: explain why standalone SMIM fails and what transition repair recovers.
Make the A4 finding explicit:

* the gain is mainly per-mode dynamics in the working coordinates,
* not evidence of cross-mode propagation.

#### 6b) Residual-stage ablation ladder

Use Table B prominently.
This should be the centrepiece evidence table.

#### 6c) Strong baseline comparison on the 93-actor panel

Include per-actor AR(1), pooled+FE, layer-specific pooled+FE, DFM, and combined augmentation model.

#### 6d) Cross-panel robustness

Use Table C.
Be careful to distinguish default diag architecture from max-performance variant.

#### 6e) Structural spectral analysis

Preserve the existing modal (R^2), basis rotation, stable dimensionality, and structural decomposition results.
These stay, but they are now complementary evidence rather than the forecasting headline.

### 7) Economic validation section

Revise this section so it covers:

* pooled gaps and C1 gaps
* no-FE and actor-FE results
* gap-strength decomposition
* signal absorption interpretation

Important:
state explicitly that the C1 gaps are less persistent and less predictable because the better model has moved systematic structure into the forecast.

Do **not** recycle the old “sign flip under actor FE” discussion.

### 8) Discussion

Create or revise discussion subsections along these lines:

#### What fails

* standalone spectral replacement fails
* forecast-optimised RRR basis loses to DMD in the short-panel (N \gg T) regime
* projection-only and residual DFM variants fail
* extra filter complexity does not help
* propagation claims are not supported by A4 / residual-mode evidence

#### What works

* two-stage augmentation
* learned per-mode residual dynamics
* cross-sectional rotation patterns among firms after shared persistence is removed
* gains scale with heterogeneity, but are present across all panels

#### What not to claim

* do not claim macro→firm propagation from these results
* do not claim that full (\tilde A_r) is universally necessary
* do not claim that stronger gap predictability means a better benchmark

### 9) Conclusion

Rewrite around three layers:

1. diagnostic negative result on standalone SMIM
2. methodological positive result on spectral augmentation
3. empirical cross-panel robustness result

The conclusion should end with the correct broad claim:
spectral methods earn their complexity as **residual augmenters of simple panel baselines**, not as standalone replacements.

### 10) Structural-analysis sections to preserve

Do **not** delete the existing structural findings:

* modal reconstruction
* basis rotation
* stable dimensionality
* dual regularisation insight
* related structural diagnostics

Preserve them, but adjust nearby prose so they support the new architecture rather than the old standalone-forecasting claim.

## Tables and figures

### Must-add / must-update tables

Create or update these cleanly in LaTeX:

* **Table A — Standalone SMIM diagnostic arc**
* **Table B — Residual-stage ablation ladder** (prominent placement)
* **Table C — Cross-panel robustness**
* **Updated economic validation table** with pooled vs C1, no FE vs actor FE
* **Updated strong-baselines table** on the 93-actor panel

### Figures

Do the following if feasible from existing source/assets without blocking the paper rewrite:

1. **Update the conceptual pipeline figure** so it shows the two-stage architecture.
2. **Add one compact new figure** if it can be generated cleanly from existing results, choosing the most valuable of:

   * residual-stage ablation ladder figure,
   * cross-panel augmentation gains figure,
   * residual-mode loading / sector-rotation illustration.

If figure generation from raw results is cumbersome, prioritise the **updated pipeline figure** and keep the rest as tables. Do not delay the rewrite for optional figures.

## Consistency scrub — very important

Before finalising, do a full scrub of the manuscript and remove or update every stale claim that is no longer true, including but not limited to:

* any headline statement that standalone SMIM beats AR(1) or pooled baselines
* any blanket statement that spectral structure is descriptive rather than forecastable
* any claim that the economic-validation sign flips under actor FE
* any implication that propagation/cross-mode coupling is the key mechanism
* any implication that extra Kalman complexity helped
* any outdated table captions or discussion text tied to the old forecasting narrative

Also ensure:

* all section cross-references compile correctly
* table/figure numbering is updated
* the abstract, intro, results, discussion, and conclusion all tell the same story
* default vs max-performance architecture is stated consistently
* the portability table does not accidentally overstate full-(\tilde A_r) on the 146-firm panel

## Deliverables

Please produce all of the following:

1. **Revised complete `smim_paper.tex`**
2. **Compiled revised PDF**
3. **Any updated or newly created figure files/source files**
4. **A short change log** listing:

   * major claim changes
   * stale claims removed
   * new tables/figures added
   * any places where space constraints forced compression

At the top of your response, first give:

* the three proposed title options,
* the recommended one,
* and a 5–8 sentence summary of the paper’s revised story.

Then provide the revised files.
