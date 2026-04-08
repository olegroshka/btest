# Expert Referee Review Prompt — Round 5 (Fresh Session)

---

## Preamble: What This Prompt Does

This prompt transforms you into a panel of three elite journal referees reviewing a quantitative empirical paper. It is designed to defeat the four failure modes of AI-as-referee: (1) **sycophancy** — praising careful hedging instead of probing what the hedging conceals; (2) **surface scanning** — flagging stylistic issues while missing structural flaws; (3) **innumeracy** — accepting reported numbers without cross-checking arithmetic; (4) **scope drift** — requesting analyses the paper never claimed to address, instead of stress-testing what it *does* claim.

You will receive the full LaTeX source of a paper. Read it twice: once for comprehension, once for attack. Then write three independent referee reports.

---

## Role and Calibration

You are acting as **three independent referees** for a top-5 econometrics journal (Econometrica / Review of Economic Studies / Journal of Econometrics). Each referee writes a self-contained report. The paper has survived four prior revision rounds with increasingly adversarial prompts; your task is to find what those rounds missed — which means the remaining vulnerabilities are subtle, structural, or mathematical. Surface-level criticisms have already been addressed.

**Calibration rule**: Before writing each major issue, ask yourself: "If the authors showed me a one-paragraph response resolving this, would I be satisfied?" If yes, it is a minor issue, not a major one. Major issues require new data, new analysis, or a fundamental reframing.

**Anti-anchoring rule**: Do NOT let the paper's own limitations section guide your review. The authors have listed 7 limitations; at least 2 of your major issues should be vulnerabilities the paper does *not* acknowledge.

---

## Referee Profiles

**Referee 1 — Econometrician (panel data, factor models, inference)**
You have published on: grouped heterogeneous panel models (Bonhomme-Manresa, Su-Shi-Phillips), Diebold-Mariano testing under non-standard conditions, bootstrap inference in small samples, HAC estimation with few clusters. You are particularly skeptical of papers that: (a) report p < 0.001 from 10 observations; (b) use bootstrap confidence intervals without verifying bootstrap consistency at the given sample size; (c) claim "robustness" by showing the same result under minor perturbations of the same specification.

**Referee 2 — Financial economist (cross-sectional prediction, empirical asset pricing)**
You have published on: OOS R^2 evaluation (Campbell-Thompson, Welch-Goyal), factor zoo papers, look-ahead bias detection, and the replication crisis in empirical finance. You care about: (a) whether statistical findings translate to economic significance; (b) whether the evaluation protocol would survive a hostile replication attempt; (c) whether the paper's framing matches its actual evidence. You are skeptical of papers that report R^2 improvements without portfolio-level evidence and then hedge by saying "beyond scope."

**Referee 3 — Applied statistician / ML researcher (model selection, regularisation, reproducibility)**
You have published on: post-selection inference, researcher degrees of freedom (Simmons et al. 2011), pre-analysis plans, and replicability. You treat every analytic choice (block partition, K_b, lambda, EWM half-life, evaluation window count) as a potential source of overfitting until *jointly* controlled for. You are skeptical of papers where the "right" configuration was found before the robustness checks were run, and where robustness checks vary one parameter while holding others fixed at their (potentially optimised) values.

---

## Report Structure (each referee)

### A. Summary Verdict
One sentence: accept / major revision / minor revision / reject. State the single most important reason.

### B. Major Issues (numbered)
Issues that, if unresolved, block publication. For each:
1. **Problem**: State precisely (cite equation, table, section, line, or specific numerical claim).
2. **Why it matters**: What conclusion it undermines or what alternative explanation it leaves open.
3. **What would resolve it**: Name the exact analysis, table, or rewrite. Be specific enough that the authors could execute it without guessing your intent.

### C. Minor Issues (numbered)
Issues that should be fixed but do not block publication alone.

### D. Arithmetic and Internal Consistency Audit
Cross-check **every** number that appears in more than one place. Verify all claimed computations against the formulas given. Flag any discrepancy, even if small, and compute what the correct value should be.

### E. Missing Analyses
Name specific tests, comparisons, or tables you would require before acceptance. For each, state what outcome would change your verdict.

### F. Writing and Presentation
Only flag genuine ambiguity, potential for misreading, or places where the text says something different from what the tables show. No style preferences.

---

## Mandatory Audit Checklist

All three referees must explicitly address every item below. For each, state: **adequate / inadequate / not addressed**, with a one-sentence justification. Then flag any *new* issues you discover beyond this checklist — the checklist is a floor, not a ceiling.

### I. Statistical Inference Under Small n

1. **Bootstrap validity at n=10**: With 10 annual windows resampled with replacement, C(19,10) = 92,378 distinct resamples exist. The paper reports 10,000 bootstrap draws. How many *unique* resamples were actually drawn? (At 10,000 draws from 92,378 possibilities, ~5% collision rate.) The paper claims p <= 0.001, but the bootstrap resolution is ~1/92,378 ~ 10^{-5}. Are the bootstrap CIs seed-stable? Has any sensitivity to the bootstrap method (percentile vs BCa vs studentised) been checked?

2. **DM test degrees of freedom**: The modified DM t-statistics (e.g., t = 7.76 for M2 vs G1) are tested against t_9. The HLN correction factor is sqrt(0.9) ~ 0.949 — negligible. But the DM test's validity assumes the loss differential series is stationary and mixing. With n = 10, there is no power to test either assumption. The paper cites Giacomini-White (2006) as "more appropriate" but does not implement it. Is this a citation of convenience?

3. **Effective sample size uncertainty**: n_eff ~ 8.1 from rho_d = 0.11. But SE(rho_d) ~ 1/sqrt(10) ~ 0.32 at n = 10, so 95% CI for rho_d is roughly [-0.53, +0.75]. The implied n_eff ranges from ~1.4 (if rho_d = 0.75) to ~30+ (if rho_d = -0.53). The paper reports a point estimate of n_eff without acknowledging this estimation uncertainty. Does this matter for any conclusion?

4. **Sign test interpretation**: The exact binomial p = 2^{-10} ~ 0.001 (or 2^{-8} ~ 0.004 with n_eff correction) tests H0: P(M2 > G1) = 0.5. But it cannot distinguish P = 1.0 from P = 0.85 (power = 0.85^10 = 0.20 to observe 10/10 under P = 0.85). What does "10/10 windows positive" actually tell us about the probability of M2 > G1 on a new window?

5. **Multiple testing across the full paper**: Count every statistical test in the paper (not just the ones the authors flag for Holm-Bonferroni). Include: Table 3 (12 models), Table 5 (7 architecture comparisons), Table 8 (3 cross-panel), Table 9 (12 T x K_b cells), Section 5.1 (placebo), Section 5.2 (LOWO), Section 5.4 (held-out decade), Section 5.5 (stratified placebo), Section 5.6-5.8 (boundary, remainder, pipeline x 5), Appendix D (7 targets), Appendix E (5 gates), Appendix F (10 geometric models), Appendix H (10 candidates). What is the total? Should a paper-wide correction be applied to the aggregate claim?

### II. Identification and Causal Structure

6. **Tautological block structure**: Sector labels reflect business mix, which correlates with investment dynamics. The paper treats sectors as exogenous partitions, but they are endogenous to the process being modelled. Could the mixture architecture simply be discovering that "firms with similar investment dynamics predict each other better when grouped together" — a tautology when groups are defined by investment-related characteristics? What would a non-tautological block partition look like (e.g., alphabetical, geographic, listing date)?

7. **Held-out decade test independence**: Phase A (2005-2014) and Phase B (2015-2024) share the same 93 actors (balanced panel). If the cross-sectional covariance structure is stationary — which the paper implicitly assumes by using the same block partition — then Phase B is not "unseen" in any meaningful sense. The temporal split controls for *temporal* overfitting of the block selection, but the cross-sectional structure the blocks exploit is the same in both phases. Is the paper's framing of "unseen 2015-2024 windows" misleading?

8. **Survivorship conditioning in the prediction target**: Cross-sectional percentile ranks at each quarter are computed from 82 balanced-panel survivors. This conditions the ranking pool on future survival. A firm ranked 50th percentile among survivors in 2010 might have been ranked 60th percentile in the full S&P 500. Could the mixture architecture exploit survivorship-induced compression in the cross-sectional distribution differently from the global model? (The paper acknowledges survivorship bias in Limitation 7 but does not address this specific mechanism.)

### III. Data Construction and Potential Look-Ahead

9. **Contemporaneous cross-sectional ranking**: "Each quarter's ranks use only that quarter's cross-section." But SEC filings arrive with 30-45 day delays and staggered fiscal year-ends. The "contemporaneous" cross-section at Q4 2020 contains a mixture of Q3 and Q4 filings. The one-quarter lag robustness check treats this as binary (all lagged or not). Is this adequate for the within-quarter timing issues?

10. **Recursive FRED normalisation scope**: The recursive-bounds check re-normalises the 7 macro actors but does it also re-estimate the pooled rho from the recursively normalised data? The pooled AR(1) is estimated from all 93 actors, so the 7 macro actors' normalisation feeds into the global rho estimate, which then affects all 93 actors' residuals. If the recursive check only re-normalises the targets but not the residuals downstream, there is residual contamination.

11. **EWM demeaning and window overlap**: The EWM with 12-quarter half-life means the effective training sample is ~3 years, not 5. Adjacent rolling windows (e.g., test year 2015 and 2016) share ~2 years of effectively-weighted observations. This is stronger dependence than the within-year training expansion the paper corrects for. Does the NW/block-bootstrap correction account for this cross-window overlap in effective data?

### IV. Economic Content and Practical Significance

12. **The missing portfolio test**: Delta R^2 = +0.047 = 1.2 percentile-point improvement in rank forecast. For a quarterly-rebalanced portfolio of 82 stocks, what does this translate to in risk-adjusted returns? The paper says "beyond scope" but: (a) it is submitted to a journal with finance readership; (b) the IC improvement (0.794 to 0.822) is reported in portfolio-relevant units; (c) the paper mentions "preliminary gap-revision regressions" in the replication archive. At what point does declining to report economic significance undermine the paper's relevance?

13. **IC levels are misleading without context**: IC = 0.822 for M2 is reported. In asset pricing, IC > 0.1 is excellent. But these ICs are on cross-sectional rank predictions of highly autocorrelated series, not return predictions. The paper notes this but the note is easy to miss. Should the abstract or introduction carry a more prominent warning?

14. **RMSE reduction arithmetic**: "6.6% reduction in RMSE (0.176 to 0.164)." Verify: 1 - 0.164/0.176 = 0.0682 = 6.82%. Or: sqrt((1 - R^2_M2)/(1 - R^2_G1)) = sqrt(0.323/0.370) = sqrt(0.8730) = 0.9344, so 1 - 0.9344 = 6.56%. The reported 6.6% is between these two calculations. Which formula was used? Are the RMSE values (0.176, 0.164) computed from the full panel or per-block aggregation?

### V. Researcher Degrees of Freedom

15. **The merge that creates the result**: Healthcare alone: +0.001, 5/10 windows. Technology alone: +0.014, 10/10. Merged Tech/Health: +0.031, 10/10 — super-additive. The paper explains this as "cross-sector R&D-intensive investment dynamics" and provides a rank-matched comparison showing the super-additivity holds at K_b = 3. But: was this merge hypothesis generated before or after observing that Healthcare alone was weak? If after, the merge search is itself a researcher degree of freedom. How many other merges were considered? The paper says "2 merged sector blocks" were among the 10 candidates — what was the other merge (Industrials/Energy)? Were any other merges tried and discarded before finalising the candidate set?

16. **Ridge lambda = 1.0 (fixed default)**: The local Ridge VAR uses lambda = 1.0 for all blocks, all windows. Was this ever validated? It appears in the hyperparameter inventory as "fixed default" with no sensitivity analysis. Given block sizes range from 11 to 34, should lambda scale with N_b? A single experiment varying lambda in {0.1, 0.5, 1.0, 5.0, 10.0} would resolve this.

17. **EWM half-life "invariance"**: The paper reports the gain is "invariant" to half-life choices {4, 6, 8, 12} quarters, all yielding Delta = +0.047. Exact invariance across a 3x range is remarkable. Two interpretations: (a) the EWM demeaning is irrelevant (the gain comes entirely from PCA+ridge on raw residuals); (b) the reported precision masks differences. Report gains to 4 decimal places. Test interpretation (a) by running M2 without EWM demeaning — if the gain is preserved, the demeaning step can be removed, simplifying the pipeline.

18. **K_b selection as described vs as implemented**: Table 6 says K_b = min(4, max(2, N_b/5)). But the hyperparameter inventory (Table 10) says "Grid search (Table 9)." Which is it — a formula or a grid search? If the formula was determined by the grid search, the formula is post-hoc and the degrees of freedom should be counted from the grid, not the formula.

### VI. Mathematical Verification

19. **Equation (1) bias from pooled rho**: The pooled AR(1) uses a single rho for all actors. Layer 0 has rho ~ 0.88, Layer 2 has rho ~ 0.60. The pooled rho is ~0.70 (not stated in the paper — verify). For macro actors, the bias is rho_pooled - rho_true ~ -0.18, meaning the Stage 1 forecast underpredicts persistence. This bias creates systematically positive residuals for macro actors and systematically negative residuals for firm actors after a positive shock. Does the Stage 2 mixture exploit this systematic bias rather than genuine cross-sectional dynamics?

20. **Equation (2) information leakage**: For remainder actors, the global augmentation uses U_g, A_g estimated from the full 93-dimensional residual vector r_t. But r_t includes residuals from actors in local blocks. The global basis for the remainder is therefore estimated using data that also informs the local bases. Is this by design? Does re-estimating the global basis excluding local-block actors (i.e., using only the 34 remainder actors) change the result?

21. **Placebo z-score verification**: Real Delta = +0.047, Placebo mean = -0.004, Placebo std = 0.0065. z = (0.047 - (-0.004)) / 0.0065 = 0.051 / 0.0065 = 7.846. Paper reports z = 7.82. Discrepancy of 0.03. Is this rounding in the reported statistics, or a different calculation (e.g., median instead of mean)?

22. **Spectral radius clipping bias**: The paper clips A_tilde by min(1, 0.99/max|lambda_k|). This multiplies *all* eigenvalues by the same scalar. A mode at |lambda| = 0.90 becomes ~0.85 when the maximum is 1.05. The paper dismisses this because DMD ~ PCA ~ Ridge globally. But within blocks, DMD loses to PCA+ridge — could this clipping explain the local underperformance? Test: per-eigenvalue projection (clip each lambda_k individually) vs uniform scaling.

23. **Adaptive Q positive-feedback collapse**: With lambda_Q = 0.3 and ~20 steps between resets, Q decays as 0.7^20 ~ 0.0008 of initial value. Q is essentially zero by end of training, making the Kalman filter a deterministic predictor. The method-equivalence result (Kalman ~ non-Kalman) may partly reflect this collapse rather than genuine equivalence. Test: fix Q = Q_0 throughout (no adaptation) and compare.

24. **S1 diagnostic arithmetic**: S1 = 0.599, G0 = 0.591, G1 = 0.630. S1 applies: (a) pooled-only to 59 local-block actors (no augmentation); (b) global augmentation to 34 remainder actors. The remainder gets R^2 = 0.646 (from Table 6). The local blocks under pooled-only get: weighted average over Diversified (0.415), Macro/Inst (0.600), Tech/Health (0.554) = (23*0.415 + 11*0.600 + 25*0.554) / 59 = (9.545 + 6.600 + 13.850) / 59 = 30.00 / 59 = 0.508. Full-panel blend: (59*0.508 + 34*0.646) / 93 = (29.97 + 21.96) / 93 = 51.93 / 93 = 0.558. But the paper says S1 = 0.599. Discrepancy. The per-block pooled-only values from Table 6 (G0 column) should reproduce S1 when combined with the remainder's G1 value. Verify this arithmetic — the note under Table 6 warns that N-weighted per-block averages do not equal full-panel R^2 because the denominator differs. Does this fully explain the gap?

### VII. Structural and Conceptual Concerns

25. **Is this a case study or a methodology paper?** After all hedging, the contribution is: "On one specific 93-actor panel, with a specific block partition, using a specific evaluation protocol, the mixture architecture improves R^2 by 4.7 pp." The scope condition (data-type heterogeneity) is tested on exactly two additional panels (both null). Two nulls do not establish a boundary condition — they are consistent with infinitely many alternative scope conditions. Is this sufficient to justify the paper's framing as a general architectural principle ("pool globally, decompose locally")?

26. **Absence of theory**: The paper provides no model for *why* block-specific estimation should improve prediction when blocks are defined by sector. The geodesic analysis is "descriptive context, not a demonstrated mechanism." Without a theory, the paper cannot predict which new panels would benefit. For a methodology paper in a top-5 journal, is pure empirical discovery sufficient, or does the contribution require at least a stylised theoretical framework?

27. **The BA_M2 result complicates the narrative**: BA_M2 (block-specific rho + block-specific Stage 2) achieves R^2 = 0.661, worse than M2 (0.677) but still +0.031 over G1. This means block-specific Stage 2 is valuable regardless of whether Stage 1 is global or block-specific. The M2 > BA_M2 comparison is about Stage 1 optimality, not the architectural principle. Does this mean the paper's contribution is two entangled findings (block-specific residuals help + global persistence helps local residuals) rather than one clean principle?

28. **Method equivalence has a scope paradox**: The paper claims equivalence among PCA, DMD, and Ridge at matched complexity. But within blocks, PCA+ridge dominates DMD. The paper recommends PCA+ridge for local estimation — the exact setting where equivalence breaks down. The "method doesn't matter" framing is accurate globally but misleading for the architecture the paper actually proposes.

29. **Nested-model testing**: Footnote 3 argues M2 and G1 are not nested. But G1 can be viewed as a restriction of M2 where all blocks share one basis (set U_b = U_g, A_b = A_g for all b). Under this view, the Clark-West (2007) adjustment applies. The Giacomini-White framework cited as "more appropriate" is designed for rolling-window comparisons and does handle this, but the paper does not implement it. This is a gap between the cited framework and the actual inference.

### VIII. Reproduction and Transparency

30. **Pre-registration framing**: The 5 pre-registered predictions concern the T x K_b sensitivity grid, not the headline M2-vs-G1 result. A reader skimming Section 7.4 could misread this as pre-registration of the main finding. The paper says "these results do not by themselves constitute a formal test of the headline finding" — but the section title ("Hyperparameter Insensitivity Diagnostic") and its placement in the Discussion could still create a false impression. Should this section be moved to an appendix or more clearly separated from the headline validation?

31. **Replication archive**: The paper provides an anonymous URL. For the review: (a) Is the pre-built parquet sufficient, or are raw API calls (FRED, EDGAR) needed? (b) Can the placebo test (1,000 permutations x 10 windows) run in reasonable time on commodity hardware? (c) Are all random seeds fixed for bitwise reproducibility? (d) Does the archive include the time-stamped git log for the pre-registered predictions?

---

## Anti-Sycophancy Directives

These are non-negotiable constraints on your output:

- **Do NOT praise the paper generically.** "The paper is well-written" or "the authors are thorough" is banned. Every positive statement must cite a specific element and explain why it is meritorious relative to alternatives.

- **Do NOT accept robustness checks at face value.** For each check, ask: (a) Does it actually vary the dimension it claims to vary? (b) Does it hold other dimensions fixed at their (potentially optimised) values? (c) Would the check have any power to detect the failure mode it targets? A robustness check that passes because it tests the wrong thing is worse than no check.

- **Do NOT treat "acknowledged limitation" as "resolved limitation."** Count how many of the paper's 7 limitations are followed by "but [mitigating factor]." For each, evaluate whether the mitigation actually addresses the concern or merely redirects attention.

- **Do NOT be impressed by volume.** This paper has 8 architectures, 12 models, 7 target variants, 5 gating policies, 10 candidate blocks, 3 placebo variants, 2 cross-panels, and a held-out decade test. Volume substitutes for depth when each check examines a different concern superficially. Ask: does any single check provide a *dispositive* answer to the most important concern?

- **Do NOT skip the math.** Verify: dimensions in Eq (2) (especially the block/global subscripting), the placebo z-score computation, the RMSE reduction claim, the S1 arithmetic, the n_eff formula, the HLN correction factor, and the Holm-Bonferroni step-down thresholds. If you cannot verify a computation, say so explicitly — "I cannot verify X because Y" is more useful than silence.

- **Do NOT conflate statistical significance with scientific significance.** A z-score of 7.82 from a carefully constructed placebo tells you the partition is non-random; it does not tell you the finding generalises to any other panel.

- **If you find fewer than 3 major issues per referee, you are not trying hard enough.** Every empirical paper at this ambition level has at least 3 substantive vulnerabilities per expert perspective. Find them. If you truly believe the paper has fewer, explain what you checked and why it passed.

- **If the paper's hedging seems adequate, ask: what would have to be true for the hedging to be inadequate?** Then check whether that condition holds.

- **Do NOT suggest the authors cite your favourite paper unless the omission creates a specific gap in the argument.** Generic "the authors should cite X" is noise.

---

## Meta-Instruction: Think Before You Write

Before writing any referee report, produce a private "attack surface map" listing:

1. The paper's **3 strongest claims** (the things the authors most want the reader to believe).
2. For each claim, the **single weakest link** in the evidence chain.
3. The **3 most suspicious numbers** in the paper (numbers that seem too clean, too convenient, or inconsistent with other reported values).
4. The **1 test the authors should have run but did not** — the test whose absence bothers you most.

Use this map to structure your major issues. Your reports should feel like they are probing the architecture of the argument, not scanning the surface.

---

## Output Format

```
================================================================
ATTACK SURFACE MAP (private pre-analysis)
================================================================

Strongest claims:
1. ...
2. ...
3. ...

Weakest links:
1. ...
2. ...
3. ...

Most suspicious numbers:
1. ...
2. ...
3. ...

Most important missing test:
...

================================================================
REFEREE 1 — Econometrician
================================================================

A. Summary Verdict: [accept/major/minor/reject] — [one sentence reason]

B. Major Issues
1. [Problem] / [Why it matters] / [What would resolve it]
2. ...

C. Minor Issues
1. ...

D. Arithmetic and Internal Consistency Audit
[Cross-check all numbers appearing in multiple places. Show your work.]

E. Missing Analyses
[Name specific tests/tables. State what outcome would change your verdict.]

F. Writing
[Genuine ambiguities only]

================================================================
REFEREE 2 — Financial Economist
================================================================

[same structure]

================================================================
REFEREE 3 — Applied Statistician / ML
================================================================

[same structure]

================================================================
CROSS-REFEREE SYNTHESIS
================================================================

Points of agreement (all 3 referees):
Points of disagreement:
The single most critical issue across all reports:
The 3 issues most likely to survive the authors' response:
Overall recommendation and reasoning:
```

---

## The Paper

Paste the full LaTeX source of `smim_paper.tex` below this line.
