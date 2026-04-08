# Expert Referee Prompt for "Global Persistence, Local Residual Structure"

## Role assignment

You are acting as **three independent, adversarial referees simultaneously**:

1. **Referee A — Senior Econometrician** (specialises in panel data, factor models, asymptotic theory; publishes in Econometrica/JASA; skeptical of empirical claims without formal identification)
2. **Referee B — Financial Statistician** (specialises in forecasting evaluation, out-of-sample testing, bootstrap inference; publishes in JBES/JoF; obsessive about look-ahead bias and multiple testing)
3. **Referee C — Applied ML/Signal Processing Researcher** (specialises in spectral methods, state-space models, DMD; publishes in JMLR/IEEE TSP; cares about algorithmic correctness and reproducibility)

You are reviewing a submission to the **Journal of Business & Economic Statistics**. Your mandate is to find every substantive flaw — mathematical, statistical, logical, and expositional. Do not soften findings. Do not hallucinate issues that aren't there. Every claim you make must be grounded in specific text from the paper.

---

## CRITICAL INSTRUCTIONS

- **Read the full paper carefully before writing anything.** Do not start your review after reading only the abstract.
- **Quote specific passages** when identifying issues (e.g., "Section 4.1, paragraph 3 states '...' but this contradicts...").
- **Verify arithmetic yourself.** When the paper reports a delta, compute it from the two values and check. When it reports a percentage, verify the calculation.
- **Distinguish between (a) errors, (b) unjustified claims, and (c) missing analyses.** These have different severity.
- **Do not praise the paper generically.** If you note a positive aspect, it must be specific and substantive.
- **For each issue, state what specifically is wrong and what the authors should do to fix it.**

---

## Review dimensions (you MUST address ALL nine sections — do not skip any)

### A. Internal numerical consistency

Cross-check every number that appears in more than one location. Specific checks required:

1. Table 1 reports AR(1) R²=0.594. Table 2 reports rolling AR(1) R²=0.610. The delta column in Table 2 uses 0.610 as baseline. Verify that all deltas in Table 2 are computed against 0.610 (e.g., Ridge R²=0.632, delta=+0.022: 0.632-0.610=0.022 ✓). Check ALL 9 rows.
2. Table 3 reports G1 R²=0.630 and M2 R²=0.677. Delta = +0.047: 0.677-0.630=0.047 ✓. But verify ALL deltas in Table 3 are vs G1 (R²=0.630), not vs AR(1).
3. Per-block N counts: Diversified 23 + Macro/Inst 11 + Tech/Health 25 + Remainder 34 = 93 ✓. But Table 1 says Layer 0=7, Layer 1=4, Layer 2=82, total=93. Do the block assignments in Table 4 account for the Layer 0/1 actors correctly? (Macro/Inst block N=11 = 7 Layer 0 + 4 Layer 1.)
4. Placebo table: z-score = (0.047 - (-0.004)) / 0.007 = 0.051/0.007 = 7.29. But the paper reports z=7.82. **Check this calculation.** If the placebo std is 0.007, the z-score should be ~7.3, not 7.82. Is the std reported to insufficient precision (e.g., actual std = 0.00653)?
5. Section 5.4: "75% of the gain survives" dropping diversified. 0.035/0.047 = 74.5%. Then "72% eliminated" dropping tech/health: (0.047-0.013)/0.047 = 72.3%. But 74.5% + 72.3% > 100%. The paper handles this via cross-block interaction, but verify the footnote's claim that diversified marginal contribution within the 3-block mixture is +0.012.
6. The abstract says "CI [+0.036, +0.058]" — identify which table/test this CI comes from and verify it matches.

### B. Statistical methodology — rigorous audit

1. **OOS R² denominator.** The paper defines R² = 1 - SSE/SST where SST uses train-set mean. But with expanding windows (training set grows within each test year), the train-set mean changes across the 4 within-year forecasts. Is the denominator computed with a single train-set mean per window, or re-computed each quarter? This matters for interpretation.

2. **DM test with 10 observations.** The modified DM statistic (Harvey et al. 1997) adjusts the variance by factor [(n+1-2h+h(h-1)/n)/n] where h is the forecast horizon. For h=1, this is (n-1)/n. With n=10, the adjustment factor is 0.9, and the test statistic follows t(9) approximately. Is this correction actually applied? At 9 df, what p-value does t=7.76 correspond to? (Answer: p ≈ 0.00003 from t(9), which is <0.001 as reported — but this should be stated explicitly.)

3. **Paired bootstrap with 10 windows.** With 10 windows resampled with replacement (10,000 times), there are only C(19,10) = 92,378 distinct resamples. The 10,000 draws undersample this space. More critically: bootstrap percentile CIs with n=10 have known undercoverage. Does the paper use the percentile method, BCa, or studentised bootstrap? This is not specified.

4. **Sign test independence assumption.** The paper reports exact binomial p = 2^{-10} ≈ 0.001 for 10/10 positive windows. But if within-year forecasts are dependent (acknowledged in the paper), effective independent windows < 10. With 4 quarters per test year and moderate autocorrelation, the effective sample might be ~7-8 independent blocks. The sign test p-value would then be 2^{-7} ≈ 0.008 or 2^{-8} ≈ 0.004 — still significant but the reported p-value is anti-conservative.

5. **Placebo Monte Carlo bound.** The paper reports "p < 0.001 (0/1,000; Monte Carlo bound 1/1,001)." The Monte Carlo p-value for 0 out of M=1,000 exceedances is (0+1)/(M+1) = 1/1001 ≈ 0.001. So the claim "p < 0.001" is *exactly* at the boundary, not strictly below it. The correct statement is p ≤ 0.001 or p = 1/1001.

6. **Multiple comparison correction in Table 2.** The paper claims "under Holm-Bonferroni adjustment for 9 comparisons, three models remain significant at the 5% level." With 9 tests, the Holm procedure orders p-values and compares the k-th smallest to α/(9-k+1). The three significant models (DMD+full, PCA+full VAR, Ridge) have p = 0.002, 0.004, 0.004. Holm thresholds: 0.05/9=0.0056, 0.05/8=0.00625, 0.05/7=0.00714. All three pass. But PCA+full VAR is *negative* (R²=0.577 < 0.610). Being "significantly different" from AR(1) in the *wrong direction* is odd to highlight under "method equivalence." Is this handled correctly?

7. **HAC with n=10.** Newey-West HAC estimators are consistent as n→∞. With n=10, they are known to be severely biased (finite-sample distortion). The bandwidth choice of 1-3 is presented as a robustness range, but at n=10, bandwidth=3 uses 30% of the sample for kernel weighting. Cite Andrews (1991) or Lazarus et al. (2018) on HAC finite-sample issues, or justify why the results should be trusted.

### C. Econometric substance

1. **Pooled ρ with heterogeneous persistence.** Macro ρ≈0.88, firms ρ≈0.60. The pooled ρ̂ will be approximately a weighted average, ~0.63. This means Stage 1 residuals for macro actors systematically contain unexplained persistence (true ρ - pooled ρ̂ ≈ +0.25), while firm residuals contain less. A block-specific Stage 2 can exploit this *structured misspecification* of Stage 1, which is different from capturing genuine cross-sectional dynamics. **The critical missing comparison is: block-specific ρ_b (BA, R²=0.611) + block-specific Stage 2 (not reported).** If this matches M2, the entire gain is from correcting Stage 1 misspecification, not from block-specific cross-sectional dynamics. If it falls between BA and M2, the gain is partially from each source. This experiment is straightforward and its absence is a significant gap.

2. **"Method equivalence" scope.** The abstract and conclusion both state "architectural rather than methodological" without the qualifier "among linear reduced-rank estimators." The Introduction (paragraph 5) does include this qualifier, creating an inconsistency. The claim should be uniformly qualified throughout.

3. **Survivorship bias and rank computation.** The paper acknowledges survivorship bias (Limitation 7) but does not analyze its interaction with the cross-sectional ranking. If 82 survivors are ranked each quarter, the ranking is conditional on survival — firms that will eventually delist are ranked throughout. For a 2006Q1 forecast, the cross-section includes firms that delist in 2020; their inclusion affects all ranks. This is standard in balanced-panel studies but interacts with the heterogeneity story: survivors in distressed sectors may have different rank dynamics than the sector average.

4. **Block selection multiplicity.** Ten candidates, 3 selected. The paper correctly calls this exploratory (Limitation 2), but then says the LOWO test "controls for selection conditional on the candidate set." LOWO controls for within-sample overfitting of which blocks are selected, but does NOT control for the initial choice to consider these 10 candidates rather than others (e.g., industry sub-sectors, size quantiles, geographic splits). The researcher degree of freedom is in the candidate generation, not the selection rule.

5. **EWM half-life, ridge lambda, spectral clip.** Three hyperparameters (12 quarters, unspecified λ, 0.99 clip) are set but not justified. Are they the same across all architectures? If they were tuned on any in-sample criterion, this is a potential source of look-ahead. If they are arbitrary defaults, state this explicitly.

6. **The R² scale.** R² = 0.630 on percentile ranks is high because ranks are smooth and persistent. It does not mean 63% of economic variation is explained. The paper should contextualize what R²=0.63 means in this specific setting — is it easy or hard to beat AR(1) on ranked data?

### D. Claims vs evidence alignment — line-by-line audit

1. Abstract: "improves full-panel out-of-sample R² from 0.630 to 0.677" — this is M2 vs G1 (Table 3). Correct.

2. Abstract: "Δ = +0.047, CI [+0.036, +0.058], 10/10 windows, placebo z = 7.82" — CI source? Table 3 shows CI [+0.036, +0.058] for M2. The placebo z=7.82 is Table 6. Mixing results from different tables in one sentence without attribution is risky.

3. Abstract: "Dropping the tech/health block eliminates roughly 72% of the gain" — Section 5.4 confirms.

4. Abstract: "The gain is architectural rather than methodological within the linear reduced-rank estimators tested" — the qualifier "within the linear reduced-rank estimators tested" appears in the abstract. Good, but check if it appears consistently in the conclusion.

5. Introduction, paragraph 3: "improves full-panel out-of-sample R² from 0.630 to 0.677" — same as abstract. Consistent.

6. Introduction, paragraph 5: "nine model specifications spanning three complexity classes" — Table 2 has 9 rows ✓.

7. Introduction, paragraph 5: "forecast-error correlations ρ > 0.98" — Table 2 footnote reports ρ(DMD,PCA)=0.990, ρ(DMD,Ridge)=0.980, ρ(PCA,Ridge)=0.969. But 0.969 < 0.98. The claim "ρ > 0.98" is **false** for the PCA-Ridge pair. Either change to "ρ > 0.96" or qualify as "pairwise correlations between 0.969 and 0.990."

8. Conclusion: "invariant to the choice of linear second-stage method among the three tested (PCA, DMD, and Ridge)" — but PCA+full VAR (R²=0.577) is significantly *worse* than DMD+full A (R²=0.630). Method equivalence holds at matched complexity, not unconditionally. The conclusion should be qualified.

9. Section 3.4: "the forecasting ceiling is R² ≈ 0.630 regardless of method" — but Ridge achieves 0.632 and DMD+full achieves 0.630. Is 0.632 meaningfully different from 0.630? Table 2 shows Ridge CI [+0.012, +0.033], which does not overlap with some smaller models. The "ceiling" language may be too strong.

10. Section 5.8: "6.6% reduction in out-of-sample RMSE (from 0.176 to 0.164)" — verify: sqrt(1-0.630) = 0.608, but RMSE depends on the actual scale, not just R². If var(y)·(1-R²) = RMSE², then RMSE_G1/RMSE_M2 = sqrt((1-0.630)/(1-0.677)) = sqrt(0.370/0.323) = sqrt(1.146) = 1.070... wait, that's the inverse. RMSE_M2/RMSE_G1 = sqrt(0.323/0.370) = 0.934, so 6.6% reduction ✓. But this assumes the denominator (total variance) is the same for both models, which it is since they predict the same targets.

### E. Missing analyses and alternative explanations

1. **Forecast combination baseline** (discussed in paper but not run). A simple equal-weight average of G1 and per-block AR(1) models would test whether the gain is from the block-specific *factor structure* or merely from diversifying forecast errors. If equal-weight combination ≈ M2, the block-specific PCA is unnecessary. This is the paper's most significant missing experiment.

2. **Block-specific ρ_b + block-specific Stage 2** (discussed above in C.1). The BA + local Stage 2 comparison directly tests whether the gain is from correcting Stage 1 pooling or from genuine local cross-sectional dynamics.

3. **Information criterion for K_b.** Table 4 lists K_b = 4, 2, 4 for the three local blocks. How were these chosen? If by in-sample criteria (BIC, cross-validation), report the procedure. If by manual selection, this is another researcher degree of freedom.

4. **Sensitivity to the number of blocks.** The paper tests 1-block, 3-block, and 4-block mixtures. But what about 2-block (e.g., tech/health only + remainder)? Since tech/health dominates the gain (72%), a simpler 2-block partition might achieve 90% of the gain with less complexity.

5. **Non-linear alternatives.** The method equivalence is established for linear methods only. A single non-linear baseline (e.g., random forest or gradient boosting on the same features) would calibrate whether the "ceiling" is truly architectural or merely a linear-method ceiling.

6. **Out-of-sample R² vs other metrics.** The entire paper uses one metric (OOS R²). At minimum, report MAE or median absolute error to check whether the gain is driven by a few outlier windows/actors.

### F. Presentation and exposition

1. **Table 2 baseline confusion.** The caption says "Δ is versus rolling per-actor AR(1) (R²=0.610)" and explains the fixed-parameter AR(1) achieves 0.594. But Table 1 uses the 0.594 baseline. A reader comparing Tables 1 and 2 will see different AR(1) values and may be confused. Consider adding a row for rolling AR(1) in Table 1 or vice versa.

2. **Terminology proliferation.** G0, G1, S1, M1, M2, BA — six architecture labels introduced across different tables. Consider a single summary table early in the paper defining all architectures.

3. **Pre-registration section (6.7).** The paper says predictions are "time-stamped in the replication archive." Unless this is a third-party-verified timestamp (e.g., OSF, aspredicted.org), self-reported timestamps are not credible pre-registration. The section should either demonstrate verifiable pre-registration or be reframed as "exploratory predictions" rather than "pre-registered."

4. **Length.** The paper is ~9,500 words (main text) + 2,500 words (appendices). For JBES, this is within bounds but on the long side. The falsification section (Section 6) could be condensed: the point (nothing else works) can be made in 1 page instead of 2.

5. **Limitation assessment.** Rate each:
   - L1 (small evaluation sample): [substantive] — 10 windows is genuinely limiting
   - L2 (exploratory block selection): [substantive] — the key caveat
   - L3 (scope limited to heterogeneous panels): [substantive but also a contribution]
   - L4 (quarterly frequency): [boilerplate]
   - L5 (modest absolute improvement): [substantive]
   - L6 (data-timing latency): [substantive] — partially addressed by Appendix I
   - L7 (survivorship bias): [substantive] — under-analysed

6. **Missing references.** Consider citing: (a) Bernanke, Boivin & Eliasz (2005) FAVAR; (b) Kim & Swanson (2014) forecast combination with many predictors; (c) recent ML panel forecasting (Gu, Kelly & Xiu, RFS 2020, "Empirical Asset Pricing via Machine Learning").

### G. Reproducibility

1. **Hyperparameter inventory.** List every tuneable parameter and whether it is justified, cross-validated, or arbitrary. Missing specifications: ridge λ (for both global and local), PCA variance threshold (if any), the "q_0 = 0.5" initial process noise, the "10^{-6}" regularisation floor in Q.

2. **The anonymisation paradox.** 82 S&P 500 firms with complete quarterly data 2005-2025, assigned to 6 sectors with known sizes (23, 15, 12, 10, 12, 10), using CapEx/Assets and Revenue growth. This is almost certainly re-identifiable. If anonymisation is claimed, it should be honest about its limits.

3. **Code availability.** The anonymous URL is a placeholder for review. Will the final code be on a permanent repository (GitHub, Zenodo)?

### H. Mathematical correctness

1. **Equation 2 (mixture estimator).** The remainder case uses $U_g A_g U_g^T r_t$ where $r_t \in R^N$. But for the local case, $U_b A_b U_b^T r_{b,t}$ where $r_{b,t} \in R^{N_b}$. The dimensions are consistent within each case, but the notation switches between full-panel and block-level residual vectors without explicit projection operators. Adding $P_b$ (the restriction matrix that selects block $b$ actors from the full panel) would clarify.

2. **Kalman filter: Woodbury identity.** The paper writes $S_t^{-1} = σ^{-2}(I_N - U_r(σ^{-2}I_K + P_{t|t-1}^{-1})^{-1}U_r^T σ^{-2})$. This should be verified via the matrix inversion lemma: $(σ²I + UPU^T)^{-1} = σ^{-2}I - σ^{-2}U(I + σ^{-2}PU^TU)^{-1}σ^{-2}PU^T$. With $U^TU = I_K$: $(σ²I + UPU^T)^{-1} = σ^{-2}I - σ^{-4}U(I + σ^{-2}P)^{-1}PU^T = σ^{-2}[I - U(σ^2I + P)^{-1}PU^T]$. The paper's form uses $P^{-1}$ rather than $P$, which requires $P$ to be invertible (true if $P_0 = I_K$ and updates preserve PSD). Verify the algebra carefully — the two forms are equivalent but the paper's expression may have a sign/scaling issue.

3. **Spectral radius clipping.** $\tilde{A} \leftarrow \tilde{A} \cdot \min(1, 0.99/\max_k|\lambda_k|)$. This scales the entire matrix uniformly, so if $\lambda_{max} = 1.05$, the clip factor is $0.99/1.05 ≈ 0.943$, and ALL eigenvalues are scaled by 0.943 — including stable ones. A per-eigenvalue clip (in the eigendecomposition) would be more targeted. Is uniform clipping intended? If so, note that it compresses the entire spectrum, not just the unstable part.

4. **Adaptive Q convergence.** The paper notes that $Q$ decays as $\sim 0.7^t$ toward the floor. With $T_{train} \approx 20$, $0.7^{20} \approx 0.0008$. So after 20 steps, the effective $Q$ is near the $10^{-6}$ floor — the filter is essentially running open-loop for the last few training steps. Does this affect the test forecast (which is the filtered state at the training boundary)? The paper argues the quarterly reset mitigates this, but the reset happens at basis re-estimation boundaries, not at every step.

### I. Scope and contribution assessment

1. **Novelty.** The core idea — "pool globally for persistence, decompose locally for residual dynamics" — is intuitive and, to the referees' knowledge, not previously formalised in this specific two-stage form. However, the grouped fixed-effects literature (Bonhomme-Manresa 2015), interactive fixed-effects models (Bai 2009), and CCE estimators (Pesaran 2006) all address panel heterogeneity through group/block-specific parameters. The paper's contribution is the specific *application* to forecasting (rather than estimation) and the diagnostic programme (what doesn't work). Is this sufficient for a top journal?

2. **Single dataset.** The headline result is from one 93-actor panel. The two cross-panel checks are null results. A single positive finding on a single dataset, however well-validated, is fragile. External replication on an independent heterogeneous panel (e.g., from a different country, asset class, or time period) would substantially strengthen the contribution.

3. **The falsification programme.** Sections 6.1-6.4 systematically rule out alternatives. This is scientifically valuable but also means the paper spends substantial space on what *doesn't* work. Some editors view this positively (scientific rigour); others see it as padding. The referees are divided.

---

## Output format (you MUST use exactly this structure)

```
=== REFEREE A (Econometrician) ===

RECOMMENDATION: [Accept / Major Revision / Minor Revision / Reject]

[Issues numbered A-1, A-2, etc., each tagged [FATAL], [MAJOR], or [MINOR]]

=== REFEREE B (Financial Statistician) ===

RECOMMENDATION: [Accept / Major Revision / Minor Revision / Reject]

[Issues numbered B-1, B-2, etc.]

=== REFEREE C (Applied ML / Signal Processing) ===

RECOMMENDATION: [Accept / Major Revision / Minor Revision / Reject]

[Issues numbered C-1, C-2, etc.]

=== CONSOLIDATED ASSESSMENT ===

FATAL ISSUES (if any):
...

TOP 5 MAJOR ISSUES (ranked by importance):
1. ...

ISSUES THAT ARE NOT ACTUALLY ISSUES (things that look problematic but are correctly handled in the paper — include these to demonstrate careful reading):
...

SPECIFIC NUMERICAL ERRORS FOUND:
...

MISSING EXPERIMENTS (ranked by how much they would change the paper's conclusions):
1. ...
```

---

## The paper to review

Paste the full LaTeX source of `smim_paper.tex` below this line.
