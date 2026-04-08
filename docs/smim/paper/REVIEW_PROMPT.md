# Expert Referee Prompt for "Global Persistence, Local Residual Structure"

---

## Instructions

You are acting as **three independent referees** for a top-tier econometrics journal (Econometrica / JASA / Journal of Econometrics / Review of Financial Studies). Each referee has a distinct expertise profile. You must produce three separate, self-contained reports. Do not soften criticism out of politeness. Do not summarise the paper back to the author. Go straight to evaluation.

---

## Referee Profiles

**Referee 1 — Econometrician (panel data, factor models, inference)**
Focus: statistical methodology, asymptotic validity, finite-sample inference, identification, degrees-of-freedom corrections, bootstrap validity, multiple-testing adjustments, nested vs non-nested model comparison, information criteria.

**Referee 2 — Financial economist (asset pricing, cross-sectional prediction, empirical design)**
Focus: economic content, data construction, look-ahead bias, survivorship, evaluation protocol (OOS R^2 definition, denominator choice), practical significance, competing benchmarks, connection to asset pricing literature, portfolio implications.

**Referee 3 — Applied statistician / machine learning (model selection, regularisation, reproducibility)**
Focus: hyperparameter tuning on test data, researcher degrees of freedom, block selection procedure, placebo design adequacy, replicability, overfitting diagnostics, information leakage, sensitivity to arbitrary choices, code/data reproducibility.

---

## What to Evaluate

For each referee, produce a structured report with these sections:

### A. Summary Verdict
One sentence: accept / major revision / minor revision / reject, with the single most important reason.

### B. Major Issues (numbered, each with a concrete fix)
Issues that, if unresolved, would block publication. For each:
- State the problem precisely (cite the specific table, equation, section, or claim)
- Explain *why* it matters (what conclusion could be wrong)
- Propose a specific remedy or experiment that would resolve it

### C. Minor Issues (numbered)
Issues that should be fixed but would not block publication alone.

### D. Consistency and Internal Logic Audit
Check for:
- Numbers that appear in multiple places but don't match
- Claims in the text that contradict what the tables show
- Denominators, sample sizes, degrees of freedom that don't add up
- Confidence intervals that are inconsistent with the reported t-statistics and sample sizes
- Any circular reasoning (e.g., using a result to justify a design choice that produced the result)

### E. Missing Analyses
What would you want to see that isn't in the paper? Be specific (name the test, the comparison, the table you'd want).

### F. Writing and Presentation
Only flag issues that cause genuine ambiguity or could mislead a reader. Do not comment on style preferences.

---

## Specific Audit Checklist (all three referees should address these)

The following are known high-risk areas for this type of paper. Check each one explicitly and state whether the paper handles it adequately, inadequately, or does not address it:

1. **OOS R^2 definition**: The paper uses test-window mean in the denominator (not training-set mean). Does this inflate or deflate R^2? Is the Campbell-Thompson (2008) convention discussed? Is the differential Delta R^2 truly invariant to denominator choice as claimed?

2. **Block selection is post-hoc**: 10 candidate blocks were evaluated, 3 were selected. The LOWO procedure controls for within-candidate overfitting but not for the candidate-generation step. Is the paper's own Limitation 2 adequate, or does this require a stronger control (e.g., a held-out decade)?

3. **Bootstrap with n=10**: With only 10 annual windows, the bootstrap has limited resolution (~92k distinct resamples). Are the reported CIs and p-values reliable? Is the sign test (exact binomial) a more appropriate primary test?

4. **Placebo test design**: The 1,000 random partitions control for block *sizes* but use the same actor pool. Could the placebo be gamed by the specific actor composition? Should the placebo also permute actor identities across layers?

5. **Survivorship bias**: 82 firms with complete 20-year data. The paper acknowledges this (Limitation 7). But does the balanced-panel requirement mechanically create the cross-sectional rank persistence (rho ~ 0.60) that the architecture exploits? Would the result survive on an unbalanced panel?

6. **FRED normalisation look-ahead**: Full-sample min-max for 7 macro actors. The robustness check (recursive bounds, Delta=+0.048) uses only 7 of 93 actors. Is this sufficient to rule out contamination of the pooled rho?

7. **Filing-lag test**: Lagging firms by 1 quarter but keeping macro at t=0 creates a structural timing mismatch. Does this test actually approximate real-time availability, or does it introduce a different bias?

8. **Method equivalence claim**: The paper claims DMD ~ PCA ~ Ridge at matched complexity. But PCA+full VAR (R^2=0.577) is significantly *worse* than DMD+full A-tilde (R^2=0.630). The paper attributes this to overfitting. Is this attribution convincing, or does it undermine the equivalence claim?

9. **Gradient boosting baseline**: GBM achieves R^2=0.661 globally, between the linear ceiling (0.630) and M2 (0.677). This is mentioned in one sentence. Should this be a major comparison? What if GBM + block-specific estimation exceeds M2?

10. **The tech/health block drives 72% of the gain**: Is this a robust finding or a fragile single-cluster effect? What happens if the tech/health block is split into tech-only and health-only? The candidate table shows tech alone at +0.014 (10/10) but health at +0.001 (5/10) — the merged block gain (+0.031) appears super-additive. Is this explained?

11. **Equation 1 (pooled AR(1))**: A single global rho for actors with persistence ranging from 0.60 to 0.88. The paper defends this by saying block-specific rho_b hurts (BA_M2 < M2). But the BA baseline (rho_b only, no Stage 2) at R^2=0.611 is *above* G0 at 0.591. Is the global rho truly optimal for Stage 1, or just better than the tested alternative?

12. **Multiple testing**: The paper tests 8 architectures against G1. Holm-Bonferroni is applied. Is this sufficient? Should the falsification programme (Table 6: 9 models, 10 geometric models, 7 targets, 5 gates) also receive multiple-testing correction for the aggregate claim that "nothing else works"?

13. **Pre-registered predictions**: 4/5 failed, 1 borderline. The paper presents this transparently. But the predictions concern hyperparameter sensitivity, not the headline result. Does this transparency actually strengthen or weaken the paper's credibility?

14. **Replication**: The paper cites an anonymous repository. Is the description of the computational pipeline sufficient for independent replication without the code? Could a reader reproduce Table 3 from the paper text alone?

15. **Economic significance**: Delta R^2 = +0.047 corresponds to 1.2 percentile-point improvement in forecast precision. Is this economically meaningful? The paper punts on portfolio implications. Should it at least report a Sharpe ratio or information coefficient?

---

## Deep Mathematical Audit (Referee 1 must address all; Referees 2-3 address where relevant)

16. **Verify placebo z-score arithmetic**: Real Delta=+0.047, Placebo mean=-0.004, Placebo std=0.0065. z = (0.047-(-0.004))/0.0065 = 0.051/0.0065 = 7.846. Paper reports 7.82. Is std reported with sufficient precision, or is there a rounding discrepancy?

17. **Cross-check per-block N**: Diversified(23) + Macro/Inst(11) + Tech/Health(25) + Remainder(34) = 93. But Macro/Inst = Layer 0 (7) + Layer 1 (4) = 11. Tech/Health = Tech (15) + Health (10) = 25. Remainder = Energy (12) + Financials (10) + Industrials (12) = 34. All check out — but verify that no Layer 0/1 actors are double-counted in sector blocks.

18. **RMSE reduction**: Paper claims 6.6% RMSE reduction (0.176 to 0.164). Verify: RMSE_M2/RMSE_G1 = sqrt((1-0.677)/(1-0.630)) = sqrt(0.323/0.370) = sqrt(0.8730) = 0.9344, i.e., 6.56% reduction. Close to 6.6% but verify the exact calculation basis.

19. **Effective sample size calculation**: Paper states rho_d = 0.11, n_eff = 10*(1-0.11)/(1+0.11) = 10*0.89/1.11 = 8.02. Paper says ~8.1. Minor rounding. But: is the (1-rho)/(1+rho) formula appropriate for an AR(1) process, or should a more general formula (e.g., Bayley-Hammersley) be used for 10 observations?

20. **Kalman filter Woodbury identity**: Verify the matrix inversion lemma application in Appendix A. The paper writes S_t^{-1} with P_{t|t-1}^{-1} inside the parentheses. Standard Woodbury gives (A + UBU^T)^{-1} = A^{-1} - A^{-1}U(B^{-1}+U^T A^{-1}U)^{-1}U^T A^{-1}. With A=sigma^2 I, B=P, U=U_r: sigma^{-2}[I - U_r(P^{-1}+sigma^{-2}I)^{-1}U_r^T sigma^{-2}]. Paper's expression matches this form — confirm or find the error.

---

## Output Format

Structure your response as:

```
================================================================
REFEREE 1 — Econometrician
================================================================

A. Summary Verdict: ...

B. Major Issues
1. ...
2. ...

C. Minor Issues
1. ...

D. Consistency Audit
...

E. Missing Analyses
...

F. Writing
...

================================================================
REFEREE 2 — Financial Economist
================================================================

[same structure]

================================================================
REFEREE 3 — Applied Statistician / ML
================================================================

[same structure]

================================================================
CROSS-REFEREE AGREEMENT / DISAGREEMENT
================================================================

List any points where referees would likely disagree, and why.
State the 3 most critical issues across all three reports.
```

---

## The Paper

Paste the full LaTeX source of `smim_paper.tex` below this line.
