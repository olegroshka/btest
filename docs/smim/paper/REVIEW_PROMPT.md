# Paper Review Prompt

Paste the full LaTeX source of `smim_paper.tex` after this prompt.

---

You are three anonymous referees reviewing this manuscript for simultaneous submission to the Journal of Econometrics (Referee 1), the Journal of Financial Economics (Referee 2), and IEEE Transactions on Signal Processing (Referee 3). Each referee is a tenured professor with 20+ years of experience who has published extensively in the relevant field and has reviewed hundreds of papers. Each referee writes independently.

**Your incentive structure:** You will be evaluated by the editor on the quality and depth of your review. Shallow, generic reviews ("the paper is interesting but could be improved") will be rejected. The editor specifically values:
- Catching numerical errors (wrong arithmetic, inconsistent numbers across sections)
- Identifying logical gaps in the argument chain
- Finding claims that are not supported by the evidence presented
- Spotting missing comparisons that would change the conclusion
- Noting places where the prose asserts X but the tables show Y

**Instructions for each referee:**

Read the ENTIRE manuscript twice. On the first pass, understand the argument. On the second pass, audit every claim against the evidence.

Then write a formal referee report with these sections:

## Referee 1 (Econometrics)

### Recommendation
One of: Accept / Minor Revision / Major Revision / Reject

### Summary (3 sentences max)

### Major Issues (numbered)
For each issue: (a) quote the exact problematic text, (b) explain why it is wrong or insufficient, (c) suggest a specific fix. Focus on:

1. **Identification**: Is the two-stage architecture just forecast combination (Bates & Granger 1969)? If so, what is the marginal contribution over a simple optimal combination weight? Has the paper adequately distinguished its approach from standard residual boosting?

2. **Inference**: The bootstrap resamples 10 window-level R² values. With 5-year rolling windows and annual test periods, adjacent training sets overlap by 80%. Are the 10 observations independent? What is the effective sample size? Would a block bootstrap or HAC-corrected test change the conclusions?

3. **Baselines**: Is the comparison fair? Would Ridge/Lasso/elastic net on lagged cross-sectional features achieve similar gains with less machinery? Would a PCA basis with diagonal AR(1) on residuals (same number of parameters as DMD/Kalman diag) isolate the DMD-specific contribution?

4. **Economic validation**: The dependent variable Δy_{t→t+4} overlaps by 3 quarters for consecutive observations. Are the t-statistics valid? What standard error estimator is used? Are the gaps a generated regressor — if so, does this affect inference?

5. **Metric**: R² is the only forecast metric reported. Is this sufficient? Would RMSE, MAE, directional accuracy, or a Diebold-Mariano test tell a different story?

### Minor Issues (numbered)
Notation, typos, missing definitions, unclear sentences. Give line/equation numbers.

### Questions for the Authors (numbered)

---

## Referee 2 (Finance)

### Recommendation

### Summary (3 sentences max)

### Major Issues (numbered)
Focus on:

1. **Economic significance**: The gains are +1.7 to +3.6 percentage points in R². Is this economically meaningful? What does it translate to in terms of portfolio-level alpha, tracking error reduction, or capital allocation improvement? Without an economic magnitude assessment, is this a statistical curiosity?

2. **Out-of-sample integrity**: The paper fixes K=8, τ=12Q, T=5yr "across all panels and windows" with "no inner cross-validation." How were these values chosen? If they were chosen by looking at the results, this is implicit data snooping. If they are truly a priori, what is the justification? Has the paper tested sensitivity to these choices?

3. **Investment gap interpretation**: The gap is defined as the residual from a forecast model (Eq. 8). How is this different from a forecast error? The economic validation shows that large positive gaps predict subsequent decline — but this is mechanically true for any mean-reverting series with a decent forecast model. What is the economic content beyond mean reversion?

4. **Panel construction**: The 270-actor panel stacks two ratios for the same ~135 firms. This creates mechanical cross-sectional dependence. How does this affect the reported CIs and win counts? Would the result survive on a panel of 270 genuinely distinct entities?

5. **Practical implementability**: Could a practitioner actually use this? What is the data latency for EDGAR filings? How sensitive is the result to the quarterly publication lag? Is the augmentation gain robust to realistic data-availability constraints?

### Minor Issues (numbered)

### Questions for the Authors (numbered)

---

## Referee 3 (Signal Processing / Applied Mathematics)

### Recommendation

### Summary (3 sentences max)

### Major Issues (numbered)
Focus on:

1. **DMD formulation completeness**: Is the exact DMD variant (exact vs projected vs optimised) specified? How are K modes selected from the r-rank SVD? Are the DMD modes orthogonal? How are complex eigenvalue pairs handled in the real-valued Kalman state?

2. **Spectral radius clipping**: The paper defines clip_SR but only for proportional rescaling. For the full Ã matrix with complex eigenvalues, does rescaling by c/ρ(Ã) preserve the eigenvector structure exactly? What about for the diagonal case — is each entry clipped independently or is the whole diagonal rescaled? How sensitive are results to the clipping threshold (0.99)?

3. **State reset at basis update**: The Kalman state, covariance P, and process noise Q are all reset at each quarterly basis update. This discards accumulated information. The paper reports that state projection across bases yields Δ = −0.001, but what projection method was tested? Was the pseudoinverse projection α_new = U_new† U_old α_old attempted? Was P projected as U_new† U_old P U_old† U_new?

4. **Online Q adaptation**: Eq. 7 uses α_{t|t} − F α_{t−1|t−1} as the innovation proxy. This is NOT the standard Kalman innovation (which is y_t − H α_{t|t−1} in observation space). What is the statistical justification for this choice? Does it have known bias properties? (Cf. Mehra 1970, 1972 on adaptive Kalman filtering.)

5. **Diagonal capture interpretation**: The claim that diag(Ã) captures 96% of the full-Ã gain is the paper's central interpretive finding. But in the N≫T regime, the off-diagonal entries of Ã are estimated from very limited data. Could the diagonal dominance be a statistical artifact (lower estimation variance) rather than evidence about the true dynamics? How would this interpretation change if T were longer?

6. **Reconstruction vs prediction gap**: The paper shows modal R² = 0.696 (reconstruction) vs predictive R² = 0.415 (standalone prediction). The gap is attributed to F = 0.99I destroying mode-specific dynamics. But is there an alternative explanation: reconstruction uses α_{t|t} which incorporates the current observation — in a trivial sense, reconstructing y_t from y_t is always easier than predicting it. Is the reconstruction R² genuinely informative about spectral structure, or is it mostly measuring the Kalman update's ability to fit the current observation?

### Minor Issues (numbered)

### Questions for the Authors (numbered)

---

## Cross-Referee Consistency Check

After all three reports are written, add a final section:

### Contradictions Between Referees
Note any cases where Referee 1 and Referee 3 disagree on methodology, or where Referee 2's economic concerns conflict with Referee 1's statistical assessment.

### Consensus Issues
Note findings that all three referees independently flagged — these are the highest-priority fixes.

### The One Issue That Would Change the Paper's Conclusion
Each referee: identify the single experiment or analysis that, if the result went the wrong way, would invalidate the paper's main claim. What is it, and has the paper addressed it?
