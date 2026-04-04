# Critical Review of SMIM Paper Before arXiv Submission

## Your role

You are a demanding referee for a top econometrics journal (Journal of Econometrics, JBES, or Quantitative Economics). You have deep expertise in:
- State-space models and Kalman filtering
- High-dimensional panel data and regularisation
- Dynamic Mode Decomposition and spectral methods
- Out-of-sample forecast evaluation methodology (Diebold-Mariano, nested CV, bootstrap inference)
- Investment and capital allocation in corporate finance

You are reviewing the paper `docs/smim/paper/smim_paper.tex` (~870 lines, 13 figures, 8 tables).

## What the paper claims

A spectral state-space framework (DMD + Kalman + dual regularisation + rolling basis) forecasts cross-sectional investment intensity on a 146-firm US CapEx/Revenue panel. Key claims:

1. **Predictive result (headline)**: Nested CV R²=0.711 vs AR(1) R²=0.669, delta=+0.042, 8/8 windows (DM t=6.4, p<0.001). K=2 modes, no operator learning — the pipeline is just DMD + Kalman + rolling basis.

2. **Cross-sectional pooling**: The advantage grows monotonically as T shrinks — from zero at T=7yr to +5.7pp at T=2yr. SMIM pools across N actors; AR(1) uses each actor alone.

3. **Structural findings (modal R², separate 93-actor panel)**: 8-dimensional spectral structure rotates at 26°/quarter. Rolling basis captures this rotation (+14.7pp modal R²). Dual regularisation is essential (spherical R + near-identity F).

4. **Economic validation**: Model-implied investment gaps predict CapEx revision (t=-6.4, actor-clustered SEs).

5. **Simplification finding**: At K=2 with quarterly basis refresh, the Kalman filter is functionally redundant — the pipeline reduces to a single DMD projection per quarter (Eq. 7 in the paper).

## Your task

Conduct a thorough review. For EACH issue you find, classify it as:

- **FATAL**: Would reject the paper outright (e.g., data leakage, circular evaluation, fundamental methodological error)
- **MAJOR**: Must be addressed before publication (e.g., missing robustness check, unclear methodology, overclaiming)
- **MINOR**: Should be addressed but not blocking (e.g., unclear exposition, missing reference, notation inconsistency)

Structure your review as:

### Summary and Recommendation
One paragraph: what the paper does, whether the claims are supported, overall recommendation.

### Fatal Issues
(hopefully none)

### Major Issues
Numbered list with specific line/table/figure references.

### Minor Issues
Numbered list.

### Questions for the Authors
Things you'd want clarified before accepting.

## What to look for specifically

1. **Internal consistency**: Do all numbers in text match tables match figures? Are the two panels (93-actor structural, 146-firm predictive) clearly distinguished everywhere?

2. **Methodological soundness**: Is the nested CV properly implemented? Is the DM test appropriate? Is the T-sweep comparison fair (same T for SMIM and AR(1))?

3. **Overclaiming**: Does the paper claim more than the evidence supports? Are scope conditions stated clearly enough?

4. **Missing comparisons**: Should the paper compare against DFM, LASSO, random forest, or other panel forecasting methods beyond AR(1)?

5. **The two-panel design**: Is it legitimate to run structural analysis on one panel and prediction on another? Does the paper explain this adequately?

6. **The K=8 vs K=2 gap**: 8 modes are structurally recoverable but only 2 are predictively useful. Is this discussed sufficiently? Does it undermine the structural claims?

7. **Figure-text-table consistency**: Do figure captions match what the figures show? Do table numbers match text references?

8. **Statistical rigour**: Are the bootstrap CIs and permutation tests appropriate for this setting? Is the DM test's HAC bandwidth reasonable?

9. **Economic interpretation**: Is the gap prediction (D2) convincing? Does the actor-FE sign flip undermine the result?

10. **The simplification to Eq. 7**: If the Kalman filter is redundant at K=2, why present the full Kalman framework? Is this a feature or a weakness?

## Key files

- Paper: `docs/smim/paper/smim_paper.tex`
- Figures: `docs/smim/paper/img/fig*.pdf` (13 figures)
- Nested CV results: `results/metrics/iter5_1v2_nested_cv.parquet`
- T-sweep results: `results/metrics/iter5_1v2_t_sweep.parquet`
- Ablation (predictive): `results/metrics/iter5_1v2_ablation.parquet`
- Ablation (structural): hardcoded in `scripts/smim/paper_figures_v2.py`
- Verification script: `scripts/smim/verify_modal_numbers.py`
- Main pipeline: `scripts/smim/run_smim_iter5_1_cv2.py`

## Context the reviewer should know

- The paper was previously submitted to SSRN with a bug (modal alpha used as predictive). This version is a complete rewrite with correct evaluation. The current version has no erratum language — it presents as a clean paper.
- The structural panel (93 actors) and predictive panel (146 firms) are different by design — explained in the "Why two panels?" paragraph after Table 5 (predictive ablation).
- All structural R² values are explicitly labeled "modal" throughout the paper and in figure axis labels.
- The AR(1) baseline was removed from all structural/modal comparisons to avoid the apples-to-oranges issue.
- The figures were ALL regenerated on 2026-04-04 with verified numbers and correct file paths.

## How to read the paper efficiently

1. Start with the abstract (line ~79) — does it accurately summarise the claims?
2. Skip to Table 1 (baselines, ~line 370) and Table 2 (inference, ~line 398) — are the numbers internally consistent?
3. Check the T-sweep figure (fig13, ~line 630) — does it support the cross-sectional pooling claim?
4. Read the predictive ablation (Table ~line 455) — does it make sense that Kalman hurts on a static basis?
5. Check the "When the Kalman Filter Is and Is Not Needed" subsection (~line 737) — is the simplification argument convincing?
6. Read the "Reading the Spectral Structure" subsection (~line 760) — is the practitioner interpretation sound?
7. Finally, scan ALL figures for consistency with captions and text.

Be rigorous. Be specific. Cite line numbers. The goal is to catch every remaining issue before this goes on arXiv.
