# Iteration 5.1: Paper Correction + CapEx/Revenue Drilldown

> Created: 2026-04-04
> Status: PLAN — to be executed in a fresh session
> Predecessor: Iterations 1-5 (this session)
> Paper: `docs/smim/paper/smim_paper.tex` (submitted SSRN 2026-04-03 — needs correction)
> Key scripts: `scripts/smim/run_smim_a1i5.py`, `scripts/smim/run_smim_ws_a_nested_cv.py`

---

## 1. The Issue: Modal vs Predictive R² (discovered 2026-04-04)

### What happened

The published paper reports R²=0.702 (nested CV, 8 windows) and R²=0.765
(frozen holdout, 2 windows) as "out-of-sample" performance.

The prediction function (`predict_quarter` in `run_smim_ws_a_nested_cv.py` line 130
and `platinum_predict_quarter` in `run_smim_dd2_v2.py` line 112) uses the
**filtered** alpha (after Kalman update with the CURRENT observation):

```python
# BEFORE (modal — uses current obs y_t through Kalman update):
pred = (a_f @ U.T + om).ravel()     # a_f = alpha_filtered

# AFTER (predictive — genuine one-step-ahead forecast):
pred = (a_p @ U.T + om).ravel()     # a_p = alpha_predicted (before update)
```

The filtered reconstruction is:

```
ŷ_t = (1 - g) × genuine_prediction + g × y_t
```

where g depends on the Kalman gain (typically 0.3-0.7). Part of the "prediction"
IS the target observation. This inflates R² and makes the comparison with AR(1)
(which IS a genuine prediction using only t-1 information) unfair.

### The corrected numbers

| Metric | Original (modal) | Corrected (predictive) | AR(1) |
|--------|-----------------|----------------------|-------|
| Nested CV (8 windows) | 0.702 | **0.489** | 0.610 |
| Frozen holdout (2 windows) | 0.765 | **0.540** | 0.674 |
| SMIM vs AR(1) | +0.093 (wins 8/8) | **-0.120 (wins 0/8)** | — |

**The fix has been applied** to both `run_smim_ws_a_nested_cv.py` and
`run_smim_dd2_v2.py` (changed `a_f` to `a_p` in the prediction return).
Both scripts now return `(pred, pred_modal, ...)` — predictive as primary,
modal as diagnostic.

### What this does NOT invalidate

1. **Dual regularisation methodology** — the ablation steps are all relative
   (modal metric throughout), so *relative* improvements remain valid. The
   ablation tells us spherical R matters more than DMD, which matters more
   than transition regularisation. This ordering is independent of the metric.

2. **Rolling basis rotation (26 deg/Q)** — a structural finding about the
   cross-sectional dynamics. Independent of any R² metric.

3. **CapEx revision prediction (D2: t=-6.4)** — the gap-to-CapEx regression
   tests whether gaps predict FUTURE outcomes. The gaps are modal (filtered),
   which if anything makes them SMALLER (better reconstruction = smaller gaps).
   Predictive gaps would be larger and might give STRONGER regression results.

4. **CapEx/Revenue result (Iter 5, delta=+3.0pp, 9/10 wins)** — this was
   always computed with predictive alpha (the Iter 5 scripts used `a_p`
   from the start). This IS a genuine predictive result.

### What this DOES invalidate

1. **The headline claim "SMIM R²=0.702 exceeds AR(1)"** — must be corrected.
   Predictive SMIM (0.489) loses to AR(1) (0.610).
2. **The inference table** (DM test, bootstrap CI, permutation p) — these
   compared modal SMIM with predictive AR(1). Must be recomputed.
3. **The baseline comparison table** — all SMIM numbers were modal.

---

## 2. The Positive Result: CapEx/Revenue (Iteration 5 Path A)

### What we have

CapEx/Revenue (investment per unit sales) with K=3, EWM=12Q, T=3yr training,
operator learning, DIAMOND rolling basis:

| Window | SMIM R² | AR(1) R² | Delta |
|--------|---------|----------|-------|
| W2015 | 0.688 | 0.642 | +0.046 |
| W2016 | 0.633 | 0.615 | +0.019 |
| W2017 | 0.661 | 0.636 | +0.025 |
| W2018 | 0.693 | 0.666 | +0.027 |
| W2019 | 0.688 | 0.626 | +0.062 |
| W2020 | 0.687 | 0.661 | +0.026 |
| W2021 | 0.709 | 0.647 | +0.062 |
| W2022 | 0.768 | 0.745 | +0.023 |
| W2023 | 0.802 | 0.809 | -0.007 |
| W2024 | 0.791 | 0.773 | +0.019 |
| **MEAN** | **0.712** | **0.682** | **+0.030** |

**Genuine predictive R², 9/10 wins.** Script: `run_smim_a1i5.py`.

### Why CapEx/Revenue works where CapEx/Assets doesn't (predictively)

| Property | CapEx/Assets (experiment_a1) | CapEx/Revenue (Iter 5) |
|----------|----------------------------|----------------------|
| Actors | 93 (mixed: US firms, UK, FRED, banks) | 146 (US firms only) |
| Persistence (rho) | 0.47 (mixed, some actors rho>0.8) | 0.28 (homogeneous, low) |
| AR(1) R² | 0.61 (strong baseline) | 0.68 (moderate baseline) |
| Optimal K | 3 (inner CV) | 3 |
| Optimal T_train | 3yr | 3yr |
| Operator learning | +4pp (not enough) | +3pp (tips the balance) |
| **Predictive delta** | **-7.5pp (loses)** | **+3.0pp (wins)** |

The key difference: CapEx/Revenue's lower persistence (rho=0.28) means AR(1) is
weaker (captures less), leaving MORE room for spectral dynamics. The 146 US-only
actors are homogeneous (all have CapEx + Revenue), avoiding the noise from
heterogeneous actor types that hurts the mixed panel.

---

### Second positive: Multi-ratio panel (Path B)

134 actors x 3 signals (CapEx/Rev + Rev/Assets + CapEx/Assets) = 402 columns.
K=5, EWM=8, T=3yr + operator learning:
- SMIM R²=0.740, AR(1)=0.725, delta=+1.5pp, 9/10 wins
- Also genuinely predictive (same script pattern as Path A)
- Script: `run_smim_iter5_paths_bc.py`

### Code verification note

**Confirmed: `run_smim_a1i5.py` uses predictive alpha.** The prediction line
(inside the rolling loop) is:
```python
pred_dm = U @ alpha_pred     # alpha_pred = F @ alpha (before Kalman update)
pred_raw = pred_dm + om.ravel()
preds_rolling.append(pred_raw)
```
This is genuine one-step-ahead prediction. No concurrent information.

### Path fix note

`run_smim_ws_a_nested_cv.py` line 33 was changed from `parent.parent` to
`parent.parent.parent` to correctly resolve the project root from
`scripts/smim/`. The original path gave `scripts/` which is wrong.

---

## 3. Revised Paper Thesis

**Before (invalidated)**: "SMIM achieves R²=0.702, exceeding AR(1) in all windows"

**After (corrected)**:

> We develop a spectral state-space framework with dual regularisation and
> rolling basis update. On quarterly CapEx/Revenue intensity for 146 US firms,
> the framework achieves genuine predictive R² of [nested-CV number], exceeding
> per-actor AR(1) in [X]/8 validation windows (delta=+[Y]pp, permutation
> p=[Z]). The framework provides high-quality spectral reconstruction
> (modal R²=0.70+) across both CapEx/Assets and CapEx/Revenue intensity
> constructions, with model-implied investment gaps predicting subsequent
> CapEx revision (t≈-6.4). The spectral basis rotates at 26°/quarter with
> stable 8-mode dimensionality.

Key changes:
- Lead with CapEx/Revenue PREDICTIVE result (genuine forecast)
- Report modal R² as reconstruction quality (valid, just not forecast)
- Keep rotation finding and economic validation
- Add operator learning to the methodology section
- Be transparent about the correction

---

## 4. Drilldown Experiments to Strengthen the CapEx/Revenue Result

### Session Plan (~4 hours)

### Phase A: Reproduce and Verify (~30 min)

| ID | Task | Purpose |
|----|------|---------|
| A-1 | Re-run `run_smim_a1i5.py` Path A best config to verify R²=0.712 | Reproducibility |
| A-2 | Confirm prediction uses `a_p` (predictive alpha), not `a_f` | No bias |
| A-3 | Run AR(1) with T=3yr AND T=10yr on same 146 actors | Ensure fair comparison |

### Phase B: Nested Cross-Validation (~1 hour)

The current result uses fixed hyperparameters (K=3, EWM=12, T=3yr) selected
from the sweep. For the paper, we need NESTED CV where inner folds select
these parameters without seeing the test data.

| ID | Task | Purpose |
|----|------|---------|
| B-1 | Implement inner CV: last 2yr of each outer-train selects K ∈ {3,5}, EWM ∈ {8,12}, T ∈ {3,5} | Proper protocol |
| B-2 | Freeze holdout: W2023-W2024 get median of CV-selected params | Leakage prevention |
| B-3 | Report nested CV R² and holdout R² | Fair headline numbers |

The nested CV R² will likely be lower than 0.712 (some windows will get
sub-optimal K/EWM), but it's the methodologically correct number.

### Phase C: Statistical Inference (~30 min)

| ID | Task | Purpose |
|----|------|---------|
| C-1 | Block bootstrap (10,000 window resamples) for mean delta-R², CI | Uncertainty |
| C-2 | Permutation test (sign-flip on 8/10 windows) | p-value |
| C-3 | Diebold-Mariano test at actor-quarter level | Per-observation inference |
| C-4 | Sign test (binomial: 9/10 wins) | Non-parametric |

### Phase D: Operator Learning Drilldown (~1 hour)

The operator learning adds ~+3pp. Can we improve it further?

Priority order (cheapest/safest first):

| Priority | ID | Experiment | Est. time | Hypothesis |
|----------|-----|-----------|-----------|-----------|
| 1 | D-5 | Ridge penalty on operator weights | 5 min | Prevent overfitting (wild weights seen in some windows) |
| 2 | D-3 | Increase Nelder-Mead budget (100 → 300 iters) | 10 min | Current budget may be insufficient |
| 3 | D-1 | Add intensity-proximity operator to library | 10 min | Actors with similar CapEx/Rev levels co-move |
| 4 | D-2 | Add sector-distance operator (intra-sector weight) | 10 min | Sector-level dynamics |
| 5 | D-6 | Try differential evolution optimiser | 15 min | Nelder-Mead may be stuck in local optima |
| 6 | D-4 | Re-learn operator each rolling step (most expensive) | 30 min | Structural operator changes over time |

Stop after first experiment that doesn't improve delta — diminishing returns.

### Phase E: Additional Ablation and Robustness (~1 hour)

| ID | Experiment | Purpose |
|----|-----------|---------|
| E-1 | Full ablation ladder for CapEx/Revenue (steps 1-7 as in Paper 1 Table 4) | Decompose R² into components |
| E-2 | Sector composition analysis: which sectors drive SMIM's advantage? | Understand the signal |
| E-3 | N sweep: run on subsets (50, 100, 146 actors) | Does more actors help? |
| E-4 | Basis rotation analysis: rotation speed, K_eff, mode birth/death | Structural characterisation |
| E-5 | CapEx revision prediction (D2) using CapEx/Revenue gaps | Economic validation |
| E-6 | Test on Revenue/Assets (rho=0.09) as second signal | Generalisation |

### Phase F: Paper Update (~1 hour)

| ID | Task |
|----|------|
| F-1 | Rewrite abstract: lead with CapEx/Revenue predictive result, keep dual reg + rotation |
| F-2 | Add Section: "Corrected Evaluation: Predictive vs Modal R²" (transparent about the fix) |
| F-3 | Update all tables with predictive R² numbers |
| F-4 | Add CapEx/Revenue as second intensity construction |
| F-5 | Update inference table with corrected bootstrap/DM results |
| F-6 | Update SSRN with corrected version (add revision note) |

---

## 4. Success Criteria for the Corrected Paper

| Level | Criterion | Status |
|-------|----------|--------|
| BRONZE | CapEx/Revenue nested-CV predictive R² > AR(1) | EXPECTED (fixed config: +3.0pp) |
| SILVER | Bootstrap CI for delta-R² excludes zero | EXPECTED if delta > 2pp |
| GOLD | Permutation p < 0.05 on 8 CV windows | EXPECTED if 7+/8 wins |
| PLATINUM | CapEx/Revenue gaps predict CapEx revision (t > 2) | NEEDS TESTING |

---

## 5. Risks

| Risk | Probability | Mitigation |
|------|-----------|-----------|
| Nested CV selects sub-optimal K/EWM in some windows | High | Report both nested-CV and fixed-config |
| Nested CV R² < AR(1) in some windows | Medium | Win ratio 7+/10 is still significant |
| Bootstrap CI includes zero (delta too small) | Low-Med | Actor-quarter DM test has more power |
| CapEx revision (D2) doesn't replicate | Medium | D2 used modal gaps; try predictive gaps |
| Operator learning overfits to sub-validation | Low | Inner CV + holdout prevents leakage |

---

## 6. CapEx/Revenue Intensity Construction Recipe

For the next session to reproduce without re-reading all scripts:

```python
# Source: EDGAR XBRL
# Tags: PaymentsToAcquirePropertyPlantAndEquipment (capex), Revenues
# Both quarterly, point-in-time via event_date

edgar = pd.read_parquet("data/smim/processed/edgar_balance_sheet.parquet")
capex = edgar[edgar["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"]
revenue = edgar[edgar["tag"] == "Revenues"]

# Quarterly aggregation (last filing per ticker per quarter)
# Merge on (ticker, quarter), compute ratio, cross-sectional percentile rank
# Result: 146 tickers, 74 quarters (2005-2025), values in [0,1]
# Median AR(1) rho = 0.283
# Coverage filter: >50% non-NaN per ticker
```

Full implementation: `run_smim_a1i5.py::build_capex_revenue_panel()`

---

## 7. Key Code References

| File | What | Line(s) |
|------|------|---------|
| `scripts/smim/run_smim_a1i5.py` | Path A CapEx/Revenue DIAMOND+OpLearn | Full script |
| `scripts/smim/run_smim_ws_a_nested_cv.py` | Nested CV (FIXED: now uses predictive alpha) | Line 130 |
| `scripts/smim/run_smim_dd2_v2.py` | dd2_v2 rolling DIAMOND (FIXED: predictive alpha) | Line 112 |
| `scripts/smim/run_smim_iter5_signal_sweep.py` | 13-signal persistence/headroom sweep | Full script |
| `scripts/smim/run_smim_iter5_paths_bc.py` | Paths B (multi-ratio), C (combined) | Full script |
| `scripts/smim/run_smim_diamond_oplearn.py` | DIAMOND+OpLearn on experiment_a1 | Full script |

---

## 7. Key Data References

| File | Contents |
|------|----------|
| `data/smim/intensities/experiment_a1_intensities.parquet` | Original 93-actor CapEx/Assets panel |
| `data/smim/intensities/iter5_dividends_raw.parquet` | Dividend data (161 tickers) |
| `data/smim/intensities/iter5_dividend_intensity.parquet` | Dividend yield rank panel |
| `data/smim/processed/edgar_balance_sheet.parquet` | EDGAR XBRL (7 tags, 765 tickers) |
| `results/metrics/iter5_path_a_best_config.parquet` | CapEx/Revenue best result (per-window) |
| `results/metrics/iter5_diamond_oplearn.parquet` | DIAMOND+OpLearn on experiment_a1 |

---

## 8. Key Document References

| Document | What to read | Why |
|----------|-------------|-----|
| `docs/smim/STATUS.md` | Current project status | Overall context |
| `docs/smim/ITERATION_5_PLAN.md` §12-16 | Iter 5 full results | CapEx/Revenue discovery + signal sweep |
| `docs/smim/ITERATION_3_PLAN.md` §20 | Iter 3 results | Daily frequency negatives |
| `docs/smim/ITERATION_4_PLAN.md` §14 | Iter 4 results | Multi-frequency negatives |
| `docs/smim/paper/smim_paper.tex` | Current paper draft | Sections to update |
| `docs/smim/CLAUDE.md` | SMIM dev context | Standing assumptions, patterns |

---

## 9. What NOT to Do

- Do NOT re-run any experiment with modal (filtered) alpha as the headline metric
- Do NOT compare modal SMIM R² with predictive AR(1) R² anywhere in the paper
- Do NOT change the CapEx/Revenue intensity construction (it's clean as-is)
- Do NOT add complexity to the operator (e.g. Granger edges) before nested CV
  confirms the base result
- Do NOT skip the nested CV — fixed hyperparameters are exploratory, nested CV
  is confirmatory
- Do NOT present experiment_a1 predictive results (they lose to AR(1)) as the
  headline — lead with CapEx/Revenue instead
- Do NOT discard the modal R² entirely — report it as "spectral reconstruction
  quality" (a legitimate and useful metric, just not a forecasting metric)
- Do NOT forget to update the SSRN submission after correction

---

## 10. Session Execution Order

```
Phase A: Verify (30 min)
  A-1: Reproduce CapEx/Revenue R²=0.712
  A-2: Confirm predictive alpha (code inspection)
  A-3: AR(1) baseline at multiple T
  → GATE: reproduction matches? Prediction is genuine?

Phase B: Nested CV (60 min)
  B-1: Build nested CV for CapEx/Revenue
  B-2: Run with inner fold selecting K, EWM, T
  B-3: Report CV + holdout numbers
  → NEW HEADLINE NUMBER for the paper

Phase C: Inference (30 min)
  C-1 through C-4: bootstrap, permutation, DM, sign test
  → CONFIDENCE INTERVALS and P-VALUES

Phase D: Operator drilldown (60 min)
  D-1 through D-6: improve operator learning
  → CAN WE PUSH DELTA ABOVE +5pp?

Phase E: Ablation + robustness (60 min)
  E-1 through E-6: decompose, characterise, validate
  → FULL ABLATION TABLE for the paper

Phase F: Paper update (60 min)
  F-1 through F-6: rewrite, update tables, upload
  → CORRECTED PAPER on SSRN
```
