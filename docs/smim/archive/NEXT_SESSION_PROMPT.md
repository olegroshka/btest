# SMIM Next Research Session: Experiment Iteration 2

> Use this as the opening prompt for the next Claude Code session.
> Copy the full content below into the conversation.

---

## Prompt

```
You are continuing research on the SMIM (Spectral Multi-layer Investment
Misallocation) framework. This is experiment iteration 2 — the first cycle
of 24 experiments is complete with a strong baseline (R²=0.524).

READ THESE FILES AT SESSION START (in this order):
  1. docs/smim/STATUS.md           <- current best config, all findings, what works/doesn't
  2. docs/smim/DRILLDOWN_PLAN.md   <- methodology, performance ladder, drill-down results
  3. docs/smim/EXPERIMENT_RESULTS.md <- detailed per-experiment findings (Sections A3-D6)
  4. docs/smim/CLAUDE.md           <- SMIM dev context, test commands, known deviations
  5. docs/smim/paper/smim_paper.tex <- current paper draft (understand what we claim)

CONTEXT SUMMARY (so you don't need to read everything immediately):

The GOLD+ configuration achieves R²=0.524 on 10-window out-of-sample prediction
of cross-sectional investment intensity (CapEx/Assets rank), beating per-actor
AR(1) at 0.425 by +10pp (DM p=0.001). The pipeline:
  Intensity panel -> EWM demeaning (tau=8Q) -> DMD (K=8 modes) ->
  Kalman filter (spherical R=cI) -> Online Q adaptation (lambda=0.3) -> Gaps

Key strengths: regularisation breakthrough (spherical R), DMD > static operators,
dynamics portability (103-106% zero-shot retention), D2 gap prediction (t=-6.95).

Key weaknesses that THIS SESSION should address:
  1. Emergence (PID synergy, TDA) does NOT fire — CV selects weight=0
  2. Directed operators collapse to PCA (correlation operator is symmetric)
  3. Event alignment fails 0/8 (rank normalisation absorbs spikes)
  4. Return-based intensity gives R²=-0.15 (only CapEx method works)
  5. Kim filter regime switching fails (symmetric initialisation)

YOUR TASK: Plan and execute experiment iteration 2 focused on:

=== RESEARCH DIRECTION 1: MAKING EMERGENCE WORK ===

The PID synergy correction was negligible because:
- T=20 quarters gives unreliable Gaussian MI estimates for K=8 mode pairs
- The quadratic correction alpha_j * alpha_k * S[j,k] is second-order (~0.01)
  vs the linear prediction U @ alpha (~0.1-1.0)

Strategies to explore:
(a) HIGHER-FREQUENCY INTENSITY PROXY: Construct monthly intensity from
    interpolated EDGAR + OHLCV-derived investment proxies. T=60 per 5yr
    window instead of T=20. More data points -> better PID estimation.
    Check: can we compute monthly CapEx proxies from quarterly EDGAR filings?

(b) RICHER EMERGENCE SIGNALS: Instead of PID between abstract spectral modes,
    compute emergence from ECONOMIC observables:
    - Cross-sectional dispersion trajectory (already shown to lead VIX in D6)
    - Sector rotation velocity (rate of change in sector-average intensity)
    - Network clustering coefficient evolution (from intensity correlation graph)
    These are directly interpretable and don't need mode-pair MI estimation.

(c) CONVOLUTION-BASED EMERGENCE: In iteration 1 we brainstormed multi-scale
    temporal convolutions to discover actor co-movement at different timescales
    (4Q/8Q/16Q). This was used for operator construction (Approach A, R²=0.281)
    but NOT for emergence detection. Apply the same convolution features as
    EMERGENCE SIGNALS: actors whose convolution features diverge from their
    cluster are exhibiting emergent behaviour.

(d) TRANSFER ENTROPY AS DIRECTED EDGES: The codebase has a working TE
    estimator (ksg_transfer_entropy in emergence/transfer_entropy.py).
    Use TE between actors' intensity series to build a genuinely DIRECTED
    operator. This would make Schur/Polar/Hermitian differ from PCA.
    Then test: does the directed spectral basis enable meaningful emergence?

=== RESEARCH DIRECTION 2: BREAKING OPERATOR SYMMETRY ===

All 5 static spectral methods gave identical R²=0.339 because the operator
(Pearson cross-correlation) is exactly symmetric. DMD bypasses this, but we
haven't tried building a genuinely asymmetric operator.

Strategies:
(a) TRANSFER ENTROPY OPERATOR: TE(i->j) != TE(j->i) by construction.
    Build the operator from pairwise TE of intensity series.
    Requires T>=30 for reliable TE estimation (we have T=20).

(b) GRANGER CAUSALITY ON INTENSITY (not OHLCV): In A1 we computed Granger
    edges from OHLCV returns (external signals). But B3 showed external
    signals are dispensable. Try Granger on the INTENSITY series itself.
    This produces a directed operator from the primary data.

(c) BEA SUPPLY CHAIN AS DIRECTED EDGES: The BEA I/O tables
    (data/smim/processed/bea_io_tables.parquet) contain directed
    industry-to-industry flows. Map actors to BEA industries and use
    the I/O coefficients as directed edge weights.

(d) NARRATIVE SENTIMENT AS EDGE MODIFIER: GDELT narrative data
    (data/smim/processed/gdelt_narrative.parquet) provides sector-level
    tone and volume. Use sentiment asymmetry between sectors as a
    directed overlay on the correlation operator.

=== RESEARCH DIRECTION 3: ALTERNATIVE INTENSITY CONSTRUCTIONS ===

The CapEx/Assets rank method works (R²=0.52) but return method doesn't
(R²=-0.15). The framework may be capturing CapEx-specific dynamics, not
general investment misallocation.

Strategies:
(a) RAW CAPEX/ASSETS RATIO (no rank): Test if the rank transformation
    helps or hurts. Raw ratios are unbounded and more volatile — they
    might enable event-level detection (D4 currently fails on ranks).

(b) CAPEX GROWTH (year-on-year change): Instead of level, use the
    rate of change in CapEx/Assets. This removes the persistent level
    component and focuses on investment ACCELERATION — closer to the
    misallocation concept.

(c) MULTI-MEASURE PANEL: Stack CapEx/Assets AND Revenue growth AND
    R&D intensity as a multi-dimensional intensity vector per actor.
    The spectral decomposition operates on the stacked panel, capturing
    co-movement across investment dimensions.

=== RESEARCH DIRECTION 4: PIPELINE ARCHITECTURE IMPROVEMENTS ===

(a) REGULARISED KIM FILTER: The symmetric initialisation problem prevents
    M>1 from working. Fix: initialise regimes with K-means on the
    training alpha trajectory, giving asymmetric starting points.

(b) ACTOR-SPECIFIC LOADINGS: Instead of shared U from DMD, allow
    per-actor weights: y_i = mu_i + w_i * (U @ alpha). The scalar w_i
    captures how strongly each actor loads on the common factors.
    Estimate w_i via cross-validation on training data.

(c) ENSEMBLE: Average predictions from multiple spectral methods
    (DMD K=8, DMD K=5, Schur K=8) to reduce variance.

(d) RECURSIVE DMD: Instead of static DMD on the training window,
    use online DMD that updates the spectral basis each quarter.
    This naturally adapts to non-stationary cross-sectional structure.

=== KNOWN DEAD ENDS (DO NOT RETRY) ===

- DO NOT add external signals (OHLCV, FRED, EDGAR) to the operator at L1 depth.
  B3 proved intensity cross-correlation captures all structure. Signals only
  matter if you're building Granger edges, and even then the gain is marginal.
- DO NOT use KimFilter M>1 without fixing the symmetric initialisation first
  (Direction 4a). EM cannot break symmetry from identical starting F/Q.
- DO NOT expect event-level detection (D4) from rank-normalised intensity.
  If you need events, you must change the intensity construction (Direction 3).
- DO NOT try longer T for SMIM. T=5yr is optimal; longer T includes stale
  correlation structure. If you want more data, increase FREQUENCY not window.
- DO NOT compare SMIM to AR(1) at the same T without noting that AR(1)
  benefits from long T while SMIM benefits from short T.

=== KEY CODE ENTRY POINTS ===

Existing implementations to reuse:
- `src/quantdsl_backtest/smim/emergence/transfer_entropy.py`:
  `ksg_transfer_entropy(x, y, k=4, lag=1)` — KSG estimator for TE
- `src/quantdsl_backtest/smim/emergence/pid.py`:
  `compute_synergy_matrix(alpha, gaps, K)` — PID synergy (batched PyTorch)
  `gaussian_mmi_pid(alpha_j, alpha_k, target)` — pairwise PID
- `src/quantdsl_backtest/smim/emergence/tda.py`:
  `sliding_window_persistence()`, `topological_complexity()`
- `src/quantdsl_backtest/smim/spectral/dmd.py`:
  `ExactDMDDecomposer().decompose_snapshots(Y, k)` — current winner
- `src/quantdsl_backtest/smim/dynamics/kim_filter.py`:
  `KimFilter(n_regimes=M).em_estimate()` — needs init fix

Data files for new experiments:
- `data/smim/processed/bea_io_tables.parquet`: 26,852 rows,
  cols: source_industry, target_industry, coefficient, year
- `data/smim/processed/gdelt_narrative.parquet`: weekly,
  cols: week_start, theme_or_actor, article_count, avg_tone, intensity
- `data/smim/processed/edgar_balance_sheet.parquet`: 461K rows,
  cols: ticker, cik, event_date, pub_date, tag, value, form_type
  Tags: CapEx, Assets, Revenue, LongTermDebt, R&D, StockholdersEquity
- `data/smim/processed/fred_signals.parquet`: 72K rows,
  cols: signal_id, event_date, value, pub_date (28 FRED series)

=== EXECUTION APPROACH ===

1. Start by reading the referenced documents to understand the full context.
2. Prioritise directions by estimated impact and feasibility.
3. Create a focused experiment plan (like DRILLDOWN_PLAN.md).
4. Execute experiments systematically, commit at each step.
5. Update STATUS.md and EXPERIMENT_RESULTS.md with findings.
6. If any direction achieves R²>0.55 or makes emergence work,
   update the paper draft.

Success criteria for iteration 2:
  - BRONZE: emergence contributes measurably (any positive delta-R² from
    synergy/TDA/emergence-aware benchmark)
  - SILVER: directed operator outperforms symmetric (Schur > PCA with
    TE or Granger intensity edges)
  - GOLD: R² > 0.55 with emergence active

IMPORTANT: Be systematic. Each experiment should test ONE hypothesis.
Commit results at each step. Update docs. Think before acting.
```

---

## Files the Next Session Needs

| File | Why |
|------|-----|
| `docs/smim/STATUS.md` | Current best config and all findings |
| `docs/smim/DRILLDOWN_PLAN.md` | Performance ladder, drill-down methodology |
| `docs/smim/EXPERIMENT_RESULTS.md` | Detailed findings for all 24 experiments |
| `docs/smim/CLAUDE.md` | SMIM dev context, test commands |
| `docs/smim/paper/smim_paper.tex` | Paper claims (what we need to strengthen) |
| `docs/smim/EXPERIMENT_PLAN.md` | Original experiment programme (for Phase B/C/D specs) |
| `docs/smim/notation.md` | Math notation reference |
| `src/quantdsl_backtest/smim/emergence/` | PID, TE, TDA implementations |
| `src/quantdsl_backtest/smim/spectral/` | All decomposition methods |
| `data/smim/processed/` | GDELT, BEA, EDGAR data files |
| `scripts/run_smim_a1.py` | Current A1 runner with GOLD+ config |
| `scripts/run_smim_drilldown.py` | Drill-down runner template |
