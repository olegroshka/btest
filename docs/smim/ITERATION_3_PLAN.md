# SMIM Iteration 3: Data Frequency, Actor Expansion, and Emergence at Scale

> Created: 2026-04-02
> Status: Plan
> Baseline: SMIM (rolling basis) R^2=0.691 at quarterly frequency
> Goal: Validate framework at daily frequency; test emergence with T=1260

---

## 1. Motivation

Iteration 2 and Drilldown 2 established two facts:

1. **The linear pipeline is at ceiling on quarterly data.** Rolling DMD captures
   all exploitable structure at T=20. Extended DMD, MI operators, and emergence
   diagnostics all fail because T=20 is too few for nonlinear estimation
   (V1: EDMD R^2=0.19 at T/P=0.45; V3: subspace angles noise-dominated at T=8).

2. **The spectral structure evolves continuously.** 8 fixed modes rotating at
   26 deg/quarter, with high-rotation events aligned to macro shocks. Rolling
   basis update captures this (+14.8pp). But at quarterly resolution we cannot
   determine whether rotation is smooth or abrupt, or whether nonlinear mode
   coupling occurs within quarters.

Higher-frequency data resolves both limitations mechanically:
- Daily T=1260 per 5yr window: T/P=28.6 for EDMD (P=44). Well-conditioned.
- KSG MI with n=1259: highly reliable for operator construction.
- PID synergy with T=1260 per mode pair: 63x more data than quarterly.
- TDA with 62+ sliding windows per year: can track topological evolution.

## 2. Data Acquisition Plan

### 2.1 SP500 Daily OHLCV (priority: HIGH)

**What**: Daily open/high/low/close/volume for ~500 US large-cap stocks.
**Source**: Yahoo Finance via existing `yf://` data source infrastructure.
**Script**: `scripts/download_sp500_to_parquet.py` (already exists).
**Ticker list**: `data/sp500_tickers.csv` (already exists).
**Coverage**: 2010-01-01 to 2025-12-31 (15 years, ~3,780 trading days).
**Storage**: `data/equities/sp500_daily/` as parquet.

**Action**: Download, validate completeness (require >90% non-NaN days per stock),
filter to ~200 stocks with complete EDGAR CapEx coverage.

### 2.2 EDGAR Quarterly Filings (already available)

**What**: `data/smim/processed/edgar_balance_sheet.parquet` — 461K rows.
**Tags**: CapEx (`PaymentsToAcquirePropertyPlantAndEquipment`), Assets, Revenue,
R&D, LongTermDebt, StockholdersEquity.
**Coverage**: 2005-2026, quarterly (10-Q) and annual (10-K).
**Usage**: Construct quarterly CapEx/Assets ratio per stock, then step-interpolate
to daily frequency.

### 2.3 FRED Daily Macro (already available)

8 daily series in `data/smim/processed/fred_signals.parquet`:
- BAA10Y (corporate credit spread)
- BAMLH0A0HYM2 (high-yield spread)
- DCOILBRENTEU, DCOILWTICO (oil prices)
- DFF (Fed funds effective rate)
- DTWEXBGS (trade-weighted USD index)
- T10Y2Y (yield curve slope)
- VIXCLS (volatility index)

### 2.4 GDELT Daily Narrative (already available)

9 daily actors in `data/smim/processed/gdelt_narrative_daily.parquet`:
- sector_energy, sector_financials, sector_healthcare, sector_technology, sector_macro
- actor_FED, actor_SEC, actor_BOE, actor_IMF
- Coverage: 2015-2025, daily tone + article count + intensity.

### 2.5 Sector ETFs (to download)

11 GICS sector SPDR ETFs — daily OHLCV via Yahoo Finance:
XLK (tech), XLF (financials), XLE (energy), XLV (healthcare),
XLI (industrials), XLY (consumer disc.), XLP (consumer staples),
XLU (utilities), XLB (materials), XLRE (real estate), XLC (communication).

## 3. Daily Intensity Construction

### Phase 1: Step-Interpolated CapEx (primary — safe, proven cross-sectional structure)

For each SP500 stock with EDGAR coverage:
1. Compute quarterly CapEx/Assets ratio from EDGAR filings
2. Cross-sectional percentile rank per quarter (same as current methodology)
3. Step-interpolate to daily: each trading day inherits the most recent
   quarterly rank (point-in-time correct using `pub_date`, not `event_date`)
4. Result: daily panel with same cross-sectional structure as quarterly,
   but 252 observations per year instead of 4

**Why this works**: The Kalman filter sees daily observations of a slowly-moving
quarterly state. It learns the observation noise model from intra-quarter
variation and applies the spectral dynamics across quarters. This is exactly
what Kalman filters are designed for.

**Validation gate**: Daily AR(1) per stock must be >0 (persistent signal).
Cross-sectional correlation with quarterly CapEx rank must be >0.9 (by
construction, since it IS the same signal step-interpolated).

### Phase 2: Multi-Source Daily Composite (secondary — richer, riskier)

Augment the step-interpolated CapEx panel with daily signals:

| Signal | Per actor? | Construction | Rationale |
|--------|-----------|-------------|-----------|
| CapEx rank (step) | Yes (per stock) | EDGAR step-interpolated | Primary: proven structure |
| 60-day momentum rank | Yes (per stock) | Rolling return rank | Market's investment view |
| Sector ETF flow | Per sector | ETF relative strength | Sector-level allocation |
| GDELT tone | Per sector | Daily normalised tone | Narrative sentiment |
| FRED macro | Global | Standardised level | Regime indicators |

Two approaches to combine:
**(a)** Single intensity per actor: weighted average of signals.
**(b)** Multi-dimensional intensity: stack as (N, T, D) tensor, D=number of signals.
    DMD operates on the stacked panel.

Start with (a), move to (b) if (a) is insufficient.

### Phase 3: Actor Space Expansion

Build incrementally — validate each layer before adding the next:

| Step | Actors | N | Source |
|------|--------|---|--------|
| 3a | SP500 equity (CapEx coverage) | ~150 | EDGAR + YF |
| 3b | + FRED daily macro | ~158 | fred_signals.parquet |
| 3c | + GDELT daily narrative | ~167 | gdelt_narrative_daily.parquet |
| 3d | + Sector ETFs | ~178 | Yahoo Finance |

**Normalisation**: All actors normalised to cross-sectional rank per date,
ensuring comparability across heterogeneous signal types.

## 4. Experiment Programme

### E3-1: Download and Construct Daily Panel

1. Download SP500 daily OHLCV
2. Match tickers to EDGAR CapEx data
3. Construct step-interpolated daily CapEx/Assets rank
4. Validate: persistence, cross-sectional correlation, coverage

**Gate**: Panel has N>=100 stocks with daily CapEx rank, 2010-2025.

### E3-2: SMIM at Daily Frequency (linear baseline)

Run the full SMIM pipeline (static + rolling) on the daily panel:
- T=5yr (1260 trading days) training, 1yr (252 days) test
- DMD K=8, spherical R, F=0.99I, Q=0.5I, online Q adapt
- 10 rolling windows (2015-2024), same as quarterly evaluation

**Metrics**: R^2 at daily frequency, comparison with daily AR(1),
comparison with quarterly R^2=0.691.

**Gate**: Daily SMIM R^2 > 0.10 (positive signal extraction).

### E3-3: Extended DMD at Daily Frequency

Rerun the V1 experiments that failed at quarterly:
- EDMD degree 2 on alpha trajectory: P=44, T=1260, T/P=28.6
- EDMD degree 3: P=164, T/P=7.7
- Diagonal-only quadratic: P=16, T/P=78.8

**Hypothesis**: With T/P >> 1, Koopman estimation becomes stable and
nonlinear mode coupling may be detectable.

**Gate**: EDMD R^2 > SMIM linear R^2 at daily (any positive delta).

### E3-4: MI Operator at Daily Frequency

Replace the Pearson correlation operator with KSG mutual information:
- Pairwise MI(y_i, y_j) for all N(N-1)/2 pairs
- T=1260 points: KSG with k=5 is highly reliable
- MI operator -> Schur decomposition -> SMIM pipeline

**Hypothesis**: MI captures nonlinear dependence that correlation misses.

Also test conditional MI for directed edges:
- CMI(Y_{i,t+1}; Y_{j,t} | Y_{i,t}) for all N(N-1) pairs
- Directed operator -> Schur decomposition (should differ from PCA)

**Gate**: MI operator R^2 > correlation operator R^2 at daily.

### E3-5: Emergence at Daily Frequency

With T=1260, retry all emergence diagnostics:

**(a) PID synergy**: C(8,2)=28 mode pairs, each estimated from 1260 points.
Gaussian MI estimates should be reliable. Test whether synergy matrix
entries exceed 2x bootstrap confidence interval.

**(b) TDA topological complexity**: Sliding window persistence on daily
alpha trajectory. Window=20 days, overlap=10 -> 62 windows/year, 310/5yr.
Track Betti numbers, persistence entropy, Wasserstein distance.
Look for complexity spikes preceding market events.

**(c) Transfer entropy between actors**: Pairwise TE from daily intensity.
N=150 actors, T=1260: 22,350 pairs. Each KSG call on 1259 points.
Build directed TE operator. Test whether directed spectral basis
outperforms symmetric (the quarterly test that failed due to noise).

**(d) Multi-resolution divergence**: Fast (20-day) vs slow (252-day) DMD.
At daily frequency, the fast window has T=20 (same as quarterly!), but
the slow window has T=252 (vs T=4 at quarterly for 1yr). The comparison
is much more stable.

**Gate (BRONZE)**: Any emergence signal gives delta-R^2 > 0.
**Gate (SILVER)**: Directed operator (TE or CMI) outperforms symmetric.

### E3-6: Basis Rotation at Daily Resolution

Track basis rotation at daily granularity:
- Rolling 252-day DMD, 1-day steps -> ~3500 DMD computations
- Subspace angle between successive daily bases
- Does the 26 deg/quarter rotation happen smoothly or in jumps?
- Do rotation spikes predict next-day/next-week forecast errors?

This is the daily-frequency version of V2-3, which was informative but
noise-limited at quarterly resolution.

### E3-7: Combined Best at Daily

Stack all daily innovations that produce positive delta-R^2.
Validate on all 10 windows with DM tests.

## 5. Execution Order and Dependencies

```
E3-1: Data construction        [no dependencies]
  |
  v
E3-2: Linear SMIM at daily     [depends on E3-1]
  |
  +---> E3-3: EDMD at daily           [depends on E3-2]
  +---> E3-4: MI operator at daily    [depends on E3-2]
  +---> E3-6: Basis rotation daily    [depends on E3-2]
  |
  v
E3-5: Emergence at daily       [depends on E3-2, benefits from E3-3/E3-4]
  |
  v
E3-7: Combined best            [depends on E3-3..E3-6]
```

E3-3, E3-4, and E3-6 can run in parallel after E3-2.

## 6. Computational Budget

| Experiment | Estimated time | Bottleneck |
|-----------|---------------|-----------|
| E3-1 Data download + construction | 30-60 min | Yahoo Finance rate limits |
| E3-2 SMIM daily (10 windows) | ~50 min | Kalman filter N=150, T=252 |
| E3-3 EDMD (3 configs x 10 windows) | ~30 min | Koopman SVD |
| E3-4 MI operator (150x150, T=1260) | ~60 min | 11,175 KSG calls |
| E3-5a PID (28 pairs, T=1260) | ~10 min | MI estimation |
| E3-5b TDA (310 windows) | ~30 min | Ripser |
| E3-5c TE (22,350 pairs) | ~120 min | KSG transfer entropy |
| E3-5d Multi-resolution | ~20 min | DMD recomputation |
| E3-6 Daily rotation tracking | ~60 min | ~3500 DMD computations |
| E3-7 Combined | ~30 min | Depends on winners |

**Total: ~7-8 hours** (can be parallelised across E3-3/E3-4/E3-6).

## 7. Success Criteria

| Level | Criterion | What it proves |
|-------|----------|---------------|
| BRONZE | Daily SMIM R^2 > 0.10 | Framework works at daily frequency |
| SILVER | EDMD > linear DMD at daily | Nonlinear mode coupling exists |
| GOLD | MI > correlation operator | Nonlinear dependence matters |
| PLATINUM | Emergence fires (PID > 0 or TE directed > symmetric) | Framework emergence claims validated |
| DIAMOND | Daily rolling SMIM R^2 > quarterly 0.691 | Higher frequency improves prediction |

## 8. Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-----------|-----------|
| Step-interpolated CapEx is trivially constant within quarters | Medium | Add OHLCV momentum as observation noise; test weekly aggregation |
| SP500 OHLCV download incomplete or rate-limited | Low | Fall back to 150 largest stocks; use existing data sources |
| Daily noise dominates signal, R^2 << 0 | Medium | Test at weekly (5-day avg) and biweekly (10-day) as fallback |
| MI operator identical to correlation for near-Gaussian data | Medium | Expected; value is in tails and nonlinear dependence |
| TE computation too slow for N=150 (22K pairs) | Medium | Subsample: random 5000 pairs, or top-100 actors only |
| Emergence still null at daily | Possible | Valid science: publish as definitive negative result |

## 9. Relation to Paper

Daily results are **supplementary/exploratory** — the paper's contribution
stands on the quarterly R^2=0.691 with dual regularisation and rolling basis.

If daily experiments succeed:
- Positive emergence -> new section in paper or follow-up publication
- EDMD/MI improvement -> strengthens nonlinear dynamics discussion
- Daily R^2 > 0.691 -> supplementary material showing frequency generalisation

If daily experiments fail:
- Definitive negative result for emergence -> strengthens the conclusion that
  cross-sectional investment dynamics are fundamentally linear with 8 rotating modes
- "Future work" discussion on what data regime might enable emergence detection

## 10. What We Do NOT Attempt

- Do NOT use pure return-based intensity (proven dead end: R^2=-0.15)
- Do NOT skip step-interpolated CapEx validation (Phase 1 must pass before Phase 2)
- Do NOT add heterogeneous actors before validating equity-only daily panel
- Do NOT attempt sub-daily (intraday) frequency — not meaningful for CapEx dynamics
- Do NOT expect daily R^2 to match quarterly R^2 — different targets, different scales
- Do NOT claim daily results replace the quarterly paper contribution
