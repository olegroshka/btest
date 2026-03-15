# SMIM Task Registry

Status key: ☐ Not started | 🔨 In progress | ✅ Done | ⏭ Skipped

## M0.0 — Repository Bootstrap
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M0.0-T1 | Create `smim/` package skeleton with `__init__.py` for all subpackages | ☐ | |
| M0.0-T2 | Create `smim/interfaces.py` with all Protocol definitions | ☐ | |
| M0.0-T3 | Create `smim/config.py` with Pydantic config models + sample YAML | ☐ | |
| M0.0-T4 | Create stub test files mirroring smim/ structure | ☐ | |

## M0.1–M0.5 — WP0: Formal Scoping
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M0.1-T1 | Data dictionary / notation sheet (docs/smim/notation.md) | ☐ | |
| M0.2-T1 | Actor taxonomy spreadsheet for chosen scope | ☐ | |
| M0.3-T1 | InvestmentIntensityMapper implementations per actor type | ☐ | |
| M0.4-T1 | Benchmark specs (formal definitions in docs/) | ☐ | |
| M0.5-T1 | Scope selection report | ☐ | |

## M1.1–M1.4 — WP1: Data Ingestion
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M1.1-T1 | Data source audit document | ☐ | |
| M1.2-T1 | Document btest's existing FRED adapter pattern → ADAPTER_GUIDE.md | ☐ | |
| M1.2-T2 | Extend FRED adapter with ALFRED vintage retrieval | ☐ | |
| M1.2-T3 | Implement EdgarAdapter (SEC XBRL) | ☐ | |
| M1.2-T4 | Implement GdeltAdapter (narrative intensity) | ☐ | |
| M1.2-T5 | Implement ImfSdmxAdapter | ☐ | |
| M1.2-T6 | Implement BeaIoAdapter (supply-chain I/O tables) | ☐ | |
| M1.2-T7 | Implement OecdAdapter | ☐ | |
| M1.3-T1 | Design PIT store schema (dual-timestamp Parquet) | ☐ | |
| M1.3-T2 | Implement PIT store read/write with pub_date filtering | ☐ | |
| M1.3-T3 | Implement leak-detection validation script | ☐ | |
| M1.4-T1 | Implement automated quality checks (missingness, range, cross-source) | ☐ | |
| M1.4-T2 | Generate coverage report from quality check results | ☐ | |

## M2.1–M2.5 — WP2: Graph Construction
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M2.1-T1 | Define EdgeEstimator protocol + base class + factory | ☐ | |
| M2.1-T2 | GrangerEdgeEstimator (VAR + BIC lag selection) | ☐ | |
| M2.1-T3 | NarrativeEdgeEstimator (TF-IDF / embedding similarity) | ☐ | |
| M2.1-T4 | SupplyChainEdgeEstimator (BEA I/O coefficients) | ☐ | |
| M2.1-T5 | AggregateOperator construction (weighted combination) | ☐ | |
| M2.1-T6 | YAML config section for edge estimation | ☐ | |
| M2.1-T7 | Integration test: edges → operator on minimal actor set | ☐ | |
| M2.1-T8 | Sparse storage: save/load edge matrices | ☐ | |
| M2.2-T1 | Degree-preserving random rewiring generator | ☐ | |
| M2.2-T2 | Null-model comparison pipeline (B=100, p-value computation) | ☐ | |
| M2.2-T3 | L1 sparsification with density control | ☐ | |
| M2.3-T1 | Dynamic factor model baseline (Stock-Watson) | ☐ | |
| M2.3-T2 | VAR baseline (BIC-selected lag) | ☐ | |
| M2.4-T1 | Aggregate operator + sparse format storage | ☐ | |
| M2.5-T1 | Ablation experiments runner | ☐ | |
| M2.5-T2 | Sensitivity report generation | ☐ | |

## M3.1–M3.4 — WP3: Modal Representation
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M3.1-T1 | SpectralDecomposer protocol + factory | ☐ | |
| M3.1-T2 | SchurDecomposer | ☐ | |
| M3.1-T3 | PolarDecomposer | ☐ | |
| M3.1-T4 | HermitianDilationDecomposer | ☐ | |
| M3.1-T5 | DirectedVariationDecomposer (Stiefel optimisation) | ☐ | |
| M3.1-T6 | Decomposition comparison harness | ☐ | |
| M3.2-T1 | Exact DMD implementation | ☐ | |
| M3.2-T2 | Extended DMD with dictionary lifting | ☐ | |
| M3.3-T1 | MDL criterion for mode selection | ☐ | |
| M3.3-T2 | LZ compressibility computation | ☐ | |
| M3.3-T3 | RG relevance scorer (layer coarse-graining) | ☐ | |
| M3.4-T1 | OOS reconstruction stability evaluation | ☐ | |
| M3.4-T2 | Comparison report generation | ☐ | |

## M4.1–M4.8 — WP4: Dynamics & Emergence
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M4.1-T1 | Kalman filter (single regime, linear Gaussian) | ☐ | |
| M4.1-T2 | EM estimation for single-regime SSM | ☐ | |
| M4.1-T3 | OOS prediction evaluation | ☐ | |
| M4.2-T1 | Kim filter core loop (M² parallel filters + collapse) | ☐ | |
| M4.2-T2 | EM for regime-switching (forward-backward smoother) | ☐ | |
| M4.2-T3 | MDL/BIC regime count selection | ☐ | |
| M4.2-T4 | Regime transition matrix estimation | ☐ | |
| M4.3-T1 | Observability matrix computation + condition number | ☐ | |
| M4.4-T1 | Predictive benchmark computation | ☐ | |
| M4.4-T2 | Modal benchmark computation | ☐ | |
| M4.5-T1 | Order parameter extraction from modal states | ☐ | |
| M4.5-T2 | Ginzburg-Landau landscape fitting | ☐ | |
| M4.5-T3 | Criticality index C_t computation | ☐ | |
| M4.6-T1 | PID synergy computation (Gaussian MMI) | ☐ | |
| M4.6-T2 | PID bootstrap confidence intervals | ☐ | |
| M4.6-T3 | Transfer entropy (KSG estimator) | ☐ | |
| M4.6-T4 | Conditional TE (controlling for intermediate layers) | ☐ | |
| M4.7-T1 | Sliding-window point cloud + VR complex | ☐ | |
| M4.7-T2 | Topological complexity T_t + Wasserstein distances | ☐ | |
| M4.8-T1 | Emergence-aware benchmark (synergistic corrections) | ☐ | |
| M4.8-T2 | Incremental R² evaluation (emergence vs baseline) | ☐ | |

## M5.1–M5.6 — WP5: Backtesting & Falsification
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M5.1-T1 | Rolling OOS evaluation harness (expanding + rolling windows) | ☐ | |
| M5.1-T2 | Metric computation (R², RMSE, Spearman ρ, NDCG) | ☐ | |
| M5.1-T3 | Bridge signals: GapSignal wrapping Δ_{i,t} for btest engine | ☐ | |
| M5.2-T1 | Shuffled-edge placebo test (B=100) | ☐ | |
| M5.2-T2 | Lag-destroyed + randomised-type tests | ☐ | |
| M5.2-T3 | Frozen-regime + symmetric-operator tests | ☐ | |
| M5.2-T4 | Block-preserving + no-network DFM tests | ☐ | |
| M5.3-T1 | Event study framework (pre/post shock windows) | ☐ | |
| M5.3-T2 | Event study for ≥5 historical shocks | ☐ | |
| M5.4-T1 | Gap persistence / mean-reversion analysis | ☐ | |
| M5.5-T1 | Model comparison (Diebold-Mariano / MCS tests) | ☐ | |
| M5.6-T1 | Evidence synthesis report generator | ☐ | |

## M6.1–M6.4 — WP6: Extensions
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M6.1-T1 | Channel decomposition (stable vs distortionary) | ☐ | |
| M6.1-T2 | Structural benchmark from stable parameters | ☐ | |
| M6.2-T1 | Actor-level state-space model (N'≤50 subset) | ☐ | |
| M6.3-T1 | Cross-sector/geography transfer experiment | ☐ | |
| M6.4-T1 | Extension evaluation report | ☐ | |
