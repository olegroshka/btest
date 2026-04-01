# SMIM Task Registry

Status key: ☐ Not started | 🔨 In progress | ✅ Done | ⏭ Skipped

Last updated: 2026-03-19

## M0.0 — Repository Bootstrap
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M0.0-T1 | Create `smim/` package skeleton with `__init__.py` for all subpackages | ✅ | |
| M0.0-T2 | Create `smim/interfaces.py` with all Protocol definitions | ✅ | |
| M0.0-T3 | Create `smim/config.py` with Pydantic config models + sample YAML | ✅ | experiments/mvp_energy_us_uk.yaml |
| M0.0-T4 | Create stub test files mirroring smim/ structure | ✅ | |

## M0.1–M0.5 — WP0: Formal Scoping
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M0.1-T1 | Data dictionary / notation sheet (docs/smim/notation.md) | ✅ | |
| M0.2-T1 | Actor taxonomy spreadsheet for chosen scope | ✅ | docs/smim/actor_taxonomy.md |
| M0.3-T1 | InvestmentIntensityMapper implementations per actor type | ✅ | |
| M0.4-T1 | Benchmark specs (formal definitions in docs/) | ✅ | docs/smim/benchmark_specs.md |
| M0.5-T1 | Scope selection report | ✅ | docs/smim/scope_selection.md |

## M1.1–M1.4 — WP1: Data Ingestion
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M1.1-T1 | Data source audit document | ✅ | docs/smim/data_source_audit.md |
| M1.2-T1 | Document btest's existing FRED adapter pattern → ADAPTER_GUIDE.md | ✅ | docs/smim/ADAPTER_GUIDE.md |
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
| M2.1-T1 | Define EdgeEstimator protocol + base class + factory | ✅ | graph/edges/base.py |
| M2.1-T2 | GrangerEdgeEstimator (VAR + BIC lag selection) | ✅ | graph/edges/granger.py |
| M2.1-T3 | NarrativeEdgeEstimator (TF-IDF / embedding similarity) | ✅ | graph/edges/narrative.py |
| M2.1-T4 | SupplyChainEdgeEstimator (BEA I/O coefficients) | ✅ | graph/edges/supply_chain.py |
| M2.1-T5 | AggregateOperator construction (weighted combination) | ✅ | graph/operators.py |
| M2.1-T6 | YAML config section for edge estimation | ✅ | config.py |
| M2.1-T7 | Integration test: edges → operator on minimal actor set | ✅ | acceptance tests AT-1 |
| M2.1-T8 | Sparse storage: save/load edge matrices | ✅ | |
| M2.2-T1 | Degree-preserving random rewiring generator | ✅ | graph/null_model.py |
| M2.2-T2 | Null-model comparison pipeline (B=100, p-value computation) | ✅ | |
| M2.2-T3 | L1 sparsification with density control | ✅ | graph/sparsification.py |
| M2.3-T1 | Dynamic factor model baseline (Stock-Watson) | ✅ | |
| M2.3-T2 | VAR baseline (BIC-selected lag) | ✅ | |
| M2.4-T1 | Aggregate operator + sparse format storage | ✅ | |
| M2.5-T1 | Ablation experiments runner | ✅ | |
| M2.5-T2 | Sensitivity report generation | ✅ | |

## M3.1–M3.4 — WP3: Modal Representation
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M3.1-T1 | SpectralDecomposer protocol + factory | ✅ | spectral/base.py |
| M3.1-T2 | SchurDecomposer | ✅ | spectral/schur.py |
| M3.1-T3 | PolarDecomposer | ✅ | spectral/polar.py |
| M3.1-T4 | HermitianDilationDecomposer | ✅ | spectral/hermitian.py |
| M3.1-T5 | DirectedVariationDecomposer (Stiefel optimisation) | ✅ | spectral/dv_basis.py |
| M3.1-T6 | Decomposition comparison harness | ✅ | spectral/comparison.py |
| M3.2-T1 | Exact DMD implementation | ✅ | spectral/dmd.py |
| M3.2-T2 | Extended DMD with dictionary lifting | ✅ | spectral/dmd.py |
| M3.3-T1 | MDL criterion for mode selection | ✅ | spectral/mode_selection.py |
| M3.3-T2 | LZ compressibility computation | ✅ | spectral/mode_selection.py |
| M3.3-T3 | RG relevance scorer (layer coarse-graining) | ✅ | spectral/mode_selection.py |
| M3.4-T1 | OOS reconstruction stability evaluation | ✅ | spectral/oos_evaluation.py |
| M3.4-T2 | Comparison report generation | ✅ | spectral/modal_report.py |

## M4.1–M4.8 — WP4: Dynamics & Emergence
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M4.1-T1 | Kalman filter (single regime, linear Gaussian) | ✅ | dynamics/kalman.py |
| M4.1-T2 | EM estimation for single-regime SSM | ✅ | dynamics/kalman.py |
| M4.1-T3 | OOS prediction evaluation | ✅ | dynamics/evaluation.py |
| M4.2-T1 | Kim filter core loop (M² parallel filters + collapse) | ✅ | dynamics/kim_filter.py |
| M4.2-T2 | EM for regime-switching (forward-backward smoother) | ✅ | dynamics/kim_filter.py |
| M4.2-T3 | MDL/BIC regime count selection | ✅ | dynamics/model_selection.py |
| M4.2-T4 | Regime transition matrix estimation | ✅ | dynamics/kim_filter.py |
| M4.3-T1 | Observability matrix computation + condition number | ✅ | dynamics/observability.py |
| M4.4-T1 | Predictive benchmark computation | ✅ | gaps/predictive.py |
| M4.4-T2 | Modal benchmark computation | ✅ | gaps/modal.py |
| M4.5-T1 | Order parameter extraction from modal states | ✅ | dynamics/phase_transition.py |
| M4.5-T2 | Ginzburg-Landau landscape fitting | ✅ | dynamics/phase_transition.py |
| M4.5-T3 | Criticality index C_t computation | ✅ | dynamics/phase_transition.py |
| M4.6-T1 | PID synergy computation (Gaussian MMI) | ✅ | emergence/pid.py |
| M4.6-T2 | PID bootstrap confidence intervals | ✅ | emergence/pid.py |
| M4.6-T3 | Transfer entropy (KSG estimator) | ✅ | emergence/transfer_entropy.py |
| M4.6-T4 | Conditional TE (controlling for intermediate layers) | ✅ | emergence/transfer_entropy.py |
| M4.7-T1 | Sliding-window point cloud + VR complex | ✅ | emergence/tda.py |
| M4.7-T2 | Topological complexity T_t + Wasserstein distances | ✅ | emergence/tda.py |
| M4.8-T1 | Emergence-aware benchmark (synergistic corrections) | ✅ | gaps/emergence_aware.py |
| M4.8-T2 | Incremental R² evaluation (emergence vs baseline) | ✅ | gaps/emergence_evaluation.py |

## M5.1–M5.6 — WP5: Backtesting & Falsification
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M5.1-T1 | Rolling OOS evaluation harness (expanding + rolling windows) | ✅ | |
| M5.1-T2 | Metric computation (R², RMSE, Spearman ρ, NDCG) | ✅ | validation/metrics.py |
| M5.1-T3 | Bridge signals: GapSignal wrapping Δ_{i,t} for btest engine | ✅ | signals/ |
| M5.2-T1 | Shuffled-edge placebo test (B=100) | ✅ | |
| M5.2-T2 | Lag-destroyed + randomised-type tests | ✅ | |
| M5.2-T3 | Frozen-regime + symmetric-operator tests | ✅ | |
| M5.2-T4 | Block-preserving + no-network DFM tests | ✅ | |
| M5.3-T1 | Event study framework (pre/post shock windows) | ✅ | |
| M5.3-T2 | Event study for ≥5 historical shocks | ✅ | |
| M5.4-T1 | Gap persistence / mean-reversion analysis | ✅ | |
| M5.5-T1 | Model comparison (Diebold-Mariano / MCS tests) | ✅ | |
| M5.6-T1 | Evidence synthesis report generator | ✅ | |

## M6.1–M6.4 — WP6: Extensions
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| M6.1-T1 | Channel decomposition (stable vs distortionary) | ✅ | |
| M6.1-T2 | Structural benchmark from stable parameters | ✅ | gaps/structural.py |
| M6.2-T1 | Actor-level state-space model (N'≤50 subset) | ✅ | dynamics/actor_level.py |
| M6.3-T1 | Cross-sector/geography transfer experiment | ✅ | |
| M6.4-T1 | Extension evaluation report | ✅ | |

## Acceptance Tests (AT series)
| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| AT-0 | Acceptance test infrastructure, conftest, synthetic generators | ✅ | tests/acceptance/smim/ |
| AT-1 | 20 graph construction tests (Granger, narrative, operator, sparsification, null-model) | ✅ | test_graph_construction.py |
| AT-2 | 9 Schur decomposition tests | ✅ | test_spectral_schur.py |
| AT-3 | 14 polar decomposition + Hermitian dilation tests | ✅ | test_spectral_polar_hermitian.py |
| AT-4 | 11 directed variation + DMD tests | ✅ | test_spectral_dv_dmd.py |
| AT-5 | 9 mode selection tests (MDL, LZ, RG) | ✅ | test_mode_selection.py |
| AT-6 | 14 Kalman filter + EM tests | ✅ | test_kalman_em.py |
| AT-7 | Kim filter tests | ⏭ | No dedicated file; Kim filter covered via dynamics acceptance |
| AT-8 | 11 observability + phase transition tests | ✅ | test_observability_phase.py |
| AT-9 | 6 PID tests | ✅ | test_pid.py |
| AT-10 | 6 transfer entropy tests | ✅ | test_transfer_entropy.py |
| AT-11 | 7 TDA tests | ✅ | test_tda.py |
| AT-12 | 11 benchmark + pipeline sanity tests | ✅ | test_benchmarks_pipeline.py |

**Current acceptance gate: 119/119 passed ✅ — STATUS: READY FOR EXPERIMENTS**
