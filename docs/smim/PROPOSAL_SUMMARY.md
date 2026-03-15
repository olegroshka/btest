# SMIM Research Proposal — Condensed Reference

This is the condensed reference for the mathematical architecture. Read this when you
need to understand WHY a component exists or WHAT equation it implements. For full
derivations, see research_proposal_v5.tex.

## Core Object

Investment gap: Δ_{i,t} = y_{i,t} - y*_{i,t}
- y_{i,t}: actor-specific investment intensity (normalised to [0,1])
- y*_{i,t}: regime-conditional benchmark
- Δ > 0: over-investment, Δ < 0: under-investment

## Architecture Pipeline

```
Actors → Investment Intensities → Directed Multilayer Graph → Spectral Decomposition
→ Modal State-Space (regime-switching) → Benchmarks → Gaps → Emergence Diagnostics
```

## Four-Layer Hierarchy

- Layer 0: Exogenous (global shocks, commodity, geopolitical)
- Layer 1: Upstream (central banks, regulators, think tanks, IMF, OECD)
- Layer 2: Transmission (large firms, banks, sector leaders)
- Layer 3: Downstream (SMEs, municipalities, households, retail)

## Seven Relation Channels (C1–C7)

C1: Regulatory, C2: Financial, C3: Fiscal, C4: Narrative,
C5: Supply-chain, C6: Imitation, C7: Market-implied

## Aggregate Operator

A_t = Σ_r ω_{r,z_t} A_t^{(r)}   — regime-dependent channel weights

## Spectral Decomposition Options (compared in WP3)

1. Schur: A = QTQ^H (Q unitary, T upper triangular)
2. Directed variation: minimise dispersion subject to DV ordering
3. Polar: A = UP (U orthogonal, P symmetric PSD)
4. Hermitian dilation: H(A) = [[0, A], [A^T, 0]], eigenvalues ±σ_k

## State-Space Model (modal space)

State:  α_t = F^{(z_t)} α_{t-1} + G^{(z_t)} u_t + η_t,   η ~ N(0, Q^{(z_t)})
Obs:    s_t = U_{K*} α_t + B^{(z_t)} x_t + ε_t,           ε ~ N(0, R)
Regime: z_t ∈ {1,...,M}, Markov with transition matrix P

Filtering: Kim filter (M² parallel Kalman filters, moment-matching collapse).
Estimation: EM algorithm. Model selection: MDL/BIC.

## Mode Selection (three criteria, ALL must hold)

1. MDL: including mode reduces total description length
2. Compressibility: LZ compressibility ρ_k > ρ_min
3. RG relevance: mode contribution stable under layer coarse-graining

## Phase Transitions

Order parameter: ψ_t = Γ(α_t) ∈ R^d
Free energy: F(ψ; θ_t) = Σ_k [a_k/2 ψ_k² + b_k/4 ψ_k⁴] + cross terms - fields
Gradient flow: dψ = -∇F dt + σ dW
Criticality index: C_t = (Var ratio) × (ACF1 ratio) over consecutive windows

## Benchmarks (5 classes — always label which one)

1. Predictive: y*_{pred} = E[y_{i,t} | F_{t-1}]
2. Structural: y*_{str} from stable channels, excluding distortions
3. Modal: per-mode decomposition of gap
4. Equilibrium: Nash/welfare optimal (Phase III only)
5. Emergence-aware: adds synergistic corrections + C_t + T_t

## Emergence Diagnostics

- PID synergy: S_{jk} from partial information decomposition of mode pairs
- Transfer entropy: TE_{ℓ→m} directed information flow between layers
- Causal emergence: EI_macro > EI_micro means macro description is superior
- Information geometry: Fisher distance spikes = distributional regime shifts
- Topological complexity: T_t from persistent homology on sliding windows of α_t

## Falsification Tests (7 tests, each with B ≥ 100 null instances)

1. Shuffled-edge placebos (degree-preserving rewiring)
2. Lag-destroyed histories (temporal permutation)
3. Randomised actor-type assignments
4. Frozen-regime baselines (single regime)
5. Symmetric-operator baselines
6. Block-preserving rewiring (within-layer)
7. No-network dynamic factor baselines

## Minimum Viable Specification (Appendix J — all 11 must be met)

1. One domain, one geography
2. Actor ontology with ≥2 types per layer
3. Investment-intensity normalisation defined + sensitivity-tested
4. ≥2 edge families separately estimated
5. One directed operator + one DMD baseline compared
6. One no-regime + one switching-regime SSM estimated
7. One predictive + one structural benchmark reported
8. ≥1 emergence diagnostic computed
9. Rolling OOS + ≥1 placebo + ≥1 event study
10. All results labelled with benchmark class
11. Point-in-time discipline verified
