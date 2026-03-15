# SMIM Notation Sheet

Every mathematical symbol used in the SMIM framework is defined here.
Use this as the single source of truth when reading code or writing new components.

---

## Quick-Reference: 15 Most Important Symbols

| Symbol (LaTeX) | Python variable | Shape | Meaning |
|---|---|---|---|
| $y_{i,t}$ | `intensities[i, t]` | `(N, T)` float | Normalised investment intensity of actor $i$ at time $t$, always in $[0,1]$ |
| $y^*_{i,t}$ | `gap_result.benchmarks[i, t]` | `(N, T)` float | Regime-conditional benchmark for actor $i$ at time $t$ |
| $\Delta_{i,t}$ | `gap_result.gaps[i, t]` | `(N, T)` float | Investment gap: $\Delta = y - y^*$ (positive = over-invest) |
| $A_t^{(r)}$ | `adj_channel` | sparse `(N, N)` | Directed adjacency matrix for relation channel $r$ at time $t$ |
| $A_t$ | `operator` | sparse `(N, N)` | Aggregate operator: $\sum_r \omega_{r,z_t} A_t^{(r)}$ |
| $U_t$ | `modal_frame.basis` | `(N, K^*)` float | Modal frame: columns are retained eigenmodes |
| $\alpha_t$ | `alpha_t` | `(K^*,)` or `(T, K^*)` | Modal state vector (coordinates in modal basis) |
| $z_t$ | `regime_labels` | `(T,)` int | Discrete regime label at time $t$, $z_t \in \{1,\ldots,M\}$ |
| $K^*$ | `modal_frame.K` | scalar int | Number of retained modes (selected by MDL/compressibility/RG) |
| $N$ | `actors.N` | scalar int | Total number of actors in the registry |
| $M$ | `filtered_state.regimes_selected` | scalar int | Number of regimes selected by MDL/BIC |
| $\psi_t$ | `order_param` | `(d,)` float | Ginzburg-Landau order parameter vector |
| $C_t$ | `criticality` | scalar float | Criticality index (variance × autocorrelation ratio) |
| $S_{jk}$ | `synergy_matrix[j, k]` | `(K^*, K^*)` float | PID synergy between modes $j$ and $k$ |
| $F^{(z)}$ | `transition_matrices[z]` | `(K^*, K^*)` float | State transition matrix for regime $z$ |

---

## 1  Actors and Investment Intensities

### Actors

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $N$ | `actors.N` | `int` | Total number of actors in the system | `smim/data/actor_registry.py` |
| $i, j$ | loop index | `int ∈ [0,N)` | Actor index | throughout |
| $\ell$ | `actor.layer.value` | `int ∈ {0,1,2,3}` | Layer assignment of an actor | `smim/interfaces.py` |
| $\mathcal{L}_\ell$ | `registry.actors_in_layer(Layer(ℓ))` | `list[Actor]` | Set of actors in layer $\ell$ | `smim/data/actor_registry.py` |

### Investment Intensities

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $y_{i,t}$ | `intensities[i, t]` | `(N, T)` float64 | Normalised investment intensity of actor $i$ at time $t$; $y \in [0,1]$ | `smim/data/intensity_mappers.py` |
| $T$ | `intensities.shape[1]` | `int` | Number of time steps in the observation window | throughout |
| $s_t$ | `observations[t, :]` | `(N,)` float | Row of observation matrix at time $t$; equal to $y_{\cdot,t}$ | `smim/dynamics/` |

---

## 2  Directed Multilayer Graph

### Channel Matrices

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $r$ | loop index | `int ∈ {1,…,7}` | Relation channel index (C1–C7 in proposal) | `smim/interfaces.py` |
| $A_t^{(r)}$ | `adj_channel` | sparse CSR `(N, N)` | Directed adjacency matrix for channel $r$; $A[j,i]$ = influence of $i$ on $j$ | `smim/graph/edges/` |
| $\omega_{r,z}$ | `channel_weights[r, z]` | `(R, M)` float | Regime-dependent weight of channel $r$ in regime $z$ | `smim/graph/operators.py` |
| $A_t$ | `operator` | sparse CSR `(N, N)` | Aggregate operator: $A_t = \sum_r \omega_{r,z_t} A_t^{(r)}$ | `smim/graph/operators.py` |
| $R$ | `len(channels)` | `int` | Number of active relation channels | `smim/graph/operators.py` |

### Sparsification

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\rho_{\text{graph}}$ | `target_density` | `float ∈ (0,1)` | Target graph density after L1 sparsification | `smim/graph/edges/` |
| $\mathcal{E}_{\text{ret}}$ | — | `float ∈ (0,1)` | Retained spectral energy fraction after sparsification; must exceed 0.80 (A3) | `smim/graph/operators.py` |

---

## 3  Spectral Decomposition

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $K$ | `modal_frame.basis.shape[1]` before selection | `int` | Total number of candidate modes before selection | `smim/spectral/` |
| $K^*$ | `modal_frame.K` | `int ≤ K` | Number of retained modes; $K^* \leq 0.1N$ required by G3 | `smim/spectral/` |
| $U_t$ | `modal_frame.basis` | `(N, K^*)` float | Modal frame matrix; columns $u_k$ are the retained eigenmodes | `smim/spectral/` |
| $\lambda_k$ | `modal_frame.eigenvalues[k]` | `(K,)` complex | $k$-th eigenvalue of $A_t$; may be complex for directed operators | `smim/spectral/` |
| $\sigma_k$ | `abs(modal_frame.eigenvalues[k])` | `float` | Spectral radius of mode $k$ | `smim/spectral/` |
| $\rho_k$ | stored in `modal_frame.metadata["compressibility"]` | `float ∈ [0,1]` | LZ compressibility of mode $k$'s amplitude time series; must exceed $\rho_{\min}$ | `smim/spectral/` |

### Mode Selection Criteria (Definition 3 — all three must hold)

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $L_{\text{total}}$ | `mdl_total` | `float` | Total MDL description length; mode $k$ retained iff including it reduces $L_{\text{total}}$ | `smim/spectral/` |
| $\rho_{\min}$ | `config.spectral.mode_selection.compressibility_min` | `float` | Minimum compressibility threshold (default 0.10) | `smim/config.py` |
| $\text{RG}(k)$ | `rg_relevance[k]` | `bool` | True if mode $k$'s contribution is stable under layer coarse-graining | `smim/spectral/` |

---

## 4  State-Space Dynamics

### State-Space Model

The model in modal space (Proposal Section 5.3):

$$\alpha_t = F^{(z_t)} \alpha_{t-1} + G^{(z_t)} u_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q^{(z_t)})$$

$$s_t = U_{K^*} \alpha_t + B^{(z_t)} x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R)$$

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\alpha_t$ | `alpha_t` or `filtered_state.alpha_filtered[t,:]` | `(K^*,)` float | Modal state vector: coordinates of the system state in the modal basis | `smim/dynamics/` |
| $\hat{\alpha}_{t\|t}$ | `filtered_state.alpha_filtered[t,:]` | `(T, K^*)` float | Filtered modal state: posterior mean given observations up to $t$ | `smim/dynamics/` |
| $\hat{\alpha}_{t\|t-1}$ | `filtered_state.alpha_predicted[t,:]` | `(T, K^*)` float | Predicted modal state: one-step-ahead prediction | `smim/dynamics/` |
| $F^{(z)}$ | `transition_matrices[z]` | `(K^*, K^*)` float | State transition matrix for regime $z$ | `smim/dynamics/` |
| $G^{(z)}$ | `input_matrices[z]` | `(K^*, d_u)` float | Input/shock matrix for regime $z$ | `smim/dynamics/` |
| $Q^{(z)}$ | `process_noise[z]` | `(K^*, K^*)` float, PSD | Process noise covariance for regime $z$ | `smim/dynamics/` |
| $R$ | `obs_noise` | `(N, N)` float, PSD | Observation noise covariance (shared across regimes) | `smim/dynamics/` |
| $u_t$ | `inputs[t,:]` | `(d_u,)` float | Exogenous input vector at time $t$ | `smim/dynamics/` |
| $B^{(z)}$ | `local_covariate_matrices[z]` | `(N, d_x)` float | Local covariate loading matrix for regime $z$ | `smim/dynamics/` |
| $x_t$ | `local_covariates[t,:]` | `(d_x,)` float | Local covariate vector at time $t$ | `smim/dynamics/` |
| $\eta_t$ | — | `(K^*,)` float, latent | Process noise draw at time $t$ | `smim/dynamics/` |
| $\varepsilon_t$ | — | `(N,)` float, latent | Observation noise draw at time $t$ | `smim/dynamics/` |

### Regime Variables

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $z_t$ | `regime_labels[t]` | `(T,)` int | Discrete regime at time $t$; $z_t \in \{1,\ldots,M\}$ | `smim/dynamics/` |
| $M$ | `filtered_state.regimes_selected` | `int` | Number of regimes, selected by MDL/BIC from candidates $\{1,2,3,4\}$ | `smim/dynamics/` |
| $\mathbf{P}$ | `regime_transition` | `(M, M)` float, row-stochastic | Markov regime transition probability matrix | `smim/dynamics/` |
| $P(z_t\|Y_T)$ | `filtered_state.regime_probs[t,:]` | `(T, M)` float | Smoothed regime probability at time $t$ | `smim/dynamics/` |
| $\ell$ | `filtered_state.log_likelihood` | `float` | Total log-likelihood of observations under estimated model | `smim/dynamics/` |

### Observability

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\mathcal{O}_z$ | `observability[z]` | `(N \cdot K^*, K^*)` float | Observability matrix for regime $z$: $[U_{K^*}; U_{K^*} F^{(z)}; \ldots]$ | `smim/dynamics/` |
| $\kappa(\mathcal{O}_z)$ | `condition_numbers[z]` | `float` | Condition number of $\mathcal{O}_z$; must be $< 10^6$ (A4) | `smim/dynamics/` |

### Phase Transitions

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\psi_t$ | `order_param` | `(d,)` float | Ginzburg-Landau order parameter: low-dimensional projection $\psi_t = \Gamma(\alpha_t)$ | `smim/dynamics/phase_transition.py` |
| $d$ | `config.dynamics.order_parameter_dims` | `int ∈ {1,2,3}` | Dimension of the order parameter vector | `smim/config.py` |
| $\mathcal{F}(\psi; \theta_t)$ | `free_energy` | scalar float | GL free energy landscape: $\sum_k [a_k/2 \cdot \psi_k^2 + b_k/4 \cdot \psi_k^4] + \text{cross} - \text{fields}$ | `smim/dynamics/phase_transition.py` |
| $C_t$ | `criticality` | scalar float | Criticality index: (variance ratio) × (first-order ACF ratio) over consecutive windows | `smim/dynamics/phase_transition.py` |
| $w$ | `config.dynamics.criticality_window_quarters` | `int` | Window size (in quarters) for computing $C_t$ | `smim/config.py` |

---

## 5  Emergence Diagnostics

### Partial Information Decomposition (PID)

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $S_{jk}$ | `synergy_matrix[j, k]` | `(K^*, K^*)` float | PID synergy: information about the target that is only accessible from modes $j$ and $k$ jointly | `smim/emergence/pid.py` |
| $U_{jk}$ | `unique_matrix[j, k]` | `(K^*, K^*)` float | PID unique information: info about target exclusively from mode $j$ (or $k$) | `smim/emergence/pid.py` |

### Transfer Entropy

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\text{TE}_{\ell \to m}$ | `te_profile[ℓ, m]` | `(4, 4)` float | Directed transfer entropy from layer $\ell$ to layer $m$ | `smim/emergence/` |
| $\text{CTE}_{\ell \to m \| \mathbf{c}}$ | `cte_profile[ℓ, m]` | `(4, 4)` float | Conditional TE controlling for intermediary layers $\mathbf{c}$ | `smim/emergence/` |

### Topological Complexity

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $T_t$ | `topological_complexity[t]` | `(T,)` float | Topological complexity: total persistence of the sliding-window persistence diagram of $\alpha_t$ | `smim/emergence/` |
| $W_p$ | `wasserstein_dist[t]` | `float` | Wasserstein $p$-distance between consecutive persistence diagrams | `smim/emergence/` |

---

## 6  Benchmarks and Gaps

### Benchmarks

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $y^*_{\text{pred},i,t}$ | `gap_result.benchmarks` (class=PREDICTIVE) | `(N, T)` float | Predictive benchmark: $\mathbb{E}[y_{i,t} \| \mathcal{F}_{t-1}]$, one-step-ahead conditional mean | `smim/gaps/` |
| $y^*_{\text{str},i,t}$ | `gap_result.benchmarks` (class=STRUCTURAL) | `(N, T)` float | Structural benchmark: counterfactual $y$ from stable channels only, removing distortionary channels | `smim/gaps/` |
| $y^*_{\text{mode},i,t}$ | `gap_result.benchmarks` (class=MODAL) | `(N, T)` float | Modal benchmark: per-mode attribution of gaps | `smim/gaps/` |
| $y^*_{\text{em},i,t}$ | `gap_result.benchmarks` (class=EMERGENCE\_AWARE) | `(N, T)` float | Emergence-aware benchmark: adds synergistic corrections and $C_t$ term | `smim/gaps/` |

### Gaps

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $\Delta_{i,t}$ | `gap_result.gaps[i, t]` | `(N, T)` float | Investment gap: $\Delta_{i,t} = y_{i,t} - y^*_{i,t}$; positive = over-invest, negative = under-invest | `smim/gaps/` |
| $\Delta^{\text{mode}}_{k,i,t}$ | `gap_result.modal_attribution[i, t, k]` | `(N, T, K^*)` float | Per-mode contribution to the gap from mode $k$ | `smim/gaps/` |

---

## 7  Falsification and Validation

| Symbol | Python | Shape/Type | Definition | Module |
|---|---|---|---|---|
| $B$ | `n_null_instances` | `int ≥ 100` | Number of null model instances per falsification test; minimum 100 (proposal requirement) | `smim/validation/` |
| $p$ | `falsification_result.p_value` | `float ∈ [0,1]` | Empirical p-value: fraction of null instances exceeding the observed statistic | `smim/validation/` |
| $\hat{\theta}_{\text{obs}}$ | `falsification_result.observed_statistic` | `float` | Test statistic computed on real data | `smim/validation/` |
| $\{\hat{\theta}^{(b)}\}$ | `falsification_result.null_distribution` | `(B,)` float | Distribution of test statistics under the null hypothesis | `smim/validation/` |

---

## 8  Standing Assumptions (from smim/CLAUDE.md)

| Label | Statement | Checked by |
|---|---|---|
| A1 | Point-in-time: $\text{pub\_date}(x) \leq t$ for all data $x$ used at time $t$ | `smim/data/pit_store.py` |
| A2 | Typed comparability: normalisation is per `ActorType` via `InvestmentIntensityMapper` | `smim/data/intensity_mappers.py` |
| A3 | Sparse propagation: $\mathcal{E}_{\text{ret}} > 0.80$ after sparsification | `smim/graph/operators.py` |
| A4 | Stable modes: eigenmode rank correlation $> 0.5$ across $\geq 80\%$ of rolling windows | `smim/spectral/` |
| A5 | Regime persistence: average regime duration $> 8$ quarters | `smim/dynamics/` |
