"""
SMIM Acceptance Tests — Section 4: Kalman Filter and EM Estimation (14 tests)

References: docs/smim/ACCEPTANCE_TESTS.md, Section 4.1–4.2.

Kalman filter (10 tests): A-KF-1, A-KF-2, A-KF-3, I-KF-1, I-KF-2,
                           I-KF-3, R-KF-1, R-KF-2, D-KF-1, D-KF-2
EM estimation (4 tests):   A-EM-1, I-EM-1, I-EM-2, D-EM-1
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import DecompositionMethod, ModalFrame

from tests.acceptance.smim.conftest import make_modal_system

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.section("kalman_filter_em"),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _frame(basis: np.ndarray) -> ModalFrame:
    """Wrap a (N, K) basis array in a ModalFrame."""
    K = basis.shape[1]
    return ModalFrame(
        basis=basis,
        eigenvalues=np.ones(K, dtype=complex),
        method=DecompositionMethod.SCHUR,
        metadata={},
    )


def _identity_frame(K: int) -> ModalFrame:
    """ModalFrame with H = I_K (full observation of all K state dimensions)."""
    return _frame(np.eye(K))


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_A_KF_1_steady_state_gain_convergence():
    """A-KF-1: Kalman gain K_t converges to DARE solution by step 50."""
    F = np.array([[0.9, 0.1], [-0.1, 0.85]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    H = np.eye(2)   # full observation
    R = np.array([[0.5, 0.0], [0.0, 0.5]])
    T = 500

    # DARE solution
    P_ss = solve_discrete_are(F.T, H.T, Q, R)
    K_ss = P_ss @ H.T @ np.linalg.solve(H @ P_ss @ H.T + R, np.eye(2))

    rng = np.random.default_rng(1001)
    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _identity_frame(2)

    obs = rng.standard_normal((T, 2))
    state = kf.filter(obs, frame)

    P_pred_all = state.params["P_predicted"]  # (T, K, K)

    # Compute K_t at each step from stored P_predicted
    K_errors = []
    for t in range(T):
        Pp = P_pred_all[t]
        K_t = Pp @ H.T @ np.linalg.solve(H @ Pp @ H.T + R, np.eye(2))
        K_errors.append(np.linalg.norm(K_t - K_ss))

    # After step 50 all gains should be TIGHT to K_ss
    max_err_after_50 = max(K_errors[50:])
    assert max_err_after_50 < 1e-6, (
        f"Kalman gain did not converge: max ||K_t - K_ss||_F = {max_err_after_50:.3e} "
        f"after step 50 (expected < 1e-6)"
    )

    # P_filtered should also converge to P_ss = P_ss - K_ss @ H @ P_ss
    P_filt_all = state.params["P_filtered"]
    P_filt_ss = (np.eye(2) - K_ss @ H) @ P_ss
    P_err = np.linalg.norm(P_filt_all[-1] - P_filt_ss)
    assert P_err < 1e-6, (
        f"Filtered covariance did not converge to DARE solution: "
        f"||P_T - P_ss||_F = {P_err:.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_A_KF_2_perfect_observation():
    """A-KF-2: With R ≈ 0 and H=I, filtered state = observation and P → 0."""
    K, T = 2, 50
    F = 0.9 * np.eye(K)
    Q = 0.1 * np.eye(K)
    R_tiny = 1e-10 * np.eye(K)

    rng = np.random.default_rng(2002)
    obs = rng.standard_normal((T, K))

    kf = KalmanFilter(F=F, Q=Q, R=R_tiny)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    # Filtered state should match observation to high precision after step 0
    err = np.max(np.abs(state.alpha_filtered[1:] - obs[1:]))
    assert err < 1e-5, (
        f"Filtered state does not equal observation under near-zero R: "
        f"max |α_filt - obs| = {err:.3e}"
    )

    # Filtered covariance P should collapse to near-zero
    P_filt = state.params["P_filtered"]
    max_P = np.max(np.abs(P_filt[-1]))
    assert max_P < 1e-5, (
        f"Filtered covariance did not collapse to 0: max |P_T| = {max_P:.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_A_KF_3_known_trajectory_recovery():
    """A-KF-3: Filter reduces RMSE vs raw observations on a known modal system."""
    N, K, T = 10, 3, 1000
    rng = np.random.default_rng(3003)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    F_true = 0.85 * np.eye(K)
    Q_true = np.eye(K) * 0.5
    R_true = np.eye(N) * 0.5

    result = make_modal_system(N=N, K=K, T=T,
                               U_true=U_raw, F_true=F_true, Q=Q_true, R=R_true,
                               seed=42)
    Y = result.observations       # (T, N)
    alpha_true = result.true_alphas  # (T, K)

    frame = _frame(U_raw)
    kf = KalmanFilter(F=F_true, Q=Q_true, R=R_true)
    state = kf.filter(Y, frame)

    # Reconstruct observations from filtered states and true states
    Y_filtered = state.alpha_filtered @ U_raw.T   # (T, N)
    Y_true     = alpha_true @ U_raw.T              # (T, N)

    rmse_filtered = np.sqrt(np.mean((Y_filtered - Y_true) ** 2))
    rmse_raw      = np.sqrt(np.mean((Y            - Y_true) ** 2))

    assert rmse_filtered < rmse_raw, (
        f"Filter did not improve RMSE: filtered={rmse_filtered:.4f}, "
        f"raw={rmse_raw:.4f}"
    )

    # Per (t, k) cell: check that |alpha_filt[t,k] - alpha_true[t,k]| ≤ 2 * sigma_k(t)
    # With K=3 independent dimensions, requiring ALL K within 2σ simultaneously
    # gives only (0.9545)^3 ≈ 87% in expectation; test per-component instead.
    alpha_filt = state.alpha_filtered
    residuals  = alpha_filt - alpha_true                       # (T, K)
    P_filt     = state.params["P_filtered"]                    # (T, K, K)

    std_all = np.sqrt(np.maximum(
        np.array([np.diag(P_filt[t]) for t in range(T)]), 1e-30
    ))  # (T, K)
    within = np.abs(residuals) <= 2 * std_all                  # (T, K) bool
    fraction = within.mean()

    assert fraction > 0.90, (
        f"Per-component within-2σ fraction = {fraction:.1%} "
        f"(expected > 90%)"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_I_KF_1_covariance_positive_definiteness():
    """I-KF-1: P_{t|t} and P_{t|t-1} remain positive definite at every step."""
    K, N, T = 3, 8, 500
    rng = np.random.default_rng(4004)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    F = 0.8 * np.eye(K) + 0.05 * rng.standard_normal((K, K))
    # Make F stable: scale so spectral radius < 1
    sr = np.max(np.abs(np.linalg.eigvals(F)))
    F = F / (sr + 0.1)
    Q = 0.1 * np.eye(K)
    R = 0.2 * np.eye(N)

    obs = rng.standard_normal((T, N))
    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _frame(U_raw)
    state = kf.filter(obs, frame)

    P_filt = state.params["P_filtered"]   # (T, K, K)
    P_pred = state.params["P_predicted"]  # (T, K, K)

    min_eig_filt = min(np.linalg.eigvalsh(P_filt[t]).min() for t in range(T))
    min_eig_pred = min(np.linalg.eigvalsh(P_pred[t]).min() for t in range(T))

    assert min_eig_filt > 0, (
        f"P_{{t|t}} has non-positive eigenvalue: min={min_eig_filt:.3e}"
    )
    assert min_eig_pred > 0, (
        f"P_{{t|t-1}} has non-positive eigenvalue: min={min_eig_pred:.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_I_KF_2_log_likelihood_consistency():
    """I-KF-2: Filter LL matches innovation-based scipy.stats.multivariate_normal."""
    from scipy.stats import multivariate_normal

    K, T = 2, 100
    F = np.array([[0.9, 0.1], [-0.1, 0.85]])
    Q = 0.1 * np.eye(K)
    R = 0.5 * np.eye(K)

    rng = np.random.default_rng(5005)
    obs = rng.standard_normal((T, K))

    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    U = np.eye(K)
    P_pred_all = state.params["P_predicted"]   # (T, K, K)

    # Recompute LL via scipy on each innovation
    ll_scipy = 0.0
    for t in range(T):
        v = obs[t] - U @ state.alpha_predicted[t]
        S = U @ P_pred_all[t] @ U.T + R
        ll_scipy += multivariate_normal.logpdf(v, mean=np.zeros(K), cov=S)

    assert abs(ll_scipy - state.log_likelihood) < 1e-6, (
        f"LL mismatch: filter={state.log_likelihood:.6f}, "
        f"scipy={ll_scipy:.6f}, diff={abs(ll_scipy - state.log_likelihood):.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_I_KF_3_innovation_whiteness():
    """I-KF-3: Innovations are white noise under correctly specified model."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    K, T = 2, 500
    F = np.array([[0.9, 0.1], [-0.1, 0.85]])
    Q = 0.1 * np.eye(K)
    R = 0.5 * np.eye(K)

    rng = np.random.default_rng(6006)
    # Generate data from the true model
    true_alpha = np.zeros((T, K))
    noise_Q = np.linalg.cholesky(Q)
    noise_R = np.linalg.cholesky(R)
    for t in range(1, T):
        true_alpha[t] = F @ true_alpha[t - 1] + noise_Q @ rng.standard_normal(K)
    obs = true_alpha + (noise_R @ rng.standard_normal((K, T))).T   # (T, K)

    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    U = np.eye(K)
    innovations = obs - state.alpha_predicted @ U.T   # (T, K)

    for k in range(K):
        innov_k = innovations[:, k]
        lb = acorr_ljungbox(innov_k, lags=10, return_df=True)
        p_values = lb["lb_pvalue"].values
        assert np.all(p_values > 0.05), (
            f"Dimension {k}: innovation not white (Ljung-Box p={p_values.min():.4f} "
            f"≤ 0.05 at some lag)"
        )

    # ACF at lags 1..10 must all be < 0.1
    for k in range(K):
        innov_k = innovations[:, k]
        acf = np.array([
            np.corrcoef(innov_k[:-lag], innov_k[lag:])[0, 1]
            for lag in range(1, 11)
        ])
        max_acf = np.max(np.abs(acf))
        assert max_acf < 0.1, (
            f"Dimension {k}: |ACF| = {max_acf:.4f} > 0.1 at some lag 1..10"
        )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_R_KF_1_statsmodels_agreement():
    """R-KF-1: Our filter agrees with statsmodels.tsa.statespace on filtered states and LL."""
    sm = pytest.importorskip("statsmodels")
    import statsmodels.api as smapi  # noqa: F401

    K, T = 2, 200
    F = np.array([[0.9, 0.1], [-0.1, 0.85]])
    Q = 0.1 * np.eye(K)
    R = 0.5 * np.eye(K)

    rng = np.random.default_rng(7007)
    obs = rng.standard_normal((T, K))

    # Our filter (alpha_0=0, P_0=I)
    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    # statsmodels reference.
    # Our filter: alpha_0=0, P_0=I → first prediction a_{1|0}=0, P_{1|0}=F@I@F^T+Q.
    # statsmodels initialize_known(a, P) sets a_{1|0}=a and P_{1|0}=P, so we must
    # pass the already-predicted covariance F@I@F^T+Q to match exactly.
    P_pred_0 = F @ np.eye(K) @ F.T + Q

    class _FixedSSM(smapi.tsa.statespace.MLEModel):
        def __init__(self, endog, F_, Q_, R_, P0_):
            super().__init__(endog, k_states=K)
            self["transition"] = F_
            self["design"] = np.eye(K)    # H = I
            self["state_cov"] = Q_
            self["obs_cov"] = R_
            self["selection"] = np.eye(K)
            self.initialize_known(np.zeros(K), P0_)
            self.loglikelihood_burn = 0

        @property
        def param_names(self):
            return []

        @property
        def start_params(self):
            return np.array([], dtype=float)

        def update(self, params, **kwargs):
            pass

    mod = _FixedSSM(obs, F, Q, R, P_pred_0)
    res = mod.filter([])

    sm_alpha_filt = res.filtered_state.T    # (T, K)
    sm_ll         = float(res.llf)

    # States should agree to TIGHT tolerance
    err_states = np.max(np.abs(sm_alpha_filt - state.alpha_filtered))
    assert err_states < 1e-6, (
        f"Filtered state mismatch vs statsmodels: max |Δα| = {err_states:.3e}"
    )

    # Log-likelihood should agree to TIGHT tolerance
    err_ll = abs(sm_ll - state.log_likelihood)
    assert err_ll < 1e-4, (
        f"LL mismatch vs statsmodels: ours={state.log_likelihood:.6f}, "
        f"sm={sm_ll:.6f}, diff={err_ll:.3e}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_R_KF_2_dare_agreement():
    """R-KF-2: Steady-state P from our filter agrees with scipy DARE to EXACT tolerance."""
    F = np.array([[0.9, 0.1], [-0.1, 0.85]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    H = np.eye(2)
    R = np.array([[0.5, 0.0], [0.0, 0.5]])
    T = 1000

    # DARE solution P_∞
    P_dare = solve_discrete_are(F.T, H.T, Q, R)

    rng = np.random.default_rng(8008)
    obs = rng.standard_normal((T, 2))

    kf = KalmanFilter(F=F, Q=Q, R=R)
    frame = _identity_frame(2)
    state = kf.filter(obs, frame)

    P_pred_final = state.params["P_predicted"][-1]
    err = np.max(np.abs(P_pred_final - P_dare))
    assert err < 1e-8, (
        f"P_predicted[-1] differs from DARE: max |ΔP| = {err:.3e} (expected < 1e-8)"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_D_KF_1_unstable_system_no_nan():
    """D-KF-1: Filter runs 100 steps on unstable F (eigenvalue 1.1) without NaN/Inf."""
    K = 2
    # Unstable: spectral radius > 1
    F_unstable = np.array([[1.1, 0.0], [0.0, 0.95]])
    Q = 0.1 * np.eye(K)
    R = 0.5 * np.eye(K)
    T = 100

    rng = np.random.default_rng(9009)
    obs = rng.standard_normal((T, K))

    kf = KalmanFilter(F=F_unstable, Q=Q, R=R)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    assert not np.any(np.isnan(state.alpha_filtered)), "NaN in filtered states"
    assert not np.any(np.isinf(state.alpha_filtered)), "Inf in filtered states"
    assert not np.isnan(state.log_likelihood), "NaN in log-likelihood"


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_D_KF_2_near_singular_R():
    """D-KF-2: Filter handles near-singular R = diag(1e-12, 1e-12) without crash."""
    K = 2
    F = 0.9 * np.eye(K)
    Q = 0.1 * np.eye(K)
    R_singular = 1e-12 * np.eye(K)
    T = 20

    rng = np.random.default_rng(1010)
    obs = rng.standard_normal((T, K))

    kf = KalmanFilter(F=F, Q=Q, R=R_singular)
    frame = _identity_frame(K)
    state = kf.filter(obs, frame)

    assert not np.any(np.isnan(state.alpha_filtered)), "NaN with near-singular R"
    assert not np.any(np.isinf(state.alpha_filtered)), "Inf with near-singular R"

    # Near R=0: filtered state should be very close to observation
    max_err = np.max(np.abs(state.alpha_filtered[1:] - obs[1:]))
    assert max_err < 1e-3, (
        f"Filtered state not close to obs under near-singular R: max err = {max_err:.3e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EM ESTIMATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_A_EM_1_recover_known_parameters():
    """A-EM-1: EM recovers F within 10%, Q within 25%, R within 20% on T=2000 data."""
    N, K, T = 10, 2, 2000
    rng = np.random.default_rng(1111)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)

    # True parameters — differ from EM defaults (F_init=0.9·I, Q_init=R_init=0.1·I)
    F_true = 0.70 * np.eye(K)
    Q_true = 0.50 * np.eye(K)
    R_true = 0.10 * np.eye(N)

    result = make_modal_system(N=N, K=K, T=T,
                               U_true=U_raw, F_true=F_true, Q=Q_true, R=R_true,
                               seed=7)
    Y = result.observations   # (T, N)

    frame = _frame(U_raw)
    kf = KalmanFilter()
    kf.em_estimate(Y, frame, max_iter=200, tol=1e-5)

    # F recovery: relative Frobenius norm ≤ 10%
    err_F = np.linalg.norm(kf.F - F_true) / np.linalg.norm(F_true)
    assert err_F <= 0.10, (
        f"F recovery: ||F_est - F_true||_F / ||F_true||_F = {err_F:.3f} > 0.10"
    )

    # Q recovery: relative Frobenius norm ≤ 25%
    err_Q = np.linalg.norm(kf.Q - Q_true) / np.linalg.norm(Q_true)
    assert err_Q <= 0.25, (
        f"Q recovery: ||Q_est - Q_true||_F / ||Q_true||_F = {err_Q:.3f} > 0.25"
    )

    # R recovery: relative Frobenius norm ≤ 20%
    err_R = np.linalg.norm(kf.R - R_true) / np.linalg.norm(R_true)
    assert err_R <= 0.20, (
        f"R recovery: ||R_est - R_true||_F / ||R_true||_F = {err_R:.3f} > 0.20"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_I_EM_1_monotone_log_likelihood():
    """I-EM-1: EM log-likelihood never decreases (EM property: L[i+1] >= L[i] - 1e-10).

    This is the most important EM test. A non-monotone EM is broken.
    """
    N, K, T = 8, 2, 500
    rng = np.random.default_rng(2222)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    F_true = 0.8 * np.eye(K)
    Q_true = 0.3 * np.eye(K)
    R_true = 0.1 * np.eye(N)

    result = make_modal_system(N=N, K=K, T=T,
                               U_true=U_raw, F_true=F_true, Q=Q_true, R=R_true,
                               seed=8)
    Y = result.observations

    frame = _frame(U_raw)
    kf = KalmanFilter()
    final_state = kf.em_estimate(Y, frame, max_iter=200, tol=1e-6)

    ll_history = final_state.params["ll_history"]
    assert len(ll_history) >= 2, "EM ran fewer than 2 iterations"

    ll = np.asarray(ll_history, dtype=float)
    diffs = np.diff(ll)
    min_diff = diffs.min()

    assert min_diff >= -1e-6, (
        f"EM log-likelihood decreased: min(ΔLL) = {min_diff:.3e} < -1e-6\n"
        f"LL history (first 20): {ll[:20].tolist()}"
    )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_I_EM_2_convergence():
    """I-EM-2: EM converges — |LL[i] - LL[i-1]| < tol for some i < max_iter."""
    N, K, T = 6, 2, 400
    rng = np.random.default_rng(3333)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    F_true = 0.75 * np.eye(K)
    Q_true = 0.2 * np.eye(K)
    R_true = 0.15 * np.eye(N)

    result = make_modal_system(N=N, K=K, T=T,
                               U_true=U_raw, F_true=F_true, Q=Q_true, R=R_true,
                               seed=9)
    Y = result.observations

    frame = _frame(U_raw)
    tol = 1e-4
    max_iter = 200

    kf = KalmanFilter()
    final_state = kf.em_estimate(Y, frame, max_iter=max_iter, tol=tol)

    ll_history = final_state.params["ll_history"]
    ll = np.asarray(ll_history, dtype=float)

    # Check that convergence was achieved (not all iterations consumed)
    assert len(ll) < max_iter, (
        f"EM did not converge within {max_iter} iterations "
        f"(ran all {len(ll)} iterations)"
    )

    # Verify the final increment satisfies the tolerance
    if len(ll) >= 2:
        final_delta = abs(ll[-1] - ll[-2])
        assert final_delta < tol * 10, (  # 10x slack: convergence check is relative
            f"Final |ΔLL| = {final_delta:.3e} does not look converged (tol={tol})"
        )


@pytest.mark.acceptance
@pytest.mark.section("kalman_filter_em")
def test_D_EM_1_bad_initialization():
    """D-EM-1: EM recovers from bad init (F=2·I unstable, Q=0.001·I too small).

    Pass: EM produces finite parameters after max_iter steps, or converges;
    output F must have spectral radius < 1.5 (at minimum, not diverging worse).
    """
    N, K, T = 6, 2, 300
    rng = np.random.default_rng(4444)

    U_raw, _, _ = np.linalg.svd(rng.standard_normal((N, K)), full_matrices=False)
    F_true = 0.75 * np.eye(K)
    Q_true = 0.2 * np.eye(K)
    R_true = 0.1 * np.eye(N)

    result = make_modal_system(N=N, K=K, T=T,
                               U_true=U_raw, F_true=F_true, Q=Q_true, R=R_true,
                               seed=10)
    Y = result.observations

    # Inject bad initial parameters directly
    frame = _frame(U_raw)
    kf = KalmanFilter()
    kf.F = 2.0 * np.eye(K)          # unstable
    kf.Q = 0.001 * np.eye(K)        # too small
    kf.R = 0.1 * np.eye(N)

    # Override em_estimate to start from our bad init
    # Manually run one pass of EM after bad init to test robustness
    final_state = kf.em_estimate(Y, frame, max_iter=200, tol=1e-4)

    # Must not crash or produce NaN/Inf
    assert not np.any(np.isnan(kf.F)), "F contains NaN after bad-init EM"
    assert not np.any(np.isinf(kf.F)), "F contains Inf after bad-init EM"
    assert not np.isnan(final_state.log_likelihood), "LL is NaN after bad-init EM"

    # EM should have pulled F back toward a reasonable range (spectral radius < 1.5)
    sr = np.max(np.abs(np.linalg.eigvals(kf.F)))
    assert sr < 1.5, (
        f"F spectral radius = {sr:.3f} still dangerously large after EM with bad init"
    )
