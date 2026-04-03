#!/usr/bin/env python
"""
Workstream C -- Ablation Ladder Under Corrected Protocol.

Re-runs every step of the performance ladder on the same 10 outer windows
used in Workstream A.  Each step toggles one component; the rest remain at
the previous step's setting.  For every step we report:

  - per-window R^2 and MSE,
  - mean Delta-R^2 vs the previous step AND vs the baseline,
  - bootstrap 95 pct CI for both deltas,
  - fraction of windows where the step helps.

The script also adds the AR(1) baseline as step -1.

Outputs:
  results/metrics/ws_c_ablation_per_window.csv
  results/metrics/ws_c_ablation_summary.csv

Usage::
    uv run python scripts/run_smim_ws_c_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

# -- Paths -----------------------------------------------------------------
INTENSITIES_PATH = (
    PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
)
REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

TEST_YEARS = list(range(2015, 2025))
K_MAX = 15
N_BOOT = 5_000
SEED = 42


# =========================================================================
# Configuration for each ablation step
# =========================================================================

@dataclass
class AblationConfig:
    """One row of the ablation ladder."""
    name: str
    T_yr: int           # training window length in years
    K: int              # mode count
    use_dmd: bool       # True = DMD, False = Schur on correlation operator
    ewm_demean: bool    # True = EWM (tau=8), False = full-period mean
    spherical_R: bool   # True = R = (tr(R)/N)*I, False = full EM R
    online_Q: bool      # True = recursive Q adaptation (lambda=0.3)
    fixed_F: bool       # True = F=0.99*I, False = EM-estimated F
    fixed_Q0: bool      # True = Q0=0.5*I, False = EM Q
    rolling_basis: bool # True = recompute DMD each quarter


LADDER = [
    # Step 0: original baseline
    AblationConfig("S0: baseline (T=10,K=3,Schur,full-demean)",
                   T_yr=10, K=3, use_dmd=False, ewm_demean=False,
                   spherical_R=False, online_Q=False, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 1: +EWM demeaning
    AblationConfig("S1: +EWM demean",
                   T_yr=10, K=3, use_dmd=False, ewm_demean=True,
                   spherical_R=False, online_Q=False, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 2: +shorter T, higher K
    AblationConfig("S2: +T=5yr,K=5",
                   T_yr=5, K=5, use_dmd=False, ewm_demean=True,
                   spherical_R=False, online_Q=False, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 3: +spherical R
    AblationConfig("S3: +spherical R",
                   T_yr=5, K=5, use_dmd=False, ewm_demean=True,
                   spherical_R=True, online_Q=False, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 4: +DMD basis
    AblationConfig("S4: +DMD",
                   T_yr=5, K=5, use_dmd=True, ewm_demean=True,
                   spherical_R=True, online_Q=False, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 5: +online Q + K=8
    AblationConfig("S5: +online Q,K=8",
                   T_yr=5, K=8, use_dmd=True, ewm_demean=True,
                   spherical_R=True, online_Q=True, fixed_F=False,
                   fixed_Q0=False, rolling_basis=False),
    # Step 6: +transition regularisation
    AblationConfig("S6: +F=0.99I,Q0=0.5I",
                   T_yr=5, K=8, use_dmd=True, ewm_demean=True,
                   spherical_R=True, online_Q=True, fixed_F=True,
                   fixed_Q0=True, rolling_basis=False),
    # Step 7: +rolling basis
    AblationConfig("S7: +rolling basis",
                   T_yr=5, K=8, use_dmd=True, ewm_demean=True,
                   spherical_R=True, online_Q=True, fixed_F=True,
                   fixed_Q0=True, rolling_basis=True),
]


# =========================================================================
# Data loading
# =========================================================================

def load_panel():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(
        index="period", columns="actor_id", values="intensity_value"
    )
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    return wide[[c for c in actor_ids if c in wide.columns]]


# =========================================================================
# Building blocks
# =========================================================================

def ewm_mean(otr: np.ndarray, halflife: int = 8) -> np.ndarray:
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / halflife)
    return (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def full_mean(otr: np.ndarray) -> np.ndarray:
    return np.nanmean(otr, axis=0, keepdims=True)


def build_corr_operator(obs_train: np.ndarray) -> np.ndarray:
    """Pearson cross-correlation with threshold sparsification."""
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0
    return corr


def extract_basis_schur(otr_dm: np.ndarray, N: int, K: int):
    """Schur basis from correlation operator."""
    op = build_corr_operator(otr_dm)
    try:
        mf = SchurDecomposer().decompose(op, k=min(K, N - 2))
        U = mf.basis[:, : min(K, mf.K)].real
        return U
    except Exception:
        return None


def extract_basis_dmd(otr_dm: np.ndarray, N: int, K: int):
    """DMD basis from snapshot pairs."""
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(
            otr_dm.T, k=min(K_MAX, N)
        )
        U = mf.basis[:, : min(K, mf.K)].real
        return U
    except Exception:
        return None


def estimate_R(otr_dm: np.ndarray, N: int, U: np.ndarray, spherical: bool):
    """Kalman EM to get R; optionally sphericalise."""
    k_act = U.shape[1]
    mf = ModalFrame(
        basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd"
    )
    kf = KalmanFilter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = kf.em_estimate(otr_dm, mf, max_iter=50)
        R_em = st.params["R"]
        F_em = st.params.get("F", np.eye(k_act) * 0.9)
        Q_em = st.params.get("Q", np.eye(k_act) * 0.1)
        if spherical:
            tr_R = np.trace(R_em)
            if not np.isfinite(tr_R) or tr_R <= 0:
                resid = otr_dm - otr_dm @ U @ U.T
                tr_R = float(np.mean(resid ** 2)) * N
            R_out = np.eye(N) * tr_R / N
        else:
            R_out = R_em
        return R_out, F_em, Q_em
    except Exception:
        resid = otr_dm - otr_dm @ U @ U.T
        scale = max(float(np.mean(resid ** 2)), 1e-6)
        R_out = np.eye(N) * scale
        F_em = np.eye(k_act) * 0.9
        Q_em = np.eye(k_act) * 0.1
        return R_out, F_em, Q_em


def predict_quarter(
    obs_dm_q, om, N, U, F, Q_run, R, alpha, P, online_Q: bool
):
    """One-quarter Kalman predict-update, optional online Q."""
    k = U.shape[1]
    a_p = F @ alpha
    P_p = F @ P @ F.T + Q_run
    v = obs_dm_q - U @ a_p
    S = U @ P_p @ U.T + R
    try:
        Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
    except Exception:
        Kg = np.zeros((k, N))
    a_f = a_p + Kg @ v
    P_f = (np.eye(k) - Kg @ U) @ P_p
    pred = (a_f @ U.T + om).ravel()
    if online_Q:
        innov = a_f - F @ alpha
        Q_new = 0.7 * Q_run + 0.3 * np.outer(innov, innov)
        Q_new = (Q_new + Q_new.T) / 2 + np.eye(k) * 1e-6
    else:
        Q_new = Q_run
    return pred, a_f, P_f, Q_new


# =========================================================================
# Run one ablation config on one test year
# =========================================================================

def run_step(wide: pd.DataFrame, ty: int, cfg: AblationConfig) -> dict | None:
    """Returns dict with window, r2, mse, config name, or None on failure."""
    train_start = pd.Timestamp(f"{ty - cfg.T_yr}-01-01")
    train_end = pd.Timestamp(f"{ty - 1}-12-31")
    test_quarters = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")

    if train_start < pd.Timestamp("2005-01-01"):
        return None

    all_data = wide[
        (wide.index >= train_start) & (wide.index <= test_quarters[-1])
    ]
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())
    N = len(valid)

    otr = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None

    # -- Demeaning --
    om = ewm_mean(otr) if cfg.ewm_demean else full_mean(otr)
    otr_dm = otr - om

    # -- Basis extraction --
    if cfg.use_dmd:
        U = extract_basis_dmd(otr_dm, N, cfg.K)
    else:
        U = extract_basis_schur(otr_dm, N, cfg.K)
    if U is None:
        return None
    k_act = U.shape[1]

    # -- R, F, Q estimation --
    R, F_em, Q_em = estimate_R(otr_dm, N, U, cfg.spherical_R)

    if cfg.fixed_F:
        F_mat = np.eye(k_act) * 0.99
    else:
        F_mat = F_em[:k_act, :k_act] if F_em.shape[0] >= k_act else np.eye(k_act) * 0.9

    if cfg.fixed_Q0:
        Q_init = np.eye(k_act) * 0.5
    else:
        Q_init = Q_em[:k_act, :k_act] if Q_em.shape[0] >= k_act else np.eye(k_act) * 0.1

    # -- Test prediction --
    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    Q_run = Q_init.copy()
    U_current = U.copy()
    om_current = om.copy()
    R_current = R.copy()
    F_current = F_mat.copy()

    preds, actuals = [], []

    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        q_dm = q_data[0] - om_current.ravel()

        pred, alpha_new, P_new, Q_new = predict_quarter(
            q_dm, om_current, N, U_current, F_current, Q_run,
            R_current, alpha_c, P_c, cfg.online_Q,
        )
        preds.append(pred)
        actuals.append(q_data[0])

        # Rolling basis update
        if cfg.rolling_basis:
            otr_exp = all_data[
                (all_data.index >= train_start) & (all_data.index <= q_date)
            ].values.astype(np.float64)
            om_new = ewm_mean(otr_exp) if cfg.ewm_demean else full_mean(otr_exp)
            otr_dm_new = otr_exp - om_new

            if cfg.use_dmd:
                U_new = extract_basis_dmd(otr_dm_new, N, cfg.K)
            else:
                U_new = extract_basis_schur(otr_dm_new, N, cfg.K)

            if U_new is not None:
                U_current = U_new
                k_new = U_new.shape[1]
                alpha_c = U_new.T @ q_dm
                P_c = np.eye(k_new)
                Q_run = np.eye(k_new) * 0.5 if cfg.fixed_Q0 else Q_init[:k_new, :k_new]
                F_current = np.eye(k_new) * 0.99 if cfg.fixed_F else F_mat[:k_new, :k_new]
                R_current, _, _ = estimate_R(otr_dm_new, N, U_new, cfg.spherical_R)
                om_current = om_new
            else:
                alpha_c, P_c, Q_run = alpha_new, P_new, Q_new
        else:
            alpha_c, P_c, Q_run = alpha_new, P_new, Q_new

    if not preds:
        return None

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)
    r2 = float(oos_r_squared(preds_arr.ravel(), actuals_arr.ravel()))
    mse = float(np.mean((preds_arr - actuals_arr) ** 2))

    return {"window": f"W{ty}", "r2": r2, "mse": mse, "step": cfg.name}


# =========================================================================
# AR(1) baseline
# =========================================================================

def run_ar1(wide, ty, T_yr=10):
    train_start = pd.Timestamp(f"{ty - T_yr}-01-01")
    train_end = pd.Timestamp(f"{ty - 1}-12-31")
    test_quarters = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")

    all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
    valid = all_data.columns[all_data.notna().any()]
    all_data = all_data[valid].fillna(all_data[valid].mean())

    train = all_data[
        (all_data.index >= train_start) & (all_data.index <= train_end)
    ].values.astype(np.float64)
    N = train.shape[1]
    rho = np.zeros(N)
    mu = np.zeros(N)
    for j in range(N):
        y = train[:, j]
        y = y[~np.isnan(y)]
        if len(y) > 2:
            mu[j] = y.mean()
            y_dm = y - mu[j]
            if np.var(y_dm) > 0:
                rho[j] = np.corrcoef(y_dm[:-1], y_dm[1:])[0, 1]

    preds, actuals = [], []
    prev = train[-1] if len(train) > 0 else np.zeros(N)
    for q_date in test_quarters:
        q_data = all_data.loc[[q_date]].values.astype(np.float64)
        if q_data.shape[0] == 0:
            continue
        pred = mu + rho * (prev - mu)
        preds.append(pred)
        actuals.append(q_data[0])
        prev = q_data[0]

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)
    r2 = float(oos_r_squared(preds_arr.ravel(), actuals_arr.ravel()))
    mse = float(np.mean((preds_arr - actuals_arr) ** 2))
    return {"window": f"W{ty}", "r2": r2, "mse": mse, "step": "AR(1) T=10yr"}


# =========================================================================
# Bootstrap CI helper
# =========================================================================

def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(values)
    means = np.array(
        [np.mean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(means, 100 * alpha)),
        "ci_hi": float(np.percentile(means, 100 * (1 - alpha))),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 90)
    print("  WORKSTREAM C: ABLATION LADDER UNDER CORRECTED PROTOCOL")
    print("=" * 90)

    wide = load_panel()
    print(f"  Panel: {wide.shape[0]} quarters x {wide.shape[1]} actors")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Collect per-window results for every step -------------------------
    all_rows = []

    # AR(1) baseline
    print("\n  Running AR(1) baseline...")
    for ty in TEST_YEARS:
        res = run_ar1(wide, ty)
        if res:
            all_rows.append(res)

    # Ablation ladder
    for step_idx, cfg in enumerate(LADDER):
        t0 = time.time()
        print(f"\n  Running {cfg.name} ...")
        for ty in TEST_YEARS:
            try:
                res = run_step(wide, ty, cfg)
                if res:
                    all_rows.append(res)
            except Exception as e:
                print(f"    W{ty}: FAILED ({e})")
        elapsed = time.time() - t0
        # Quick summary
        step_rows = [r for r in all_rows if r["step"] == cfg.name]
        if step_rows:
            mean_r2 = np.mean([r["r2"] for r in step_rows])
            print(f"    Mean R^2 = {mean_r2:.4f}  ({len(step_rows)} windows, {elapsed:.1f}s)")

    # -- Build per-window DataFrame ----------------------------------------
    pw_df = pd.DataFrame(all_rows)
    pw_df.to_csv(RESULTS_DIR / "ws_c_ablation_per_window.csv", index=False)

    # -- Build summary table -----------------------------------------------
    steps = ["AR(1) T=10yr"] + [c.name for c in LADDER]
    summary_rows = []

    prev_r2s = None
    base_r2s = None

    for step_name in steps:
        sdf = pw_df[pw_df["step"] == step_name].sort_values("window")
        if sdf.empty:
            continue
        r2s = sdf["r2"].values
        mses = sdf["mse"].values
        n_win = len(r2s)

        row = {
            "step": step_name,
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
            "mean_mse": float(np.mean(mses)),
            "n_windows": n_win,
        }

        if base_r2s is None:
            base_r2s = r2s  # AR(1) is the baseline for "vs baseline"

        if prev_r2s is not None and len(prev_r2s) == len(r2s):
            # Incremental effect vs previous step
            delta_prev = r2s - prev_r2s
            bp = boot_ci(delta_prev)
            row["delta_r2_vs_prev"] = bp["mean"]
            row["ci_lo_prev"] = bp["ci_lo"]
            row["ci_hi_prev"] = bp["ci_hi"]
            row["frac_improved_prev"] = float(np.mean(delta_prev > 0))

            # Effect vs AR(1) baseline
            delta_base = r2s - base_r2s[: len(r2s)]
            bb = boot_ci(delta_base)
            row["delta_r2_vs_ar1"] = bb["mean"]
            row["ci_lo_ar1"] = bb["ci_lo"]
            row["ci_hi_ar1"] = bb["ci_hi"]
        else:
            row["delta_r2_vs_prev"] = 0.0
            row["ci_lo_prev"] = 0.0
            row["ci_hi_prev"] = 0.0
            row["frac_improved_prev"] = 0.0
            row["delta_r2_vs_ar1"] = 0.0
            row["ci_lo_ar1"] = 0.0
            row["ci_hi_ar1"] = 0.0

        summary_rows.append(row)
        prev_r2s = r2s

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "ws_c_ablation_summary.csv", index=False)

    # -- Print summary table -----------------------------------------------
    print("\n" + "=" * 90)
    print("  ABLATION SUMMARY (with bootstrap 95% CI)")
    print("=" * 90)
    print(f"  {'Step':<35s}  {'R^2':>6s}  {'Delta':>7s}  {'95% CI':>17s}  {'Frac+':>5s}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        ci_str = f"[{r['ci_lo_prev']:+.4f}, {r['ci_hi_prev']:+.4f}]"
        print(
            f"  {r['step']:<35s}  {r['mean_r2']:6.4f}  "
            f"{r['delta_r2_vs_prev']:+7.4f}  {ci_str:>17s}  "
            f"{r['frac_improved_prev']:5.0%}"
        )

    print(f"\n  Saved: {RESULTS_DIR / 'ws_c_ablation_per_window.csv'}")
    print(f"  Saved: {RESULTS_DIR / 'ws_c_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
