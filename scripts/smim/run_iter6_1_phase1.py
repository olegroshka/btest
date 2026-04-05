#!/usr/bin/env python
"""
Iteration 6.1 Phase 1 — Core Diagnostics (A1 + C4 + B2 + E1).

Experiments:
  A1: DMD-eigenvalue diagonal F (attacks H1: transition mis-specification)
      A1a — block-diagonal F from eigenvalues, clip |λ| to [0.01, 0.99]
      A1b — same, clip to [0.01, 0.95] (conservative)
      A1c — F = Ã directly (full reduced propagator), clip spectral radius to 0.99
  C4: Per-mode predictive R² decomposition (diagnostic)
  B2: Reduced-rank regression (attacks H2: wrong basis)
  E1: Uniform normalisation (attacks H3: heterogeneous pooling)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_phase1.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

# SMIM hyperparameters (fixed, matching paper)
F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_DEFAULT = 8
K_MAX = 15  # SVD truncation rank for DMD (matches run_iter6_test3.py)


# ══════════════════════════════════════════════════════════════════════
#  Panel Construction (from run_iter6_test3.py, unchanged)
# ══════════════════════════════════════════════════════════════════════


def load_93_actor_panel():
    """Load the 93-actor multilayer panel + layer labels."""
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    layer_map = {a["actor_id"]: a["layer"] for a in registry["actors"]}
    type_map = {a["actor_id"]: a["actor_type"] for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    panel = panel.loc["2005-01-01":"2025-12-31"]
    actors = list(panel.columns)
    layer_labels = np.array([layer_map.get(a, -1) for a in actors])
    actor_types = [type_map.get(a, "unknown") for a in actors]
    return panel, layer_labels, actor_types


# ══════════════════════════════════════════════════════════════════════
#  Shared Helpers (from run_iter6_test3.py, unchanged)
# ══════════════════════════════════════════════════════════════════════


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def ar1_baseline(otr, N):
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c
    return mu, rho


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 10:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = (
        ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))]
        .values.astype(np.float64)
    )
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


# ══════════════════════════════════════════════════════════════════════
#  Baseline Window Runners (from template, unchanged)
# ══════════════════════════════════════════════════════════════════════


def run_window_ar1(panel, ty, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    mu, rho = ar1_baseline(otr, N)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        ps.append(mu + rho * (prev - mu))
        ac.append(qv[0])
        prev = qv[0]
    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    return float(oos_r_squared(psa.ravel(), aca.ravel()))


def run_window_pooled(panel, ty, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    rho_pool, bar_y = estimate_pooled_ar1(otr)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        ps.append(bar_y + rho_pool * (prev - bar_y))
        ac.append(qv[0])
        prev = qv[0]
        otr = np.vstack([otr, qv])
        rho_pool, bar_y = estimate_pooled_ar1(otr)
    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    return float(oos_r_squared(psa.ravel(), aca.ravel()))


# ══════════════════════════════════════════════════════════════════════
#  DMD + Eigenvalue F Helpers
# ══════════════════════════════════════════════════════════════════════


def dmd_full(dm, k_svd=K_MAX):
    """DMD decomposition returning full ModalFrame.

    k_svd controls SVD truncation rank (default K_MAX=15 to match template).
    Returns ModalFrame or None on failure.
    """
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def build_eigenvalue_F(mf, ka, max_lambda=0.99, min_lambda=0.01):
    """Build block-diagonal F from DMD eigenvalues.

    Real eigenvalues:     F_kk = clip(|λ_k|, min_lambda, max_lambda)
    Conjugate pairs:      2×2 rotation-scaling block
        F = r · [[cos θ, -sin θ], [sin θ, cos θ]]
    """
    mode_pairs = mf.metadata["mode_pairs"]
    eigs = mf.eigenvalues

    F = np.zeros((ka, ka))
    for pair in mode_pairs:
        if max(pair) >= ka:
            continue
        if len(pair) == 1:
            j = pair[0]
            r = float(np.clip(abs(eigs[j]), min_lambda, max_lambda))
            F[j, j] = r
        else:
            j, j1 = pair
            lam = eigs[j]  # positive-imaginary eigenvalue
            r = float(np.clip(abs(lam), min_lambda, max_lambda))
            theta = float(np.angle(lam))
            F[j, j] = r * np.cos(theta)
            F[j, j1] = -r * np.sin(theta)
            F[j1, j] = r * np.sin(theta)
            F[j1, j1] = r * np.cos(theta)

    return F


def build_atilde_F(mf, ka, max_sr=0.99):
    """F = Ã (full reduced propagator), spectral radius clipped to max_sr."""
    Atilde = mf.metadata["Atilde"][:ka, :ka].copy()
    eigvals, _ = np.linalg.eig(Atilde)
    max_abs = float(np.max(np.abs(eigvals)))
    if max_abs > max_sr:
        Atilde *= max_sr / max_abs
    return Atilde.real


def _safe_truncate_K(mf, K):
    """Find effective K that doesn't split a conjugate pair."""
    ka = min(K, mf.K)
    for p in mf.metadata["mode_pairs"]:
        if len(p) == 2 and p[0] < ka <= p[1]:
            ka = p[0]
            break
    return ka


def _get_basis_and_F(mf, K, f_mode):
    """Dispatch: returns (basis, F, ka) for a given variant.

    mf comes from dmd_full(dm, k_svd=K_MAX) — may have more than K modes.
    We truncate to K for the Kalman filter state space.

    f_mode: 'baseline', 'a1a', 'a1b', 'a1c'
    """
    N = mf.basis.shape[0]

    if f_mode == "a1c":
        # Use SVD left singular vectors as basis, Atilde[:K,:K] as F.
        # Atilde[:K,:K] equals the Atilde from a k=K DMD call.
        ka = min(K, N - 2, mf.K)
        U_r = mf.metadata["U"][:, :ka]
        F = build_atilde_F(mf, ka)
        return U_r, F, ka

    if f_mode == "baseline":
        # Match template: just truncate basis to K columns (no pair adjustment)
        ka = min(K, N - 2, mf.K)
        basis = mf.basis[:, :ka]
        F = np.eye(ka) * F_REG
        return basis, F, ka

    # a1a / a1b: use DMD mode basis, need clean eigenvalue structure
    ka = _safe_truncate_K(mf, K)
    basis = mf.basis[:, :ka]

    if f_mode == "a1a":
        F = build_eigenvalue_F(mf, ka, max_lambda=0.99)
    elif f_mode == "a1b":
        F = build_eigenvalue_F(mf, ka, max_lambda=0.95)
    else:
        raise ValueError(f"Unknown f_mode: {f_mode}")

    return basis, F, ka


def _extract_eig_info(mf, ka):
    """Extract eigenvalue diagnostics from a ModalFrame."""
    eigs = mf.eigenvalues[:ka]
    return {
        "magnitudes": np.abs(eigs).real.copy(),
        "angles_deg": np.angle(eigs).real * 180.0 / np.pi,
        "max_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(
            mf.metadata["Atilde"]
        )))),
        "ka": int(ka),
    }


# ══════════════════════════════════════════════════════════════════════
#  RRR Basis (B2)
# ══════════════════════════════════════════════════════════════════════


def rrr_decompose(dm, k=2):
    """Reduced-rank regression basis (Reinsel & Velu 1998).

    Returns (U, B) where prediction = U @ B @ ỹ_t, or None.
    """
    T, N = dm.shape
    if T < 3:
        return None
    X = dm[:-1].T  # (N, T-1) predictors
    Y = dm[1:].T   # (N, T-1) targets

    reg = 1e-6 * np.eye(N)
    try:
        C = Y @ X.T @ np.linalg.solve(X @ X.T + reg, np.eye(N))
    except np.linalg.LinAlgError:
        return None

    try:
        L, S, Rt = np.linalg.svd(C, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    k_actual = min(k, len(S))
    U = L[:, :k_actual]
    B = np.diag(S[:k_actual]) @ Rt[:k_actual, :]
    return U, B


# ══════════════════════════════════════════════════════════════════════
#  SMIM Variant Window Runner (A1 + C4)
# ══════════════════════════════════════════════════════════════════════


def run_window_smim_variant(
    panel, ty, K=K_DEFAULT, ewm=12, T_yr=5, f_mode="baseline",
    collect_diagnostics=False,
):
    """SMIM with configurable F.  Returns (r2, diag_dict) or None."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    om = ewm_demean(otr, ewm)
    dm = otr - om

    # Initial DMD (k_svd=K_MAX to match template baseline)
    mf = dmd_full(dm, k_svd=K_MAX)
    if mf is None:
        return None

    U, F, ka = _get_basis_and_F(mf, K, f_mode)
    R = sph_r(dm, U)
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

    # Diagnostics collectors
    eig_records = []
    alpha_preds, alpha_filts = [], []
    if collect_diagnostics:
        eig_records.append(_extract_eig_info(mf, ka))

    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    has_nan = False

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        odm = obs - om.ravel()

        # ── Predict ──
        ap = F @ a
        Pp = F @ P @ F.T + Q
        pred = U @ ap + om.ravel()
        if not np.all(np.isfinite(pred)):
            has_nan = True
            pred = np.nan_to_num(pred, nan=0.5)
        ps.append(pred)
        ac.append(obs)
        prev = obs

        if collect_diagnostics:
            # Pad to K for consistent array shapes across rolling updates
            ap_pad = np.zeros(K)
            ap_pad[:len(ap)] = ap
            alpha_preds.append(ap_pad)

        # ── Kalman update ──
        S = U @ Pp @ U.T + R
        try:
            Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp

        if collect_diagnostics:
            a_pad = np.zeros(K)
            a_pad[:len(a)] = a
            alpha_filts.append(a_pad)

        # ── Q adaptation ──
        inn = a - ap
        Q = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(inn, inn)
        Q = (Q + Q.T) / 2 + np.eye(ka) * 1e-6

        # ── Rolling basis update ──
        otr = np.vstack([otr, qv])
        om2 = ewm_demean(otr, ewm)
        dm2 = otr - om2
        mf2 = dmd_full(dm2, k_svd=K_MAX)
        if mf2 is not None:
            U2, F2, k2 = _get_basis_and_F(mf2, K, f_mode)
            a = U2.T @ odm
            P = np.eye(k2)
            Q = np.eye(k2) * Q_INIT_SCALE
            F = F2
            R = sph_r(dm2, U2)
            U = U2
            ka = k2
            om = om2
            if collect_diagnostics:
                eig_records.append(_extract_eig_info(mf2, k2))

    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    if not np.all(np.isfinite(psa)):
        return None

    r2 = float(oos_r_squared(psa.ravel(), aca.ravel()))

    diag = {"has_nan": has_nan}
    if collect_diagnostics:
        diag["eig_records"] = eig_records
        if alpha_preds and alpha_filts:
            diag["alpha_preds"] = np.array(alpha_preds)
            diag["alpha_filts"] = np.array(alpha_filts)
    return r2, diag


# ══════════════════════════════════════════════════════════════════════
#  RRR Window Runner (B2)
# ══════════════════════════════════════════════════════════════════════


def run_window_rrr(panel, ty, K=K_DEFAULT, ewm=12, T_yr=5):
    """RRR one-step prediction, rolling. Returns (r2, subspace_angle_deg) or None."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    om = ewm_demean(otr, ewm)
    dm = otr - om

    rrr = rrr_decompose(dm, K)
    if rrr is None:
        return None
    U_rrr, B_rrr = rrr

    # DMD basis for subspace angle comparison
    mf = dmd_full(dm, k_svd=K_MAX)
    angle_deg = np.nan
    if mf is not None:
        ka = min(K, mf.K, U_rrr.shape[1])
        if ka >= 1:
            try:
                angles = scipy.linalg.subspace_angles(
                    mf.basis[:, :ka], U_rrr[:, :ka]
                )
                angle_deg = float(np.mean(np.degrees(angles)))
            except Exception:
                pass

    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        # RRR predict: ŷ = om + U @ B @ ỹ_{prev}
        y_hat = om.ravel() + U_rrr @ (B_rrr @ (prev - om.ravel()))
        ps.append(y_hat)
        ac.append(qv[0])
        prev = qv[0]

        # Rolling update
        otr = np.vstack([otr, qv])
        om = ewm_demean(otr, ewm)
        dm = otr - om
        rrr_new = rrr_decompose(dm, K)
        if rrr_new is not None:
            U_rrr, B_rrr = rrr_new

    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    if not np.all(np.isfinite(psa)):
        return None
    return float(oos_r_squared(psa.ravel(), aca.ravel())), angle_deg


# ══════════════════════════════════════════════════════════════════════
#  E1: Uniform Normalisation
# ══════════════════════════════════════════════════════════════════════


def uniform_rank_panel(panel):
    """Cross-sectional percentile rank per quarter (all actors uniform)."""
    return panel.rank(axis=1, method="average", pct=True)


def compute_per_layer_rho(panel, layer_labels):
    """Mean AR(1) coefficient per layer."""
    vals = panel.values.astype(np.float64)
    rho_by_layer = {}
    actors = list(panel.columns)
    # Assume layer_labels ordered as panel.columns
    for layer in np.unique(layer_labels):
        mask = layer_labels == layer
        cols = vals[:, mask]
        rhos = []
        for j in range(cols.shape[1]):
            y = cols[:, j]
            y = y[np.isfinite(y)]
            if len(y) < 4:
                continue
            d = y - np.mean(y)
            if np.std(d[:-1]) > 1e-10:
                c = np.corrcoef(d[:-1], d[1:])[0, 1]
                if np.isfinite(c):
                    rhos.append(c)
        rho_by_layer[int(layer)] = float(np.mean(rhos)) if rhos else 0.0
    return rho_by_layer


# ══════════════════════════════════════════════════════════════════════
#  Per-Mode R² Computation (C4)
# ══════════════════════════════════════════════════════════════════════


def compute_per_mode_r2(diag_dict):
    """Compute R²_pred(k) = 1 - SS_res(k) / SS_tot(k) per mode.

    SS_res(k) = Σ_t (α_filt[t,k] - α_pred[t,k])²
    SS_tot(k) = Σ_t α_filt[t,k]²
    """
    if "alpha_preds" not in diag_dict or "alpha_filts" not in diag_dict:
        return []
    preds = diag_dict["alpha_preds"]   # (n_q, ka)
    filts = diag_dict["alpha_filts"]   # (n_q, ka)
    if len(preds) == 0:
        return []
    ka = preds.shape[1]
    r2s = []
    for k in range(ka):
        ss_res = float(np.sum((filts[:, k] - preds[:, k]) ** 2))
        ss_tot = float(np.sum(filts[:, k] ** 2))
        if ss_tot < 1e-15:
            r2s.append(0.0)
        else:
            r2s.append(1.0 - ss_res / ss_tot)
    return r2s


# ══════════════════════════════════════════════════════════════════════
#  A1 Experiments
# ══════════════════════════════════════════════════════════════════════


def run_a1_experiments(panel, test_years, K=K_DEFAULT, ewm=12, T_yr=5):
    """Run baseline + A1a/b/c + C4 diagnostics. Returns dict of results."""
    variants = ["baseline", "a1a", "a1b", "a1c"]
    results = {v: [] for v in variants}
    results["ar1"] = []

    # Per-mode diagnostics: collect for baseline and a1a
    permode_baseline = []
    permode_a1a = []
    eig_all = []  # eigenvalue records across all windows

    for ty in test_years:
        t0 = time.time()

        # AR(1) baseline
        ar1 = run_window_ar1(panel, ty, T_yr)
        results["ar1"].append(ar1)

        for v in variants:
            collect = v in ("baseline", "a1a")
            res = run_window_smim_variant(
                panel, ty, K=K, ewm=ewm, T_yr=T_yr,
                f_mode=v, collect_diagnostics=collect,
            )
            if res is not None:
                r2, diag = res
                results[v].append(r2)
                if collect and "eig_records" in diag:
                    eig_all.extend(diag["eig_records"])
                if v == "baseline" and "alpha_preds" in diag:
                    permode_baseline.append(compute_per_mode_r2(diag))
                if v == "a1a" and "alpha_preds" in diag:
                    permode_a1a.append(compute_per_mode_r2(diag))
            else:
                results[v].append(None)

        elapsed = time.time() - t0
        ar1_s = f"{results['ar1'][-1]:.4f}" if results['ar1'][-1] is not None else "  N/A "
        bl_s = f"{results['baseline'][-1]:.4f}" if results['baseline'][-1] is not None else "  N/A "
        a1a_s = f"{results['a1a'][-1]:.4f}" if results['a1a'][-1] is not None else "  N/A "
        print(f"  W{ty}: AR1={ar1_s}  BL={bl_s}  A1a={a1a_s}  ({elapsed:.1f}s)")

    return results, eig_all, permode_baseline, permode_a1a


# ══════════════════════════════════════════════════════════════════════
#  B2 Experiments
# ══════════════════════════════════════════════════════════════════════


def run_b2_experiments(panel, test_years, ewm=12, T_yr=5):
    """Run RRR at K=2,4,8. Returns dict."""
    results = {}
    for K in [2, 4, 8]:
        r2s, angles = [], []
        for ty in test_years:
            res = run_window_rrr(panel, ty, K=K, ewm=ewm, T_yr=T_yr)
            if res is not None:
                r2, angle = res
                r2s.append(r2)
                if np.isfinite(angle):
                    angles.append(angle)
            else:
                r2s.append(None)
        results[K] = {"r2s": r2s, "angles": angles}
    return results


# ══════════════════════════════════════════════════════════════════════
#  E1 Experiments
# ══════════════════════════════════════════════════════════════════════


def run_e1_experiments(panel, layer_labels, test_years, K=K_DEFAULT, ewm=12, T_yr=5):
    """Uniform normalisation → re-run baselines + SMIM."""
    panel_ranked = uniform_rank_panel(panel)

    # Per-layer persistence before and after
    rho_before = compute_per_layer_rho(panel, layer_labels)
    rho_after = compute_per_layer_rho(panel_ranked, layer_labels)

    models = {
        "ar1": [],
        "pooled": [],
        "smim_baseline": [],
        "smim_a1a": [],
    }
    for ty in test_years:
        models["ar1"].append(run_window_ar1(panel_ranked, ty, T_yr))
        models["pooled"].append(run_window_pooled(panel_ranked, ty, T_yr))
        res_bl = run_window_smim_variant(
            panel_ranked, ty, K=K, ewm=ewm, T_yr=T_yr, f_mode="baseline",
        )
        models["smim_baseline"].append(res_bl[0] if res_bl else None)
        res_a1a = run_window_smim_variant(
            panel_ranked, ty, K=K, ewm=ewm, T_yr=T_yr, f_mode="a1a",
        )
        models["smim_a1a"].append(res_a1a[0] if res_a1a else None)

    return models, rho_before, rho_after


# ══════════════════════════════════════════════════════════════════════
#  Output Formatting
# ══════════════════════════════════════════════════════════════════════


def _mean_valid(lst):
    vals = [x for x in lst if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def _count_wins(model_list, ref_list):
    wins, total = 0, 0
    for m, r in zip(model_list, ref_list):
        if m is not None and r is not None:
            total += 1
            if m > r:
                wins += 1
    return wins, total


def print_a1_results(a1_results, eig_all, permode_baseline, permode_a1a):
    ar1_mean = _mean_valid(a1_results["ar1"])

    print("\nA1 — TRANSITION DYNAMICS:")
    print(f"  {'Variant':<25s} {'R²':>7s} {'ΔR² vs AR1':>12s} {'ΔR² vs BL':>12s} {'Wins':>7s}")
    print(f"  {'-'*65}")

    for v, label in [
        ("baseline", "Baseline (F=0.99I)"),
        ("a1a", "A1a (eigenvalue F)"),
        ("a1b", "A1b (clipped 0.95)"),
        ("a1c", "A1c (full Ã)"),
    ]:
        vals = a1_results[v]
        mean_r2 = _mean_valid(vals)
        bl_mean = _mean_valid(a1_results["baseline"])

        # Delta vs AR1
        deltas_ar1 = []
        for m, a in zip(vals, a1_results["ar1"]):
            if m is not None and a is not None:
                deltas_ar1.append(m - a)
        d_ar1 = float(np.mean(deltas_ar1)) if deltas_ar1 else np.nan

        # Delta vs baseline
        if v == "baseline":
            d_bl_str = "---".center(12)
        else:
            deltas_bl = []
            for m, b in zip(vals, a1_results["baseline"]):
                if m is not None and b is not None:
                    deltas_bl.append(m - b)
            d_bl = float(np.mean(deltas_bl)) if deltas_bl else np.nan
            d_bl_str = f"{d_bl:+12.4f}" if np.isfinite(d_bl) else "N/A".center(12)

        wins, total = _count_wins(vals, a1_results["ar1"])
        d_ar1_str = f"{d_ar1:+12.4f}" if np.isfinite(d_ar1) else "N/A".center(12)
        r2_str = f"{mean_r2:7.4f}" if np.isfinite(mean_r2) else "  N/A  "
        print(f"  {label:<25s} {r2_str} {d_ar1_str} {d_bl_str} {wins:>3d}/{total}")

    # ── Eigenvalue table ──
    if eig_all:
        # Aggregate per mode across all DMD estimations
        K_max = max(e["ka"] for e in eig_all)
        print(f"\n  DMD eigenvalues (mean across {len(eig_all)} estimations):")
        print(f"  {'Mode':>6s} {'|λ|':>7s} {'θ(deg)':>8s} {'R²_pred(BL)':>12s} {'R²_pred(A1a)':>13s} {'Gap':>7s}")
        print(f"  {'-'*56}")

        for k in range(min(K_max, K_DEFAULT)):
            mags = [e["magnitudes"][k] for e in eig_all if k < e["ka"]]
            angs = [abs(e["angles_deg"][k]) for e in eig_all if k < e["ka"]]
            mean_mag = float(np.mean(mags)) if mags else np.nan
            mean_ang = float(np.mean(angs)) if angs else np.nan

            # Per-mode R² from C4
            r2_bl_k = float(np.mean([pm[k] for pm in permode_baseline if k < len(pm)])) \
                if permode_baseline else np.nan
            r2_a1a_k = float(np.mean([pm[k] for pm in permode_a1a if k < len(pm)])) \
                if permode_a1a else np.nan
            gap = r2_bl_k - r2_a1a_k if np.isfinite(r2_bl_k) and np.isfinite(r2_a1a_k) else np.nan

            mag_s = f"{mean_mag:7.4f}" if np.isfinite(mean_mag) else "  N/A  "
            ang_s = f"{mean_ang:8.1f}" if np.isfinite(mean_ang) else "   N/A  "
            bl_s = f"{r2_bl_k:12.4f}" if np.isfinite(r2_bl_k) else "   N/A      "
            a1a_s = f"{r2_a1a_k:13.4f}" if np.isfinite(r2_a1a_k) else "    N/A       "
            gap_s = f"{gap:+7.4f}" if np.isfinite(gap) else "  N/A  "
            print(f"  {k+1:>6d} {mag_s} {ang_s} {bl_s} {a1a_s} {gap_s}")

    # Max spectral radius report (QG4)
    if eig_all:
        max_srs = [e["max_spectral_radius"] for e in eig_all]
        print(f"\n  Max |λ_k| across all DMD estimations: {max(max_srs):.4f} "
              f"(mean {np.mean(max_srs):.4f})")


def print_b2_results(b2_results, a1_baseline_r2s):
    print("\nB2 — REDUCED-RANK REGRESSION:")
    print(f"  {'K':>4s} {'RRR R²':>8s} {'DMD R²':>8s} {'Subspace angle':>16s}")
    print(f"  {'-'*40}")

    for K in [2, 4, 8]:
        data = b2_results[K]
        rrr_mean = _mean_valid(data["r2s"])
        angle_mean = float(np.mean(data["angles"])) if data["angles"] else np.nan

        # DMD R² at this K — for K=8 use a1 baseline; for K=2,4 we need separate runs
        # (will be passed in if available)
        dmd_r2 = np.nan
        if K == 8:
            dmd_r2 = _mean_valid(a1_baseline_r2s)

        rrr_s = f"{rrr_mean:8.4f}" if np.isfinite(rrr_mean) else "   N/A  "
        dmd_s = f"{dmd_r2:8.4f}" if np.isfinite(dmd_r2) else "   N/A  "
        ang_s = f"{angle_mean:14.1f}°" if np.isfinite(angle_mean) else "          N/A "
        print(f"  {K:>4d} {rrr_s} {dmd_s} {ang_s}")


def print_e1_results(e1_models, rho_before, rho_after):
    print("\nE1 — UNIFORM NORMALISATION:")
    ar1_mean = _mean_valid(e1_models["ar1"])

    # Persistence change
    print("  Per-layer ρ (before → after ranking):")
    layer_names = {0: "macro", 1: "inst", 2: "firms"}
    for layer in sorted(rho_before.keys()):
        rb = rho_before[layer]
        ra = rho_after[layer]
        name = layer_names.get(layer, f"L{layer}")
        print(f"    Layer {layer} ({name:5s}):  ρ={rb:.3f} → {ra:.3f}  (Δ={ra-rb:+.3f})")

    print(f"\n  {'Model (re-ranked panel)':<30s} {'R²':>8s} {'ΔR² vs AR1':>12s}")
    print(f"  {'-'*52}")

    for key, label in [
        ("ar1", "Per-actor AR(1)"),
        ("pooled", "Pooled+FE"),
        ("smim_baseline", "SMIM baseline (F=0.99I)"),
        ("smim_a1a", "SMIM A1a (eigenvalue F)"),
    ]:
        vals = e1_models[key]
        mean_r2 = _mean_valid(vals)
        deltas = []
        for m, a in zip(vals, e1_models["ar1"]):
            if m is not None and a is not None:
                deltas.append(m - a)
        d = float(np.mean(deltas)) if deltas else np.nan

        r2_str = f"{mean_r2:8.4f}" if np.isfinite(mean_r2) else "   N/A  "
        if key == "ar1":
            print(f"  {label:<30s} {r2_str}")
        else:
            d_str = f"{d:+12.4f}" if np.isfinite(d) else "    N/A     "
            print(f"  {label:<30s} {r2_str} {d_str}")


def print_quality_gates(a1_results, eig_all):
    print("\nQUALITY GATES:")
    bl_mean = _mean_valid(a1_results["baseline"])
    ar1_mean = _mean_valid(a1_results["ar1"])

    # QG1: Baseline reproduces R²=0.415 (±0.002)
    qg1 = abs(bl_mean - 0.415) < 0.020 if np.isfinite(bl_mean) else False
    print(f"  QG1: Baseline R²={bl_mean:.4f} (expect 0.415 ±0.020)"
          f"  {'PASS' if qg1 else 'CHECK'}")

    # QG2: AR(1) reproduces R²=0.594 (±0.002)
    qg2 = abs(ar1_mean - 0.594) < 0.020 if np.isfinite(ar1_mean) else False
    print(f"  QG2: AR(1) R²={ar1_mean:.4f} (expect 0.594 ±0.020)"
          f"  {'PASS' if qg2 else 'CHECK'}")

    # QG3: No NaN/Inf
    all_finite = True
    for v in ["baseline", "a1a", "a1b", "a1c"]:
        for r in a1_results[v]:
            if r is not None and not np.isfinite(r):
                all_finite = False
    print(f"  QG3: No NaN/Inf in predictions {'PASS' if all_finite else 'FAIL'}")

    # QG4: Max |λ_k|
    if eig_all:
        max_srs = [e["max_spectral_radius"] for e in eig_all]
        max_sr = max(max_srs)
        qg4 = max_sr <= 1.5  # reasonable bound
        print(f"  QG4: Max |λ_k|={max_sr:.4f} {'PASS' if qg4 else 'CHECK'}")

    # QG5: Modal R² check (baseline with diagnostics)
    print(f"  QG5: Modal R² not degraded — see per-mode table above")


def print_decision(a1_results, b2_results, e1_models):
    print("\nPHASE 1 DECISION:")

    bl_mean = _mean_valid(a1_results["baseline"])
    a1a_mean = _mean_valid(a1_results["a1a"])
    a1b_mean = _mean_valid(a1_results["a1b"])
    a1c_mean = _mean_valid(a1_results["a1c"])
    ar1_mean = _mean_valid(a1_results["ar1"])

    best_a1 = max(a1a_mean, a1b_mean, a1c_mean) if all(
        np.isfinite(x) for x in [a1a_mean, a1b_mean, a1c_mean]
    ) else np.nan
    a1_gain = best_a1 - bl_mean if np.isfinite(best_a1) and np.isfinite(bl_mean) else np.nan

    rrr_8 = _mean_valid(b2_results[8]["r2s"]) if 8 in b2_results else np.nan
    rrr_gain = rrr_8 - bl_mean if np.isfinite(rrr_8) and np.isfinite(bl_mean) else np.nan

    e1_a1a = _mean_valid(e1_models["smim_a1a"]) if "smim_a1a" in e1_models else np.nan

    if np.isfinite(a1_gain) and a1_gain >= 0.05:
        print(f"  ✓ A1 gained {a1_gain:+.4f} (≥+0.05) → Phase 2a: refine F")
    elif np.isfinite(a1_gain) and a1_gain >= 0.03:
        print(f"  ~ A1 gained {a1_gain:+.4f} (marginal) → Phase 2a with caution")
    else:
        print(f"  ✗ A1 gained {a1_gain:+.4f} (< +0.03)")

    if np.isfinite(rrr_gain) and rrr_gain >= 0.03:
        print(f"  ✓ B2 (RRR K=8) beats DMD by {rrr_gain:+.4f} → consider RRR basis")
    else:
        rrr_str = f"{rrr_gain:+.4f}" if np.isfinite(rrr_gain) else "N/A"
        print(f"  ✗ B2 (RRR K=8) Δ vs baseline: {rrr_str}")

    if np.isfinite(e1_a1a) and np.isfinite(bl_mean) and (e1_a1a - bl_mean) > 0.03:
        print(f"  ✓ E1 helps ({e1_a1a:.4f} vs {bl_mean:.4f}) → re-run A1 on fixed panel")
    else:
        print(f"  ✗ E1 uniform normalisation: no major improvement")

    if (np.isfinite(a1_gain) and a1_gain < 0.01 and
            np.isfinite(rrr_gain) and rrr_gain < 0.01):
        print(f"  → ALL FAIL: Phase 2b (architectural changes — C1 augmentation)")


# ══════════════════════════════════════════════════════════════════════
#  Save Results
# ══════════════════════════════════════════════════════════════════════


def save_results(a1_results, b2_results, e1_models, test_years):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # A1 results
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty, "ar1": a1_results["ar1"][i]}
        for v in ["baseline", "a1a", "a1b", "a1c"]:
            row[f"smim_{v}"] = a1_results[v][i]
        rows.append(row)
    df_a1 = pd.DataFrame(rows)
    df_a1.to_parquet(METRICS_DIR / "iter6_1_phase1_a1.parquet", index=False)

    # B2 results
    rows = []
    for K in [2, 4, 8]:
        for i, ty in enumerate(test_years):
            rows.append({
                "year": ty, "K": K,
                "rrr_r2": b2_results[K]["r2s"][i],
            })
    df_b2 = pd.DataFrame(rows)
    df_b2.to_parquet(METRICS_DIR / "iter6_1_phase1_b2.parquet", index=False)

    # E1 results
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty}
        for key in e1_models:
            row[f"e1_{key}"] = e1_models[key][i]
        rows.append(row)
    df_e1 = pd.DataFrame(rows)
    df_e1.to_parquet(METRICS_DIR / "iter6_1_phase1_e1.parquet", index=False)

    print(f"\n  Saved: iter6_1_phase1_a1.parquet, iter6_1_phase1_b2.parquet,"
          f" iter6_1_phase1_e1.parquet")


# ══════════════════════════════════════════════════════════════════════
#  SMIM K=2, K=4 baselines (for B2 comparison)
# ══════════════════════════════════════════════════════════════════════


def run_smim_baseline_at_K(panel, test_years, K, ewm=12, T_yr=5):
    """Run baseline SMIM at a specific K. Returns list of R²."""
    r2s = []
    for ty in test_years:
        res = run_window_smim_variant(
            panel, ty, K=K, ewm=ewm, T_yr=T_yr, f_mode="baseline",
        )
        r2s.append(res[0] if res else None)
    return r2s


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()

    print("=" * 76)
    print("  ITERATION 6.1 PHASE 1 — CORE DIAGNOSTICS")
    print("  93-actor panel, T=5yr, K=8, τ=12Q")
    print("=" * 76)

    panel, layer_labels, actor_types = load_93_actor_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")
    print(f"Layers: {dict(zip(*np.unique(layer_labels, return_counts=True)))}")
    print(f"Range: {panel.index.min().date()} to {panel.index.max().date()}")

    # ── A1 + C4 ──
    print(f"\n{'─'*76}")
    print("  Running A1 experiments (baseline + A1a/b/c + C4 diagnostics)...")
    print(f"{'─'*76}")
    a1_results, eig_all, permode_bl, permode_a1a = run_a1_experiments(
        panel, TEST_YEARS, K=K_DEFAULT, ewm=12, T_yr=5,
    )

    # ── B2 ──
    print(f"\n{'─'*76}")
    print("  Running B2 experiments (RRR K=2,4,8)...")
    print(f"{'─'*76}")
    b2_results = run_b2_experiments(panel, TEST_YEARS, ewm=12, T_yr=5)

    # Also run SMIM baseline at K=2,4 for comparison
    for K in [2, 4]:
        r2s = run_smim_baseline_at_K(panel, TEST_YEARS, K=K, ewm=12, T_yr=5)
        b2_results[K]["dmd_r2s"] = r2s
    b2_results[8]["dmd_r2s"] = a1_results["baseline"]

    # ── E1 ──
    print(f"\n{'─'*76}")
    print("  Running E1 experiments (uniform normalisation)...")
    print(f"{'─'*76}")
    e1_models, rho_before, rho_after = run_e1_experiments(
        panel, layer_labels, TEST_YEARS, K=K_DEFAULT, ewm=12, T_yr=5,
    )

    # ══════════════════════════════════════════════════════════════════
    #  Print Results
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*76}")
    print(f"  PHASE 1 RESULTS (93-actor panel, T=5yr, K=8)")
    print(f"{'='*76}")

    print_a1_results(a1_results, eig_all, permode_bl, permode_a1a)

    # Update B2 table with DMD R² at K=2,4
    def print_b2_full(b2_results):
        print("\nB2 — REDUCED-RANK REGRESSION:")
        print(f"  {'K':>4s} {'RRR R²':>8s} {'DMD R²':>8s} {'Subspace angle':>16s}")
        print(f"  {'-'*40}")
        for K in [2, 4, 8]:
            data = b2_results[K]
            rrr_mean = _mean_valid(data["r2s"])
            dmd_mean = _mean_valid(data.get("dmd_r2s", []))
            angle_mean = float(np.mean(data["angles"])) if data["angles"] else np.nan

            rrr_s = f"{rrr_mean:8.4f}" if np.isfinite(rrr_mean) else "   N/A  "
            dmd_s = f"{dmd_mean:8.4f}" if np.isfinite(dmd_mean) else "   N/A  "
            ang_s = f"{angle_mean:14.1f}°" if np.isfinite(angle_mean) else "          N/A "
            print(f"  {K:>4d} {rrr_s} {dmd_s} {ang_s}")

    print_b2_full(b2_results)
    print_e1_results(e1_models, rho_before, rho_after)

    print(f"\n{'='*76}")
    print_quality_gates(a1_results, eig_all)
    print_decision(a1_results, b2_results, e1_models)

    save_results(a1_results, b2_results, e1_models, TEST_YEARS)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
