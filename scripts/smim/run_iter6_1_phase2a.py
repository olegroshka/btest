#!/usr/bin/env python
"""
Iteration 6.1 Phase 2a — Refine the Transition Matrix (A2 + A4 + C1).

Follows Phase 1 finding: A1c (full Ã) gained +0.071 over baseline.

Experiments:
  A2: Shrinkage F_γ = γÃ + (1-γ)0.99I, sweep γ∈{0,0.1,0.25,0.5,0.75,1.0}
  A4: Low-rank-plus-diagonal F: D + rank-r SVD of off-diagonal(Ã)
  C1: Spectral augmentation — pooled AR(1)+FE → DMD on residuals → combined

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_phase2a.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

F_REG = 0.99
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3
K_DEFAULT = 8
K_MAX = 15
GAMMA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


# ══════════════════════════════════════════════════════════════════════
#  Shared infrastructure (copied from Phase 1 — not modifying originals)
# ══════════════════════════════════════════════════════════════════════


def load_93_actor_panel():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    layer_map = {a["actor_id"]: a["layer"] for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    actors = list(panel.columns)
    layer_labels = np.array([layer_map.get(a, -1) for a in actors])
    return panel, layer_labels


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
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


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
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def dmd_full(dm, k_svd=K_MAX):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


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
    return float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))


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
    return float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))


# ══════════════════════════════════════════════════════════════════════
#  F-building helpers for A2, A4 (all in U_r coordinate system)
# ══════════════════════════════════════════════════════════════════════


def _clip_spectral_radius(F, max_sr=0.99):
    """Scale F so that max|eigenvalue| ≤ max_sr."""
    eigvals = np.linalg.eigvals(F)
    max_abs = float(np.max(np.abs(eigvals)))
    if max_abs > max_sr:
        F = F * (max_sr / max_abs)
    return F


def _get_Atilde_K(mf, K):
    """Return Ã[:K,:K] (real) and effective ka."""
    ka = min(K, mf.basis.shape[0] - 2, mf.K)
    return mf.metadata["Atilde"][:ka, :ka].real.copy(), ka


def build_shrinkage_F(mf, K, gamma):
    """F_γ = γ·Ã + (1-γ)·0.99I, spectral-radius-clipped to 0.99."""
    A, ka = _get_Atilde_K(mf, K)
    F = gamma * A + (1 - gamma) * F_REG * np.eye(ka)
    return _clip_spectral_radius(F), ka


def build_low_rank_F(mf, K, rank):
    """F = diag(Ã) + rank-r SVD of off-diagonal(Ã), clipped."""
    A, ka = _get_Atilde_K(mf, K)
    D = np.diag(np.diag(A))
    off_diag = A - D
    if rank <= 0:
        F = D
    elif rank >= ka:
        F = A  # full Ã
    else:
        try:
            U, S, Vt = np.linalg.svd(off_diag, full_matrices=False)
            approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
            F = D + approx
        except np.linalg.LinAlgError:
            F = D
    return _clip_spectral_radius(F), ka


def _F_diagnostics(F):
    """Return (spectral_radius, condition_number) for F."""
    eigvals = np.linalg.eigvals(F)
    sr = float(np.max(np.abs(eigvals)))
    try:
        cn = float(np.linalg.cond(F))
    except Exception:
        cn = np.nan
    return sr, cn


# ══════════════════════════════════════════════════════════════════════
#  Generic SMIM window runner (U_r basis, arbitrary F builder)
# ══════════════════════════════════════════════════════════════════════


def _run_smim_Ur(panel, ty, K, ewm, T_yr, build_F_fn):
    """Run SMIM with U_r basis and F from build_F_fn(mf, K) → (F, ka).

    Returns (r2, diag_dict) or None.
    diag_dict has: spectral_radii, cond_numbers (lists per DMD estimation).
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    om = ewm_demean(otr, ewm)
    dm = otr - om

    mf = dmd_full(dm, k_svd=K_MAX)
    if mf is None:
        return None

    F, ka = build_F_fn(mf, K)
    U = mf.metadata["U"][:, :ka]
    R = sph_r(dm, U)
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

    sr, cn = _F_diagnostics(F)
    srs, cns = [sr], [cn]

    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        odm = obs - om.ravel()

        ap = F @ a
        Pp = F @ P @ F.T + Q
        pred = U @ ap + om.ravel()
        if not np.all(np.isfinite(pred)):
            pred = np.nan_to_num(pred, nan=0.5)
        ps.append(pred)
        ac.append(obs)
        prev = obs

        S = U @ Pp @ U.T + R
        try:
            Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp

        inn = a - ap
        Q = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(inn, inn)
        Q = (Q + Q.T) / 2 + np.eye(ka) * 1e-6

        otr = np.vstack([otr, qv])
        om2 = ewm_demean(otr, ewm)
        dm2 = otr - om2
        mf2 = dmd_full(dm2, k_svd=K_MAX)
        if mf2 is not None:
            F2, k2 = build_F_fn(mf2, K)
            U2 = mf2.metadata["U"][:, :k2]
            a = U2.T @ odm
            P = np.eye(k2)
            Q = np.eye(k2) * Q_INIT_SCALE
            F = F2
            R = sph_r(dm2, U2)
            U = U2
            ka = k2
            om = om2
            sr2, cn2 = _F_diagnostics(F2)
            srs.append(sr2)
            cns.append(cn2)

    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    if not np.all(np.isfinite(psa)):
        return None
    r2 = float(oos_r_squared(psa.ravel(), aca.ravel()))
    return r2, {"spectral_radii": srs, "cond_numbers": cns}


# ══════════════════════════════════════════════════════════════════════
#  A2: Shrinkage Sweep
# ══════════════════════════════════════════════════════════════════════


def run_a2_experiments(panel, test_years, K=K_DEFAULT, ewm=12, T_yr=5):
    """Sweep γ ∈ GAMMA_GRID. Returns dict[gamma] → list-of-per-window-r2."""
    results = {g: [] for g in GAMMA_GRID}
    diags = {g: [] for g in GAMMA_GRID}

    for ty in test_years:
        t0 = time.time()
        line_parts = [f"  W{ty}:"]
        for g in GAMMA_GRID:
            res = _run_smim_Ur(
                panel, ty, K, ewm, T_yr,
                build_F_fn=lambda mf, K_, g_=g: build_shrinkage_F(mf, K_, g_),
            )
            if res is not None:
                r2, diag = res
                results[g].append(r2)
                diags[g].append(diag)
            else:
                results[g].append(None)
                diags[g].append(None)
            line_parts.append(f"γ={g:.2f}:{results[g][-1]:.3f}" if results[g][-1] is not None else f"γ={g:.2f}:N/A")
        print("  ".join(line_parts) + f"  ({time.time()-t0:.1f}s)")

    return results, diags


# ══════════════════════════════════════════════════════════════════════
#  A4: Low-Rank-Plus-Diagonal F
# ══════════════════════════════════════════════════════════════════════


def run_a4_experiments(panel, test_years, K=K_DEFAULT, ewm=12, T_yr=5):
    """Test D, D+rank1, D+rank2, full Ã. Returns dict[label] → per-window-r2."""
    configs = [
        ("diag", 0),
        ("D+rank1", 1),
        ("D+rank2", 2),
        ("full_A", K),
    ]
    results = {label: [] for label, _ in configs}
    diags = {label: [] for label, _ in configs}

    for ty in test_years:
        t0 = time.time()
        line_parts = [f"  W{ty}:"]
        for label, rank in configs:
            res = _run_smim_Ur(
                panel, ty, K, ewm, T_yr,
                build_F_fn=lambda mf, K_, r_=rank: build_low_rank_F(mf, K_, r_),
            )
            if res is not None:
                r2, diag = res
                results[label].append(r2)
                diags[label].append(diag)
            else:
                results[label].append(None)
                diags[label].append(None)
            v = results[label][-1]
            line_parts.append(f"{label}:{v:.3f}" if v is not None else f"{label}:N/A")
        print("  ".join(line_parts) + f"  ({time.time()-t0:.1f}s)")

    return results, diags


# ══════════════════════════════════════════════════════════════════════
#  C1: Spectral Augmentation on Residuals
# ══════════════════════════════════════════════════════════════════════


def _compute_training_residuals(otr):
    """Pooled AR(1)+FE in-sample one-step residuals.

    Returns: residuals (T-1, N), rho, bar_y
    """
    rho, bar_y = estimate_pooled_ar1(otr)
    # ŷ_{t+1} = bar_y + rho * (y_t - bar_y)
    predicted = bar_y + rho * (otr[:-1] - bar_y)  # (T-1, N)
    residuals = otr[1:] - predicted               # (T-1, N)
    return residuals, rho, bar_y


def run_window_c1(panel, ty, K=K_DEFAULT, ewm=12, T_yr=5):
    """C1: pooled AR(1)+FE → DMD on residuals → combined prediction.

    Returns dict with pooled_r2, residual_r2, combined_r2, resid_persistence,
    or None.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # ── Stage 1: Pooled AR(1)+FE ──
    residuals, rho_pool, bar_y = _compute_training_residuals(otr)

    # ── Stage 2: DMD on residuals ──
    om_r = ewm_demean(residuals, ewm)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None

    # Use full Ã on residuals (A1c-style — best from Phase 1)
    A_r, ka = _get_Atilde_K(mf_r, K)
    F_r = _clip_spectral_radius(A_r)
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    # Residual persistence (diagnostic)
    resid_rhos = []
    for j in range(N):
        col = residuals[:, j]
        if len(col) >= 4 and np.std(col[:-1]) > 1e-10:
            c = np.corrcoef(col[:-1], col[1:])[0, 1]
            if np.isfinite(c):
                resid_rhos.append(c)
    resid_persistence = float(np.mean(resid_rhos)) if resid_rhos else 0.0

    # ── Rolling test ──
    ps_ar, ps_resid, ps_combined, ac = [], [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # Stage 1 predict
        y_ar = bar_y + rho_pool * (prev - bar_y)
        ps_ar.append(y_ar)

        # Stage 2 predict (residual Kalman)
        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        ps_resid.append(resid_pred)

        # Combined
        ps_combined.append(y_ar + resid_pred)
        ac.append(obs)

        # Actual residual for Kalman update
        actual_resid = obs - y_ar
        odm_r = actual_resid - om_r.ravel()

        # Kalman update on residual
        S_r = U_r @ Pp_r @ U_r.T + R_r
        try:
            Kg_r = Pp_r @ U_r.T @ np.linalg.solve(S_r, np.eye(N))
        except Exception:
            Kg_r = np.zeros((ka, N))
        a_r = ap_r + Kg_r @ (odm_r - U_r @ ap_r)
        P_r = (np.eye(ka) - Kg_r @ U_r) @ Pp_r

        inn_r = a_r - ap_r
        Q_r = (1 - LAMBDA_Q) * Q_r + LAMBDA_Q * np.outer(inn_r, inn_r)
        Q_r = (Q_r + Q_r.T) / 2 + np.eye(ka) * 1e-6

        prev = obs

        # Rolling update: re-estimate both stages
        otr = np.vstack([otr, qv])
        residuals_new, rho_pool, bar_y = _compute_training_residuals(otr)
        om_r = ewm_demean(residuals_new, ewm)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            A_r2, k2 = _get_Atilde_K(mf_r2, K)
            F_r = _clip_spectral_radius(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2)
            Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2)
            U_r = U_r2
            ka = k2

    if not ps_ar:
        return None
    ar_a = np.array(ps_ar)
    resid_a = np.array(ps_resid)
    comb_a = np.array(ps_combined)
    act_a = np.array(ac)

    if not np.all(np.isfinite(comb_a)):
        return None

    actual_resids = act_a - ar_a  # actual residuals for R² of residual model

    return {
        "pooled_r2": float(oos_r_squared(ar_a.ravel(), act_a.ravel())),
        "residual_r2": float(oos_r_squared(resid_a.ravel(), actual_resids.ravel())),
        "combined_r2": float(oos_r_squared(comb_a.ravel(), act_a.ravel())),
        "resid_persistence": resid_persistence,
    }


def run_c1_experiments(panel, test_years, K=K_DEFAULT, ewm=12, T_yr=5):
    results = []
    for ty in test_years:
        t0 = time.time()
        res = run_window_c1(panel, ty, K, ewm, T_yr)
        results.append(res)
        if res:
            print(f"  W{ty}: pooled={res['pooled_r2']:.4f}  resid={res['residual_r2']:.4f}"
                  f"  combined={res['combined_r2']:.4f}  ρ_resid={res['resid_persistence']:.3f}"
                  f"  ({time.time()-t0:.1f}s)")
        else:
            print(f"  W{ty}: FAILED  ({time.time()-t0:.1f}s)")
    return results


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


def print_a2_results(a2_results, a2_diags, ar1_r2s, test_years):
    print("\n" + "=" * 76)
    print("  A2 — SHRINKAGE SWEEP: F_γ = γÃ + (1-γ)·0.99I")
    print("=" * 76)

    ar1_mean = _mean_valid(ar1_r2s)
    bl_r2s = a2_results[0.0]
    bl_mean = _mean_valid(bl_r2s)
    a1c_r2s = a2_results[1.0]

    # Summary table
    print(f"\n  {'γ':>5s} {'R²':>7s} {'ΔR² AR1':>9s} {'ΔR² BL':>9s} {'ΔR² A1c':>9s}"
          f" {'W/AR1':>6s} {'W/BL':>5s} {'SR':>6s} {'κ(F)':>8s}")
    print(f"  {'-'*68}")

    for g in GAMMA_GRID:
        vals = a2_results[g]
        m = _mean_valid(vals)
        d_ar1 = m - ar1_mean if np.isfinite(m) else np.nan
        d_bl = m - bl_mean if np.isfinite(m) else np.nan
        a1c_mean = _mean_valid(a1c_r2s)
        d_a1c = m - a1c_mean if np.isfinite(m) else np.nan
        w_ar1, t_ar1 = _count_wins(vals, ar1_r2s)
        w_bl, t_bl = _count_wins(vals, bl_r2s)

        # Mean spectral radius and condition number
        srs = [d["spectral_radii"] for d in a2_diags[g] if d is not None]
        cns = [d["cond_numbers"] for d in a2_diags[g] if d is not None]
        sr_mean = float(np.mean([np.mean(s) for s in srs])) if srs else np.nan
        cn_mean = float(np.mean([np.mean(c) for c in cns])) if cns else np.nan

        r2_s = f"{m:7.4f}" if np.isfinite(m) else "  N/A  "
        print(f"  {g:5.2f} {r2_s} {d_ar1:+9.4f} {d_bl:+9.4f} {d_a1c:+9.4f}"
              f" {w_ar1:>3d}/{t_ar1} {w_bl:>2d}/{t_bl} {sr_mean:6.3f} {cn_mean:8.1f}")

    # Per-window detail
    print(f"\n  Per-window R² (γ grid):")
    print(f"  {'Year':>6s}", end="")
    for g in GAMMA_GRID:
        print(f"  {'γ='+str(g):>8s}", end="")
    print(f"  {'AR(1)':>8s}")
    print(f"  {'-'*(8 + 10*len(GAMMA_GRID) + 10)}")
    for i, ty in enumerate(test_years):
        print(f"  {ty:>6d}", end="")
        for g in GAMMA_GRID:
            v = a2_results[g][i]
            print(f"  {v:8.4f}" if v is not None else "     N/A ", end="")
        a1 = ar1_r2s[i]
        print(f"  {a1:8.4f}" if a1 is not None else "     N/A ")

    # CI: best γ vs baseline
    best_g = max(GAMMA_GRID, key=lambda g: _mean_valid(a2_results[g]))
    best_vals = a2_results[best_g]
    deltas = [b - bl for b, bl in zip(best_vals, bl_r2s) if b is not None and bl is not None]
    if len(deltas) >= 3:
        lo, hi = bootstrap_ci(np.array(deltas))
        print(f"\n  Best γ={best_g:.2f}: mean Δ vs BL = {np.mean(deltas):+.4f}"
              f"  CI [{lo:+.4f}, {hi:+.4f}]")
    # CI: best γ vs AR(1)
    deltas_ar1 = [b - a for b, a in zip(best_vals, ar1_r2s) if b is not None and a is not None]
    if len(deltas_ar1) >= 3:
        lo2, hi2 = bootstrap_ci(np.array(deltas_ar1))
        print(f"  Best γ={best_g:.2f}: mean Δ vs AR1 = {np.mean(deltas_ar1):+.4f}"
              f"  CI [{lo2:+.4f}, {hi2:+.4f}]")


def print_a4_results(a4_results, a4_diags, ar1_r2s, test_years):
    print("\n" + "=" * 76)
    print("  A4 — LOW-RANK-PLUS-DIAGONAL F: D + rank-r SVD(off-diag Ã)")
    print("=" * 76)

    ar1_mean = _mean_valid(ar1_r2s)
    bl_mean = _mean_valid(a4_results.get("diag", []))
    a1c_r2s = a4_results.get("full_A", [])
    a1c_mean = _mean_valid(a1c_r2s)
    a1c_gain = a1c_mean - bl_mean if np.isfinite(a1c_mean) and np.isfinite(bl_mean) else 0.0

    configs = ["diag", "D+rank1", "D+rank2", "full_A"]
    labels = {"diag": "Diagonal only", "D+rank1": "D + rank-1", "D+rank2": "D + rank-2", "full_A": "Full Ã (=A1c)"}

    print(f"\n  {'Variant':<18s} {'R²':>7s} {'ΔR² AR1':>9s} {'ΔR² diag':>9s}"
          f" {'%A1c gain':>10s} {'W/AR1':>6s}")
    print(f"  {'-'*62}")

    for cfg in configs:
        vals = a4_results[cfg]
        m = _mean_valid(vals)
        d_ar1 = m - ar1_mean if np.isfinite(m) else np.nan
        d_diag = m - bl_mean if np.isfinite(m) and np.isfinite(bl_mean) else np.nan
        pct = (d_diag / a1c_gain * 100) if np.isfinite(d_diag) and abs(a1c_gain) > 1e-6 else np.nan
        w, t = _count_wins(vals, ar1_r2s)
        r2_s = f"{m:7.4f}" if np.isfinite(m) else "  N/A  "
        pct_s = f"{pct:9.0f}%" if np.isfinite(pct) else "     N/A  "
        print(f"  {labels[cfg]:<18s} {r2_s} {d_ar1:+9.4f} {d_diag:+9.4f}"
              f" {pct_s} {w:>3d}/{t}")

    # Per-window detail
    print(f"\n  Per-window R²:")
    print(f"  {'Year':>6s}", end="")
    for cfg in configs:
        print(f"  {cfg:>10s}", end="")
    print(f"  {'AR(1)':>8s}")
    print(f"  {'-'*(8 + 12*len(configs) + 10)}")
    for i, ty in enumerate(test_years):
        print(f"  {ty:>6d}", end="")
        for cfg in configs:
            v = a4_results[cfg][i]
            print(f"  {v:10.4f}" if v is not None else "       N/A ", end="")
        a1 = ar1_r2s[i]
        print(f"  {a1:8.4f}" if a1 is not None else "     N/A ")

    # CI: D+rank1 vs diag
    r1_vals = a4_results["D+rank1"]
    d_vals = a4_results["diag"]
    deltas = [r - d for r, d in zip(r1_vals, d_vals) if r is not None and d is not None]
    if len(deltas) >= 3:
        lo, hi = bootstrap_ci(np.array(deltas))
        print(f"\n  D+rank1 vs diag: Δ = {np.mean(deltas):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")


def print_c1_results(c1_results, ar1_r2s, test_years):
    print("\n" + "=" * 76)
    print("  C1 — SPECTRAL AUGMENTATION ON POOLED+FE RESIDUALS")
    print("=" * 76)

    ar1_mean = _mean_valid(ar1_r2s)

    print(f"\n  {'Year':>6s} {'Pooled':>8s} {'Resid R²':>9s} {'Combined':>9s} {'AR(1)':>8s} {'ρ_resid':>8s}")
    print(f"  {'-'*52}")
    pooled_all, resid_all, combined_all, rho_resid_all = [], [], [], []
    for i, ty in enumerate(test_years):
        res = c1_results[i]
        if res is not None:
            pooled_all.append(res["pooled_r2"])
            resid_all.append(res["residual_r2"])
            combined_all.append(res["combined_r2"])
            rho_resid_all.append(res["resid_persistence"])
            a1 = ar1_r2s[i] if ar1_r2s[i] is not None else np.nan
            print(f"  {ty:>6d} {res['pooled_r2']:8.4f} {res['residual_r2']:9.4f}"
                  f" {res['combined_r2']:9.4f} {a1:8.4f} {res['resid_persistence']:8.3f}")
        else:
            print(f"  {ty:>6d}      N/A")

    p_mean = _mean_valid(pooled_all)
    r_mean = _mean_valid(resid_all)
    c_mean = _mean_valid(combined_all)
    rho_mean = _mean_valid(rho_resid_all)

    print(f"  {'-'*52}")
    print(f"  {'Mean':>6s} {p_mean:8.4f} {r_mean:9.4f} {c_mean:9.4f} {ar1_mean:8.4f} {rho_mean:8.3f}")

    # Key comparisons
    delta_combined_pooled = c_mean - p_mean if np.isfinite(c_mean) and np.isfinite(p_mean) else np.nan
    delta_combined_ar1 = c_mean - ar1_mean if np.isfinite(c_mean) else np.nan
    print(f"\n  Combined vs pooled+FE:  Δ = {delta_combined_pooled:+.4f}")
    print(f"  Combined vs per-actor AR(1): Δ = {delta_combined_ar1:+.4f}")

    # CI
    d_cp = [c - p for c, p in zip(combined_all, pooled_all) if np.isfinite(c) and np.isfinite(p)]
    if len(d_cp) >= 3:
        lo, hi = bootstrap_ci(np.array(d_cp))
        print(f"  Combined−pooled CI: [{lo:+.4f}, {hi:+.4f}]")

    d_ar1 = [c - a for c, a in zip(combined_all, ar1_r2s)
             if c is not None and a is not None and np.isfinite(c)]
    if len(d_ar1) >= 3:
        lo2, hi2 = bootstrap_ci(np.array(d_ar1))
        print(f"  Combined−AR(1) CI:  [{lo2:+.4f}, {hi2:+.4f}]")


# ══════════════════════════════════════════════════════════════════════
#  Save
# ══════════════════════════════════════════════════════════════════════


def save_results(a2_results, a4_results, c1_results, ar1_r2s, test_years):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # A2
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty, "ar1": ar1_r2s[i]}
        for g in GAMMA_GRID:
            row[f"gamma_{g:.2f}"] = a2_results[g][i]
        rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_1_phase2a_a2.parquet", index=False)

    # A4
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty, "ar1": ar1_r2s[i]}
        for cfg in a4_results:
            row[f"a4_{cfg}"] = a4_results[cfg][i]
        rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_1_phase2a_a4.parquet", index=False)

    # C1
    rows = []
    for i, ty in enumerate(test_years):
        row = {"year": ty, "ar1": ar1_r2s[i]}
        res = c1_results[i]
        if res:
            row.update({"c1_" + k: v for k, v in res.items()})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_1_phase2a_c1.parquet", index=False)

    print(f"\n  Saved: iter6_1_phase2a_{{a2,a4,c1}}.parquet")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════


def main():
    t_start = time.time()

    print("=" * 76)
    print("  ITERATION 6.1 PHASE 2a — REFINE THE TRANSITION MATRIX")
    print("  A2 (shrinkage) + A4 (low-rank) + C1 (augmentation)")
    print("=" * 76)

    panel, layer_labels = load_93_actor_panel()
    print(f"\nPanel: {panel.shape[0]}Q × {panel.shape[1]} actors")

    # Shared baselines
    print("\n  Computing baselines...")
    ar1_r2s = [run_window_ar1(panel, ty, T_yr=5) for ty in TEST_YEARS]
    pooled_r2s = [run_window_pooled(panel, ty, T_yr=5) for ty in TEST_YEARS]
    print(f"  AR(1) mean R² = {_mean_valid(ar1_r2s):.4f}")
    print(f"  Pooled+FE mean R² = {_mean_valid(pooled_r2s):.4f}")

    # ── A2 ──
    print(f"\n{'─'*76}")
    print("  A2: Shrinkage sweep γ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}")
    print(f"{'─'*76}")
    a2_results, a2_diags = run_a2_experiments(panel, TEST_YEARS)

    # ── A4 ──
    print(f"\n{'─'*76}")
    print("  A4: Low-rank-plus-diagonal F")
    print(f"{'─'*76}")
    a4_results, a4_diags = run_a4_experiments(panel, TEST_YEARS)

    # ── C1 ──
    print(f"\n{'─'*76}")
    print("  C1: Spectral augmentation on pooled+FE residuals")
    print(f"{'─'*76}")
    c1_results = run_c1_experiments(panel, TEST_YEARS)

    # ══════════════════════════════════════════════════════════════════
    #  Print consolidated results
    # ══════════════════════════════════════════════════════════════════
    print_a2_results(a2_results, a2_diags, ar1_r2s, TEST_YEARS)
    print_a4_results(a4_results, a4_diags, ar1_r2s, TEST_YEARS)
    print_c1_results(c1_results, ar1_r2s, TEST_YEARS)

    # ══════════════════════════════════════════════════════════════════
    #  Compact comparison & verdict
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  COMPACT COMPARISON TABLE")
    print("=" * 76)

    ar1_m = _mean_valid(ar1_r2s)
    pool_m = _mean_valid(pooled_r2s)
    bl_m = _mean_valid(a2_results[0.0])   # γ=0 = baseline
    a1c_m = _mean_valid(a2_results[1.0])  # γ=1 = A1c

    best_g = max(GAMMA_GRID, key=lambda g: _mean_valid(a2_results[g]))
    best_a2 = _mean_valid(a2_results[best_g])

    diag_m = _mean_valid(a4_results["diag"])
    dr1_m = _mean_valid(a4_results["D+rank1"])
    dr2_m = _mean_valid(a4_results["D+rank2"])
    full_m = _mean_valid(a4_results["full_A"])

    c1_combined = _mean_valid([r["combined_r2"] for r in c1_results if r])
    c1_pooled = _mean_valid([r["pooled_r2"] for r in c1_results if r])

    rows = [
        ("Per-actor AR(1)", ar1_m),
        ("Pooled+FE", pool_m),
        ("SMIM baseline (F=0.99I)", bl_m),
        (f"A2 best (γ={best_g:.2f})", best_a2),
        ("A1c (full Ã, =γ=1)", a1c_m),
        ("A4 diagonal only", diag_m),
        ("A4 D + rank-1", dr1_m),
        ("A4 D + rank-2", dr2_m),
        ("A4 full Ã", full_m),
        ("C1 pooled+FE alone", c1_pooled),
        ("C1 combined (pooled+resid)", c1_combined),
    ]

    print(f"\n  {'Model':<32s} {'R²':>7s} {'ΔR² AR1':>9s} {'ΔR² BL':>9s}")
    print(f"  {'-'*60}")
    for label, r2 in rows:
        d_ar1 = r2 - ar1_m if np.isfinite(r2) else np.nan
        d_bl = r2 - bl_m if np.isfinite(r2) and np.isfinite(bl_m) else np.nan
        r2_s = f"{r2:7.4f}" if np.isfinite(r2) else "  N/A  "
        d_ar1_s = f"{d_ar1:+9.4f}" if np.isfinite(d_ar1) else "    N/A  "
        d_bl_s = f"{d_bl:+9.4f}" if np.isfinite(d_bl) else "    N/A  "
        print(f"  {label:<32s} {r2_s} {d_ar1_s} {d_bl_s}")

    # ── Parsimony verdict ──
    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)

    a1c_gain_over_diag = full_m - diag_m if np.isfinite(full_m) and np.isfinite(diag_m) else 0
    r1_gain_over_diag = dr1_m - diag_m if np.isfinite(dr1_m) and np.isfinite(diag_m) else 0
    r2_gain_over_diag = dr2_m - diag_m if np.isfinite(dr2_m) and np.isfinite(diag_m) else 0

    if abs(a1c_gain_over_diag) > 1e-6:
        pct_r1 = r1_gain_over_diag / a1c_gain_over_diag * 100
        pct_r2 = r2_gain_over_diag / a1c_gain_over_diag * 100
    else:
        pct_r1 = pct_r2 = 0

    print(f"\n  A1c gain (full Ã over diagonal): {a1c_gain_over_diag:+.4f}")
    print(f"  Rank-1 recovers: {pct_r1:.0f}%  ({r1_gain_over_diag:+.4f})")
    print(f"  Rank-2 recovers: {pct_r2:.0f}%  ({r2_gain_over_diag:+.4f})")

    if pct_r1 >= 70:
        print(f"\n  → PARSIMONY: Rank-1 captures ≥70% of the A1c gain.")
        print(f"    Strong evidence for a single dominant cross-mode channel.")
    elif pct_r2 >= 70:
        print(f"\n  → PARSIMONY: Rank-2 captures ≥70% of the A1c gain.")
        print(f"    Two cross-mode channels dominate.")
    else:
        print(f"\n  → NO PARSIMONY: Low-rank approximations recover <70% of A1c gain.")
        print(f"    The off-diagonal coupling is distributed, not low-rank.")

    if np.isfinite(c1_combined) and np.isfinite(c1_pooled):
        c1_delta = c1_combined - c1_pooled
        if c1_delta > 0.01:
            print(f"\n  → C1: Combined model beats pooled+FE by {c1_delta:+.4f}.")
            print(f"    SMIM adds spectral value on top of the best linear baseline.")
        elif c1_delta > 0:
            print(f"\n  → C1: Combined model marginally better (+{c1_delta:.4f}).")
        else:
            print(f"\n  → C1: No spectral value-add in residuals ({c1_delta:+.4f}).")

    # Narrative recommendation
    if np.isfinite(best_a2) and best_a2 > ar1_m:
        print(f"\n  NARRATIVE: 'transition repair' — SMIM beats AR(1) after Ã-based F.")
    elif np.isfinite(c1_combined) and c1_combined > c1_pooled + 0.01:
        print(f"\n  NARRATIVE: 'spectral augmentation' — SMIM adds to pooled+FE"
              f" but does not beat AR(1) standalone.")
    else:
        print(f"\n  NARRATIVE: 'structural-negative' — predictive gains are modest,"
              f" paper should frame around the honest negative + diagnostic value.")

    save_results(a2_results, a4_results, c1_results, ar1_r2s, TEST_YEARS)

    print(f"\n  Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
