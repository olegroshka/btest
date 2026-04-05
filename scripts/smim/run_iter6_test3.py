#!/usr/bin/env python
"""
Iteration 6 Test 3 — 93-Actor Multilayer Panel: The Decisive Test.

Tests SMIM predictive R² on the heterogeneous 93-actor panel spanning
macro shocks (Layer 0), institutions (Layer 1), and firms (Layer 2).

Models:
  1. Per-actor AR(1)
  2. Pooled AR(1) + FE (single rho)
  3. Layer-specific pooled AR(1) + FE (3 rho values)
  4. DFM K=2, K=4, K=8
  5. SMIM K=2, K=4, K=8 (predictive)
  6. SMIM K=8 (modal, sanity check — expect ~0.696)

Runs at both T=5yr (paper structural config) and T=3yr (headline comparison).

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_test3.py
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

# SMIM hyperparameters (fixed, matching paper)
F_REG, Q_INIT_SCALE, LAMBDA_Q, K_MAX = 0.99, 0.5, 0.3, 15


# ══════════════════════════════════════════════════════════════════════
#  Panel Construction
# ══════════════════════════════════════════════════════════════════════


def load_93_actor_panel():
    """Load the 93-actor multilayer panel + layer labels.

    Intensities are already normalised to [0,1] per actor (minmax for macro/inst,
    cross-sectional rank for firms). Used as-is to match the structural analysis.

    Returns:
        panel: (T, 93) DataFrame, quarterly, sorted by time
        layer_labels: (93,) array of ints (0, 1, 2)
        actor_types: (93,) list of actor_type strings
    """
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # Build lookup tables
    layer_map = {a["actor_id"]: a["layer"] for a in registry["actors"]}
    type_map = {a["actor_id"]: a["actor_type"] for a in registry["actors"]}

    # Pivot to wide format
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Use data from 2005 onwards (maximise training)
    panel = panel.loc["2005-01-01":"2025-12-31"]

    # Get ordered column lists
    actors = list(panel.columns)
    layer_labels = np.array([layer_map.get(a, -1) for a in actors])
    actor_types = [type_map.get(a, "unknown") for a in actors]

    return panel, layer_labels, actor_types


# ══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def ar1_baseline(otr, N):
    """Per-actor AR(1) — per-column rho."""
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
    """Single shared rho via within-transformation OLS."""
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def estimate_layer_pooled_ar1(otr, layer_labels):
    """Within-transformation with layer-specific rho.

    Returns:
        rho_per_actor: (N,) array with layer-specific rho per actor
        bar_y: (N,) actor means
        rho_by_layer: dict {layer: rho}
    """
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    rho_per_actor = np.zeros(otr.shape[1])
    rho_by_layer = {}
    for layer in np.unique(layer_labels):
        mask = layer_labels == layer
        cols = tilde[:, mask]
        num = np.sum(cols[1:] * cols[:-1])
        den = np.sum(cols[:-1] ** 2)
        rho_layer = float(num / den) if den > 1e-12 else 0.0
        rho_per_actor[mask] = rho_layer
        rho_by_layer[int(layer)] = rho_layer
    return rho_per_actor, bar_y, rho_by_layer


def pca_basis(dm, k=2):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        C = dm.T @ dm / dm.shape[0]
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        return eigvecs[:, : min(k, N - 2)].real
    except Exception:
        return None


def estimate_var1(factors):
    K = factors.shape[1]
    X = factors[:-1]
    Y = factors[1:]
    try:
        B = np.linalg.solve(X.T @ X, X.T @ Y)
        A = B.T
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.5
    return A


def dmd_basis(dm, k=2):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, N))
        return mf.basis[:, : min(min(k, N - 2), mf.K)].real
    except Exception:
        return None


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def perm_test(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    obs = d.mean()
    cnt = sum(
        1 for _ in range(n) if (d * rng.choice([-1, 1], len(d))).mean() >= obs
    )
    return (cnt + 1) / (n + 1)


# ══════════════════════════════════════════════════════════════════════
#  Window runners
# ══════════════════════════════════════════════════════════════════════


def _prepare_window(panel, ty, T_yr=5):
    """Slice panel into training + test quarters. Returns None if insufficient data."""
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


def run_window_ar1(panel, ty, T_yr=5):
    """Per-actor AR(1). Returns (r2, per_actor_preds, actuals) or None."""
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
    return float(oos_r_squared(psa.ravel(), aca.ravel())), psa, aca


def run_window_pooled(panel, ty, T_yr=5):
    """Pooled AR(1) + FE (single rho), rolling."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    rho_pool, bar_y = estimate_pooled_ar1(otr)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    rho_hist = [rho_pool]
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        ps.append(bar_y + rho_pool * (prev - bar_y))
        ac.append(qv[0])
        prev = qv[0]
        otr = np.vstack([otr, qv])
        rho_pool, bar_y = estimate_pooled_ar1(otr)
        rho_hist.append(rho_pool)
    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    return float(oos_r_squared(psa.ravel(), aca.ravel())), float(np.mean(rho_hist))


def run_window_layer_pooled(panel, ty, layer_labels, T_yr=5):
    """Layer-specific pooled AR(1) + FE, rolling."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    # Build layer labels for valid columns in this window
    col_list = list(v)
    all_actors = list(panel.columns)
    col_idx = [all_actors.index(c) for c in col_list]
    ll = layer_labels[col_idx]

    rho_vec, bar_y, rho_by_layer = estimate_layer_pooled_ar1(otr, ll)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    rho_history = [rho_by_layer]

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        ps.append(bar_y + rho_vec * (prev - bar_y))
        ac.append(qv[0])
        prev = qv[0]
        otr = np.vstack([otr, qv])
        rho_vec, bar_y, rho_by_layer = estimate_layer_pooled_ar1(otr, ll)
        rho_history.append(rho_by_layer)

    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    r2 = float(oos_r_squared(psa.ravel(), aca.ravel()))

    # Average rho per layer across rolling windows
    avg_rho = {}
    for layer in np.unique(ll):
        vals = [rh.get(int(layer), 0.0) for rh in rho_history if int(layer) in rh]
        avg_rho[int(layer)] = float(np.mean(vals)) if vals else 0.0

    return r2, avg_rho


def run_window_dfm(panel, ty, K=2, ewm=12, T_yr=5):
    """DFM (PCA + VAR(1)), rolling."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    om = ewm_demean(otr, ewm)
    dm = otr - om
    U = pca_basis(dm, K)
    if U is None:
        return None
    factors = dm @ U
    if factors.shape[0] < 3:
        return None
    A = estimate_var1(factors)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        f_prev = U.T @ (prev - om.ravel())
        f_hat = A @ f_prev
        y_hat = om.ravel() + U @ f_hat
        ps.append(y_hat)
        ac.append(qv[0])
        prev = qv[0]
        otr = np.vstack([otr, qv])
        om = ewm_demean(otr, ewm)
        dm = otr - om
        U_new = pca_basis(dm, K)
        if U_new is not None:
            U = U_new
            factors = dm @ U
            if factors.shape[0] >= 3:
                A = estimate_var1(factors)
    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    if not np.all(np.isfinite(psa)):
        return None
    return float(oos_r_squared(psa.ravel(), aca.ravel()))


def run_window_smim(panel, ty, K=2, ewm=12, T_yr=5, return_modal=False):
    """SMIM (DMD + Kalman), rolling.

    Returns predictive R² by default. If return_modal=True, also returns
    modal (filtered) R² as a sanity check.
    """
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    om = ewm_demean(otr, ewm)
    dm = otr - om
    U = dmd_basis(dm, K)
    if U is None:
        return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

    ps_pred, ps_modal, ac = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        odm = obs - om.ravel()

        # Predictive step: α_{t|t-1}
        ap = F @ a
        Pp = F @ P @ F.T + Q
        ps_pred.append(U @ ap + om.ravel())
        ac.append(obs)
        prev = obs

        # Kalman update: α_{t|t}
        S = U @ Pp @ U.T + R
        try:
            Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp

        # Modal (filtered) prediction uses α_{t|t}
        if return_modal:
            ps_modal.append(U @ a + om.ravel())

        # Q adaptation
        inn = a - ap
        Q = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(inn, inn)
        Q = (Q + Q.T) / 2 + np.eye(ka) * 1e-6

        # Rolling basis update
        otr = np.vstack([otr, qv])
        om2 = ewm_demean(otr, ewm)
        dm2 = otr - om2
        U2 = dmd_basis(dm2, K)
        if U2 is not None:
            k2 = U2.shape[1]
            a = U2.T @ odm
            P = np.eye(k2)
            Q = np.eye(k2) * Q_INIT_SCALE
            F = np.eye(k2) * F_REG
            R = sph_r(dm2, U2)
            U = U2
            ka = k2
            om = om2

    if not ps_pred:
        return None
    psa_pred = np.array(ps_pred)
    aca = np.array(ac)
    if not np.all(np.isfinite(psa_pred)):
        return None

    r2_pred = float(oos_r_squared(psa_pred.ravel(), aca.ravel()))

    if return_modal and ps_modal:
        psa_modal = np.array(ps_modal)
        r2_modal = float(oos_r_squared(psa_modal.ravel(), aca.ravel()))
        return r2_pred, r2_modal
    return r2_pred


def run_window_ar1_per_layer(panel, ty, layer_labels, T_yr=5):
    """Per-actor AR(1) decomposed by layer. Returns dict {layer: r2}."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    all_actors = list(panel.columns)
    col_idx = [all_actors.index(c) for c in v]
    ll = layer_labels[col_idx]

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

    result = {}
    for layer in np.unique(ll):
        mask = ll == layer
        p_layer = psa[:, mask].ravel()
        a_layer = aca[:, mask].ravel()
        result[int(layer)] = float(oos_r_squared(p_layer, a_layer))
    return result


def run_window_smim_per_layer(panel, ty, layer_labels, K=8, ewm=12, T_yr=5):
    """SMIM predictive R² decomposed by layer. Returns dict {layer: r2}."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep

    all_actors = list(panel.columns)
    col_idx = [all_actors.index(c) for c in v]
    ll = layer_labels[col_idx]

    om = ewm_demean(otr, ewm)
    dm = otr - om
    U = dmd_basis(dm, K)
    if U is None:
        return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

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
        ps.append(U @ ap + om.ravel())
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
        U2 = dmd_basis(dm2, K)
        if U2 is not None:
            k2 = U2.shape[1]
            a = U2.T @ odm
            P = np.eye(k2)
            Q = np.eye(k2) * Q_INIT_SCALE
            F = np.eye(k2) * F_REG
            R = sph_r(dm2, U2)
            U = U2
            ka = k2
            om = om2

    if not ps:
        return None
    psa, aca = np.array(ps), np.array(ac)
    if not np.all(np.isfinite(psa)):
        return None

    result = {}
    for layer in np.unique(ll):
        mask = ll == layer
        p_layer = psa[:, mask].ravel()
        a_layer = aca[:, mask].ravel()
        result[int(layer)] = float(oos_r_squared(p_layer, a_layer))
    return result


# ══════════════════════════════════════════════════════════════════════
#  Main runner
# ══════════════════════════════════════════════════════════════════════


def run_config(panel, layer_labels, actor_types, T_yr, ewm=12):
    """Run all models at one configuration."""
    label = f"T={T_yr}yr, tau={ewm}Q"
    print(f"\n{'='*80}")
    print(f"  CONFIG: {label}  (93-actor panel)")
    print(f"{'='*80}")

    rows = []
    layer_r2_ar1 = []
    layer_r2_smim = []

    for ty in TEST_YEARS:
        t1 = time.time()
        row = {"year": ty}

        # 1. Per-actor AR(1)
        res_ar1 = run_window_ar1(panel, ty, T_yr)
        row["ar1"] = res_ar1[0] if res_ar1 else None

        # 2. Pooled+FE (single rho)
        res_pool = run_window_pooled(panel, ty, T_yr)
        if res_pool:
            row["pooled_single"], row["rho_single"] = res_pool
        else:
            row["pooled_single"] = row["rho_single"] = None

        # 3. Layer-specific pooled+FE
        res_layer = run_window_layer_pooled(panel, ty, layer_labels, T_yr)
        if res_layer:
            row["pooled_layer"], rho_layers = res_layer
            for l, rho in rho_layers.items():
                row[f"rho_L{l}"] = rho
        else:
            row["pooled_layer"] = None

        # 4. DFM K=2, K=4, K=8
        row["dfm_k2"] = run_window_dfm(panel, ty, K=2, ewm=ewm, T_yr=T_yr)
        row["dfm_k4"] = run_window_dfm(panel, ty, K=4, ewm=ewm, T_yr=T_yr)
        row["dfm_k8"] = run_window_dfm(panel, ty, K=8, ewm=ewm, T_yr=T_yr)

        # 5. SMIM K=2, K=4 (predictive only)
        row["smim_k2"] = run_window_smim(panel, ty, K=2, ewm=ewm, T_yr=T_yr)
        row["smim_k4"] = run_window_smim(panel, ty, K=4, ewm=ewm, T_yr=T_yr)

        # 6. SMIM K=8 (predictive + modal sanity)
        res_k8 = run_window_smim(panel, ty, K=8, ewm=ewm, T_yr=T_yr, return_modal=True)
        if res_k8 is not None:
            if isinstance(res_k8, tuple):
                row["smim_k8"], row["smim_k8_modal"] = res_k8
            else:
                row["smim_k8"] = res_k8
                row["smim_k8_modal"] = None
        else:
            row["smim_k8"] = row["smim_k8_modal"] = None

        # Per-layer breakdown (AR1 and SMIM K=8)
        ar1_pl = run_window_ar1_per_layer(panel, ty, layer_labels, T_yr)
        if ar1_pl:
            layer_r2_ar1.append(ar1_pl)
        smim_pl = run_window_smim_per_layer(panel, ty, layer_labels, K=8, ewm=ewm, T_yr=T_yr)
        if smim_pl:
            layer_r2_smim.append(smim_pl)

        rows.append(row)
        elapsed = time.time() - t1

        ar1_s = f"{row['ar1']:.4f}" if row['ar1'] is not None else "  N/A "
        pool_s = f"{row['pooled_single']:.4f}" if row['pooled_single'] is not None else "  N/A "
        layer_s = f"{row['pooled_layer']:.4f}" if row['pooled_layer'] is not None else "  N/A "
        smim8_s = f"{row['smim_k8']:.4f}" if row['smim_k8'] is not None else "  N/A "
        modal_s = f"{row.get('smim_k8_modal', None):.4f}" if row.get('smim_k8_modal') is not None else "  N/A "
        print(
            f"  W{ty}: AR1={ar1_s}  Pool={pool_s}  Layer={layer_s}"
            f"  SMIM8={smim8_s}  modal={modal_s}  ({elapsed:.1f}s)"
        )

    df = pd.DataFrame(rows)

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print(f"  93-ACTOR PANEL SUMMARY ({label})")
    print(f"{'='*80}")
    print(f"  {'Model':<30s} {'R2':>8s} {'DR2 vs AR1':>12s} {'Wins':>6s}")
    print(f"  {'-'*60}")

    ar1_mean = df["ar1"].dropna().mean()
    print(f"  {'Per-actor AR(1)':<30s} {ar1_mean:8.4f}")

    model_cols = [
        ("pooled_single", "Pooled+FE (single rho)"),
        ("pooled_layer", "Layer-specific pooled+FE"),
        ("dfm_k2", "DFM K=2"),
        ("dfm_k4", "DFM K=4"),
        ("dfm_k8", "DFM K=8"),
        ("smim_k2", "SMIM K=2 (predictive)"),
        ("smim_k4", "SMIM K=4 (predictive)"),
        ("smim_k8", "SMIM K=8 (predictive)"),
    ]
    for col, label_name in model_cols:
        vals = df[col].dropna()
        ar1_vals = df.loc[vals.index, "ar1"]
        if len(vals) == 0:
            print(f"  {label_name:<30s} {'N/A':>8s}")
            continue
        mean_r2 = vals.mean()
        delta = vals.values - ar1_vals.values
        mean_delta = delta.mean()
        wins = int((delta > 0).sum())
        n = len(vals)
        print(f"  {label_name:<30s} {mean_r2:8.4f} {mean_delta:+12.4f} {wins:>3d}/{n}")

    # Modal sanity check
    modal_vals = df["smim_k8_modal"].dropna()
    if len(modal_vals) > 0:
        print(f"\n  {'SMIM K=8 (modal, sanity)':<30s} {modal_vals.mean():8.4f}")

    # ── Layer-specific rho ──
    print(f"\n  LAYER-SPECIFIC RHO:")
    for layer in sorted(set(layer_labels)):
        col = f"rho_L{layer}"
        if col in df.columns:
            vals = df[col].dropna()
            n_actors = (layer_labels == layer).sum()
            if len(vals) > 0:
                print(f"    Layer {layer}: rho={vals.mean():.4f} ({n_actors} actors)")

    # ── Statistical Inference ──
    print(f"\n  STATISTICAL INFERENCE:")
    for col, label_name in [
        ("pooled_single", "Pooled+FE single"),
        ("pooled_layer", "Layer-specific pool"),
        ("dfm_k8", "DFM K=8"),
        ("smim_k2", "SMIM K=2"),
        ("smim_k4", "SMIM K=4"),
        ("smim_k8", "SMIM K=8"),
    ]:
        vals = df[col].dropna()
        ar1_vals = df.loc[vals.index, "ar1"]
        if len(vals) < 3:
            continue
        delta = vals.values - ar1_vals.values
        lo, hi = bootstrap_ci(delta)
        pp = perm_test(delta)
        wins = int((delta > 0).sum())
        print(
            f"    {label_name:<22s} D={delta.mean():+.4f}  CI [{lo:+.4f},{hi:+.4f}]"
            f"  wins={wins}/{len(delta)}  perm p={pp:.4f}"
        )

    # Key comparison: SMIM K=8 vs layer-specific pooled
    print(f"\n  KEY COMPARISONS:")
    for smim_col, smim_label in [
        ("smim_k4", "SMIM K=4"),
        ("smim_k8", "SMIM K=8"),
    ]:
        smim_vals = df[smim_col].dropna()
        layer_vals = df.loc[smim_vals.index, "pooled_layer"]
        if len(smim_vals) >= 3 and not layer_vals.isna().any():
            d_sl = smim_vals.values - layer_vals.values
            lo_sl, hi_sl = bootstrap_ci(d_sl)
            pp_sl = perm_test(d_sl)
            wins_sl = int((d_sl > 0).sum())
            print(
                f"    {smim_label} vs Layer-Pool: D={d_sl.mean():+.4f}"
                f"  CI [{lo_sl:+.4f},{hi_sl:+.4f}]"
                f"  wins={wins_sl}/{len(d_sl)}  perm p={pp_sl:.4f}"
            )

    # SMIM vs single pooled
    for smim_col, smim_label in [
        ("smim_k4", "SMIM K=4"),
        ("smim_k8", "SMIM K=8"),
    ]:
        smim_vals = df[smim_col].dropna()
        pool_vals = df.loc[smim_vals.index, "pooled_single"]
        if len(smim_vals) >= 3 and not pool_vals.isna().any():
            d_sp = smim_vals.values - pool_vals.values
            lo_sp, hi_sp = bootstrap_ci(d_sp)
            pp_sp = perm_test(d_sp)
            wins_sp = int((d_sp > 0).sum())
            print(
                f"    {smim_label} vs Single-Pool: D={d_sp.mean():+.4f}"
                f"  CI [{lo_sp:+.4f},{hi_sp:+.4f}]"
                f"  wins={wins_sp}/{len(d_sp)}  perm p={pp_sp:.4f}"
            )

    # ── Modal-Predictive Gap ──
    pred_vals = df["smim_k8"].dropna()
    modal_vals_clean = df.loc[pred_vals.index, "smim_k8_modal"].dropna()
    if len(pred_vals) > 0 and len(modal_vals_clean) > 0:
        pred_mean = pred_vals.mean()
        modal_mean = modal_vals_clean.mean()
        gap = modal_mean - pred_mean
        print(f"\n  MODAL-PREDICTIVE GAP (K=8):")
        print(f"    Modal R² (K=8):      {modal_mean:.4f}  (expect ~0.696)")
        print(f"    Predictive R² (K=8): {pred_mean:.4f}")
        print(f"    Gap:                  {gap:.4f}")
        if gap < 0.10:
            print(f"    Interpretation: Gap < 0.10 -> most spectral structure is forecastable (STRONG)")
        elif gap < 0.25:
            print(f"    Interpretation: Gap 0.10-0.25 -> partial forecastability (MODERATE)")
        else:
            print(f"    Interpretation: Gap > 0.25 -> spectral structure is mostly descriptive (WEAK)")

    # ── Per-Layer Breakdown ──
    if layer_r2_ar1 and layer_r2_smim:
        print(f"\n  PER-LAYER R² BREAKDOWN (K=8):")
        all_layers = sorted(set(layer_labels))
        layer_names = {0: "macro", 1: "institutions", 2: "firms"}
        for layer in all_layers:
            ar1_l = [d.get(int(layer), np.nan) for d in layer_r2_ar1 if int(layer) in d]
            smim_l = [d.get(int(layer), np.nan) for d in layer_r2_smim if int(layer) in d]
            n_actors = (layer_labels == layer).sum()
            if ar1_l and smim_l:
                ar1_m = np.nanmean(ar1_l)
                smim_m = np.nanmean(smim_l)
                d_l = smim_m - ar1_m
                name = layer_names.get(int(layer), f"layer_{layer}")
                print(
                    f"    Layer {layer} ({name:12s}, {n_actors:2d} actors):"
                    f"  AR1={ar1_m:.4f}  SMIM={smim_m:.4f}  D={d_l:+.4f}"
                )

    # ── Quality Gates ──
    print(f"\n  QUALITY GATES:")
    n_actors = len(panel.columns)
    qg1 = n_actors == 93
    print(f"    QG1: Panel has {n_actors} actors (expect 93) {'PASS' if qg1 else 'FAIL'}")

    unique_layers = sorted(set(layer_labels))
    layer_counts = {l: (layer_labels == l).sum() for l in unique_layers}
    rho_l0 = df.get("rho_L0", pd.Series(dtype=float)).dropna()
    rho_l2 = df.get("rho_L2", pd.Series(dtype=float)).dropna()
    if len(rho_l0) > 0 and len(rho_l2) > 0:
        qg2 = rho_l0.mean() != rho_l2.mean()
        print(f"    QG2: Persistence heterogeneity: rho_L0={rho_l0.mean():.3f}"
              f" vs rho_L2={rho_l2.mean():.3f} {'PASS' if qg2 else 'FAIL'}")

    if len(modal_vals_clean) > 0:
        qg3_pass = abs(modal_vals_clean.mean() - 0.696) < 0.15
        print(f"    QG3: SMIM modal R² at K=8 = {modal_vals_clean.mean():.4f}"
              f" (expect ~0.696) {'PASS' if qg3_pass else 'CHECK'}")

    if len(pred_vals) > 0 and len(modal_vals_clean) > 0:
        qg4 = pred_vals.mean() <= modal_vals_clean.mean() + 0.01
        print(f"    QG4: Predictive R² ({pred_vals.mean():.4f})"
              f" <= Modal R² ({modal_vals_clean.mean():.4f})"
              f" {'PASS' if qg4 else 'FAIL'}")

    all_finite = True
    for col in ["ar1", "pooled_single", "pooled_layer", "smim_k8"]:
        vals = df[col].dropna()
        if len(vals) > 0 and not all(np.isfinite(v) for v in vals):
            all_finite = False
    print(f"    QG5: No NaN/Inf in predictions {'PASS' if all_finite else 'FAIL'}")

    ar1_check = df["ar1"].dropna()
    qg6 = len(ar1_check) >= 8 and ar1_check.mean() > 0.0
    print(f"    QG6: AR(1) R²={ar1_check.mean():.4f} over {len(ar1_check)} windows"
          f" {'PASS' if qg6 else 'FAIL'}")

    # ── Verdict ──
    print(f"\n  VERDICT ({label}):")
    smim_k8_vals = df["smim_k8"].dropna()
    smim_k4_vals = df["smim_k4"].dropna()
    layer_pool_vals = df["pooled_layer"].dropna()
    single_pool_vals = df["pooled_single"].dropna()

    verdict = "INCONCLUSIVE"

    # Check SMIM K>=4 vs layer-specific pooled
    for kk, kv in [("K=8", smim_k8_vals), ("K=4", smim_k4_vals)]:
        common = kv.index.intersection(layer_pool_vals.index)
        if len(common) >= 3:
            d_test = kv.loc[common].values - layer_pool_vals.loc[common].values
            lo_t, hi_t = bootstrap_ci(d_test)
            wins_t = int((d_test > 0).sum())
            if d_test.mean() > 0.01 and lo_t > 0 and wins_t >= 7:
                verdict = f"SMIM WINS ({kk} beats layer-specific pool)"
                break
            elif d_test.mean() > 0:
                # Check vs single pool
                common2 = kv.index.intersection(single_pool_vals.index)
                d_test2 = kv.loc[common2].values - single_pool_vals.loc[common2].values
                lo_t2, _ = bootstrap_ci(d_test2)
                if d_test2.mean() > 0 and lo_t2 > 0:
                    verdict = f"PARTIAL ({kk} beats single-pool but not layer-pool)"

    if verdict == "INCONCLUSIVE":
        # Check if SMIM at least beats single pooled
        for kk, kv in [("K=8", smim_k8_vals), ("K=4", smim_k4_vals)]:
            common = kv.index.intersection(single_pool_vals.index)
            if len(common) >= 3:
                d_test = kv.loc[common].values - single_pool_vals.loc[common].values
                if d_test.mean() <= 0:
                    verdict = "SMIM MATCHES/LOSES"
                    break

    print(f"    {verdict}")
    if "WINS" in verdict:
        print(f"    -> Session 3 = paper rewrite (Scenario A)")
    elif "PARTIAL" in verdict:
        print(f"    -> Session 3 = decide reframe (Scenario A-half)")
    elif "MATCHES" in verdict or "LOSES" in verdict:
        print(f"    -> Session 3 = decide negative result or shelve (Scenario C/D)")
    else:
        print(f"    -> Examine results more carefully")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"t{T_yr}yr"
    df.to_parquet(METRICS_DIR / f"iter6_test3_{tag}.parquet", index=False)
    print(f"\n  Saved: iter6_test3_{tag}.parquet")

    return df


def main():
    t0 = time.time()
    panel, layer_labels, actor_types = load_93_actor_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    print(f"Layers: {dict(zip(*np.unique(layer_labels, return_counts=True)))}")
    print(f"Time range: {panel.index.min()} to {panel.index.max()}")

    # Persistence characterisation
    print(f"\n  PERSISTENCE CHARACTERISATION:")
    vals = panel.loc["2010-01-01":].fillna(panel.loc["2010-01-01":].mean())
    for layer in sorted(set(layer_labels)):
        mask = layer_labels == layer
        cols = [c for c, m in zip(panel.columns, mask) if m]
        rhos = []
        for c in cols:
            s = vals[c].values
            if len(s) > 4 and np.std(s[:-1]) > 1e-10:
                r = np.corrcoef(s[:-1], s[1:])[0, 1]
                if np.isfinite(r):
                    rhos.append(r)
        if rhos:
            print(
                f"    Layer {layer} ({len(cols):2d} actors): "
                f"median rho={np.median(rhos):.3f}  "
                f"range=[{min(rhos):.3f}, {max(rhos):.3f}]"
            )

    # Run at both T=5yr and T=3yr
    df_5yr = run_config(panel, layer_labels, actor_types, T_yr=5, ewm=12)
    df_3yr = run_config(panel, layer_labels, actor_types, T_yr=3, ewm=12)

    print(f"\n{'='*80}")
    print(f"  TOTAL TIME: {time.time() - t0:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
