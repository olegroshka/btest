#!/usr/bin/env python
"""
Referee Round 2 — Four additional experiments addressing reviewer concerns.

  Exp 1: M2 vs BA_M2 direct DM test (from saved metrics)
  Exp 2: Campbell-Thompson R²_OOS (training-set mean denominator)
  Exp 3: Single-stage Ridge with block dummies
  Exp 4: EWM half-life sensitivity (hl = 4, 6, 8, 12)

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_referee_round2.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))

K_DEFAULT = 8
K_MAX = 15
Q_INIT_SCALE = 0.5
LAMBDA_Q = 0.3


# ══════════════════════════════════════════════════════════════════════
#  Shared infrastructure (from run_iter6_4b.py)
# ══════════════════════════════════════════════════════════════════════

def load_panel_and_meta():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    meta = {a["actor_id"]: a for a in registry["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel, meta


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    return float(num / den) if den > 1e-12 else 0.0, bar_y


def paired_t_test(d):
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    if se < 1e-15:
        return np.nan, np.nan
    return float(m / se), float(2 * t_dist.sf(abs(m / se), df=len(d) - 1))


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.array([x for x in d if np.isfinite(x)])
    if len(d) < 3:
        return np.nan, np.nan
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def define_blocks(panel, meta):
    actors = list(panel.columns)
    blocks = {}
    blocks["SEC_diversified"] = [a for a in actors
                                  if meta.get(a, {}).get("sector") == "diversified"]
    blocks["LAYER_macro_inst"] = [a for a in actors
                                   if meta.get(a, {}).get("layer", -1) in (0, 1)]
    blocks["MERGED_tech_health"] = [a for a in actors
                                     if meta.get(a, {}).get("sector") in ("technology", "healthcare")]
    local_actors = set()
    for b_actors in blocks.values():
        local_actors.update(b_actors)
    blocks["REMAINDER"] = [a for a in actors if a not in local_actors]
    return blocks


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 5:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def fit_local_pca_ridge(dm_block, K_b):
    N_b = dm_block.shape[1]
    ka = min(K_b, N_b - 2)
    C = dm_block.T @ dm_block / dm_block.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    U_pca = eigvecs[:, idx][:, :ka]
    factors = dm_block @ U_pca
    X, Y = factors[:-1], factors[1:]
    try:
        A = (np.linalg.solve(X.T @ X + 1.0 * np.eye(ka), X.T @ Y)).T
    except np.linalg.LinAlgError:
        A = np.eye(ka) * 0.5
    return U_pca, A


from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer


def dmd_full(dm, k_svd=K_MAX):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def _clip_sr(F, max_sr=0.99):
    ev = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(ev)))
    return F * (max_sr / mx) if mx > max_sr else F


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


# ══════════════════════════════════════════════════════════════════════
#  Experiment 1: M2 vs BA_M2 direct DM test (from saved per-window R²)
# ══════════════════════════════════════════════════════════════════════

def exp1_m2_vs_bam2():
    print("\n" + "=" * 90)
    print("  EXP 1: M2 vs BA_M2 — Direct DM Test")
    print("=" * 90)

    path = METRICS_DIR / "iter6_4b.parquet"
    if not path.exists():
        print("  ERROR: iter6_4b.parquet not found. Run run_iter6_4b.py first.")
        return {}

    df = pd.read_parquet(path)
    m2 = df[df["architecture"] == "M2"].sort_values("year")["full_r2"].values
    bam2 = df[df["architecture"] == "BA_M2"].sort_values("year")["full_r2"].values

    if len(m2) != len(bam2) or len(m2) == 0:
        print("  ERROR: mismatched or empty arrays")
        return {}

    deltas = m2 - bam2
    mean_d = np.mean(deltas)
    t_stat, p_val = paired_t_test(deltas)
    ci = bootstrap_ci(deltas)
    wins = sum(1 for d in deltas if d > 0)

    # Also compute lag-1 autocorrelation of the loss differentials (for paper fix 2)
    if len(deltas) > 2:
        d_dm = deltas - deltas.mean()
        rho_d = float(np.sum(d_dm[1:] * d_dm[:-1]) / np.sum(d_dm ** 2)) if np.sum(d_dm ** 2) > 0 else 0.0
    else:
        rho_d = 0.0

    print(f"\n  M2 per-window R²:     {m2}")
    print(f"  BA_M2 per-window R²:  {bam2}")
    print(f"  Δ(M2 - BA_M2):       {deltas}")
    print(f"\n  Mean Δ = {mean_d:+.4f}")
    print(f"  t = {t_stat:.3f}   p = {p_val:.4f}")
    print(f"  Bootstrap CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  Windows positive: {wins}/{len(deltas)}")
    print(f"\n  Lag-1 autocorrelation of loss diff (ρ_d): {rho_d:.3f}")

    return {"mean_delta": mean_d, "t": t_stat, "p": p_val,
            "ci_lo": ci[0], "ci_hi": ci[1], "wins": wins, "rho_d": rho_d}


# ══════════════════════════════════════════════════════════════════════
#  Experiment 1b: Lag-1 autocorrelation of M2-G1 loss differentials
#  (for the "effective independent blocks" claim in the paper)
# ══════════════════════════════════════════════════════════════════════

def exp1b_loss_diff_autocorrelation():
    print("\n" + "=" * 90)
    print("  EXP 1b: M2 vs G1 — Loss Differential Autocorrelation")
    print("=" * 90)

    path = METRICS_DIR / "iter6_4b.parquet"
    if not path.exists():
        print("  ERROR: iter6_4b.parquet not found.")
        return {}

    df = pd.read_parquet(path)
    m2 = df[df["architecture"] == "M2"].sort_values("year")["full_r2"].values
    g1 = df[df["architecture"] == "G1"].sort_values("year")["full_r2"].values

    deltas = m2 - g1  # these are R² differentials (proxy for loss differentials)
    d_dm = deltas - deltas.mean()
    rho_d = float(np.sum(d_dm[1:] * d_dm[:-1]) / np.sum(d_dm ** 2)) if np.sum(d_dm ** 2) > 0 else 0.0
    n_eff = len(deltas) * (1 - rho_d) / (1 + rho_d)

    print(f"\n  Δ(M2 - G1) per window: {deltas}")
    print(f"  Lag-1 autocorrelation ρ_d = {rho_d:.3f}")
    print(f"  n_eff = n × (1 - ρ_d)/(1 + ρ_d) = {len(deltas)} × {(1-rho_d)/(1+rho_d):.3f} = {n_eff:.1f}")
    print(f"  Sign-test p at n_eff = {int(round(n_eff))}: 2^(-{int(round(n_eff))}) = {2**(-int(round(n_eff))):.6f}")

    return {"rho_d": rho_d, "n_eff": n_eff}


# ══════════════════════════════════════════════════════════════════════
#  Experiment 2: Campbell-Thompson R²_OOS (training-set mean denominator)
# ══════════════════════════════════════════════════════════════════════

def ct_r_squared(predicted, actual, train_mean):
    """Campbell-Thompson OOS R² using training-set mean in denominator."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - train_mean) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def run_window_ct(panel, ty, blocks, T_yr=5):
    """Run G1 and M2 for one window, returning both standard and CT R²."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    actor_block = {}
    for bname, bactors in blocks.items():
        for a in bactors:
            if a in v_list:
                actor_block[a] = bname
    for a in v_list:
        if a not in actor_block:
            actor_block[a] = "REMAINDER"

    block_indices = {}
    for bname in blocks:
        block_indices[bname] = [v_list.index(a) for a in v_list if actor_block.get(a) == bname]
    local_block_names = [b for b in blocks if b != "REMAINDER"]

    # Stage 1
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Global augmentation
    om_r = ewm_demean(residuals, 12)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None

    ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
    A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
    F_r = _clip_sr(A_r)
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    # Local models
    local_models = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        if len(bidx) < 5:
            local_models[bname] = None
            continue
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        N_b = len(bidx)
        K_b = min(4, max(2, N_b // 5))
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)
        local_models[bname] = {
            "U_pca": U_pca, "A_pca": A_pca, "om_b": om_b, "K_b": K_b,
            "bidx": bidx, "N_b": N_b,
        }

    # Training-set mean (for CT R²)
    train_mean = otr.mean(axis=0)  # per-actor training mean

    preds_g1, preds_m2 = [], []
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_pool = bar_y + rho * (prev - bar_y)

        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_g1 = y_pool + resid_pred_global

        # M2
        y_m2 = y_g1.copy()
        prev_resid = prev - (bar_y + rho * (
            np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y
        )) if otr.shape[0] >= 2 else np.zeros(N)
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            f_pca = lm["U_pca"].T @ prev_resid_b
            local_pred = lm["U_pca"] @ (lm["A_pca"] @ f_pca) + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_m2[bidx] = y_pool[bidx] + local_pred
            else:
                y_m2[bidx] = y_pool[bidx]

        preds_g1.append(y_g1)
        preds_m2.append(y_m2)
        actuals.append(obs)

        # Kalman update + rolling re-estimation
        actual_resid = obs - y_pool
        odm_r = actual_resid - om_r.ravel()
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

        otr = np.vstack([otr, qv])
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, 12)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            k2 = min(K_DEFAULT, mf_r2.basis.shape[0] - 2, mf_r2.K)
            A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
            F_r = _clip_sr(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2

        residuals = residuals_new
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]
            lm["U_pca"], lm["A_pca"] = fit_local_pca_ridge(dm_b, K_b)
            lm["om_b"] = om_b

    if not actuals:
        return None

    act_a = np.array(actuals)
    pred_g1_a = np.array(preds_g1)
    pred_m2_a = np.array(preds_m2)

    from quantdsl_backtest.smim.validation.metrics import oos_r_squared

    # Standard R² (test-window mean denominator)
    std_g1 = float(oos_r_squared(pred_g1_a.ravel(), act_a.ravel()))
    std_m2 = float(oos_r_squared(pred_m2_a.ravel(), act_a.ravel()))

    # CT R² (training-set mean denominator)
    # train_mean is per-actor; tile to match (T_test, N)
    tm_tiled = np.tile(train_mean, (act_a.shape[0], 1))
    ct_g1 = ct_r_squared(pred_g1_a.ravel(), act_a.ravel(), tm_tiled.ravel())
    ct_m2 = ct_r_squared(pred_m2_a.ravel(), act_a.ravel(), tm_tiled.ravel())

    return {
        "std_g1": std_g1, "std_m2": std_m2, "std_delta": std_m2 - std_g1,
        "ct_g1": ct_g1, "ct_m2": ct_m2, "ct_delta": ct_m2 - ct_g1,
    }


def exp2_campbell_thompson():
    print("\n" + "=" * 90)
    print("  EXP 2: Campbell-Thompson R²_OOS (Training-Set Mean Denominator)")
    print("=" * 90)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta)

    results = []
    for ty in TEST_YEARS:
        r = run_window_ct(panel, ty, blocks)
        if r is not None:
            results.append(r)
            print(f"  W{ty}: Std G1={r['std_g1']:.4f} M2={r['std_m2']:.4f} Δ={r['std_delta']:+.4f}"
                  f"  |  CT G1={r['ct_g1']:.4f} M2={r['ct_m2']:.4f} Δ={r['ct_delta']:+.4f}")

    if not results:
        return {}

    std_deltas = [r["std_delta"] for r in results]
    ct_deltas = [r["ct_delta"] for r in results]

    print(f"\n  Standard R² (test-window mean):  mean Δ = {np.mean(std_deltas):+.4f}")
    print(f"  Campbell-Thompson R² (train mean): mean Δ = {np.mean(ct_deltas):+.4f}")
    print(f"  Difference in Δ: {np.mean(ct_deltas) - np.mean(std_deltas):+.4f}")

    t_ct, p_ct = paired_t_test(ct_deltas)
    ci_ct = bootstrap_ci(ct_deltas)
    wins_ct = sum(1 for d in ct_deltas if d > 0)
    print(f"\n  CT Δ: t = {t_ct:.3f}  p = {p_ct:.4f}  CI = [{ci_ct[0]:+.4f}, {ci_ct[1]:+.4f}]"
          f"  W = {wins_ct}/{len(ct_deltas)}")

    # CT absolute levels
    ct_g1_mean = np.mean([r["ct_g1"] for r in results])
    ct_m2_mean = np.mean([r["ct_m2"] for r in results])
    print(f"\n  CT absolute levels: G1 = {ct_g1_mean:.4f}  M2 = {ct_m2_mean:.4f}")

    return {"std_delta": np.mean(std_deltas), "ct_delta": np.mean(ct_deltas),
            "ct_t": t_ct, "ct_p": p_ct, "ct_ci": ci_ct, "ct_wins": wins_ct}


# ══════════════════════════════════════════════════════════════════════
#  Experiment 3: Single-Stage Ridge with Block Dummies
# ══════════════════════════════════════════════════════════════════════

def run_window_block_dummy_ridge(panel, ty, blocks, T_yr=5, alpha=1.0):
    """Single-stage Ridge: y_{t+1} = C · [y_t; block_dummies]."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    actor_block = {}
    for bname, bactors in blocks.items():
        for a in bactors:
            if a in v_list:
                actor_block[a] = bname
    for a in v_list:
        if a not in actor_block:
            actor_block[a] = "REMAINDER"

    # Build block dummy matrix (N × n_blocks)
    block_names = sorted(set(actor_block.values()))
    n_blocks = len(block_names)
    block_map = {b: i for i, b in enumerate(block_names)}
    D = np.zeros((N, n_blocks))
    for j, a in enumerate(v_list):
        D[j, block_map[actor_block[a]]] = 1.0

    # Build feature matrix: [y_t, y_t * d_1, ..., y_t * d_B]
    # This allows block-specific slopes
    def make_features(yt):
        # yt: (N,)  →  [yt, yt*d1, ..., yt*dB, d1, ..., dB]
        interactions = yt[:, None] * D  # (N, n_blocks)
        return np.concatenate([yt[:, None], interactions, D], axis=1)  # (N, 1 + 2*n_blocks)

    n_feat = 1 + 2 * n_blocks

    # Fit on training data
    X_list, Y_list = [], []
    for t in range(otr.shape[0] - 1):
        X_list.append(make_features(otr[t]))
        Y_list.append(otr[t + 1])

    X_train = np.vstack(X_list)  # (T*N, n_feat)
    Y_train = np.concatenate(Y_list)  # (T*N,)

    # Ridge regression
    try:
        C = np.linalg.solve(X_train.T @ X_train + alpha * N * np.eye(n_feat),
                            X_train.T @ Y_train)  # (n_feat,)
    except np.linalg.LinAlgError:
        return None

    # Test predictions
    preds_bd, preds_g0 = [], []
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    # Also compute pooled-only baseline for comparison
    rho, bar_y = estimate_pooled_ar1(otr)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # Block-dummy Ridge prediction
        feat = make_features(prev)
        y_bd = feat @ C

        # Pooled-only baseline
        y_g0 = bar_y + rho * (prev - bar_y)

        preds_bd.append(y_bd)
        preds_g0.append(y_g0)
        actuals.append(obs)

        prev = obs
        otr = np.vstack([otr, qv])

        # Re-fit Ridge on expanded training set
        X_list.append(make_features(otr[-2]))
        Y_list.append(obs)
        X_train = np.vstack(X_list)
        Y_train = np.concatenate(Y_list)
        try:
            C = np.linalg.solve(X_train.T @ X_train + alpha * N * np.eye(n_feat),
                                X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            pass

        rho, bar_y = estimate_pooled_ar1(otr)

    if not actuals:
        return None

    from quantdsl_backtest.smim.validation.metrics import oos_r_squared
    act_a = np.array(actuals)
    return {
        "BD": float(oos_r_squared(np.array(preds_bd).ravel(), act_a.ravel())),
        "G0": float(oos_r_squared(np.array(preds_g0).ravel(), act_a.ravel())),
    }


def exp3_block_dummy_ridge():
    print("\n" + "=" * 90)
    print("  EXP 3: Single-Stage Ridge with Block Dummies")
    print("=" * 90)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta)

    results = []
    for ty in TEST_YEARS:
        r = run_window_block_dummy_ridge(panel, ty, blocks)
        if r is not None:
            results.append(r)
            print(f"  W{ty}: BlockDummy={r['BD']:.4f}  G0={r['G0']:.4f}")

    if not results:
        return {}

    bd_r2 = [r["BD"] for r in results]
    g0_r2 = [r["G0"] for r in results]

    # Compare to G1 and M2 from saved metrics
    path = METRICS_DIR / "iter6_4b.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        g1_r2 = df[df["architecture"] == "G1"].sort_values("year")["full_r2"].values
        m2_r2 = df[df["architecture"] == "M2"].sort_values("year")["full_r2"].values

        deltas_vs_g1 = np.array(bd_r2) - g1_r2[:len(bd_r2)]
        t_g1, p_g1 = paired_t_test(deltas_vs_g1)
        ci_g1 = bootstrap_ci(deltas_vs_g1)
        wins_g1 = sum(1 for d in deltas_vs_g1 if d > 0)

        print(f"\n  Block-Dummy Ridge: mean R² = {np.mean(bd_r2):.4f}")
        print(f"  G1 (saved):        mean R² = {np.mean(g1_r2):.4f}")
        print(f"  M2 (saved):        mean R² = {np.mean(m2_r2):.4f}")
        print(f"\n  BD vs G1: Δ = {np.mean(deltas_vs_g1):+.4f}  t = {t_g1:.3f}  p = {p_g1:.4f}"
              f"  CI = [{ci_g1[0]:+.4f}, {ci_g1[1]:+.4f}]  W = {wins_g1}/{len(deltas_vs_g1)}")

        deltas_vs_m2 = np.array(bd_r2) - m2_r2[:len(bd_r2)]
        print(f"  BD vs M2: Δ = {np.mean(deltas_vs_m2):+.4f}")

    return {"bd_mean": np.mean(bd_r2), "per_window": bd_r2}


# ══════════════════════════════════════════════════════════════════════
#  Experiment 4: EWM Half-Life Sensitivity
# ══════════════════════════════════════════════════════════════════════

def run_window_m2_hl(panel, ty, blocks, hl, T_yr=5):
    """Run M2 architecture with a specific EWM half-life."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    actor_block = {}
    for bname, bactors in blocks.items():
        for a in bactors:
            if a in v_list:
                actor_block[a] = bname
    for a in v_list:
        if a not in actor_block:
            actor_block[a] = "REMAINDER"

    block_indices = {}
    for bname in blocks:
        block_indices[bname] = [v_list.index(a) for a in v_list if actor_block.get(a) == bname]
    local_block_names = [b for b in blocks if b != "REMAINDER"]

    # Stage 1
    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    # Global augmentation with custom hl
    om_r = ewm_demean(residuals, hl)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None

    ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
    A_r = mf_r.metadata["Atilde"][:ka, :ka].real.copy()
    F_r = _clip_sr(A_r)
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    # Local models with custom hl
    local_models = {}
    for bname in local_block_names:
        bidx = block_indices[bname]
        if len(bidx) < 5:
            local_models[bname] = None
            continue
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, hl)
        dm_b = block_resids - om_b
        N_b = len(bidx)
        K_b = min(4, max(2, N_b // 5))
        U_pca, A_pca = fit_local_pca_ridge(dm_b, K_b)
        local_models[bname] = {
            "U_pca": U_pca, "A_pca": A_pca, "om_b": om_b, "K_b": K_b,
            "bidx": bidx, "N_b": N_b,
        }

    preds_g1, preds_m2 = [], []
    actuals = []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_pool = bar_y + rho * (prev - bar_y)

        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred_global = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred_global)):
            resid_pred_global = np.zeros(N)
        y_g1 = y_pool + resid_pred_global

        y_m2 = y_g1.copy()
        prev_resid = prev - (bar_y + rho * (
            np.nan_to_num(otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y
        )) if otr.shape[0] >= 2 else np.zeros(N)
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            f_pca = lm["U_pca"].T @ prev_resid_b
            local_pred = lm["U_pca"] @ (lm["A_pca"] @ f_pca) + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_m2[bidx] = y_pool[bidx] + local_pred
            else:
                y_m2[bidx] = y_pool[bidx]

        preds_g1.append(y_g1)
        preds_m2.append(y_m2)
        actuals.append(obs)

        # Kalman update
        actual_resid = obs - y_pool
        odm_r = actual_resid - om_r.ravel()
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

        otr = np.vstack([otr, qv])
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, hl)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            k2 = min(K_DEFAULT, mf_r2.basis.shape[0] - 2, mf_r2.K)
            A_r2 = mf_r2.metadata["Atilde"][:k2, :k2].real.copy()
            F_r = _clip_sr(A_r2)
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2

        residuals = residuals_new
        for bname in local_block_names:
            lm = local_models.get(bname)
            if lm is None:
                continue
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, hl)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]
            lm["U_pca"], lm["A_pca"] = fit_local_pca_ridge(dm_b, K_b)
            lm["om_b"] = om_b

    if not actuals:
        return None

    from quantdsl_backtest.smim.validation.metrics import oos_r_squared
    act_a = np.array(actuals)
    return {
        "g1": float(oos_r_squared(np.array(preds_g1).ravel(), act_a.ravel())),
        "m2": float(oos_r_squared(np.array(preds_m2).ravel(), act_a.ravel())),
    }


def exp4_ewm_sensitivity():
    print("\n" + "=" * 90)
    print("  EXP 4: EWM Half-Life Sensitivity")
    print("=" * 90)

    panel, meta = load_panel_and_meta()
    blocks = define_blocks(panel, meta)

    half_lives = [4, 6, 8, 12]
    all_results = {}

    for hl in half_lives:
        print(f"\n  Half-life = {hl} quarters:")
        hl_results = []
        for ty in TEST_YEARS:
            r = run_window_m2_hl(panel, ty, blocks, hl)
            if r is not None:
                hl_results.append(r)
                print(f"    W{ty}: G1={r['g1']:.4f}  M2={r['m2']:.4f}  Δ={r['m2']-r['g1']:+.4f}")
        all_results[hl] = hl_results

    print(f"\n  {'HL':>4s}  {'G1':>8s}  {'M2':>8s}  {'Δ(M2-G1)':>10s}  {'t':>7s}  {'p':>7s}  {'W':>5s}")
    print(f"  {'-'*55}")

    for hl in half_lives:
        res = all_results[hl]
        if not res:
            continue
        g1_vals = [r["g1"] for r in res]
        m2_vals = [r["m2"] for r in res]
        deltas = [r["m2"] - r["g1"] for r in res]
        t_stat, p_val = paired_t_test(deltas)
        wins = sum(1 for d in deltas if d > 0)
        marker = " ★" if hl == 12 else ""
        print(f"  {hl:>4d}  {np.mean(g1_vals):8.4f}  {np.mean(m2_vals):8.4f}"
              f"  {np.mean(deltas):+10.4f}  {t_stat:7.2f}  {p_val:7.4f}  {wins:>2d}/{len(deltas)}{marker}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 100)
    print("  REFEREE ROUND 2 — FOUR ADDITIONAL EXPERIMENTS")
    print("=" * 100)

    # Exp 1: M2 vs BA_M2 direct DM test
    r1 = exp1_m2_vs_bam2()

    # Exp 1b: Loss differential autocorrelation
    r1b = exp1b_loss_diff_autocorrelation()

    # Exp 2: Campbell-Thompson R²
    r2 = exp2_campbell_thompson()

    # Exp 3: Single-stage Ridge with block dummies
    r3 = exp3_block_dummy_ridge()

    # Exp 4: EWM half-life sensitivity
    exp4_ewm_sensitivity()

    print(f"\n\n{'=' * 100}")
    print(f"  SUMMARY FOR PAPER")
    print(f"{'=' * 100}")

    if r1:
        sig = "significant" if r1["p"] < 0.05 else "NOT significant"
        print(f"\n  Exp 1 (M2 vs BA_M2): Δ = {r1['mean_delta']:+.4f}, t = {r1['t']:.2f},"
              f" p = {r1['p']:.4f} → {sig}")

    if r1b:
        print(f"\n  Exp 1b (M2-G1 ρ_d): {r1b['rho_d']:.3f} → n_eff = {r1b['n_eff']:.1f}")

    if r2:
        print(f"\n  Exp 2 (CT R²): Δ_standard = {r2['std_delta']:+.4f},"
              f" Δ_CT = {r2['ct_delta']:+.4f} → difference = {r2['ct_delta']-r2['std_delta']:+.4f}")

    if r3:
        print(f"\n  Exp 3 (Block-Dummy Ridge): mean R² = {r3['bd_mean']:.4f}")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
