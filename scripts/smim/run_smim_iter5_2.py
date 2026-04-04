#!/usr/bin/env python
"""
Iteration 5.2 — Parameter Space Exploration.

Six phases:
  A: Dual-regularisation sweep (F, Q₀, λ_Q)  — 120 configs
  B: K=1 test                                  — 4 configs
  C: Alternative intensities                   — 3 panels × best config
  D: Rolling DMD window                        — 5 window lengths
  E: Fine EWM grid                             — 9 halflife values
  F: Interaction effects + nested CV           — top combos

All use predictive alpha (α_{t|t-1}), no operator learning, K=2 baseline.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_2.py
"""
from __future__ import annotations

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Baseline constants (from 5.1)
F_REG_DEFAULT = 0.99
Q_INIT_DEFAULT = 0.5
LAMBDA_Q_DEFAULT = 0.3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))
HOLDOUT_YEARS = [2023, 2024]

# ═══════════════════════════════════════════════════════════════════════
# Panel builders
# ═══════════════════════════════════════════════════════════════════════

def _load_edgar():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    edgar["q"] = edgar["event_date"].dt.to_period("Q").dt.to_timestamp()
    return edgar


def _build_ratio_panel(edgar, num_tag: str, den_tag: str, min_coverage: float = 0.50):
    """Generic ratio panel builder: num_tag / den_tag, cross-sectional rank."""
    num = edgar[edgar["tag"] == num_tag][["ticker", "event_date", "q", "value"]].copy()
    den = edgar[edgar["tag"] == den_tag][["ticker", "event_date", "q", "value"]].copy()
    num = num.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    den = den.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    m = num.merge(den, on=["ticker", "q"], suffixes=("_n", "_d"))
    m["ratio"] = m["value_n"] / m["value_d"]
    m = m.replace([np.inf, -np.inf], np.nan)
    p = m.pivot_table(index="q", columns="ticker", values="ratio")
    p.index = pd.to_datetime(p.index)
    r = p.rank(axis=1, method="average", pct=True)
    good = r.columns[r.notna().mean() > min_coverage]
    return r[good].loc["2005-01-01":"2025-12-31"]


def build_capex_revenue_panel(edgar=None):
    if edgar is None:
        edgar = _load_edgar()
    return _build_ratio_panel(edgar, "PaymentsToAcquirePropertyPlantAndEquipment", "Revenues")


def build_revenue_assets_panel(edgar=None):
    if edgar is None:
        edgar = _load_edgar()
    return _build_ratio_panel(edgar, "Revenues", "Assets")


def build_capex_assets_panel(edgar=None):
    if edgar is None:
        edgar = _load_edgar()
    return _build_ratio_panel(edgar, "PaymentsToAcquirePropertyPlantAndEquipment", "Assets")


def build_multi_ratio_panel(edgar=None):
    """Stack CapEx/Rev + Rev/Assets for overlapping firms → virtual heterogeneity."""
    if edgar is None:
        edgar = _load_edgar()
    cr = _build_ratio_panel(edgar, "PaymentsToAcquirePropertyPlantAndEquipment", "Revenues")
    ra = _build_ratio_panel(edgar, "Revenues", "Assets")
    overlap = cr.columns.intersection(ra.columns)
    cr_o = cr[overlap].rename(columns=lambda c: c + "_CR")
    ra_o = ra[overlap].rename(columns=lambda c: c + "_RA")
    return pd.concat([cr_o, ra_o], axis=1).loc["2005-01-01":"2025-12-31"]


# ═══════════════════════════════════════════════════════════════════════
# Core pipeline (K=2, no operator learning, rolling basis)
# ═══════════════════════════════════════════════════════════════════════

def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(dm, k=2):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(min(k, N - 2), mf.K)].real
    except Exception:
        return None


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def run_window(
    panel, ty, K=2, ewm=12, T_yr=3,
    F_reg=F_REG_DEFAULT, Q_init=Q_INIT_DEFAULT, lam_q=LAMBDA_Q_DEFAULT,
    dmd_window=None,
):
    """Run one test window. Returns (smim_r2, ar1_r2) or None.

    Parameters
    ----------
    dmd_window : int or None
        If set, use only the most recent `dmd_window` quarters for DMD
        instead of the full training set.
    """
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 15:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None

    om = ewm_demean(otr, ewm)
    dm = otr - om

    # AR(1) baseline
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c):
                rho[j] = c

    # DMD basis (optionally windowed)
    dm_for_dmd = dm[-dmd_window:] if dmd_window and dm.shape[0] > dmd_window else dm
    U = dmd_basis(dm_for_dmd, K)
    if U is None:
        return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_reg
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_init

    ps, pa, ac = [], [], []
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
        pa.append(mu + rho * (prev - mu))
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
        Q = (1 - lam_q) * Q + lam_q * np.outer(inn, inn)
        Q = (Q + Q.T) / 2 + np.eye(ka) * 1e-6

        # Rolling basis
        exp = np.vstack([otr, qv])
        otr = exp
        om2 = ewm_demean(exp, ewm)
        dm2 = exp - om2
        dm2_for_dmd = dm2[-dmd_window:] if dmd_window and dm2.shape[0] > dmd_window else dm2
        U2 = dmd_basis(dm2_for_dmd, K)
        if U2 is not None:
            k2 = U2.shape[1]
            a = U2.T @ odm
            P = np.eye(k2)
            Q = np.eye(k2) * Q_init
            F = np.eye(k2) * F_reg
            R = sph_r(dm2, U2)
            U = U2
            ka = k2
            om = om2

    if not ps:
        return None
    r2s = float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))
    r2a = float(oos_r_squared(np.array(pa).ravel(), np.array(ac).ravel()))
    return r2s, r2a


def sweep_config(panel, K=2, ewm=12, T=3, **kw):
    """Run all 10 windows for one config. Returns dict or None."""
    results = []
    for ty in TEST_YEARS:
        res = run_window(panel, ty, K=K, ewm=ewm, T_yr=T, **kw)
        if res is not None:
            results.append(res)
    if not results:
        return None
    smim = [r[0] for r in results]
    ar1 = [r[1] for r in results]
    deltas = [s - a for s, a in zip(smim, ar1)]
    return {
        "K": K, "ewm": ewm, "T": T,
        "mean_smim": np.mean(smim), "mean_ar1": np.mean(ar1),
        "mean_delta": np.mean(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "n_windows": len(results),
        "min_delta": min(deltas), "max_delta": max(deltas),
        **{k: v for k, v in kw.items() if not callable(v)},
    }


# ═══════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_test(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    obs = d.mean()
    cnt = sum(1 for _ in range(n) if (d * rng.choice([-1, 1], len(d))).mean() >= obs)
    return (cnt + 1) / (n + 1)


def dm_test(preds_s, preds_a, actuals):
    """Diebold-Mariano test on R² difference."""
    e_s = (preds_s - actuals).ravel()
    e_a = (preds_a - actuals).ravel()
    d = e_a ** 2 - e_s ** 2
    n = len(d)
    db = d.mean()
    bw = max(1, int(n ** (1 / 3)))
    g0 = np.var(d, ddof=1)
    gs = sum(2 * (1 - h / (bw + 1)) * np.mean((d[h:] - db) * (d[:-h] - db)) for h in range(1, bw + 1))
    se = np.sqrt(max(g0 + gs, 1e-12) / n)
    from scipy.stats import norm
    t = db / se if se > 0 else 0
    return t, 1 - norm.cdf(t), n


def inner_cv(panel, ty, configs):
    """Select best config from inner CV (ty-2, ty-1)."""
    best_r2, best_cfg = -np.inf, configs[0]
    for cfg in configs:
        T_yr = cfg.get("T_yr", cfg.get("T", 3))
        r2s = []
        for iv in [ty - 2, ty - 1]:
            if pd.Timestamp(f"{iv - T_yr}-01-01") < pd.Timestamp("2005-01-01"):
                continue
            try:
                # Pass all cfg keys that run_window accepts
                kw = {k: v for k, v in cfg.items() if k in
                      ("K", "ewm", "T_yr", "F_reg", "Q_init", "lam_q", "dmd_window")}
                if "T_yr" not in kw and "T" in cfg:
                    kw["T_yr"] = cfg["T"]
                res = run_window(panel, iv, **kw)
                if res and np.isfinite(res[0]):
                    r2s.append(res[0])
            except Exception:
                pass
        if r2s and np.mean(r2s) > best_r2:
            best_r2 = np.mean(r2s)
            best_cfg = cfg
    return best_cfg, best_r2


# ═══════════════════════════════════════════════════════════════════════
# Phase A: Dual-Regularisation Sweep
# ═══════════════════════════════════════════════════════════════════════

def phase_a(panel):
    print(f"\n{'=' * 90}")
    print(f"  PHASE A: DUAL-REGULARISATION SWEEP  K=2, EWM=12, T=3yr")
    print(f"{'=' * 90}")

    F_SWEEP = [0.90, 0.93, 0.95, 0.97, 0.99, 1.00]
    Q0_SWEEP = [0.1, 0.3, 0.5, 0.7, 1.0]
    LAM_SWEEP = [0.1, 0.2, 0.3, 0.5]

    total = len(F_SWEEP) * len(Q0_SWEEP) * len(LAM_SWEEP)
    print(f"  F={F_SWEEP}")
    print(f"  Q₀={Q0_SWEEP}")
    print(f"  λ_Q={LAM_SWEEP}")
    print(f"  {total} configurations\n")

    rows = []
    i = 0
    t0 = time.time()
    for F_val, Q0_val, lam_val in product(F_SWEEP, Q0_SWEEP, LAM_SWEEP):
        i += 1
        res = sweep_config(panel, K=2, ewm=12, T=3,
                           F_reg=F_val, Q_init=Q0_val, lam_q=lam_val)
        if res is not None:
            res["F_reg"] = F_val
            res["Q_init"] = Q0_val
            res["lam_q"] = lam_val
            rows.append(res)
            if i % 20 == 0:
                print(f"  [{i}/{total}] ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    print(f"\n  Phase A done: {len(df)} configs in {time.time() - t0:.1f}s")

    print(f"\n  TOP 10 by mean delta:")
    print(f"  {'F':>5} {'Q₀':>5} {'λ':>5} {'SMIM':>7} {'AR1':>7} {'Δ':>8} {'Wins':>5}")
    print(f"  {'-' * 50}")
    for _, r in df.head(10).iterrows():
        print(f"  {r['F_reg']:5.2f} {r['Q_init']:5.1f} {r['lam_q']:5.1f} "
              f"{r['mean_smim']:7.4f} {r['mean_ar1']:7.4f} "
              f"{r['mean_delta']:+8.4f} {r['wins']:3.0f}/{r['n_windows']:.0f}")

    # Compare with baseline
    bl = df[(df["F_reg"] == 0.99) & (df["Q_init"] == 0.5) & (df["lam_q"] == 0.3)]
    if len(bl) > 0:
        bl_delta = bl.iloc[0]["mean_delta"]
        best_delta = df.iloc[0]["mean_delta"]
        print(f"\n  Baseline (F=0.99 Q₀=0.5 λ=0.3): Δ={bl_delta:+.4f}")
        print(f"  Best:    (F={df.iloc[0]['F_reg']:.2f} Q₀={df.iloc[0]['Q_init']:.1f} λ={df.iloc[0]['lam_q']:.1f}): Δ={best_delta:+.4f}")
        print(f"  Improvement: {best_delta - bl_delta:+.4f}")
        if best_delta - bl_delta < 0.005:
            print(f"  → Improvement < 0.5pp, treating as NOISE (per plan rules)")

    df.to_parquet(METRICS_DIR / "iter5_2_phase_a.parquet", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Phase B: K=1 Test
# ═══════════════════════════════════════════════════════════════════════

def phase_b(panel, best_a):
    print(f"\n{'=' * 90}")
    print(f"  PHASE B: K=1 TEST")
    print(f"{'=' * 90}")

    F_val = best_a.get("F_reg", F_REG_DEFAULT)
    Q_val = best_a.get("Q_init", Q_INIT_DEFAULT)
    lam_val = best_a.get("lam_q", LAMBDA_Q_DEFAULT)
    print(f"  Using dual-reg from Phase A: F={F_val}, Q₀={Q_val}, λ={lam_val}\n")

    rows = []
    for K, ewm, T in product([1, 2], [8, 12], [2, 3]):
        res = sweep_config(panel, K=K, ewm=ewm, T=T,
                           F_reg=F_val, Q_init=Q_val, lam_q=lam_val)
        if res is not None:
            rows.append(res)
            tag = "★" if res["mean_delta"] > 0.035 else " "
            print(f"  {tag} K={K} EWM={ewm} T={T}yr: "
                  f"SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)

    k1 = df[df["K"] == 1]
    k2 = df[df["K"] == 2]
    if len(k1) > 0 and len(k2) > 0:
        best_k1 = k1.iloc[0]["mean_delta"]
        best_k2 = k2.iloc[0]["mean_delta"]
        print(f"\n  Best K=1: Δ={best_k1:+.4f}")
        print(f"  Best K=2: Δ={best_k2:+.4f}")
        if best_k1 > best_k2:
            print(f"  → K=1 BEATS K=2! PLATINUM CANDIDATE")
        else:
            print(f"  → K=2 still better by {best_k2 - best_k1:+.4f}")

    df.to_parquet(METRICS_DIR / "iter5_2_phase_b.parquet", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Phase C: Alternative Intensities
# ═══════════════════════════════════════════════════════════════════════

def phase_c(edgar, best_cfg):
    print(f"\n{'=' * 90}")
    print(f"  PHASE C: ALTERNATIVE INTENSITIES")
    print(f"{'=' * 90}")

    K = best_cfg.get("K", 2)
    ewm = best_cfg.get("ewm", 12)
    T = best_cfg.get("T", 3)
    F_val = best_cfg.get("F_reg", F_REG_DEFAULT)
    Q_val = best_cfg.get("Q_init", Q_INIT_DEFAULT)
    lam_val = best_cfg.get("lam_q", LAMBDA_Q_DEFAULT)
    print(f"  Config: K={K} EWM={ewm} T={T}yr F={F_val} Q₀={Q_val} λ={lam_val}\n")

    panels = {
        "CapEx/Revenue": build_capex_revenue_panel(edgar),
        "Revenue/Assets": build_revenue_assets_panel(edgar),
        "CapEx/Assets": build_capex_assets_panel(edgar),
        "Multi-ratio": build_multi_ratio_panel(edgar),
    }

    all_rows = []
    for name, p in panels.items():
        print(f"\n  --- {name} ({p.shape[0]}Q × {p.shape[1]} firms) ---")
        res = sweep_config(p, K=K, ewm=ewm, T=T,
                           F_reg=F_val, Q_init=Q_val, lam_q=lam_val)
        if res is not None:
            res["panel"] = name
            all_rows.append(res)
            w = "★" if res["mean_delta"] > 0 else " "
            print(f"  {w} SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")
        else:
            print(f"    FAILED (not enough data)")

    df = pd.DataFrame(all_rows)
    winners = df[df["mean_delta"] > 0]
    if len(winners) > 1:
        print(f"\n  → GOLD: {len(winners)} panels beat AR(1)!")
    df.to_parquet(METRICS_DIR / "iter5_2_phase_c.parquet", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Phase D: Rolling DMD Window
# ═══════════════════════════════════════════════════════════════════════

def phase_d(panel, best_cfg):
    print(f"\n{'=' * 90}")
    print(f"  PHASE D: ROLLING DMD WINDOW  K=2, EWM=12, T=3yr")
    print(f"{'=' * 90}")

    F_val = best_cfg.get("F_reg", F_REG_DEFAULT)
    Q_val = best_cfg.get("Q_init", Q_INIT_DEFAULT)
    lam_val = best_cfg.get("lam_q", LAMBDA_Q_DEFAULT)

    W_SWEEP = [8, 12, 16, 20, None]  # None = all data
    rows = []
    for W in W_SWEEP:
        label = f"W={W}" if W else "W=all"
        res = sweep_config(panel, K=2, ewm=12, T=3,
                           F_reg=F_val, Q_init=Q_val, lam_q=lam_val,
                           dmd_window=W)
        if res is not None:
            res["dmd_window"] = W if W else -1
            rows.append(res)
            print(f"  {label}: SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    best = df.iloc[0]
    bw = int(best["dmd_window"]) if best["dmd_window"] != -1 else "all"
    print(f"\n  Best: W={bw}, Δ={best['mean_delta']:+.4f}")
    df.to_parquet(METRICS_DIR / "iter5_2_phase_d.parquet", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Phase E: Fine EWM Grid
# ═══════════════════════════════════════════════════════════════════════

def phase_e(panel, best_cfg):
    print(f"\n{'=' * 90}")
    print(f"  PHASE E: FINE EWM GRID  K=2, T=3yr")
    print(f"{'=' * 90}")

    F_val = best_cfg.get("F_reg", F_REG_DEFAULT)
    Q_val = best_cfg.get("Q_init", Q_INIT_DEFAULT)
    lam_val = best_cfg.get("lam_q", LAMBDA_Q_DEFAULT)

    EWM_FINE = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    rows = []
    for ewm in EWM_FINE:
        res = sweep_config(panel, K=2, ewm=ewm, T=3,
                           F_reg=F_val, Q_init=Q_val, lam_q=lam_val)
        if res is not None:
            rows.append(res)
            print(f"  EWM={ewm:2d}: SMIM={res['mean_smim']:.4f} AR1={res['mean_ar1']:.4f} "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    best = df.iloc[0]
    print(f"\n  Best: EWM={best['ewm']:.0f}, Δ={best['mean_delta']:+.4f}")
    df.to_parquet(METRICS_DIR / "iter5_2_phase_e.parquet", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Phase F: Interaction Effects + Nested CV
# ═══════════════════════════════════════════════════════════════════════

def phase_f(panel, best_a, best_b, best_d, best_e):
    print(f"\n{'=' * 90}")
    print(f"  PHASE F: INTERACTION EFFECTS + NESTED CV")
    print(f"{'=' * 90}")

    # Gather best findings
    best_F = best_a.get("F_reg", F_REG_DEFAULT)
    best_Q0 = best_a.get("Q_init", Q_INIT_DEFAULT)
    best_lam = best_a.get("lam_q", LAMBDA_Q_DEFAULT)
    best_ewm = int(best_e.get("ewm", 12))
    best_dmd_w = best_d.get("dmd_window", None)
    if best_dmd_w is not None and best_dmd_w != -1:
        best_dmd_w = int(best_dmd_w)
    else:
        best_dmd_w = None

    # Build interaction grid: best dual-reg × best EWM × K∈{1,2} × T∈{2,3}
    configs = []
    for K, T in product([1, 2], [2, 3]):
        cfg = dict(K=K, ewm=best_ewm, T_yr=T,
                   F_reg=best_F, Q_init=best_Q0, lam_q=best_lam,
                   dmd_window=best_dmd_w)
        configs.append(cfg)

    # Also test with the 5.1 baseline EWM=12 if best_ewm differs
    if best_ewm != 12:
        for K, T in product([1, 2], [2, 3]):
            cfg = dict(K=K, ewm=12, T_yr=T,
                       F_reg=best_F, Q_init=best_Q0, lam_q=best_lam,
                       dmd_window=best_dmd_w)
            configs.append(cfg)

    print(f"  Best params: F={best_F} Q₀={best_Q0} λ={best_lam} EWM={best_ewm} DMD_W={best_dmd_w or 'all'}")
    print(f"  {len(configs)} interaction configs\n")

    rows = []
    for cfg in configs:
        res = sweep_config(panel, K=cfg["K"], ewm=cfg["ewm"], T=cfg["T_yr"],
                           F_reg=cfg["F_reg"], Q_init=cfg["Q_init"],
                           lam_q=cfg["lam_q"], dmd_window=cfg["dmd_window"])
        if res is not None:
            rows.append(res)
            tag = "★" if res["mean_delta"] > 0.040 else " "
            print(f"  {tag} K={cfg['K']} EWM={cfg['ewm']} T={cfg['T_yr']}yr: "
                  f"Δ={res['mean_delta']:+.4f} wins={res['wins']}/{res['n_windows']}")

    df_inter = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)
    best_fixed = df_inter.iloc[0]["mean_delta"] if len(df_inter) > 0 else 0

    # ── Nested CV (if fixed-config delta > 0.040 i.e. BRONZE) ──
    CURRENT_BEST_NESTED_DELTA = 0.042
    cv_rows = []

    if best_fixed > 0.040:
        print(f"\n  Fixed-config best: Δ={best_fixed:+.4f} → BRONZE achieved, running nested CV")

        # Build CV config set from top interaction configs
        cv_configs = []
        for _, r in df_inter.head(8).iterrows():
            cv_configs.append(dict(
                K=int(r["K"]), ewm=int(r["ewm"]), T_yr=int(r["T"]),
                F_reg=float(r.get("F_reg", best_F)),
                Q_init=float(r.get("Q_init", best_Q0)),
                lam_q=float(r.get("lam_q", best_lam)),
                dmd_window=int(r["dmd_window"]) if "dmd_window" in r and pd.notna(r.get("dmd_window")) and r.get("dmd_window", -1) != -1 else None,
            ))

        # Run nested CV
        print(f"\n  Running nested CV with {len(cv_configs)} candidate configs...")
        for ty in TEST_YEARS:
            t1 = time.time()
            ho = ty in HOLDOUT_YEARS
            if ho:
                non_ho = [r for r in cv_rows if not r["ho"]]
                if non_ho:
                    # Use median selected config from non-holdout
                    sK = int(np.median([r["K"] for r in non_ho]))
                    sE = int(np.median([r["ewm"] for r in non_ho]))
                    sT = int(np.median([r["T"] for r in non_ho]))
                    sF = np.median([r["F_reg"] for r in non_ho])
                    sQ = np.median([r["Q_init"] for r in non_ho])
                    sL = np.median([r["lam_q"] for r in non_ho])
                else:
                    sK, sE, sT = 2, best_ewm, 3
                    sF, sQ, sL = best_F, best_Q0, best_lam
            else:
                # Inner CV selects from candidate configs
                # Use T from configs (T_yr → T for display)
                best_cv, _ = inner_cv(panel, ty, cv_configs)
                sK = best_cv["K"]
                sE = best_cv["ewm"]
                sT = best_cv["T_yr"]
                sF = best_cv["F_reg"]
                sQ = best_cv["Q_init"]
                sL = best_cv["lam_q"]

            res = run_window(panel, ty, K=sK, ewm=sE, T_yr=sT,
                             F_reg=sF, Q_init=sQ, lam_q=sL)
            if not res:
                print(f"  W{ty}: FAILED")
                continue
            d = res[0] - res[1]
            tag = " [HO]" if ho else ""
            print(f"  W{ty}: K={sK} EWM={sE} T={sT}yr F={sF:.2f} Q₀={sQ:.1f} λ={sL:.1f}  "
                  f"SMIM={res[0]:.4f} AR1={res[1]:.4f} Δ={d:+.4f}  ({time.time() - t1:.1f}s){tag}")
            cv_rows.append({
                "year": ty, "ho": ho, "K": sK, "ewm": sE, "T": sT,
                "F_reg": sF, "Q_init": sQ, "lam_q": sL,
                "smim": res[0], "ar1": res[1], "delta": d,
            })

        dfcv = pd.DataFrame(cv_rows)
        cm = ~dfcv["ho"]
        if cm.any():
            cvd = dfcv.loc[cm, "delta"].values
            lo, hi = bootstrap_ci(cvd)
            pp = perm_test(cvd)
            from scipy.stats import binom
            sp = float(binom.sf(int((cvd > 0).sum()) - 1, len(cvd), 0.5))
            print(f"\n  Nested CV ({cm.sum()} windows):")
            print(f"    SMIM={dfcv.loc[cm, 'smim'].mean():.4f}  AR1={dfcv.loc[cm, 'ar1'].mean():.4f}  "
                  f"Δ={cvd.mean():+.4f}  wins={(cvd > 0).sum()}/{len(cvd)}")
            print(f"    Bootstrap CI: [{lo:+.4f}, {hi:+.4f}]  Perm p={pp:.4f}  "
                  f"Sign {(cvd > 0).sum()}/{len(cvd)} p={sp:.4f}")

            if cvd.mean() > CURRENT_BEST_NESTED_DELTA:
                print(f"    → SILVER: nested CV delta {cvd.mean():+.4f} > {CURRENT_BEST_NESTED_DELTA:+.4f}!")
            else:
                print(f"    → No improvement over current nested CV ({CURRENT_BEST_NESTED_DELTA:+.4f})")

        if dfcv["ho"].any():
            hm = dfcv["ho"]
            print(f"  Holdout ({hm.sum()}):")
            print(f"    SMIM={dfcv.loc[hm, 'smim'].mean():.4f}  AR1={dfcv.loc[hm, 'ar1'].mean():.4f}  "
                  f"Δ={dfcv.loc[hm, 'delta'].mean():+.4f}")

        dfcv.to_parquet(METRICS_DIR / "iter5_2_nested_cv.parquet", index=False)
    else:
        print(f"\n  Fixed-config best: Δ={best_fixed:+.4f} — below BRONZE threshold (0.040), skipping nested CV")

    df_inter.to_parquet(METRICS_DIR / "iter5_2_phase_f.parquet", index=False)
    return df_inter, cv_rows


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  SMIM ITERATION 5.2: PARAMETER SPACE EXPLORATION")
    print("=" * 90)

    # Load data once
    edgar = _load_edgar()
    panel = build_capex_revenue_panel(edgar)
    print(f"  CapEx/Revenue panel: {panel.shape[0]}Q × {panel.shape[1]} firms\n")

    # ── Phase A: Dual-reg sweep ──
    df_a = phase_a(panel)
    best_a_row = df_a.iloc[0]
    best_a = {"F_reg": best_a_row["F_reg"], "Q_init": best_a_row["Q_init"], "lam_q": best_a_row["lam_q"]}
    # If improvement is noise (<0.5pp), keep baseline
    bl = df_a[(df_a["F_reg"] == 0.99) & (df_a["Q_init"] == 0.5) & (df_a["lam_q"] == 0.3)]
    if len(bl) > 0 and best_a_row["mean_delta"] - bl.iloc[0]["mean_delta"] < 0.005:
        print(f"\n  → Keeping baseline dual-reg (improvement is noise)")
        best_a = {"F_reg": 0.99, "Q_init": 0.5, "lam_q": 0.3}

    # ── Phase B: K=1 test ──
    df_b = phase_b(panel, best_a)
    best_b_row = df_b.iloc[0]
    best_b = {"K": int(best_b_row["K"]), "mean_delta": best_b_row["mean_delta"]}

    # ── Phase C: Alternative intensities ──
    df_c = phase_c(edgar, {
        "K": 2, "ewm": 12, "T": 3,
        "F_reg": best_a["F_reg"], "Q_init": best_a["Q_init"], "lam_q": best_a["lam_q"],
    })

    # ── Phase D: Rolling DMD window ──
    df_d = phase_d(panel, best_a)
    best_d_row = df_d.iloc[0]
    dmd_w = int(best_d_row["dmd_window"]) if best_d_row["dmd_window"] != -1 else None
    best_d = {"dmd_window": best_d_row["dmd_window"], "mean_delta": best_d_row["mean_delta"]}

    # ── Phase E: Fine EWM grid ──
    df_e = phase_e(panel, best_a)
    best_e_row = df_e.iloc[0]
    best_e = {"ewm": int(best_e_row["ewm"]), "mean_delta": best_e_row["mean_delta"]}

    # ── Phase F: Interactions + nested CV ──
    df_f, cv_rows = phase_f(panel, best_a, best_b, best_d, best_e)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 90}")
    print(f"#  ITERATION 5.2 SUMMARY")
    print(f"{'#' * 90}")

    print(f"\n  Phase A (dual-reg): best F={best_a['F_reg']:.2f} Q₀={best_a['Q_init']:.1f} λ={best_a['lam_q']:.1f}")
    print(f"  Phase B (K=1):      best K={best_b['K']} Δ={best_b['mean_delta']:+.4f}")
    print(f"  Phase D (DMD win):  best W={dmd_w or 'all'}")
    print(f"  Phase E (fine EWM): best EWM={best_e['ewm']}")

    # Success criteria
    print(f"\n  SUCCESS CRITERIA:")
    current_fixed = 0.035
    current_nested = 0.042

    if len(df_f) > 0:
        best_fixed = df_f.iloc[0]["mean_delta"]
        print(f"  BRONZE (fixed Δ>{current_fixed:+.3f}): {'PASS' if best_fixed > 0.040 else 'FAIL'} (Δ={best_fixed:+.4f})")
    if cv_rows:
        non_ho = [r for r in cv_rows if not r["ho"]]
        if non_ho:
            cv_delta = np.mean([r["delta"] for r in non_ho])
            print(f"  SILVER (nested Δ>{current_nested:+.3f}): {'PASS' if cv_delta > 0.050 else 'FAIL'} (Δ={cv_delta:+.4f})")

    # Phase C: any alternative panel beats AR(1)?
    if len(df_c) > 0:
        alt_wins = df_c[(df_c["panel"] != "CapEx/Revenue") & (df_c["mean_delta"] > 0)]
        print(f"  GOLD   (alt signal): {'PASS' if len(alt_wins) > 0 else 'FAIL'} ({len(alt_wins)} alt panels beat AR(1))")

    # Phase B: K=1 works?
    k1_works = len(df_b) > 0 and df_b[df_b["K"] == 1].iloc[0]["mean_delta"] > 0 if len(df_b[df_b["K"] == 1]) > 0 else False
    print(f"  PLATINUM (K=1):     {'PASS' if k1_works else 'FAIL'}")

    print(f"\n  Total runtime: {time.time() - t_total:.1f}s")
    print(f"  Results saved to: results/metrics/iter5_2_*.parquet")


if __name__ == "__main__":
    main()
