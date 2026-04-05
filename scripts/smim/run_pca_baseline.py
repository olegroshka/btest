#!/usr/bin/env python
"""
PCA cross-sectional baseline for SMIM paper.

Runs two PCA-based baselines on the 146-firm CapEx/Revenue panel
using the same nested CV protocol as the headline SMIM evaluation:

  Baseline A: PCA rolling, full pipeline-matched
    - Replace DMD with PCA eigendecomposition of symmetric cross-correlation
    - Keep everything else identical: EWM demeaning, spherical R, Kalman,
      rolling basis refresh, same grids

  Baseline B: PCA projection-only (no Kalman)
    - ŷ_t = μ̂ + 0.99 · U_pca U_pca^T · ỹ_{t-1}
    - Same rolling PCA basis, but no Kalman filter

Also loads existing SMIM + AR(1) results for comparison.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_pca_baseline.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
F_REG, Q_INIT_SCALE, LAMBDA_Q = 0.99, 0.5, 0.3
TEST_YEARS = list(range(2015, 2025))
HOLDOUT_YEARS = [2023, 2024]
K_GRID = [2, 3]
EWM_GRID = [8, 12]
T_GRID = [3, 5]


def build_panel():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    capex = edgar[edgar["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"][
        ["ticker", "event_date", "value"]
    ].copy()
    rev = edgar[edgar["tag"] == "Revenues"][["ticker", "event_date", "value"]].copy()
    for df in [capex, rev]:
        df["q"] = df["event_date"].dt.to_period("Q").dt.to_timestamp()
    capex = (
        capex.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    )
    rev = rev.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    m = capex.merge(rev, on=["ticker", "q"], suffixes=("_c", "_r"))
    m["ratio"] = m["value_c"] / m["value_r"]
    m = m.replace([np.inf, -np.inf], np.nan)
    p = m.pivot_table(index="q", columns="ticker", values="ratio")
    p.index = pd.to_datetime(p.index)
    r = p.rank(axis=1, method="average", pct=True)
    good = r.columns[r.notna().mean() > 0.50]
    return r[good].loc["2005-01-01":"2025-12-31"]


def ewm_demean(obs, hl=8):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def pca_basis(dm, k=2):
    """Extract top-k PCA eigenvectors from the symmetric cross-correlation."""
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        # Cross-sectional covariance: C = (1/T) * dm^T dm
        C = dm.T @ dm / dm.shape[0]
        # Eigendecomposition of symmetric matrix
        eigvals, eigvecs = np.linalg.eigh(C)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        # Take top-k
        k_use = min(k, N - 2)
        U = eigvecs[:, :k_use]
        return U.real
    except Exception:
        return None


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res**2), 1e-8)


def ar1_baseline(otr, N):
    """Fit per-actor AR(1) on training data."""
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


def run_window_pca_kalman(panel, ty, K=2, ewm=12, T_yr=3):
    """PCA Baseline A: full Kalman pipeline with PCA basis instead of DMD."""
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
    otr = (
        ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty-1}-12-31"))]
        .values.astype(np.float64)
    )
    if otr.shape[0] < 4:
        return None

    om = ewm_demean(otr, ewm)
    dm = otr - om

    # AR(1)
    mu, rho = ar1_baseline(otr, N)

    # PCA basis instead of DMD
    U = pca_basis(dm, K)
    if U is None:
        return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

    ps, pa, ac = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        odm = obs - om.ravel()

        # Kalman predict
        ap = F @ a
        Pp = F @ P @ F.T + Q
        ps.append(U @ ap + om.ravel())  # PCA Kalman prediction
        pa.append(mu + rho * (prev - mu))  # AR(1) prediction
        ac.append(obs)
        prev = obs

        # Kalman update
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

        # Rolling PCA basis update (same protocol as SMIM)
        exp = np.vstack([otr, qv])
        otr = exp
        om2 = ewm_demean(exp, ewm)
        dm2 = exp - om2
        U2 = pca_basis(dm2, K)
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
    psa, paa, aca = np.array(ps), np.array(pa), np.array(ac)
    return (
        float(oos_r_squared(psa.ravel(), aca.ravel())),
        float(oos_r_squared(paa.ravel(), aca.ravel())),
        psa,
        paa,
        aca,
    )


def run_window_pca_projection(panel, ty, K=2, ewm=12, T_yr=3):
    """PCA Baseline B: projection-only, no Kalman filter.

    ŷ_t = μ̂ + 0.99 · U U^T · ỹ_{t-1}
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
    otr = (
        ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty-1}-12-31"))]
        .values.astype(np.float64)
    )
    if otr.shape[0] < 4:
        return None

    om = ewm_demean(otr, ewm)
    dm = otr - om

    # AR(1)
    mu, rho = ar1_baseline(otr, N)

    # PCA basis
    U = pca_basis(dm, K)
    if U is None:
        return None

    # Orthonormalise U for clean projection
    U_orth, _ = np.linalg.qr(U)
    U_orth = U_orth[:, : U.shape[1]]

    ps, pa, ac = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # PCA projection prediction: ŷ = μ̂ + 0.99 * U U^T * (y_{t-1} - μ̂)
        y_prev_dm = prev - om.ravel()
        pca_pred = om.ravel() + F_REG * (U_orth @ (U_orth.T @ y_prev_dm))
        ps.append(pca_pred)
        pa.append(mu + rho * (prev - mu))
        ac.append(obs)
        prev = obs

        # Rolling PCA basis update
        exp = np.vstack([otr, qv])
        otr = exp
        om2 = ewm_demean(exp, ewm)
        dm2 = exp - om2
        U2 = pca_basis(dm2, K)
        if U2 is not None:
            U_orth2, _ = np.linalg.qr(U2)
            U_orth = U_orth2[:, : U2.shape[1]]
            om = om2

    if not ps:
        return None
    psa, paa, aca = np.array(ps), np.array(pa), np.array(ac)
    return (
        float(oos_r_squared(psa.ravel(), aca.ravel())),
        float(oos_r_squared(paa.ravel(), aca.ravel())),
        psa,
        paa,
        aca,
    )


def inner_cv_pca(panel, ty, run_fn):
    """Inner CV to select K, ewm, T — same protocol as SMIM."""
    best_r2, best = -np.inf, {"K": 2, "ewm": 12, "T": 3}
    for K in K_GRID:
        for ewm in EWM_GRID:
            for T in T_GRID:
                r2s = []
                for iv in [ty - 2, ty - 1]:
                    if pd.Timestamp(f"{iv-T}-01-01") < pd.Timestamp("2005-01-01"):
                        continue
                    try:
                        res = run_fn(panel, iv, K, ewm, T)
                        if res and np.isfinite(res[0]):
                            r2s.append(res[0])
                    except Exception:
                        pass
                if r2s and np.mean(r2s) > best_r2:
                    best_r2 = np.mean(r2s)
                    best = {"K": K, "ewm": ewm, "T": T}
    return best, best_r2


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


def dm_test(e_s, e_a):
    d = e_a.ravel() ** 2 - e_s.ravel() ** 2
    n = len(d)
    db = d.mean()
    bw = max(1, int(n ** (1 / 3)))
    g0 = np.var(d, ddof=1)
    gs = sum(
        2 * (1 - h / (bw + 1)) * np.mean((d[h:] - db) * (d[:-h] - db))
        for h in range(1, bw + 1)
    )
    se = np.sqrt(max(g0 + gs, 1e-12) / n)
    from scipy.stats import norm

    t = db / se if se > 0 else 0
    return t, 1 - norm.cdf(t), n


def run_baseline(panel, label, run_fn):
    """Run a baseline through the full nested CV + fixed-config protocol."""
    print(f"\n{'='*80}")
    print(f"  {label}: FIXED CONFIG K=2, EWM=12, T=3yr")
    print(f"{'='*80}")
    rows_fixed = []
    for ty in TEST_YEARS:
        t1 = time.time()
        res = run_fn(panel, ty, K=2, ewm=12, T_yr=3)
        if res:
            d = res[0] - res[1]
            w = "WIN" if d > 0 else "LOSS"
            print(
                f"  W{ty}: PCA={res[0]:.4f}  AR1={res[1]:.4f}  Δ={d:+.4f} {w}  ({time.time()-t1:.1f}s)"
            )
            rows_fixed.append(
                {"year": ty, "pca": res[0], "ar1": res[1], "delta": d}
            )
    df_fixed = pd.DataFrame(rows_fixed)
    da = df_fixed["delta"].values
    print(
        f"\n  Mean: PCA={df_fixed['pca'].mean():.4f}  AR1={df_fixed['ar1'].mean():.4f}"
        f"  Δ={da.mean():+.4f}  wins={(da>0).sum()}/{len(da)}"
    )
    lo, hi = bootstrap_ci(da)
    print(f"  Bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")

    # Nested CV
    print(f"\n  {label}: NESTED CV")
    cv_rows, cv_es, cv_ea = [], [], []
    for ty in TEST_YEARS:
        t1 = time.time()
        ho = ty in HOLDOUT_YEARS
        if ho:
            non_ho = [r for r in cv_rows if not r["ho"]]
            if non_ho:
                sK = int(np.median([r["K"] for r in non_ho]))
                sE = int(np.median([r["ewm"] for r in non_ho]))
                sT = int(np.median([r["T"] for r in non_ho]))
            else:
                sK, sE, sT = 2, 12, 3
        else:
            p, _ = inner_cv_pca(panel, ty, run_fn)
            sK, sE, sT = p["K"], p["ewm"], p["T"]

        res = run_fn(panel, ty, sK, sE, sT)
        if not res:
            print(f"  W{ty}: FAILED")
            continue
        d = res[0] - res[1]
        tag = " [HO]" if ho else ""
        print(
            f"  W{ty}: K={sK} EWM={sE} T={sT}yr  PCA={res[0]:.4f}  AR1={res[1]:.4f}  Δ={d:+.4f}  ({time.time()-t1:.1f}s){tag}"
        )
        cv_rows.append(
            {
                "year": ty,
                "ho": ho,
                "K": sK,
                "ewm": sE,
                "T": sT,
                "pca": res[0],
                "ar1": res[1],
                "delta": d,
            }
        )
        cv_es.append((res[2] - res[4]).ravel())
        cv_ea.append((res[3] - res[4]).ravel())

    dfcv = pd.DataFrame(cv_rows)
    cm = ~dfcv["ho"]
    hm = dfcv["ho"]
    print(f"\n  CV ({cm.sum()}):")
    print(
        f"    PCA={dfcv.loc[cm,'pca'].mean():.4f}  AR1={dfcv.loc[cm,'ar1'].mean():.4f}"
        f"  Δ={dfcv.loc[cm,'delta'].mean():+.4f}  wins={(dfcv.loc[cm,'delta']>0).sum()}/{cm.sum()}"
    )
    if hm.any():
        print(f"  Holdout ({hm.sum()}):")
        print(
            f"    PCA={dfcv.loc[hm,'pca'].mean():.4f}  AR1={dfcv.loc[hm,'ar1'].mean():.4f}"
            f"  Δ={dfcv.loc[hm,'delta'].mean():+.4f}"
        )

    # Inference
    cvd = dfcv.loc[cm, "delta"].values
    if len(cvd) >= 3:
        lo2, hi2 = bootstrap_ci(cvd)
        pp2 = perm_test(cvd)
        ci = [i for i, r in enumerate(cv_rows) if not r["ho"]]
        es = np.concatenate([cv_es[i] for i in ci])
        ea = np.concatenate([cv_ea[i] for i in ci])
        dmt, dmp, dmn = dm_test(es, ea)
        print(f"\n  Inference (CV):")
        print(f"    Bootstrap CI: [{lo2:+.4f}, {hi2:+.4f}]  excludes 0: {lo2>0}")
        print(f"    Perm p: {pp2:.4f}")
        print(f"    DM vs AR(1): t={dmt:.3f} p={dmp:.4f} (n={dmn})")

    return df_fixed, dfcv


def main():
    t0 = time.time()
    panel = build_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load SMIM results for comparison
    smim_cv_path = METRICS_DIR / "iter5_1v2_nested_cv.parquet"
    smim_fixed_path = METRICS_DIR / "iter5_1v2_phase_a.parquet"
    smim_cv = pd.read_parquet(smim_cv_path) if smim_cv_path.exists() else None
    smim_fixed = pd.read_parquet(smim_fixed_path) if smim_fixed_path.exists() else None

    # Run PCA Kalman baseline
    pca_k_fixed, pca_k_cv = run_baseline(
        panel, "PCA-KALMAN (Baseline A)", run_window_pca_kalman
    )

    # Run PCA projection-only baseline
    pca_p_fixed, pca_p_cv = run_baseline(
        panel, "PCA-PROJECTION (Baseline B)", run_window_pca_projection
    )

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"  SUMMARY: SMIM vs PCA vs AR(1)")
    print(f"{'='*80}")

    print(f"\n  Fixed config (K=2, EWM=12, T=3yr, 10 windows):")
    print(f"    {'Model':<25s} {'R²':>8s} {'ΔR² vs AR1':>12s} {'Wins':>6s}")
    print(f"    {'-'*55}")
    if smim_fixed is not None:
        print(
            f"    {'SMIM (DMD+Kalman)':<25s} {smim_fixed['smim'].mean():8.4f} "
            f"{smim_fixed['delta'].mean():+12.4f} {(smim_fixed['delta']>0).sum():>3d}/10"
        )
    print(
        f"    {'PCA+Kalman (A)':<25s} {pca_k_fixed['pca'].mean():8.4f} "
        f"{pca_k_fixed['delta'].mean():+12.4f} {(pca_k_fixed['delta']>0).sum():>3d}/10"
    )
    print(
        f"    {'PCA projection (B)':<25s} {pca_p_fixed['pca'].mean():8.4f} "
        f"{pca_p_fixed['delta'].mean():+12.4f} {(pca_p_fixed['delta']>0).sum():>3d}/10"
    )
    print(f"    {'AR(1)':<25s} {pca_k_fixed['ar1'].mean():8.4f}")

    if smim_fixed is not None:
        # SMIM vs PCA delta
        smim_v = smim_fixed.set_index("year")["smim"]
        pca_k_v = pca_k_fixed.set_index("year")["pca"]
        pca_p_v = pca_p_fixed.set_index("year")["pca"]
        common = smim_v.index.intersection(pca_k_v.index)
        d_sk = (smim_v.loc[common] - pca_k_v.loc[common]).values
        d_sp = (smim_v.loc[common] - pca_p_v.loc[common]).values
        lo_sk, hi_sk = bootstrap_ci(d_sk)
        lo_sp, hi_sp = bootstrap_ci(d_sp)
        print(f"\n  SMIM vs PCA+Kalman: ΔR²={d_sk.mean():+.4f} CI [{lo_sk:+.4f},{hi_sk:+.4f}] wins={(d_sk>0).sum()}/{len(d_sk)}")
        print(f"  SMIM vs PCA proj:   ΔR²={d_sp.mean():+.4f} CI [{lo_sp:+.4f},{hi_sp:+.4f}] wins={(d_sp>0).sum()}/{len(d_sp)}")

    # Nested CV comparison
    cm_k = ~pca_k_cv["ho"]
    cm_p = ~pca_p_cv["ho"]
    print(f"\n  Nested CV (8 windows):")
    print(f"    {'Model':<25s} {'R²':>8s} {'ΔR² vs AR1':>12s} {'Wins':>6s}")
    print(f"    {'-'*55}")
    if smim_cv is not None:
        scm = ~smim_cv["ho"]
        print(
            f"    {'SMIM (DMD+Kalman)':<25s} {smim_cv.loc[scm,'smim'].mean():8.4f} "
            f"{smim_cv.loc[scm,'delta'].mean():+12.4f} {(smim_cv.loc[scm,'delta']>0).sum():>3d}/8"
        )
    print(
        f"    {'PCA+Kalman (A)':<25s} {pca_k_cv.loc[cm_k,'pca'].mean():8.4f} "
        f"{pca_k_cv.loc[cm_k,'delta'].mean():+12.4f} {(pca_k_cv.loc[cm_k,'delta']>0).sum():>3d}/8"
    )
    print(
        f"    {'PCA projection (B)':<25s} {pca_p_cv.loc[cm_p,'pca'].mean():8.4f} "
        f"{pca_p_cv.loc[cm_p,'delta'].mean():+12.4f} {(pca_p_cv.loc[cm_p,'delta']>0).sum():>3d}/8"
    )

    # Save results
    pca_k_fixed.to_parquet(METRICS_DIR / "pca_kalman_fixed.parquet", index=False)
    pca_k_cv.to_parquet(METRICS_DIR / "pca_kalman_cv.parquet", index=False)
    pca_p_fixed.to_parquet(METRICS_DIR / "pca_projection_fixed.parquet", index=False)
    pca_p_cv.to_parquet(METRICS_DIR / "pca_projection_cv.parquet", index=False)

    print(f"\n  Total: {time.time()-t0:.1f}s")
    print(f"  Saved: pca_kalman_fixed, pca_kalman_cv, pca_projection_fixed, pca_projection_cv")


if __name__ == "__main__":
    main()
