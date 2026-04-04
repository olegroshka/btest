#!/usr/bin/env python
"""
Iteration 5.1 v2 — Nested CV with K=2, no operator learning.

Grid: K ∈ {2,3}, EWM ∈ {8,12}, T ∈ {3,5} (8 configs)
No operator learning — K=2 doesn't need it.
Also runs the T-sweep (T=2,3,4,5,7) at fixed K=2 EWM=12 for the paper figure.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1_cv2.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
F_REG, Q_INIT_SCALE, LAMBDA_Q, K_MAX = 0.99, 0.5, 0.3, 15
TEST_YEARS = list(range(2015, 2025))
HOLDOUT_YEARS = [2023, 2024]
K_GRID = [2, 3]
EWM_GRID = [8, 12]
T_GRID = [3, 5]


def build_panel():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    capex = edgar[edgar["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"][["ticker","event_date","value"]].copy()
    rev = edgar[edgar["tag"] == "Revenues"][["ticker","event_date","value"]].copy()
    for df in [capex, rev]:
        df["q"] = df["event_date"].dt.to_period("Q").dt.to_timestamp()
    capex = capex.sort_values("event_date").groupby(["ticker","q"]).last().reset_index()
    rev = rev.sort_values("event_date").groupby(["ticker","q"]).last().reset_index()
    m = capex.merge(rev, on=["ticker","q"], suffixes=("_c","_r"))
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

def dmd_basis(dm, k=2):
    N = dm.shape[1]
    if dm.shape[0] < 3: return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(min(k, N-2), mf.K)].real
    except Exception: return None

def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res**2), 1e-8)


def run_window(panel, ty, K=2, ewm=12, T_yr=3):
    """Returns (smim_r2, ar1_r2, smim_preds, ar1_preds, actuals) or None."""
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"): return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 15: return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty-1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4: return None

    om = ewm_demean(otr, ewm)
    dm = otr - om

    # AR(1)
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c): rho[j] = c

    U = dmd_basis(dm, K)
    if U is None: return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Q = np.eye(ka) * Q_INIT_SCALE

    ps, pa, ac = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        obs = qv[0]; odm = obs - om.ravel()

        ap = F @ a; Pp = F @ P @ F.T + Q
        ps.append(U @ ap + om.ravel())
        pa.append(mu + rho * (prev - mu))
        ac.append(obs); prev = obs

        S = U @ Pp @ U.T + R
        try: Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except: Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp
        inn = a - ap
        Q = (1-LAMBDA_Q)*Q + LAMBDA_Q*np.outer(inn,inn)
        Q = (Q+Q.T)/2 + np.eye(ka)*1e-6

        exp = np.vstack([otr, qv]); otr = exp
        om2 = ewm_demean(exp, ewm); dm2 = exp - om2
        U2 = dmd_basis(dm2, K)
        if U2 is not None:
            k2 = U2.shape[1]
            a = U2.T @ odm; P = np.eye(k2)
            Q = np.eye(k2)*Q_INIT_SCALE; F = np.eye(k2)*F_REG
            R = sph_r(dm2, U2); U = U2; ka = k2; om = om2

    if not ps: return None
    psa, paa, aca = np.array(ps), np.array(pa), np.array(ac)
    return (float(oos_r_squared(psa.ravel(), aca.ravel())),
            float(oos_r_squared(paa.ravel(), aca.ravel())),
            psa, paa, aca)


def inner_cv(panel, ty):
    best_r2, best = -np.inf, {"K":2,"ewm":12,"T":3}
    for K in K_GRID:
        for ewm in EWM_GRID:
            for T in T_GRID:
                r2s = []
                for iv in [ty-2, ty-1]:
                    if pd.Timestamp(f"{iv-T}-01-01") < pd.Timestamp("2005-01-01"): continue
                    try:
                        res = run_window(panel, iv, K, ewm, T)
                        if res and np.isfinite(res[0]): r2s.append(res[0])
                    except: pass
                if r2s and np.mean(r2s) > best_r2:
                    best_r2 = np.mean(r2s)
                    best = {"K":K, "ewm":ewm, "T":T}
    return best, best_r2


def bootstrap_ci(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)

def perm_test(d, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    obs = d.mean(); cnt = sum(1 for _ in range(n) if (d*rng.choice([-1,1],len(d))).mean() >= obs)
    return (cnt+1)/(n+1)

def dm_test(e_s, e_a):
    d = e_a.ravel()**2 - e_s.ravel()**2
    n = len(d); db = d.mean()
    bw = max(1, int(n**(1/3)))
    g0 = np.var(d, ddof=1)
    gs = sum(2*(1-h/(bw+1))*np.mean((d[h:]-db)*(d[:-h]-db)) for h in range(1,bw+1))
    se = np.sqrt(max(g0+gs, 1e-12)/n)
    from scipy.stats import norm
    t = db/se if se > 0 else 0
    return t, 1-norm.cdf(t), n


def main():
    t0 = time.time()
    panel = build_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── PHASE A: Fixed config K=2 EWM=12 T=3yr (no OpLearn) ──
    print(f"\n{'='*80}")
    print(f"  PHASE A: FIXED CONFIG K=2, EWM=12, T=3yr (no operator learning)")
    print(f"{'='*80}")
    rows_a = []
    for ty in TEST_YEARS:
        t1 = time.time()
        res = run_window(panel, ty, K=2, ewm=12, T_yr=3)
        if res:
            d = res[0]-res[1]
            w = "WIN" if d>0 else "LOSS"
            print(f"  W{ty}: SMIM={res[0]:.4f}  AR1={res[1]:.4f}  Δ={d:+.4f} {w}  ({time.time()-t1:.1f}s)")
            rows_a.append({"year":ty,"smim":res[0],"ar1":res[1],"delta":d})
    dfa = pd.DataFrame(rows_a)
    da = dfa["delta"].values
    print(f"\n  Mean: SMIM={dfa['smim'].mean():.4f}  AR1={dfa['ar1'].mean():.4f}  Δ={da.mean():+.4f}  wins={(da>0).sum()}/{len(da)}")
    lo,hi = bootstrap_ci(da)
    pp = perm_test(da)
    from scipy.stats import binom
    sp = float(binom.sf(int((da>0).sum())-1, len(da), 0.5))
    print(f"  Bootstrap CI: [{lo:+.4f}, {hi:+.4f}]  Perm p={pp:.4f}  Sign {(da>0).sum()}/{len(da)} p={sp:.4f}")

    # ── PHASE B: NESTED CV ──
    print(f"\n{'='*80}")
    print(f"  PHASE B: NESTED CV  K={K_GRID} EWM={EWM_GRID} T={T_GRID} (no OpLearn)")
    print(f"{'='*80}")
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
            else: sK,sE,sT = 2,12,3
        else:
            p, _ = inner_cv(panel, ty)
            sK, sE, sT = p["K"], p["ewm"], p["T"]

        res = run_window(panel, ty, sK, sE, sT)
        if not res:
            print(f"  W{ty}: FAILED"); continue
        d = res[0]-res[1]
        tag = " [HO]" if ho else ""
        print(f"  W{ty}: K={sK} EWM={sE} T={sT}yr  SMIM={res[0]:.4f}  AR1={res[1]:.4f}  Δ={d:+.4f}  ({time.time()-t1:.1f}s){tag}")
        cv_rows.append({"year":ty,"ho":ho,"K":sK,"ewm":sE,"T":sT,"smim":res[0],"ar1":res[1],"delta":d})
        cv_es.append((res[2]-res[4]).ravel())
        cv_ea.append((res[3]-res[4]).ravel())

    dfb = pd.DataFrame(cv_rows)
    cm = ~dfb["ho"]; hm = dfb["ho"]
    print(f"\n  CV ({cm.sum()}):")
    print(f"    SMIM={dfb.loc[cm,'smim'].mean():.4f}  AR1={dfb.loc[cm,'ar1'].mean():.4f}  Δ={dfb.loc[cm,'delta'].mean():+.4f}  wins={(dfb.loc[cm,'delta']>0).sum()}/{cm.sum()}")
    print(f"    K: {dfb.loc[cm,'K'].value_counts().to_dict()}  EWM: {dfb.loc[cm,'ewm'].value_counts().to_dict()}  T: {dfb.loc[cm,'T'].value_counts().to_dict()}")
    if hm.any():
        print(f"  Holdout ({hm.sum()}):")
        print(f"    SMIM={dfb.loc[hm,'smim'].mean():.4f}  AR1={dfb.loc[hm,'ar1'].mean():.4f}  Δ={dfb.loc[hm,'delta'].mean():+.4f}")

    # Inference on CV windows
    cvd = dfb.loc[cm,"delta"].values
    if len(cvd) >= 3:
        lo2,hi2 = bootstrap_ci(cvd)
        pp2 = perm_test(cvd)
        sp2 = float(binom.sf(int((cvd>0).sum())-1, len(cvd), 0.5))
        ci = [i for i,r in enumerate(cv_rows) if not r["ho"]]
        es = np.concatenate([cv_es[i] for i in ci])
        ea = np.concatenate([cv_ea[i] for i in ci])
        dmt, dmp, dmn = dm_test(es, ea)
        print(f"\n  Inference (CV):")
        print(f"    Bootstrap CI: [{lo2:+.4f}, {hi2:+.4f}]  excludes 0: {lo2>0}")
        print(f"    Perm p: {pp2:.4f}")
        print(f"    DM: t={dmt:.3f} p={dmp:.4f} (n={dmn})")
        print(f"    Sign: {(cvd>0).sum()}/{len(cvd)} p={sp2:.4f}")

    # All 10 windows
    alld = dfb["delta"].values
    lo3,hi3 = bootstrap_ci(alld)
    pp3 = perm_test(alld)
    print(f"\n  All {len(alld)} windows: Δ={alld.mean():+.4f}  CI [{lo3:+.4f},{hi3:+.4f}]  perm p={pp3:.4f}")

    # ── T-SWEEP (fixed K=2 EWM=12, varying T) ──
    print(f"\n{'='*80}")
    print(f"  T-SWEEP: K=2, EWM=12, no OpLearn (for paper figure)")
    print(f"{'='*80}")
    t_rows = []
    for T in [2, 3, 4, 5, 7]:
        ds = []
        for ty in TEST_YEARS:
            res = run_window(panel, ty, K=2, ewm=12, T_yr=T)
            if res: ds.append(res[0]-res[1])
        if ds:
            d_arr = np.array(ds)
            lo_t, hi_t = bootstrap_ci(d_arr)
            pp_t = perm_test(d_arr)
            print(f"  T={T}yr: Δ={d_arr.mean():+.4f}  CI [{lo_t:+.4f},{hi_t:+.4f}]  wins={(d_arr>0).sum()}/{len(ds)}  perm p={pp_t:.4f}")
            t_rows.append({"T":T,"mean_delta":d_arr.mean(),"ci_lo":lo_t,"ci_hi":hi_t,"wins":int((d_arr>0).sum()),"n":len(ds),"perm_p":pp_t})

    # Save
    dfa.to_parquet(METRICS_DIR / "iter5_1v2_phase_a.parquet", index=False)
    dfb.to_parquet(METRICS_DIR / "iter5_1v2_nested_cv.parquet", index=False)
    pd.DataFrame(t_rows).to_parquet(METRICS_DIR / "iter5_1v2_t_sweep.parquet", index=False)
    print(f"\n  Total: {time.time()-t0:.1f}s")
    print(f"  Saved: iter5_1v2_phase_a, iter5_1v2_nested_cv, iter5_1v2_t_sweep")

if __name__ == "__main__":
    main()
