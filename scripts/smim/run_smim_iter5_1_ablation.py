#!/usr/bin/env python
"""
Ablation ladder on the 146-firm CapEx/Revenue panel (predictive R²).

Steps:
  0: Constant per-actor mean
  1: EWM per-actor mean (τ=12Q)
  2: + DMD K=2 OLS projection (no Kalman)
  3: + Kalman with spherical R (static basis, no online Q)
  4: + Online Q adaptation (static basis)
  5: + Rolling basis update (full pipeline)

All R² are PREDICTIVE (one-step-ahead). AR(1) T=3yr baseline for comparison.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_1_ablation.py
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
F_REG, Q_INIT, LAMBDA_Q, K_MAX = 0.99, 0.5, 0.3, 15
TEST_YEARS = list(range(2015, 2025))
K, EWM_HL, T_YR = 2, 12, 3


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


def ewm_demean(obs, hl=EWM_HL):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(dm, k=K):
    N = dm.shape[1]
    if dm.shape[0] < 3: return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(min(k, N-2), mf.K)].real
    except: return None


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res**2), 1e-8)


def prep_window(panel, ty):
    """Common data prep for all ablation steps."""
    ts = pd.Timestamp(f"{ty - T_YR}-01-01")
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 15: return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty-1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4: return None
    return ad, otr, tq, N


def step0_constant_mean(panel, ty):
    """Predict per-actor full-sample mean."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    preds, acts = [], []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        preds.append(mu)
        acts.append(qv[0])
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def step1_ewm_mean(panel, ty):
    """Predict per-actor EWM mean (no spectral dynamics)."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    preds, acts = [], []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        om = ewm_demean(otr).ravel()
        preds.append(om)
        acts.append(qv[0])
        otr = np.vstack([otr, qv])
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def step2_dmd_ols(panel, ty):
    """EWM mean + DMD OLS projection (no Kalman)."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    om = ewm_demean(otr)
    dm = otr - om
    U = dmd_basis(dm)
    if U is None: return None
    # OLS: project last training obs onto basis, use as prediction
    preds, acts = [], []
    last_dm = dm[-1]
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        alpha_ols = U.T @ last_dm
        pred = U @ alpha_ols + om.ravel()
        preds.append(pred)
        acts.append(qv[0])
        last_dm = qv[0] - om.ravel()
        # Expand training for rolling mean
        otr = np.vstack([otr, qv])
        om = ewm_demean(otr)
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def step3_kalman_static(panel, ty):
    """EWM + DMD + Kalman with spherical R, static basis, NO online Q."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    om = ewm_demean(otr)
    dm = otr - om
    U = dmd_basis(dm)
    if U is None: return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    Q = np.eye(ka) * Q_INIT
    a, P = np.zeros(ka), np.eye(ka)
    preds, acts = [], []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        obs = qv[0]; odm = obs - om.ravel()
        # Predict
        ap = F @ a; Pp = F @ P @ F.T + Q
        preds.append(U @ ap + om.ravel())
        acts.append(obs)
        # Update
        S = U @ Pp @ U.T + R
        try: Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except: Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def step4_online_q(panel, ty):
    """+ Online Q adaptation (static basis)."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    om = ewm_demean(otr)
    dm = otr - om
    U = dmd_basis(dm)
    if U is None: return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Qr = np.eye(ka) * Q_INIT
    preds, acts = [], []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        obs = qv[0]; odm = obs - om.ravel()
        ap = F @ a; Pp = F @ P @ F.T + Qr
        preds.append(U @ ap + om.ravel())
        acts.append(obs)
        S = U @ Pp @ U.T + R
        try: Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except: Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp
        inn = a - ap
        Qr = (1-LAMBDA_Q)*Qr + LAMBDA_Q*np.outer(inn, inn)
        Qr = (Qr+Qr.T)/2 + np.eye(ka)*1e-6
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def step5_rolling(panel, ty):
    """+ Rolling basis update (full pipeline)."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    om = ewm_demean(otr)
    dm = otr - om
    U = dmd_basis(dm)
    if U is None: return None
    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Qr = np.eye(ka) * Q_INIT
    preds, acts = [], []
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        obs = qv[0]; odm = obs - om.ravel()
        ap = F @ a; Pp = F @ P @ F.T + Qr
        preds.append(U @ ap + om.ravel())
        acts.append(obs)
        S = U @ Pp @ U.T + R
        try: Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except: Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp
        inn = a - ap
        Qr = (1-LAMBDA_Q)*Qr + LAMBDA_Q*np.outer(inn, inn)
        Qr = (Qr+Qr.T)/2 + np.eye(ka)*1e-6
        # Rolling basis
        exp = np.vstack([otr, qv]); otr = exp
        om2 = ewm_demean(exp); dm2 = exp - om2
        U2 = dmd_basis(dm2)
        if U2 is not None:
            k2 = U2.shape[1]
            a = U2.T @ odm; P = np.eye(k2)
            Qr = np.eye(k2)*Q_INIT; F = np.eye(k2)*F_REG
            R = sph_r(dm2, U2); U = U2; ka = k2; om = om2
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def ar1_baseline(panel, ty):
    """Per-actor AR(1) T=3yr."""
    r = prep_window(panel, ty)
    if not r: return None
    ad, otr, tq, N = r
    mu = np.nan_to_num(otr.mean(0), nan=0.5)
    d = otr - mu
    rho = np.zeros(N)
    for j in range(N):
        y = d[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c): rho[j] = c
    preds, acts = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        preds.append(mu + rho * (prev - mu))
        acts.append(qv[0])
        prev = qv[0]
    if not preds: return None
    return float(oos_r_squared(np.array(preds).ravel(), np.array(acts).ravel()))


def main():
    t0 = time.time()
    panel = build_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    print(f"Config: K={K}, EWM={EWM_HL}Q, T={T_YR}yr, predictive R²\n")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("0: Constant mean", step0_constant_mean),
        ("1: + EWM mean (τ=12Q)", step1_ewm_mean),
        ("2: + DMD K=2 (OLS proj.)", step2_dmd_ols),
        ("3: + Kalman sph. R (static)", step3_kalman_static),
        ("4: + Online Q adapt. (static)", step4_online_q),
        ("5: + Rolling basis (full)", step5_rolling),
        ("AR(1) T=3yr baseline", ar1_baseline),
    ]

    rows = []
    for label, func in steps:
        r2s = []
        for ty in TEST_YEARS:
            r2 = func(panel, ty)
            if r2 is not None:
                r2s.append(r2)
        mean_r2 = np.mean(r2s) if r2s else float("nan")
        rows.append({"step": label, "mean_r2": mean_r2, "n": len(r2s)})
        print(f"  {label:35s}  R² = {mean_r2:.4f}  (n={len(r2s)})")

    # Compute deltas
    print(f"\n  ABLATION LADDER (predictive R², CapEx/Revenue, K=2):")
    print(f"  {'-'*60}")
    prev_r2 = None
    for r in rows:
        if r["step"].startswith("AR"):
            continue
        delta = f"  Δ={r['mean_r2'] - prev_r2:+.3f}" if prev_r2 is not None else ""
        print(f"  {r['step']:35s}  {r['mean_r2']:.4f}{delta}")
        prev_r2 = r['mean_r2']
    ar1_r2 = [r for r in rows if r["step"].startswith("AR")][0]["mean_r2"]
    print(f"  {'-'*60}")
    print(f"  {'AR(1) baseline':35s}  {ar1_r2:.4f}")
    print(f"  {'Full pipeline delta vs AR(1)':35s}  {rows[-2]['mean_r2'] - ar1_r2:+.4f}")

    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter5_1v2_ablation.parquet", index=False)
    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
