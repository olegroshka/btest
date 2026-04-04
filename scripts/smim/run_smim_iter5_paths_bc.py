#!/usr/bin/env python
"""
SMIM Iteration 5 — Paths B, C, and Reference.

Path B: Multi-ratio panel (134 actors x 3 signals) + operator learning
Path C: CapEx/Revenue + experiment_a1 heterogeneous actors
Reference: DIAMOND + operator learning on original experiment_a1

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_paths_bc.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
INTENSITY_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
F_REG = 0.99; Q_INIT = 0.5; LAMBDA_Q = 0.3
TEST_YEARS = list(range(2015, 2025))

edgar = pd.read_parquet(EDGAR_PATH)
edgar["event_date"] = pd.to_datetime(edgar["event_date"])


def build_ratio(nt, dt):
    n = edgar[edgar["tag"] == nt][["ticker", "event_date", "value"]].copy()
    d = edgar[edgar["tag"] == dt][["ticker", "event_date", "value"]].copy()
    n["q"] = n["event_date"].dt.to_period("Q").dt.to_timestamp()
    d["q"] = d["event_date"].dt.to_period("Q").dt.to_timestamp()
    n = n.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    d = d.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
    m = n.merge(d, on=["ticker", "q"], suffixes=("_n", "_d"))
    m["ratio"] = m["value_n"] / m["value_d"]
    m = m.replace([np.inf, -np.inf], np.nan)
    p = m.pivot_table(index="q", columns="ticker", values="ratio")
    p.index = pd.to_datetime(p.index)
    r = p.rank(axis=1, method="average", pct=True)
    g = r.notna().mean()
    return r[g[g > 0.5].index].loc["2005-01-01":"2025-12-31"]


def ewm_mean(obs, hl):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(dm, k):
    N = dm.shape[1]
    ku = min(k, N - 2)
    if dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k + 5, N))
        return mf.basis[:, :min(ku, mf.K)].real
    except Exception:
        return None


def sph_r(dm, U):
    return np.eye(U.shape[0]) * max(np.mean((dm - (dm @ U) @ U.T) ** 2), 1e-8)


def build_op_lib(dm, N):
    T = dm.shape[0]
    lib = []
    c = np.corrcoef(dm.T)
    c = np.nan_to_num(c)
    np.fill_diagonal(c, 0)
    lib.append(c)
    if T > 2:
        lag = np.zeros((N, N))
        for lo in [1, 2]:
            if T > lo:
                xp, xn = dm[:-lo], dm[lo:]
                num = (xp - xp.mean(0)).T @ (xn - xn.mean(0)) / max(len(xp) - 1, 1)
                sp = np.std(xp, 0, keepdims=True).T
                sn = np.std(xn, 0, keepdims=True)
                lag += np.nan_to_num(num / np.maximum(sp @ sn, 1e-10))
        np.fill_diagonal(lag, 0)
        lib.append(lag)
    for sc in [4]:
        if T >= sc + 1:
            ker = np.ones(sc) / sc
            sm = np.apply_along_axis(lambda x: np.convolve(x, ker, mode="valid"), 0, dm)
            if sm.shape[0] >= 2:
                nms = np.maximum(np.linalg.norm(sm, axis=0, keepdims=True), 1e-10)
                s = (sm / nms).T @ (sm / nms)
                np.fill_diagonal(s, 0)
                lib.append(s)
    return lib


def opt_op(dm, N, lib, K_use):
    T = dm.shape[0]
    ts = int(T * 0.7)
    st, sv = dm[:ts], dm[ts:]

    def obj(w):
        try:
            A = sum(wi * B for wi, B in zip(w, lib))
            shaped = st @ (np.eye(N) + 0.1 * A)
            U = dmd_basis(shaped, K_use)
            if U is None:
                return 1e6
            K = U.shape[1]
            Rs = sph_r(st, U)
            F = np.eye(K) * F_REG
            Q = np.eye(K) * Q_INIT
            a, P, Qr = np.zeros(K), np.eye(K), Q.copy()
            preds = []
            for t in range(len(sv)):
                ap = F @ a
                Pp = F @ P @ F.T + Qr
                preds.append(U @ ap)
                S = U @ Pp @ U.T + Rs
                try:
                    Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
                except Exception:
                    Kg = np.zeros((K, N))
                a = ap + Kg @ (sv[t] - U @ ap)
                P = (np.eye(K) - Kg @ U) @ Pp
                si = a - ap
                Qr = 0.7 * Qr + 0.3 * np.outer(si, si)
                Qr = (Qr + Qr.T) / 2 + np.eye(K) * 1e-8
            r2 = float(oos_r_squared(np.array(preds).ravel(), sv.ravel()))
            return -r2 if np.isfinite(r2) else 1e6
        except Exception:
            return 1e6

    x0 = np.ones(len(lib)) / len(lib)
    opt = sp_minimize(obj, x0, method="Nelder-Mead",
                      options={"maxiter": 80, "xatol": 0.02, "fatol": 0.002})
    return sum(wi * B for wi, B in zip(opt.x, lib))


def run_full(panel, ty, K_use, ewm_hl, t_yr):
    ts = pd.Timestamp(f"{ty - t_yr}-01-01")
    te = pd.Timestamp(f"{ty - 1}-12-31")
    tte = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= tte)].copy()
    cov = ad.loc[ad.index <= te].notna().mean()
    var = ad.loc[ad.index <= te].var()
    valid = cov[(cov > 0.5) & (var > 1e-10)].index
    ad = ad[valid].fillna(ad[valid].mean())
    N = len(valid)
    if N < 15:
        return None
    otr = ad[ad.index <= te].values.astype(np.float64)
    tv = ad.loc[ad.index > te].values.astype(np.float64)
    if len(otr) < 6 or len(tv) == 0:
        return None

    mu_a = otr.mean(0)
    dm_a = otr - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10
        and np.isfinite(np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1])
        else 0
        for j in range(N)
    ])

    om = ewm_mean(otr, ewm_hl)
    dm = otr - om
    lib = build_op_lib(dm, N)
    A_opt = opt_op(dm, N, lib, K_use)
    ops = np.eye(N) + 0.1 * A_opt

    U = dmd_basis(dm @ ops, K_use)
    if U is None:
        U = dmd_basis(dm, K_use)
    if U is None:
        return None
    K = U.shape[1]
    Rs = sph_r(dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT
    a, P, Qr = np.zeros(K), np.eye(K), Q.copy()
    ps, pa, ac = [], [], []
    prev = otr[-1]
    exp = otr.copy()

    for i in range(len(tv)):
        obs = tv[i]
        odm = obs - om.ravel()
        ap = F @ a
        Pp = F @ P @ F.T + Qr
        ps.append(U @ ap + om.ravel())
        pa.append(mu_a + rho * (prev - mu_a))
        ac.append(obs)
        prev = obs
        S = U @ Pp @ U.T + Rs
        try:
            Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((K, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(K) - Kg @ U) @ Pp
        si = a - ap
        Qr = (1 - LAMBDA_Q) * Qr + LAMBDA_Q * np.outer(si, si)
        Qr = (Qr + Qr.T) / 2 + np.eye(K) * 1e-6
        exp = np.vstack([exp, obs.reshape(1, -1)])
        om = ewm_mean(exp, ewm_hl)
        dm_n = exp - om
        Un = dmd_basis(dm_n @ ops, K_use)
        if Un is None:
            Un = dmd_basis(dm_n, K_use)
        if Un is not None:
            Kn = Un.shape[1]
            a = Un.T @ odm
            P = np.eye(Kn)
            Qr = np.eye(Kn) * Q_INIT
            F = np.eye(Kn) * F_REG
            Rs = sph_r(dm_n, Un)
            U, K = Un, Kn

    r2s = float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))
    r2a = float(oos_r_squared(np.array(pa).ravel(), np.array(ac).ravel()))
    return {"smim": r2s, "ar1": r2a, "delta": r2s - r2a, "N": N}


def test_panel(name, panel, configs):
    print(f"\n--- {name} ---")
    all_rows = []
    for K_use, ewm_hl, t_yr, label in configs:
        rows = []
        for ty in TEST_YEARS:
            r = run_full(panel, ty, K_use, ewm_hl, t_yr)
            if r and np.isfinite(r["ar1"]):
                rows.append(r)
        if rows:
            df = pd.DataFrame(rows)
            d = df["smim"] - df["ar1"]
            w = (d > 0).sum()
            tag = " <<<" if d.mean() > 0 else ""
            print(f"  {label:20s} (K={K_use},EWM={ewm_hl},T={t_yr}): "
                  f"SMIM={df['smim'].mean():.4f} AR1={df['ar1'].mean():.4f} "
                  f"delta={d.mean():+.4f} wins={w}/{len(df)} N~{int(df['N'].mean())}{tag}")
            for _, row in df.iterrows():
                all_rows.append({**row.to_dict(), "config": label, "panel": name})
    return all_rows


def main():
    t_total = time.time()

    cr = build_ratio("PaymentsToAcquirePropertyPlantAndEquipment", "Revenues")
    ra = build_ratio("Revenues", "Assets")
    ca = build_ratio("PaymentsToAcquirePropertyPlantAndEquipment", "Assets")

    a1 = pd.read_parquet(INTENSITY_PATH)
    a1w = a1.pivot_table(index="period", columns="actor_id", values="intensity_value")
    a1w.index = pd.to_datetime(a1w.index)
    a1w = a1w.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))

    all_results = []

    # PATH B: Multi-ratio
    print("=" * 80)
    print("  PATH B: Multi-ratio panel")
    print("=" * 80)
    common3 = sorted(set(cr.columns) & set(ra.columns) & set(ca.columns))
    mc = {}
    for a in common3:
        mc[f"{a}_cr"] = cr[a]
        mc[f"{a}_ra"] = ra[a]
        mc[f"{a}_ca"] = ca[a]
    multi = pd.DataFrame(mc)
    print(f"  {len(common3)} actors x 3 = {multi.shape[1]} columns")
    all_results.extend(test_panel("multi_ratio", multi, [
        (3, 12, 3, "K3-EWM12-T3"),
        (5, 8, 3, "K5-EWM8-T3"),
        (5, 10, 5, "K5-EWM10-T5"),
    ]))

    # PATH C: CapEx/Revenue + heterogeneous actors
    print("\n" + "=" * 80)
    print("  PATH C: CapEx/Revenue + experiment_a1 heterogeneous actors")
    print("=" * 80)
    other = [c for c in a1w.columns if c not in cr.columns]
    combined = pd.concat([cr, a1w[other]], axis=1)
    print(f"  {cr.shape[1]} CR firms + {len(other)} other = {combined.shape[1]} actors")
    all_results.extend(test_panel("cr_plus_a1", combined, [
        (3, 12, 3, "K3-EWM12-T3"),
        (5, 10, 3, "K5-EWM10-T3"),
        (5, 8, 5, "K5-EWM8-T5"),
    ]))

    # REFERENCE: Original experiment_a1 WITH operator learning
    print("\n" + "=" * 80)
    print("  REFERENCE: experiment_a1 + Operator Learning (DIAMOND upgrade)")
    print("=" * 80)
    print(f"  {a1w.shape[1]} actors")
    all_results.extend(test_panel("experiment_a1_oplearn", a1w, [
        (3, 8, 5, "K3-EWM8-T5"),
        (5, 8, 5, "K5-EWM8-T5"),
        (8, 8, 5, "K8-EWM8-T5-DIAMOND"),
    ]))

    # Save
    pd.DataFrame(all_results).to_parquet(METRICS_DIR / "iter5_paths_bc_results.parquet")
    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
