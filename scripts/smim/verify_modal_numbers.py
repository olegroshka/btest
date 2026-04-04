#!/usr/bin/env python
"""
Independent verification of ALL modal/structural numbers cited in the paper.

Reproduces from scratch on the 93-actor CapEx/Assets panel:
  1. Rolling DIAMOND modal R² (claim: 0.691)
  2. Static basis modal R² (claim: 0.543)
  3. AR(1) T=10yr baseline (claim: 0.425)
  4. Per-window values for rolling, static, AR(1)
  5. Basis rotation statistics (claim: 26°/Q, σ=6.5°)
  6. Null-simulation rotation (claim: 75.3°)
  7. Predictive R² on same panel (to confirm it loses to AR(1))

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/verify_modal_numbers.py
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INT_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REG_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"

K = 8
EWM_HL = 8
T_TRAIN_YR = 5
F_REG = 0.99
Q_INIT = 0.5
LAMBDA_Q = 0.3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))


def load_panel():
    with open(REG_PATH) as f:
        reg = json.load(f)
    df = pd.read_parquet(INT_PATH)
    actors = set(df["actor_id"].unique())
    aids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    return wide[[c for c in aids if c in wide.columns]]


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


def subspace_angle(U1, U2):
    """MEAN principal angle between two subspaces (degrees).
    Matches the convention in run_smim_ws_d_rotation_bootstrap.py."""
    k = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    svs = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svs, -1, 1))
    return float(np.degrees(np.mean(angles)))


def run_window(panel, ty, rolling=True):
    """Run one window. Returns dict with modal_r2, pred_r2, ar1_r2."""
    ts = pd.Timestamp(f"{ty - T_TRAIN_YR}-01-01")
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 10: return None

    train_end = pd.Timestamp(f"{ty-1}-12-31")
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= train_end)].values.astype(np.float64)
    if otr.shape[0] < 4: return None

    om = ewm_demean(otr)
    dm = otr - om
    U = dmd_basis(dm)
    if U is None: return None

    # AR(1) baseline (T=10yr for fair comparison with paper)
    ar1_ts = pd.Timestamp(f"{ty - 10}-01-01")
    ar1_data = panel[(panel.index >= ar1_ts) & (panel.index <= te)].copy()
    ar1_v = ar1_data.columns[ar1_data.notna().any()]
    common = [c for c in v if c in ar1_v]
    ar1_data = ar1_data[common].fillna(ar1_data[common].mean())
    ar1_train = ar1_data[(ar1_data.index >= ar1_ts) & (ar1_data.index <= train_end)].values.astype(np.float64)
    ar1_mu = np.nan_to_num(ar1_train.mean(0), nan=0.5)
    ar1_dm = ar1_train - ar1_mu
    ar1_rho = np.zeros(len(common))
    for j in range(len(common)):
        y = ar1_dm[:, j]
        if np.std(y[:-1]) > 1e-10 and np.std(y[1:]) > 1e-10:
            c = np.corrcoef(y[:-1], y[1:])[0, 1]
            if np.isfinite(c): ar1_rho[j] = c

    ka = U.shape[1]
    R = sph_r(dm, U)
    F = np.eye(ka) * F_REG
    a, P = np.zeros(ka), np.eye(ka)
    Qr = np.eye(ka) * Q_INIT

    preds_modal, preds_pred, preds_ar1, acts = [], [], [], []
    # Map common actors for AR(1) comparison
    common_idx = [list(v).index(c) for c in common if c in list(v)]
    ar1_prev = np.nan_to_num(ar1_train[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0: continue
        obs = qv[0]; odm = obs - om.ravel()

        # Predictive
        ap = F @ a; Pp = F @ P @ F.T + Qr
        preds_pred.append(U @ ap + om.ravel())

        # Kalman update
        S = U @ Pp @ U.T + R
        try: Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
        except: Kg = np.zeros((ka, N))
        a = ap + Kg @ (odm - U @ ap)
        P = (np.eye(ka) - Kg @ U) @ Pp

        # Modal (filtered)
        preds_modal.append(U @ a + om.ravel())
        acts.append(obs)

        # AR(1) — on common actors only
        ar1_pred_full = ar1_mu + ar1_rho * (ar1_prev - ar1_mu)
        preds_ar1.append(ar1_pred_full)
        ar1_q = ar1_data.loc[[qd]].values.astype(np.float64)
        if ar1_q.shape[0] > 0:
            ar1_prev = ar1_q[0]

        # Online Q
        inn = a - ap
        Qr = (1-LAMBDA_Q)*Qr + LAMBDA_Q*np.outer(inn, inn)
        Qr = (Qr+Qr.T)/2 + np.eye(ka)*1e-6

        # Rolling basis update
        if rolling:
            exp = np.vstack([otr, qv]); otr = exp
            om2 = ewm_demean(exp); dm2 = exp - om2
            U2 = dmd_basis(dm2)
            if U2 is not None:
                k2 = U2.shape[1]
                a = U2.T @ odm; P = np.eye(k2)
                Qr = np.eye(k2)*Q_INIT; F = np.eye(k2)*F_REG
                R = sph_r(dm2, U2); U = U2; ka = k2; om = om2

    if not preds_modal: return None
    pm = np.array(preds_modal); pp = np.array(preds_pred)
    ac = np.array(acts); pa = np.array(preds_ar1)

    # AR(1) R² on common actors
    ar1_r2 = float(oos_r_squared(pa.ravel(), ar1_data.loc[tq[tq.isin(ad.index)]].reindex(tq).dropna().values[:len(pa)].ravel()))

    return {
        "modal_r2": float(oos_r_squared(pm.ravel(), ac.ravel())),
        "pred_r2": float(oos_r_squared(pp.ravel(), ac.ravel())),
        "ar1_r2": float(oos_r_squared(pa.ravel(), ac[:len(pa)].ravel())) if len(common) == len(list(v)) else float("nan"),
        "N": N, "K": ka,
    }


def compute_rotation(panel):
    """Compute basis rotation across all quarterly transitions."""
    quarters = pd.date_range("2010-01-01", "2024-12-31", freq="QS")
    angles = []
    prev_U = None
    for i, q in enumerate(quarters):
        ts = q - pd.DateOffset(years=T_TRAIN_YR)
        data = panel[(panel.index >= ts) & (panel.index <= q)].copy()
        v = data.columns[data.notna().any()]
        data = data[v].fillna(data[v].mean())
        obs = data.values.astype(np.float64)
        if obs.shape[0] < 4: continue
        om = ewm_demean(obs)
        dm = obs - om
        U = dmd_basis(dm)
        if U is None: continue
        if prev_U is not None and U.shape[0] == prev_U.shape[0]:
            angle = subspace_angle(prev_U, U)
            angles.append({"quarter": q, "angle": angle})
        prev_U = U
    return angles


def null_rotation(panel, n_sims=100):
    """Null simulation: fixed basis, measure apparent rotation from noise."""
    # Use a fixed quarter's data to define the "true" basis
    ts = pd.Timestamp("2015-01-01")
    te = pd.Timestamp("2019-12-31")
    data = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = data.columns[data.notna().any()]
    data = data[v].fillna(data[v].mean())
    obs = data.values.astype(np.float64)
    N, T = obs.shape[1], obs.shape[0]
    om = ewm_demean(obs)
    dm = obs - om

    # Get "true" basis and noise level
    U_true = dmd_basis(dm)
    if U_true is None: return float("nan")
    resid = dm - (dm @ U_true) @ U_true.T
    noise_std = np.std(resid)

    rng = np.random.default_rng(42)
    null_angles = []
    for _ in range(n_sims):
        # Generate two windows from fixed basis + noise
        alpha1 = rng.standard_normal((T, U_true.shape[1]))
        alpha2 = rng.standard_normal((T, U_true.shape[1]))
        Y1 = (alpha1 @ U_true.T) + rng.standard_normal((T, N)) * noise_std
        Y2 = (alpha2 @ U_true.T) + rng.standard_normal((T, N)) * noise_std
        U1 = dmd_basis(Y1)
        U2 = dmd_basis(Y2)
        if U1 is not None and U2 is not None and U1.shape == U2.shape:
            null_angles.append(subspace_angle(U1, U2))
    return np.mean(null_angles), np.std(null_angles)


def main():
    t0 = time.time()
    panel = load_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    print(f"Config: K={K}, EWM={EWM_HL}Q, T={T_TRAIN_YR}yr\n")

    # ── 1. Rolling DIAMOND (modal + predictive) ──
    print("=" * 80)
    print("  1. ROLLING DIAMOND (K=8, EWM=8, T=5yr)")
    print("=" * 80)
    for label, rolling in [("Rolling", True), ("Static", False)]:
        print(f"\n  {label} basis:")
        modal_r2s, pred_r2s = [], []
        for ty in TEST_YEARS:
            res = run_window(panel, ty, rolling=rolling)
            if res:
                modal_r2s.append(res["modal_r2"])
                pred_r2s.append(res["pred_r2"])
                print(f"    W{ty}: modal={res['modal_r2']:.4f}  pred={res['pred_r2']:.4f}  N={res['N']}")
        print(f"    MEAN modal: {np.mean(modal_r2s):.4f}")
        print(f"    MEAN pred:  {np.mean(pred_r2s):.4f}")

    # ── 2. AR(1) baselines (both protocols) ──
    print(f"\n  AR(1) T=10yr — TWO protocols:")
    ar1_seq_r2s, ar1_static_r2s = [], []
    for ty in TEST_YEARS:
        ts = pd.Timestamp(f"{ty-10}-01-01")
        te = pd.Timestamp(f"{ty}-12-31")
        ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
        v = ad.columns[ad.notna().any()]
        ad = ad[v].fillna(ad[v].mean())
        N = len(v)
        train_end = pd.Timestamp(f"{ty-1}-12-31")
        tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
        train = ad[(ad.index >= ts) & (ad.index <= train_end)].values.astype(np.float64)

        # Fit AR(1) OLS: y_curr = a + b * y_lag
        mu = np.nan_to_num(train.mean(0), nan=0.5)
        a_coef = np.zeros(N)
        b_coef = np.zeros(N)
        for j in range(N):
            y = train[:, j]
            y_lag, y_curr = y[:-1], y[1:]
            if np.std(y_lag) < 1e-10:
                a_coef[j] = y.mean()
                continue
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            try:
                beta = np.linalg.lstsq(X, y_curr, rcond=None)[0]
                a_coef[j] = beta[0]
                b_coef[j] = beta[1]
            except:
                a_coef[j] = y.mean()

        # Sequential: update with actual obs each quarter
        preds_seq, acts = [], []
        prev = np.nan_to_num(train[-1], nan=0.5)
        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0: continue
            preds_seq.append(a_coef + b_coef * prev)
            acts.append(qv[0])
            prev = qv[0]

        # Static multi-step: compound predictions (original a2 protocol)
        preds_static = []
        last_val = np.nan_to_num(train[-1], nan=0.5).copy()
        for qd in tq:
            qv = ad.loc[[qd]].values.astype(np.float64)
            if qv.shape[0] == 0: continue
            pred = a_coef + b_coef * last_val
            preds_static.append(pred)
            last_val = pred  # use PREDICTED, not actual

        if preds_seq:
            r2_seq = float(oos_r_squared(np.array(preds_seq).ravel(), np.array(acts).ravel()))
            r2_sta = float(oos_r_squared(np.array(preds_static).ravel(), np.array(acts).ravel()))
            ar1_seq_r2s.append(r2_seq)
            ar1_static_r2s.append(r2_sta)
            print(f"    W{ty}: sequential={r2_seq:.4f}  static={r2_sta:.4f}  N={N}")
    print(f"    MEAN sequential: {np.mean(ar1_seq_r2s):.4f}")
    print(f"    MEAN static:     {np.mean(ar1_static_r2s):.4f}")

    # ── 3. Basis rotation ──
    print(f"\n{'='*80}")
    print(f"  3. BASIS ROTATION")
    print(f"{'='*80}")
    angles = compute_rotation(panel)
    if angles:
        a_vals = [a["angle"] for a in angles]
        print(f"  Transitions: {len(angles)}")
        print(f"  Mean rotation: {np.mean(a_vals):.1f}°  (σ={np.std(a_vals):.1f}°)")
        print(f"  Min: {np.min(a_vals):.1f}°  Max: {np.max(a_vals):.1f}°")
        high = [(str(a["quarter"].date()), a["angle"]) for a in angles if a["angle"] > 32]
        if high:
            print(f"  High rotation (>32°):")
            for q, ang in high:
                print(f"    {q}: {ang:.1f}°")

    # ── 4. Null simulation ──
    print(f"\n  Null simulation (100 sims)...")
    null_mean, null_std = null_rotation(panel, n_sims=100)
    print(f"  Null rotation: {null_mean:.1f}° (σ={null_std:.1f}°)")

    # ── 5. Summary vs paper claims ──
    print(f"\n{'='*80}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Claim':<40s} {'Paper':>8s} {'Verified':>10s} {'Match':>6s}")
    print(f"  {'-'*70}")

    checks = []
    def check(claim, paper, verified, tol=0.02):
        match = abs(paper - verified) < tol
        checks.append(match)
        sym = "OK" if match else "FAIL"
        print(f"  {claim:<40s} {paper:8.3f} {verified:10.3f} {sym:>6s}")

    rolling_modal = np.mean(modal_r2s) if modal_r2s else 0
    # Re-run static to get its mean
    static_modal_r2s = []
    for ty in TEST_YEARS:
        res = run_window(panel, ty, rolling=False)
        if res: static_modal_r2s.append(res["modal_r2"])
    static_modal = np.mean(static_modal_r2s) if static_modal_r2s else 0

    check("Rolling modal R²", 0.691, rolling_modal)
    check("Static modal R²", 0.543, static_modal)
    check("AR(1) T=10yr static (paper)", 0.425, np.mean(ar1_static_r2s))
    check("AR(1) T=10yr sequential", 0.623, np.mean(ar1_seq_r2s))
    check("Rotation mean (°)", 26.0, np.mean(a_vals) if angles else 0, tol=2.0)
    check("Rotation σ (°)", 6.5, np.std(a_vals) if angles else 0, tol=2.0)
    check("Null rotation (°)", 75.3, null_mean, tol=5.0)

    all_ok = all(checks)
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
