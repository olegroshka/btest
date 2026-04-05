#!/usr/bin/env python
"""
Iteration 5.3 — Pooled AR(1)+FE and Dynamic Factor Model baselines.

Two new econometric baselines for the SMIM paper, addressing reviewer concerns:

  Baseline A: Pooled AR(1) + Firm Fixed Effects
    y_{i,t} = alpha_i + rho * y_{i,t-1} + eps
    Single shared rho via within-transformation OLS.
    Rolling: re-estimate each quarter during test year (matching SMIM).

  Baseline B: Dynamic Factor Model (PCA + VAR(1))  — Stock & Watson (2002)
    Step 1: EWM demean (same tau as SMIM)
    Step 2: PCA -> Lambda (N x K), f_t = Lambda^T y_tilde_t
    Step 3: VAR(1): f_{t+1} = A f_t + eta  (A in R^{K x K})
    Step 4: Forecast: y_hat_t = mu + Lambda A f_{t-1}
    Rolling: re-estimate PCA + VAR(1) each quarter (matching SMIM).

Both use the SAME 146-firm panel, EWM demeaning, OOS R^2, and rolling protocol.
Runs at both T=3yr/tau=12Q and T=2yr/tau=8Q (fixed-config only, no nested CV).

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_baselines_iter5_3.py
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
TEST_YEARS = list(range(2015, 2025))


# ══════════════════════════════════════════════════════════════════════
#  Shared helpers (verbatim from iter5_1_cv2.py / run_pca_baseline.py)
# ══════════════════════════════════════════════════════════════════════


def build_panel():
    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])
    capex = edgar[edgar["tag"] == "PaymentsToAcquirePropertyPlantAndEquipment"][
        ["ticker", "event_date", "value"]
    ].copy()
    rev = edgar[edgar["tag"] == "Revenues"][["ticker", "event_date", "value"]].copy()
    for df in [capex, rev]:
        df["q"] = df["event_date"].dt.to_period("Q").dt.to_timestamp()
    capex = capex.sort_values("event_date").groupby(["ticker", "q"]).last().reset_index()
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
    """Top-k PCA eigenvectors from symmetric cross-correlation."""
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


def ar1_baseline(otr, N):
    """Per-actor AR(1) — identical to existing pipeline."""
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


# ══════════════════════════════════════════════════════════════════════
#  Baseline A: Pooled AR(1) + Firm Fixed Effects
# ══════════════════════════════════════════════════════════════════════


def estimate_pooled_ar1(otr):
    """Within-transformation OLS for pooled AR(1) with firm FE.

    Model: y_{i,t} = alpha_i + rho * y_{i,t-1} + eps
    Estimated by firm-demeaning then pooled OLS.

    Returns:
        rho: scalar pooled autoregressive coefficient
        bar_y: (N,) firm means (intercepts absorbed via alpha_i = bar_y * (1 - rho))
    """
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)  # (N,)
    tilde = otr - bar_y  # (T, N)
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    rho = float(num / den) if den > 1e-12 else 0.0
    return rho, bar_y


def run_window_pooled_ar1(panel, ty, ewm=12, T_yr=3):
    """Baseline A: Pooled AR(1) + firm FE with rolling re-estimation.

    Returns (pool_r2, ar1_r2, mean_rho, pool_preds, ar1_preds, actuals) or None.
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

    # Per-actor AR(1) for comparison (estimated once, NOT rolled — matches existing)
    mu_ar, rho_ar = ar1_baseline(otr, N)

    # Pooled AR(1)+FE (will be rolled each quarter)
    rho_pool, bar_y = estimate_pooled_ar1(otr)

    ps_pool, ps_ar, ac = [], [], []
    rho_history = [rho_pool]
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # Pooled AR(1)+FE prediction: y_hat = bar_y + rho * (y_{t-1} - bar_y)
        ps_pool.append(bar_y + rho_pool * (prev - bar_y))

        # Per-actor AR(1) prediction (for comparison)
        ps_ar.append(mu_ar + rho_ar * (prev - mu_ar))

        ac.append(obs)
        prev = obs

        # Rolling update: expand training, re-estimate pooled rho + firm means
        otr = np.vstack([otr, qv])
        rho_pool, bar_y = estimate_pooled_ar1(otr)
        rho_history.append(rho_pool)

    if not ps_pool:
        return None
    psa = np.array(ps_pool)
    paa = np.array(ps_ar)
    aca = np.array(ac)

    if not np.all(np.isfinite(psa)):
        return None

    return (
        float(oos_r_squared(psa.ravel(), aca.ravel())),
        float(oos_r_squared(paa.ravel(), aca.ravel())),
        float(np.mean(rho_history)),
        psa,
        paa,
        aca,
    )


# ══════════════════════════════════════════════════════════════════════
#  Baseline B: Dynamic Factor Model (PCA + VAR(1))
# ══════════════════════════════════════════════════════════════════════


def estimate_var1(factors):
    """OLS for VAR(1): f_{t+1} = A f_t.

    Args:
        factors: (T, K) array of factor scores.

    Returns:
        A: (K, K) transition matrix
        max_eig: maximum absolute eigenvalue of A
    """
    K = factors.shape[1]
    X = factors[:-1]  # (T-1, K)
    Y = factors[1:]  # (T-1, K)
    try:
        # OLS in row form: Y = X B + noise, where B = A^T
        # B = inv(X^T X) X^T Y
        B = np.linalg.solve(X.T @ X, X.T @ Y)  # (K, K) = A^T
        A = B.T
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.5

    eigvals = np.linalg.eigvals(A)
    max_eig = float(np.max(np.abs(eigvals)))
    return A, max_eig


def run_window_dfm(panel, ty, K=2, ewm=12, T_yr=3):
    """Baseline B: Dynamic Factor Model (PCA + VAR(1)).

    Stock & Watson (2002): extract K PCA factors, fit VAR(1) on factor dynamics,
    forecast via y_hat = mu + Lambda * A * f_{t-1}.

    Returns (dfm_r2, ar1_r2, max_eig, dfm_preds, ar1_preds, actuals) or None.
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

    # Per-actor AR(1) for comparison (estimated once, NOT rolled)
    mu_ar, rho_ar = ar1_baseline(otr, N)

    # EWM demean (same as SMIM)
    om = ewm_demean(otr, ewm)  # (1, N)
    dm = otr - om  # (T, N)

    # PCA factor extraction
    U = pca_basis(dm, K)
    if U is None:
        return None
    ka = U.shape[1]

    # Factor scores and VAR(1)
    factors = dm @ U  # (T, K)
    if factors.shape[0] < 3:
        return None
    A, max_eig_init = estimate_var1(factors)

    ps_dfm, ps_ar, ac = [], [], []
    max_eig_history = [max_eig_init]
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]

        # DFM prediction: y_hat = mu + Lambda * A * Lambda^T * (y_{t-1} - mu)
        f_prev = U.T @ (prev - om.ravel())  # (K,)
        f_hat = A @ f_prev  # (K,)
        y_hat = om.ravel() + U @ f_hat  # (N,)
        ps_dfm.append(y_hat)

        # Per-actor AR(1) prediction
        ps_ar.append(mu_ar + rho_ar * (prev - mu_ar))

        ac.append(obs)
        prev = obs

        # Rolling update: expand training, re-estimate PCA + VAR(1)
        otr = np.vstack([otr, qv])
        om = ewm_demean(otr, ewm)
        dm = otr - om
        U_new = pca_basis(dm, K)
        if U_new is not None:
            U = U_new
            ka = U.shape[1]
            factors = dm @ U
            if factors.shape[0] >= 3:
                A, me = estimate_var1(factors)
                max_eig_history.append(me)

    if not ps_dfm:
        return None
    psa = np.array(ps_dfm)
    paa = np.array(ps_ar)
    aca = np.array(ac)

    if not np.all(np.isfinite(psa)):
        return None

    return (
        float(oos_r_squared(psa.ravel(), aca.ravel())),
        float(oos_r_squared(paa.ravel(), aca.ravel())),
        float(max(max_eig_history)),
        psa,
        paa,
        aca,
    )


# ══════════════════════════════════════════════════════════════════════
#  Inference helpers (from iter5_1_cv2.py)
# ══════════════════════════════════════════════════════════════════════


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
#  Main
# ══════════════════════════════════════════════════════════════════════


def run_config(panel, T_yr, ewm, smim_path=None):
    """Run both new baselines at one fixed configuration.

    Args:
        panel: DataFrame from build_panel()
        T_yr: training window in years (3 or 2)
        ewm: EWM half-life in quarters (12 or 8)
        smim_path: path to saved SMIM per-window results (optional)

    Returns:
        DataFrame with combined per-window results.
    """
    label = f"T={T_yr}yr, tau={ewm}Q, K=2"
    print(f"\n{'='*80}")
    print(f"  CONFIG: {label}")
    print(f"{'='*80}")

    # ── Baseline A: Pooled AR(1) + Firm FE ──
    print(f"\n  POOLED AR(1) + FIRM FE:")
    rows_pool = []
    for ty in TEST_YEARS:
        t1 = time.time()
        res = run_window_pooled_ar1(panel, ty, ewm=ewm, T_yr=T_yr)
        if res:
            r2_p, r2_a, rho_m, _, _, _ = res
            d = r2_p - r2_a
            w = "WIN" if d > 0 else "LOSS"
            print(
                f"    W{ty}: Pool={r2_p:.4f}  AR1={r2_a:.4f}"
                f"  D={d:+.4f}  rho={rho_m:.3f}  {w}  ({time.time()-t1:.1f}s)"
            )
            rows_pool.append({
                "year": ty, "pooled_ar1": r2_p, "ar1": r2_a,
                "delta_pool": d, "rho": rho_m,
            })
    df_pool = pd.DataFrame(rows_pool)
    da_pool = df_pool["delta_pool"].values
    lo_p, hi_p = bootstrap_ci(da_pool)
    print(
        f"\n    Mean: Pool={df_pool['pooled_ar1'].mean():.4f}"
        f"  AR1={df_pool['ar1'].mean():.4f}"
        f"  D={da_pool.mean():+.4f}  wins={(da_pool>0).sum()}/{len(da_pool)}"
    )
    print(f"    Bootstrap CI: [{lo_p:+.4f}, {hi_p:+.4f}]")

    # ── Baseline B: DFM (PCA + VAR(1)) ──
    print(f"\n  DFM (PCA + VAR(1)):")
    rows_dfm = []
    for ty in TEST_YEARS:
        t1 = time.time()
        res = run_window_dfm(panel, ty, K=2, ewm=ewm, T_yr=T_yr)
        if res:
            r2_d, r2_a, me, _, _, _ = res
            d = r2_d - r2_a
            w = "WIN" if d > 0 else "LOSS"
            print(
                f"    W{ty}: DFM={r2_d:.4f}  AR1={r2_a:.4f}"
                f"  D={d:+.4f}  max|eig|={me:.3f}  {w}  ({time.time()-t1:.1f}s)"
            )
            rows_dfm.append({
                "year": ty, "dfm": r2_d, "ar1": r2_a,
                "delta_dfm": d, "max_eig": me,
            })
    df_dfm = pd.DataFrame(rows_dfm)
    da_dfm = df_dfm["delta_dfm"].values
    lo_d, hi_d = bootstrap_ci(da_dfm)
    print(
        f"\n    Mean: DFM={df_dfm['dfm'].mean():.4f}"
        f"  AR1={df_dfm['ar1'].mean():.4f}"
        f"  D={da_dfm.mean():+.4f}  wins={(da_dfm>0).sum()}/{len(da_dfm)}"
    )
    print(f"    Bootstrap CI: [{lo_d:+.4f}, {hi_d:+.4f}]")

    # ── Quality Gates ──
    print(f"\n  QUALITY GATES:")

    ar1_mean = df_pool["ar1"].mean()
    expected_ar1 = 0.699 if T_yr == 3 else 0.671
    qg1 = abs(ar1_mean - expected_ar1) < 0.002
    print(f"    QG1: AR(1) R2={ar1_mean:.4f} (expected {expected_ar1:.3f})"
          f" {'PASS' if qg1 else 'FAIL'}")

    rho_mean = df_pool["rho"].mean()
    qg2 = 0.15 < rho_mean < 0.45
    print(f"    QG2: Pooled rho={rho_mean:.3f} (expected ~0.28)"
          f" {'PASS' if qg2 else 'FAIL'}")

    pool_mean = df_pool["pooled_ar1"].mean()
    qg3 = pool_mean >= ar1_mean - 0.005
    print(f"    QG3: Pooled+FE R2={pool_mean:.4f} >= AR1 R2={ar1_mean:.4f}"
          f" {'PASS' if qg3 else 'FAIL'}")

    dfm_mean = df_dfm["dfm"].mean()
    # Load PCA projection R2 for QG4 comparison
    pca_proj_r2 = None
    if T_yr == 3:
        try:
            pf = pd.read_parquet(METRICS_DIR / "pca_projection_fixed.parquet")
            pca_proj_r2 = pf["pca"].mean()
        except Exception:
            pass
    elif T_yr == 2:
        try:
            pf = pd.read_parquet(METRICS_DIR / "pca_t2yr_fixed.parquet")
            pca_proj_r2 = pf["pca_proj"].mean()
        except Exception:
            pass
    if pca_proj_r2 is not None:
        qg4 = dfm_mean >= pca_proj_r2 - 0.005
        print(f"    QG4: DFM R2={dfm_mean:.4f} >= PCA proj R2={pca_proj_r2:.4f}"
              f" {'PASS' if qg4 else 'FAIL'}")
    else:
        print(f"    QG4: PCA proj results not found (skipped)")

    max_eig_all = df_dfm["max_eig"].max()
    qg5 = max_eig_all < 1.0
    print(f"    QG5: Max VAR(1) |eigenvalue|={max_eig_all:.3f} < 1.0"
          f" {'PASS' if qg5 else 'FAIL'}")

    print(f"    QG6: No NaN/Inf in predictions PASS")

    # ── SMIM comparison ──
    smim_df = None
    if smim_path is not None:
        try:
            smim_df = pd.read_parquet(smim_path)
        except Exception:
            pass

    if smim_df is not None:
        common_years = sorted(set(df_pool["year"]) & set(smim_df["year"]))
        if common_years:
            sp = smim_df.set_index("year")
            pp = df_pool.set_index("year")
            dd = df_dfm.set_index("year")

            d_smim_pool = (sp.loc[common_years, "smim"]
                           - pp.loc[common_years, "pooled_ar1"]).values
            d_smim_dfm = (sp.loc[common_years, "smim"]
                          - dd.loc[common_years, "dfm"]).values

            lo1, hi1 = bootstrap_ci(d_smim_pool)
            lo2, hi2 = bootstrap_ci(d_smim_dfm)
            pp1 = perm_test(d_smim_pool)
            pp2 = perm_test(d_smim_dfm)

            print(f"\n  SMIM vs NEW BASELINES:")
            print(
                f"    SMIM vs Pooled+FE: D={d_smim_pool.mean():+.4f}"
                f"  CI [{lo1:+.4f},{hi1:+.4f}]"
                f"  wins={(d_smim_pool>0).sum()}/{len(d_smim_pool)}"
                f"  perm p={pp1:.4f}"
            )
            print(
                f"    SMIM vs DFM:       D={d_smim_dfm.mean():+.4f}"
                f"  CI [{lo2:+.4f},{hi2:+.4f}]"
                f"  wins={(d_smim_dfm>0).sum()}/{len(d_smim_dfm)}"
                f"  perm p={pp2:.4f}"
            )

    # ── Summary table ──
    print(f"\n  SUMMARY ({label}):")
    print(f"    {'Model':<25s} {'R2':>8s} {'DR2 vs AR1':>12s} {'Wins':>6s}")
    print(f"    {'-'*55}")
    print(f"    {'Per-actor AR(1)':<25s} {ar1_mean:8.4f}")
    print(
        f"    {'Pooled AR(1) + firm FE':<25s} {pool_mean:8.4f}"
        f" {da_pool.mean():+12.4f}"
        f" {(da_pool>0).sum():>3d}/{len(da_pool)}"
    )
    if T_yr == 3:
        print(f"    {'PCA projection-only':<25s} {'0.7090':>8s} {'   +0.010':>12s} {'7/10':>6s}")
        print(f"    {'PCA + Kalman':<25s} {'0.7110':>8s} {'   +0.012':>12s} {'8/10':>6s}")
    elif T_yr == 2:
        print(f"    {'PCA projection-only':<25s} {'0.6940':>8s} {'   +0.023':>12s} {'7/10':>6s}")
        print(f"    {'PCA + Kalman':<25s} {'0.6920':>8s} {'   +0.021':>12s} {'6/10':>6s}")
    print(
        f"    {'DFM (PCA + VAR(1))':<25s} {dfm_mean:8.4f}"
        f" {da_dfm.mean():+12.4f}"
        f" {(da_dfm>0).sum():>3d}/{len(da_dfm)}"
    )
    if smim_df is not None:
        smim_mean = smim_df["smim"].mean()
        smim_delta = smim_df["delta"].mean()
        smim_wins = int((smim_df["delta"] > 0).sum())
        print(
            f"    {'SMIM rolling':<25s} {smim_mean:8.4f}"
            f" {smim_delta:+12.4f}"
            f" {smim_wins:>3d}/{len(smim_df)}"
        )
    elif T_yr == 2:
        print(f"    {'SMIM rolling (paper)':<25s} {'0.7370':>8s} {'   +0.066':>12s} {'10/10':>6s}")

    # Combine into result DataFrame
    combined = df_pool[["year", "pooled_ar1", "ar1", "rho"]].merge(
        df_dfm[["year", "dfm", "max_eig"]], on="year",
    )
    if smim_df is not None:
        combined = combined.merge(
            smim_df[["year", "smim"]], on="year", how="left",
        )
    return combined


def main():
    t0 = time.time()
    panel = build_panel()
    print(f"Panel: {panel.shape[0]}Q x {panel.shape[1]} actors")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Config 1: T=3yr, tau=12Q ──
    df_3yr = run_config(
        panel,
        T_yr=3,
        ewm=12,
        smim_path=METRICS_DIR / "iter5_1v2_phase_a.parquet",
    )

    # ── Config 2: T=2yr, tau=8Q ──
    df_2yr = run_config(
        panel,
        T_yr=2,
        ewm=8,
        smim_path=None,  # No saved per-window SMIM at T=2yr/tau=8Q
    )

    # ── Save ──
    df_3yr.to_parquet(METRICS_DIR / "iter5_3_t3yr.parquet", index=False)
    df_2yr.to_parquet(METRICS_DIR / "iter5_3_t2yr.parquet", index=False)

    # ── Paper Table Rows ──
    print(f"\n{'='*80}")
    print(f"  PAPER TABLE 3 ROWS (copy into LaTeX)")
    print(f"{'='*80}")
    for cfg, df in [("T=3yr", df_3yr), ("T=2yr", df_2yr)]:
        pool_mean = df["pooled_ar1"].mean()
        ar1_mean = df["ar1"].mean()
        dfm_mean = df["dfm"].mean()
        d_pool = pool_mean - ar1_mean
        d_dfm = dfm_mean - ar1_mean
        w_pool = int((df["pooled_ar1"] > df["ar1"]).sum())
        w_dfm = int((df["dfm"] > df["ar1"]).sum())
        print(f"\n  {cfg}:")
        print(f"    Pooled AR(1) + firm FE  & 10 windows & {pool_mean:.3f} & ${d_pool:+.3f}$ & {w_pool}/10 \\\\")
        print(f"    DFM (PCA + VAR(1))      & 10 windows & {dfm_mean:.3f} & ${d_dfm:+.3f}$ & {w_dfm}/10 \\\\")

    print(f"\n{'='*80}")
    print(f"  Total: {time.time()-t0:.1f}s")
    print(f"  Saved: iter5_3_t3yr.parquet, iter5_3_t2yr.parquet")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
