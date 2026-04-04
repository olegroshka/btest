#!/usr/bin/env python
"""
SMIM Iteration 5b: Systematic Signal Sweep.

Constructs all possible quarterly intensity ratios from EDGAR data,
measures their persistence, and runs SMIM on the promising ones.

The key insight: SMIM works when AR(1) rho is moderate (~0.5-0.7),
leaving headroom for spectral dynamics. We systematically search for
signals with this profile.

Usage::
    PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_smim_iter5_signal_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EDGAR_PATH = PROJECT_ROOT / "data" / "smim" / "processed" / "edgar_balance_sheet.parquet"
OHLCV_PATH = PROJECT_ROOT / "equities" / "smim" / "US-LC" / "ohlcv.parquet"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_USE = 8
F_REG = 0.99
Q_INIT = 0.5
LAMBDA_Q = 0.3
EWM_HL = 8
TEST_YEARS = list(range(2015, 2025))

RATIO_DEFS = [
    ("capex_assets", "PaymentsToAcquirePropertyPlantAndEquipment", "Assets",
     "Investment rate (CapEx/Assets) — the proven signal"),
    ("rnd_assets", "ResearchAndDevelopmentExpense", "Assets",
     "Innovation intensity (R&D/Assets)"),
    ("rnd_revenue", "ResearchAndDevelopmentExpense", "Revenues",
     "R&D share of sales"),
    ("debt_assets", "LongTermDebt", "Assets",
     "Leverage (Debt/Assets)"),
    ("debt_equity", "LongTermDebt", "StockholdersEquity",
     "Capital structure (Debt/Equity)"),
    ("equity_assets", "StockholdersEquity", "Assets",
     "Solvency (Equity/Assets)"),
    ("revenue_assets", "Revenues", "Assets",
     "Asset turnover (Revenue/Assets)"),
    ("capex_revenue", "PaymentsToAcquirePropertyPlantAndEquipment", "Revenues",
     "Investment intensity per unit sales"),
]

# Also: growth rates (quarter-over-quarter change)
GROWTH_DEFS = [
    ("revenue_growth", "Revenues", "YoY revenue growth"),
    ("asset_growth", "Assets", "YoY asset growth"),
    ("capex_growth", "PaymentsToAcquirePropertyPlantAndEquipment", "YoY CapEx growth"),
    ("rnd_growth", "ResearchAndDevelopmentExpense", "YoY R&D growth"),
    ("equity_growth", "StockholdersEquity", "YoY equity growth"),
]


def ewm_mean(obs, halflife=EWM_HL):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / halflife)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def dmd_basis(obs_dm, k):
    N = obs_dm.shape[1]
    k_use = min(k, N - 2)
    if obs_dm.shape[0] < 3:
        return None
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(obs_dm.T, k=min(k + 5, N))
        return mf.basis[:, :min(k_use, mf.K)].real
    except Exception:
        return None


def sph_r(obs_dm, U):
    N = U.shape[0]
    resid = obs_dm - (obs_dm @ U) @ U.T
    return np.eye(N) * max(np.mean(resid ** 2), 1e-8)


def kf_step(obs, U, F, R, alpha, P, Q):
    N, K = U.shape
    ap = F @ alpha
    Pp = F @ P @ F.T + Q
    pred = U @ ap
    S = U @ Pp @ U.T + R
    try:
        Kg = Pp @ U.T @ np.linalg.solve(S, np.eye(N))
    except np.linalg.LinAlgError:
        Kg = np.zeros((K, N))
    af = ap + Kg @ (obs - pred)
    Pf = (np.eye(K) - Kg @ U) @ Pp
    si = af - ap
    Qn = (1 - LAMBDA_Q) * Q + LAMBDA_Q * np.outer(si, si)
    Qn = (Qn + Qn.T) / 2 + np.eye(K) * 1e-8
    return pred, af, Pf, Qn


def run_smim_rolling(panel_wide, test_year, K_use=K_USE, train_years=5):
    ts = pd.Timestamp(f"{test_year - train_years}-01-01")
    te_tr = pd.Timestamp(f"{test_year - 1}-12-31")
    te_te = pd.Timestamp(f"{test_year}-12-31")

    all_d = panel_wide[(panel_wide.index >= ts) & (panel_wide.index <= te_te)].copy()
    cov = all_d.loc[all_d.index <= te_tr].notna().mean()
    valid = cov[cov > 0.50].index
    all_d = all_d[valid].fillna(all_d[valid].mean())
    N = len(valid)
    if N < 15:
        return None, None

    train = all_d[all_d.index <= te_tr].values.astype(np.float64)
    test_dates = all_d.index[all_d.index > te_tr]
    test = all_d.loc[test_dates].values.astype(np.float64)
    if len(train) < 8 or len(test) == 0:
        return None, None

    mu_a = train.mean(0)
    dm_a = train - mu_a
    rho = np.array([
        np.corrcoef(dm_a[:-1, j], dm_a[1:, j])[0, 1]
        if dm_a[:, j].std() > 1e-10 else 0
        for j in range(N)
    ])

    mu = ewm_mean(train)
    dm = train - mu
    U = dmd_basis(dm, K_use)
    if U is None:
        return None, None
    K = U.shape[1]
    R_s = sph_r(dm, U)
    F = np.eye(K) * F_REG
    Q = np.eye(K) * Q_INIT
    a = np.zeros(K)
    P = np.eye(K)
    Qr = Q.copy()

    preds_s, preds_a, acts = [], [], []
    prev = train[-1]
    exp = train.copy()

    for i in range(len(test)):
        obs = test[i]
        obs_dm = obs - mu.ravel()
        sp, a, P, Qr = kf_step(obs_dm, U, F, R_s, a, P, Qr)
        preds_s.append(sp + mu.ravel())
        preds_a.append(mu_a + rho * (prev - mu_a))
        acts.append(obs)
        prev = obs

        exp = np.vstack([exp, obs.reshape(1, -1)])
        mu = ewm_mean(exp)
        dm_new = exp - mu
        U_new = dmd_basis(dm_new, K_use)
        if U_new is not None:
            Kn = U_new.shape[1]
            a = U_new.T @ obs_dm
            P = np.eye(Kn)
            Qr = np.eye(Kn) * Q_INIT
            F = np.eye(Kn) * F_REG
            R_s = sph_r(dm_new, U_new)
            U, K = U_new, Kn

    ps = np.array(preds_s)
    pa = np.array(preds_a)
    ac = np.array(acts)
    return (
        float(oos_r_squared(ps.ravel(), ac.ravel())),
        float(oos_r_squared(pa.ravel(), ac.ravel())),
    )


def build_ratio_panel(edgar, num_tag, den_tag, min_tickers=30):
    """Build cross-sectional rank panel from EDGAR ratio."""
    num = edgar[edgar["tag"] == num_tag][["ticker", "event_date", "value"]].copy()
    den = edgar[edgar["tag"] == den_tag][["ticker", "event_date", "value"]].copy()

    # Quarterly aggregation (sum for flow items, last for stock items)
    num["quarter"] = num["event_date"].dt.to_period("Q").dt.to_timestamp()
    den["quarter"] = den["event_date"].dt.to_period("Q").dt.to_timestamp()

    # Use most recent filing per ticker per quarter
    num = num.sort_values("event_date").groupby(["ticker", "quarter"]).last().reset_index()
    den = den.sort_values("event_date").groupby(["ticker", "quarter"]).last().reset_index()

    merged = num.merge(den, on=["ticker", "quarter"], suffixes=("_num", "_den"))
    merged["ratio"] = merged["value_num"] / merged["value_den"]
    merged = merged.replace([np.inf, -np.inf], np.nan)

    panel = merged.pivot_table(index="quarter", columns="ticker", values="ratio")
    panel.index = pd.to_datetime(panel.index)

    # Cross-sectional rank
    ranked = panel.rank(axis=1, method="average", pct=True)

    # Filter coverage
    coverage = ranked.notna().mean()
    good = coverage[coverage > 0.50].index
    ranked = ranked[good]
    ranked = ranked.loc["2005-01-01":"2025-12-31"]

    if ranked.shape[1] < min_tickers:
        return None

    return ranked


def build_growth_panel(edgar, tag, min_tickers=30):
    """Build YoY growth rate cross-sectional rank panel."""
    data = edgar[edgar["tag"] == tag][["ticker", "event_date", "value"]].copy()
    data["quarter"] = data["event_date"].dt.to_period("Q").dt.to_timestamp()
    data = data.sort_values("event_date").groupby(["ticker", "quarter"]).last().reset_index()

    panel = data.pivot_table(index="quarter", columns="ticker", values="value")
    panel.index = pd.to_datetime(panel.index)

    # YoY growth: (value_Q - value_{Q-4}) / |value_{Q-4}|
    growth = (panel - panel.shift(4)) / panel.shift(4).abs().replace(0, np.nan)
    growth = growth.replace([np.inf, -np.inf], np.nan)
    growth = growth.clip(-5, 5)  # cap extreme growth

    ranked = growth.rank(axis=1, method="average", pct=True)
    coverage = ranked.notna().mean()
    good = coverage[coverage > 0.30].index
    ranked = ranked[good]
    ranked = ranked.loc["2006-01-01":"2025-12-31"]  # need 4Q lookback

    if ranked.shape[1] < min_tickers:
        return None

    return ranked


def main():
    t_total = time.time()
    print("=" * 80)
    print("  ITERATION 5b: Systematic Signal Sweep")
    print("=" * 80)

    edgar = pd.read_parquet(EDGAR_PATH)
    edgar["event_date"] = pd.to_datetime(edgar["event_date"])

    results = []

    # ================================================================
    # Build all ratio panels and characterise
    # ================================================================
    print("\n--- Ratio Signals ---")
    print(f"{'Signal':25s} {'N_act':>6s} {'N_Q':>5s} {'rho_med':>8s} {'AR1_R2':>7s} {'headroom':>9s}")
    print("-" * 75)

    panels = {}

    for name, num_tag, den_tag, desc in RATIO_DEFS:
        panel = build_ratio_panel(edgar, num_tag, den_tag)
        if panel is None:
            print(f"{name:25s}  SKIP (too few tickers)")
            continue

        rho_per = panel.apply(lambda c: c.dropna().autocorr(1))
        rho_med = rho_per.median()

        # Quick AR(1) R2 estimate on one window
        ar1_r2s = []
        for ty in [2020, 2022, 2024]:
            _, r2a = run_smim_rolling(panel, ty, K_use=3)
            if r2a is not None:
                ar1_r2s.append(r2a)
        ar1_mean = np.mean(ar1_r2s) if ar1_r2s else np.nan

        headroom = 1 - ar1_mean if not np.isnan(ar1_mean) else np.nan
        print(f"{name:25s} {panel.shape[1]:>6d} {panel.shape[0]:>5d} {rho_med:>8.4f} "
              f"{ar1_mean:>7.4f} {headroom:>9.4f}")

        panels[name] = panel
        results.append({
            "signal": name, "type": "ratio", "desc": desc,
            "n_actors": panel.shape[1], "n_quarters": panel.shape[0],
            "rho_median": rho_med, "ar1_r2": ar1_mean, "headroom": headroom,
        })

    # ================================================================
    # Build growth panels
    # ================================================================
    print(f"\n--- Growth Signals ---")
    print(f"{'Signal':25s} {'N_act':>6s} {'N_Q':>5s} {'rho_med':>8s} {'AR1_R2':>7s} {'headroom':>9s}")
    print("-" * 75)

    for name, tag, desc in GROWTH_DEFS:
        panel = build_growth_panel(edgar, tag)
        if panel is None:
            print(f"{name:25s}  SKIP (too few tickers)")
            continue

        rho_per = panel.apply(lambda c: c.dropna().autocorr(1))
        rho_med = rho_per.median()

        ar1_r2s = []
        for ty in [2020, 2022, 2024]:
            _, r2a = run_smim_rolling(panel, ty, K_use=3)
            if r2a is not None:
                ar1_r2s.append(r2a)
        ar1_mean = np.mean(ar1_r2s) if ar1_r2s else np.nan

        headroom = 1 - ar1_mean if not np.isnan(ar1_mean) else np.nan
        print(f"{name:25s} {panel.shape[1]:>6d} {panel.shape[0]:>5d} {rho_med:>8.4f} "
              f"{ar1_mean:>7.4f} {headroom:>9.4f}")

        panels[name] = panel
        results.append({
            "signal": name, "type": "growth", "desc": desc,
            "n_actors": panel.shape[1], "n_quarters": panel.shape[0],
            "rho_median": rho_med, "ar1_r2": ar1_mean, "headroom": headroom,
        })

    # ================================================================
    # Run full SMIM on promising signals (headroom > 0.20)
    # ================================================================
    res_df = pd.DataFrame(results)
    promising = res_df[res_df["headroom"] > 0.20]["signal"].tolist()

    print(f"\n{'='*80}")
    print(f"  FULL SMIM ON PROMISING SIGNALS (headroom > 20%)")
    print(f"{'='*80}")

    smim_results = []

    for name in promising:
        panel = panels[name]
        print(f"\n--- {name} ({res_df[res_df['signal']==name]['desc'].iloc[0]}) ---")

        for K_use in [3, 5, 8]:
            s_r2s, a_r2s = [], []
            for ty in TEST_YEARS:
                r2s, r2a = run_smim_rolling(panel, ty, K_use=K_use)
                if r2s is not None:
                    s_r2s.append(r2s)
                    a_r2s.append(r2a)
            if s_r2s:
                deltas = [s - a for s, a in zip(s_r2s, a_r2s)]
                wins = sum(1 for d in deltas if d > 0)
                print(f"  K={K_use}: SMIM={np.mean(s_r2s):.4f}  AR1={np.mean(a_r2s):.4f}  "
                      f"delta={np.mean(deltas):+.4f}  wins={wins}/{len(deltas)}")
                smim_results.append({
                    "signal": name, "K": K_use,
                    "smim_r2": np.mean(s_r2s), "ar1_r2": np.mean(a_r2s),
                    "delta": np.mean(deltas), "wins": wins, "n_windows": len(deltas),
                })

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'#'*80}")
    print(f"#  SIGNAL SWEEP SUMMARY")
    print(f"{'#'*80}")

    print(f"\n  All signals ranked by headroom (1 - AR1 R2):")
    res_df = res_df.sort_values("headroom", ascending=False)
    print(f"  {'Signal':25s} {'rho':>6s} {'AR1':>6s} {'headroom':>9s} {'N':>5s}")
    for _, r in res_df.iterrows():
        marker = " <<<" if r["headroom"] > 0.30 else ""
        print(f"  {r['signal']:25s} {r['rho_median']:>6.3f} {r['ar1_r2']:>6.3f} "
              f"{r['headroom']:>9.3f} {r['n_actors']:>5d}{marker}")

    if smim_results:
        sr = pd.DataFrame(smim_results)
        print(f"\n  SMIM results (best K per signal):")
        for sig in sr["signal"].unique():
            sub = sr[sr["signal"] == sig]
            best = sub.loc[sub["delta"].idxmax()]
            verdict = "SMIM WINS" if best["delta"] > 0 else "AR1 WINS"
            print(f"  {sig:25s}: best K={int(best['K'])}, delta={best['delta']:+.4f}, "
                  f"wins={int(best['wins'])}/{int(best['n_windows'])}  {verdict}")

        sr.to_parquet(METRICS_DIR / "iter5_signal_sweep_smim.parquet", index=False)

    res_df.to_parquet(METRICS_DIR / "iter5_signal_sweep_summary.parquet", index=False)
    print(f"\n  Total: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
