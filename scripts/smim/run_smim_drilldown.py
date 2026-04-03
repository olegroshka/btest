#!/usr/bin/env python
"""
Phase A-B Drill-Down: DD-1 through DD-6 (Quick Diagnostics).

DD-1: K sweep at L1 (K=1,2,3,5,8) x 10 windows
DD-2: Training window sweep at L1 (3,5,8,10,15yr)
DD-3: DMD at L1 across all 10 FULL-ROLL windows
DD-4: Lag-1 directed operator at L1
DD-5: Demeaning strategy comparison (5 variants)
DD-6: Per-actor-type R² decomposition

Output:
  results/metrics/drilldown_DD-{1..6}.parquet
  results/configs/DRILLDOWN.yaml

Usage::
    uv run python scripts/smim/run_smim_drilldown.py
    uv run python scripts/smim/run_smim_drilldown.py --only DD-1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"

K_MAX = 15
TEST_YEARS = list(range(2015, 2025))

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


# --- shared helpers ----------------------------------------------------------


def load_panel():
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_data = {a["actor_id"]: a for a in reg["actors"] if a["actor_id"] in actors}
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], actor_data


def prepare_window(wide, train_start, train_end, test_start, test_end, demean="full"):
    """Prepare train/test arrays with specified demeaning strategy."""
    tr = wide[(wide.index >= train_start) & (wide.index <= train_end)]
    te = wide[(wide.index >= test_start) & (wide.index <= test_end)]
    valid = tr.columns[tr.notna().any()]
    tr, te = tr[valid], te[valid]
    cm = tr.mean()
    otr = tr.fillna(cm).values.astype(np.float64)
    ote = te.fillna(cm).values.astype(np.float64)

    if demean == "full":
        om = otr.mean(axis=0, keepdims=True)
    elif demean == "rolling_8Q":
        om = otr[-8:].mean(axis=0, keepdims=True) if otr.shape[0] >= 8 else otr.mean(axis=0, keepdims=True)
    elif demean == "rolling_20Q":
        om = otr[-20:].mean(axis=0, keepdims=True) if otr.shape[0] >= 20 else otr.mean(axis=0, keepdims=True)
    elif demean == "ewm_8Q":
        weights = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
        om = (otr * weights[:, None]).sum(axis=0, keepdims=True) / weights.sum()
    elif demean == "none":
        om = np.zeros((1, otr.shape[1]))
    else:
        om = otr.mean(axis=0, keepdims=True)

    return otr - om, ote - om, ote, om, list(valid), otr.shape[1]


def l1_evaluate(obs_train, obs_test, obs_test_raw, obs_mean, U):
    """L1 graph-factor OLS with given basis U. Returns R² and errors."""
    K = U.shape[1]
    factors = obs_train @ U
    try:
        A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
    except np.linalg.LinAlgError:
        A = np.eye(K) * 0.9
    f_last = factors[-1]
    pred = np.zeros_like(obs_test)
    for t in range(obs_test.shape[0]):
        f_next = A.T @ f_last
        pred[t] = f_next @ U.T
        f_last = f_next
    pred_r = (pred + obs_mean).T.ravel()
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred_r, actual))
    return r2, actual - pred_r


def ar1_predict(obs_train_raw, obs_test_raw):
    """Per-actor AR(1). Returns predictions and R²."""
    N = obs_train_raw.shape[1]
    pred = np.zeros_like(obs_test_raw)
    for i in range(N):
        y = obs_train_raw[:, i]
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        try:
            beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = [y.mean(), 0]
        v = obs_train_raw[-1, i]
        for t in range(obs_test_raw.shape[0]):
            pred[t, i] = beta[0] + beta[1] * v
            v = pred[t, i]
    actual = obs_test_raw.T.ravel()
    r2 = float(oos_r_squared(pred.T.ravel(), actual))
    return pred, r2, actual - pred.T.ravel()


def build_schur_basis(obs_train, N, K, threshold=0.1):
    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < threshold] = 0.0
    mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    return mf.basis[:, :min(K, mf.K)].real


def fullroll_windows():
    return [
        (f"W{ty}", pd.Timestamp(f"{ty-10}-01-01"), pd.Timestamp(f"{ty-1}-12-31"),
         pd.Timestamp(f"{ty}-01-01"), pd.Timestamp(f"{ty}-12-31"))
        for ty in TEST_YEARS
    ]


# --- DD-1: K sweep -----------------------------------------------------------


def run_dd1(wide):
    print("\n" + "=" * 70)
    print("  DD-1: K SWEEP AT L1 DEPTH")
    print("=" * 70)

    K_values = [1, 2, 3, 5, 8]
    rows = []

    for wid, ts, te, test_s, test_e in fullroll_windows():
        otr, ote, ote_raw, om, _, N = prepare_window(wide, ts, te, test_s, test_e)
        _, ar1_r2, ar1_errors = ar1_predict(otr + om, ote_raw)

        for K in K_values:
            U = build_schur_basis(otr, N, K)
            r2, errors = l1_evaluate(otr, ote, ote_raw, om, U)
            dm_stat, dm_p = diebold_mariano_test(errors, ar1_errors)
            rows.append({"dd": "DD-1", "window": wid, "K": K, "oos_r2": r2,
                          "ar1_r2": ar1_r2, "dm_stat": dm_stat, "dm_p": dm_p})

    df = pd.DataFrame(rows)
    print(f"\n  {'K':>4}", end="")
    for wid, *_ in fullroll_windows():
        print(f"  {wid:>7}", end="")
    print("     MEAN")
    for K in K_values:
        sub = df[df["K"] == K]
        print(f"  K={K:>2}", end="")
        for _, row in sub.iterrows():
            print(f"  {row['oos_r2']:>7.4f}", end="")
        print(f"  {sub['oos_r2'].mean():>8.4f}")
    ar1_mean = df.groupby("window")["ar1_r2"].first().mean()
    print(f"  AR1 ", end="")
    for wid, *_ in fullroll_windows():
        ar1 = df[(df["window"] == wid)]["ar1_r2"].iloc[0]
        print(f"  {ar1:>7.4f}", end="")
    print(f"  {ar1_mean:>8.4f}")

    best_K = df.groupby("K")["oos_r2"].mean().idxmax()
    print(f"\n  Best K = {best_K} (mean R2 = {df[df['K']==best_K]['oos_r2'].mean():.4f})")
    return df


# --- DD-2: Training window sweep ---------------------------------------------


def run_dd2(wide):
    print("\n" + "=" * 70)
    print("  DD-2: TRAINING WINDOW SWEEP AT L1")
    print("=" * 70)

    T_years = [3, 5, 8, 10, 15]
    test_year = 2024
    test_s = pd.Timestamp("2024-01-01")
    test_e = pd.Timestamp("2024-12-31")
    rows = []

    for t_yr in T_years:
        ts = pd.Timestamp(f"{test_year - t_yr}-01-01")
        te = pd.Timestamp("2023-12-31")
        otr, ote, ote_raw, om, _, N = prepare_window(wide, ts, te, test_s, test_e)
        _, ar1_r2, ar1_err = ar1_predict(otr + om, ote_raw)

        for K in [1, 3, 5]:
            U = build_schur_basis(otr, N, K)
            r2, errors = l1_evaluate(otr, ote, ote_raw, om, U)
            rows.append({"dd": "DD-2", "T_years": t_yr, "T_quarters": otr.shape[0],
                          "K": K, "oos_r2": r2, "ar1_r2": ar1_r2, "N": N})

    df = pd.DataFrame(rows)
    print(f"\n  {'T(yr)':>6} {'T(Q)':>5} {'N':>4}  {'K=1':>8} {'K=3':>8} {'K=5':>8} {'AR(1)':>8}")
    for t_yr in T_years:
        sub = df[df["T_years"] == t_yr]
        print(f"  {t_yr:>6} {sub.iloc[0]['T_quarters']:>5} {sub.iloc[0]['N']:>4}", end="")
        for K in [1, 3, 5]:
            r2 = sub[sub["K"] == K]["oos_r2"].iloc[0]
            print(f"  {r2:>8.4f}", end="")
        print(f"  {sub.iloc[0]['ar1_r2']:>8.4f}")
    return df


# --- DD-3: DMD at L1 ---------------------------------------------------------


def run_dd3(wide):
    print("\n" + "=" * 70)
    print("  DD-3: DMD VS SCHUR AT L1 (ALL WINDOWS)")
    print("=" * 70)

    rows = []
    for wid, ts, te, test_s, test_e in fullroll_windows():
        otr, ote, ote_raw, om, _, N = prepare_window(wide, ts, te, test_s, test_e)
        _, ar1_r2, _ = ar1_predict(otr + om, ote_raw)

        for K in [1, 3, 5]:
            # Schur
            U_schur = build_schur_basis(otr, N, K)
            r2_schur, _ = l1_evaluate(otr, ote, ote_raw, om, U_schur)

            # DMD
            try:
                mf_dmd = ExactDMDDecomposer().decompose_snapshots(otr.T, k=min(K_MAX, N))
                U_dmd = mf_dmd.basis[:, :min(K, mf_dmd.K)].real
                r2_dmd, _ = l1_evaluate(otr, ote, ote_raw, om, U_dmd)
            except Exception:
                r2_dmd = float("nan")

            rows.append({"dd": "DD-3", "window": wid, "K": K,
                          "r2_schur": r2_schur, "r2_dmd": r2_dmd, "ar1_r2": ar1_r2})

    df = pd.DataFrame(rows)
    print(f"\n  K=3 comparison:")
    print(f"  {'Window':>8} {'Schur':>8} {'DMD':>8} {'AR(1)':>8} {'DMD-Schur':>10}")
    sub3 = df[df["K"] == 3]
    for _, row in sub3.iterrows():
        delta = row["r2_dmd"] - row["r2_schur"] if np.isfinite(row["r2_dmd"]) else float("nan")
        print(f"  {row['window']:>8} {row['r2_schur']:>8.4f} {row['r2_dmd']:>8.4f} "
              f"{row['ar1_r2']:>8.4f} {delta:>+10.4f}")
    mean_schur = sub3["r2_schur"].mean()
    mean_dmd = sub3["r2_dmd"].mean()
    print(f"  {'MEAN':>8} {mean_schur:>8.4f} {mean_dmd:>8.4f} {sub3['ar1_r2'].mean():>8.4f} "
          f"{mean_dmd - mean_schur:>+10.4f}")
    return df


# --- DD-4: Lag-1 directed operator -------------------------------------------


def run_dd4(wide):
    print("\n" + "=" * 70)
    print("  DD-4: LAG-1 DIRECTED OPERATOR VS CORRELATION")
    print("=" * 70)

    rows = []
    for wid, ts, te, test_s, test_e in fullroll_windows():
        otr, ote, ote_raw, om, _, N = prepare_window(wide, ts, te, test_s, test_e)
        T = otr.shape[0]

        # Correlation operator (symmetric, current)
        U_corr = build_schur_basis(otr, N, 3)
        r2_corr, _ = l1_evaluate(otr, ote, ote_raw, om, U_corr)

        # Lag-1 directed: corr(y_{i,t}, y_{j,t+1})
        xp = otr[:-1] - otr[:-1].mean(axis=0)
        xn = otr[1:] - otr[1:].mean(axis=0)
        num = (xp.T @ xn) / max(len(xp) - 1, 1)
        sp_std = np.std(otr[:-1], axis=0, keepdims=True).T
        sn_std = np.std(otr[1:], axis=0, keepdims=True)
        denom = np.maximum(sp_std @ sn_std, 1e-10)
        lag1 = np.nan_to_num(num / denom, nan=0.0)
        np.fill_diagonal(lag1, 0.0)
        lag1[np.abs(lag1) < 0.1] = 0.0
        mf_lag = SchurDecomposer().decompose(lag1, k=min(K_MAX, N))
        U_lag = mf_lag.basis[:, :min(3, mf_lag.K)].real
        r2_lag, _ = l1_evaluate(otr, ote, ote_raw, om, U_lag)

        # Combined: 0.5*corr + 0.5*lag1
        combined = np.corrcoef(otr.T)
        combined = np.nan_to_num(combined, nan=0.0)
        np.fill_diagonal(combined, 0.0)
        combined[np.abs(combined) < 0.1] = 0.0
        combined = 0.5 * combined + 0.5 * lag1
        mf_comb = SchurDecomposer().decompose(combined, k=min(K_MAX, N))
        U_comb = mf_comb.basis[:, :min(3, mf_comb.K)].real
        r2_comb, _ = l1_evaluate(otr, ote, ote_raw, om, U_comb)

        rows.append({"dd": "DD-4", "window": wid, "r2_corr": r2_corr,
                      "r2_lag1": r2_lag, "r2_combined": r2_comb})

    df = pd.DataFrame(rows)
    print(f"\n  {'Window':>8} {'Corr':>8} {'Lag-1':>8} {'Combined':>8}")
    for _, row in df.iterrows():
        print(f"  {row['window']:>8} {row['r2_corr']:>8.4f} {row['r2_lag1']:>8.4f} {row['r2_combined']:>8.4f}")
    print(f"  {'MEAN':>8} {df['r2_corr'].mean():>8.4f} {df['r2_lag1'].mean():>8.4f} {df['r2_combined'].mean():>8.4f}")
    return df


# --- DD-5: Demeaning strategy ------------------------------------------------


def run_dd5(wide):
    print("\n" + "=" * 70)
    print("  DD-5: DEMEANING STRATEGY COMPARISON")
    print("=" * 70)

    strategies = ["full", "rolling_8Q", "rolling_20Q", "ewm_8Q", "none"]
    rows = []

    for wid, ts, te, test_s, test_e in fullroll_windows():
        for strat in strategies:
            otr, ote, ote_raw, om, _, N = prepare_window(wide, ts, te, test_s, test_e, demean=strat)
            U = build_schur_basis(otr, N, 3)
            r2, _ = l1_evaluate(otr, ote, ote_raw, om, U)
            rows.append({"dd": "DD-5", "window": wid, "strategy": strat, "oos_r2": r2})

    df = pd.DataFrame(rows)
    print(f"\n  {'Strategy':<16}", end="")
    for wid, *_ in fullroll_windows():
        print(f"  {wid:>7}", end="")
    print("     MEAN")
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        print(f"  {strat:<16}", end="")
        for _, row in sub.iterrows():
            print(f"  {row['oos_r2']:>7.4f}", end="")
        print(f"  {sub['oos_r2'].mean():>8.4f}")
    return df


# --- DD-6: Per-actor-type R² ------------------------------------------------


def run_dd6(wide, actor_data):
    print("\n" + "=" * 70)
    print("  DD-6: PER-ACTOR-TYPE R² DECOMPOSITION")
    print("=" * 70)

    rows = []
    for wid, ts, te, test_s, test_e in fullroll_windows():
        otr, ote, ote_raw, om, actor_ids, N = prepare_window(wide, ts, te, test_s, test_e)
        U = build_schur_basis(otr, N, 3)

        # L1 predictions
        factors = otr @ U
        try:
            A = np.linalg.lstsq(factors[:-1], factors[1:], rcond=None)[0]
        except np.linalg.LinAlgError:
            A = np.eye(U.shape[1]) * 0.9
        f_last = factors[-1]
        pred_l1 = np.zeros_like(ote)
        for t in range(ote.shape[0]):
            fn = A.T @ f_last; pred_l1[t] = fn @ U.T; f_last = fn
        pred_l1_r = pred_l1 + om

        # AR(1) predictions
        pred_ar1, _, _ = ar1_predict(otr + om, ote_raw)

        # Per actor-type R²
        for i, aid in enumerate(actor_ids):
            atype = actor_data.get(aid, {}).get("actor_type", "unknown")
            actual_i = ote_raw[:, i]
            l1_i = pred_l1_r[:, i]
            ar1_i = pred_ar1[:, i]
            ss_tot = np.sum((actual_i - actual_i.mean()) ** 2)
            if ss_tot < 1e-10:
                continue
            r2_l1_i = 1 - np.sum((actual_i - l1_i) ** 2) / ss_tot
            r2_ar1_i = 1 - np.sum((actual_i - ar1_i) ** 2) / ss_tot
            rows.append({"dd": "DD-6", "window": wid, "actor_id": aid,
                          "actor_type": atype, "r2_l1": r2_l1_i, "r2_ar1": r2_ar1_i,
                          "delta": r2_l1_i - r2_ar1_i})

    df = pd.DataFrame(rows)
    print(f"\n  {'ActorType':<20} {'N':>4} {'L1 mean':>10} {'AR1 mean':>10} {'Delta':>10} {'L1 wins%':>10}")
    for atype, group in df.groupby("actor_type"):
        l1_wins = (group["delta"] > 0).mean()
        print(f"  {atype:<20} {len(group):>4} {group['r2_l1'].mean():>10.4f} "
              f"{group['r2_ar1'].mean():>10.4f} {group['delta'].mean():>+10.4f} {l1_wins:>9.0%}")

    # Overall
    l1_wins_all = (df["delta"] > 0).mean()
    print(f"  {'ALL':<20} {len(df):>4} {df['r2_l1'].mean():>10.4f} "
          f"{df['r2_ar1'].mean():>10.4f} {df['delta'].mean():>+10.4f} {l1_wins_all:>9.0%}")
    return df


# --- main --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase A-B Drill-Down")
    parser.add_argument("--only", type=str, default=None, help="Run only one DD (e.g., DD-1)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  PHASE A-B DRILL-DOWN  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DD-1 through DD-6: Quick Diagnostics")
    print("=" * 70)

    wide, actor_data = load_panel()

    results = {}
    run_list = [("DD-1", run_dd1), ("DD-2", run_dd2), ("DD-3", run_dd3),
                ("DD-4", run_dd4), ("DD-5", run_dd5)]

    for dd_name, dd_fn in run_list:
        if args.only and args.only != dd_name:
            continue
        try:
            if dd_name == "DD-5":
                results[dd_name] = dd_fn(wide)
            else:
                results[dd_name] = dd_fn(wide)
        except Exception as exc:
            print(f"\n  {dd_name} FAILED: {exc}")
            import traceback; traceback.print_exc()

    if not args.only or args.only == "DD-6":
        try:
            results["DD-6"] = run_dd6(wide, actor_data)
        except Exception as exc:
            print(f"\n  DD-6 FAILED: {exc}")

    # Save all
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for dd_name, df in results.items():
        path = METRICS_DIR / f"drilldown_{dd_name}.parquet"
        df.to_parquet(path, index=False)
    print(f"\n  Saved {len(results)} drill-down result files to results/metrics/")

    # Summary config
    config = {
        "experiment_id": "DRILLDOWN-PHASE1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": list(results.keys()),
    }
    with open(CONFIGS_DIR / "DRILLDOWN.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
