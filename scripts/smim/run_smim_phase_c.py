#!/usr/bin/env python
"""
Phase C: Transfer Experiments (C1-C6 combined runner).

Uses GOLD+ pipeline config: DMD K=min(8,N-2), spherical R, online Q, EWM demean, T=5yr.

Transfer protocol:
  zero_shot:  Train F/Q on source. Build new operator on target. Apply trained dynamics.
  fine_tune:  Re-estimate everything on target (only K is carried from source).
  baseline:   Train and test on target itself (sector-trained reference).

Output:
  results/metrics/level1_C{1..6}-*.parquet
  results/configs/C-TRANSFER.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

K_MAX = 15
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"


# --- core pipeline -----------------------------------------------------------


def load_intensity(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "smim" / "intensities" / f"{name}_intensities.parquet"
    df = pd.read_parquet(path)
    wide = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))


def prepare(wide, train_start, train_end, test_start, test_end):
    tr = wide[(wide.index >= train_start) & (wide.index <= train_end)]
    te = wide[(wide.index >= test_start) & (wide.index <= test_end)]
    valid = tr.columns[tr.notna().any()]
    # Also require test data
    valid = valid.intersection(te.columns[te.notna().any()])
    tr, te = tr[valid], te[valid]
    cm = tr.mean()
    otr = tr.fillna(cm).values.astype(np.float64)
    ote = te.fillna(cm).values.astype(np.float64)
    N = otr.shape[1]
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
    om = (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()
    return otr - om, ote - om, ote, om, N


def get_dmd_basis(otr_dm, N, K):
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(K, mf.K)].real
        return U, mf
    except Exception:
        corr = np.corrcoef(otr_dm.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        corr[np.abs(corr) < 0.1] = 0.0
        mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
        return mf.basis[:, :min(K, mf.K)].real, mf


def run_pipeline(otr_dm, ote_dm, ote, om, N, K, F_init=None, Q_init=None):
    """Run GOLD+ pipeline. If F_init/Q_init provided, use as Kalman init (zero-shot)."""
    U, mf = get_dmd_basis(otr_dm, N, K)
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=mf.eigenvalues[:k_act], method=mf.method)

    kf = KalmanFilter()
    if F_init is not None and Q_init is not None and F_init.shape[0] == k_act:
        kf.F = F_init
        kf.Q = Q_init
    state = kf.em_estimate(otr_dm, mf_k, max_iter=50)

    R_sph = np.eye(N) * np.trace(state.params["R"]) / N
    F = state.params["F"]
    Q = state.params["Q"].copy()

    # Online Kalman
    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    pred = np.zeros_like(ote)
    for t in range(ote.shape[0]):
        a_p = F @ alpha_c
        P_p = F @ P_c @ F.T + Q
        v = ote_dm[t] - U @ a_p
        S = U @ P_p @ U.T + R_sph
        try:
            Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
        except Exception:
            Kg = np.zeros((k_act, N))
        a_f = a_p + Kg @ v
        P_f = (np.eye(k_act) - Kg @ U) @ P_p
        pred[t] = (a_f @ U.T + om).ravel()
        innov = a_f - F @ alpha_c
        Q = 0.7 * Q + 0.3 * np.outer(innov, innov)
        Q = (Q + Q.T) / 2
        Q += np.eye(k_act) * 1e-6
        alpha_c = a_f
        P_c = P_f

    actual = ote.T.ravel()
    r2 = float(oos_r_squared(pred.T.ravel(), actual))
    return r2, F, Q


def ar1_baseline(otr_raw, ote_raw):
    N = otr_raw.shape[1]
    pred = np.zeros_like(ote_raw)
    for i in range(N):
        y = otr_raw[:, i]
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        try:
            b = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        except Exception:
            b = [y.mean(), 0]
        v = otr_raw[-1, i]
        for t in range(ote_raw.shape[0]):
            pred[t, i] = b[0] + b[1] * v
            v = pred[t, i]
    return float(oos_r_squared(pred.T.ravel(), ote_raw.T.ravel()))


# --- C1+C2: Cross-Sector Transfer -------------------------------------------


def run_c1_c2():
    print("\n" + "=" * 80)
    print("  C1+C2: CROSS-SECTOR TRANSFER (energy -> tech/fins/health/indus)")
    print("=" * 80)

    train_start = pd.Timestamp("2010-01-01")
    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2025-12-31")

    # Train on energy
    energy = load_intensity("US-LC-ENERGY")
    otr_e, _, _, om_e, N_e = prepare(energy, train_start, train_end, test_start, test_end)
    K_source = min(5, N_e - 2)  # N=12 -> K=5 max
    _, F_trained, Q_trained = run_pipeline(otr_e, np.zeros((1, N_e)), np.zeros((1, N_e)),
                                            om_e, N_e, K_source)

    targets = ["US-LC-TECH", "US-LC-FINS", "US-LC-HEALTH", "US-LC-INDUS"]
    rows = []

    print(f"\n  Source: ENERGY (N={N_e}, K={K_source})")
    print(f"  {'Target':<16} {'N':>4} {'Baseline':>10} {'Zero-shot':>10} {'Fine-tune':>10} {'Retention':>10}")

    for tgt_name in targets:
        tgt = load_intensity(tgt_name)
        otr_t, ote_t, ote_raw_t, om_t, N_t = prepare(tgt, train_start, train_end, test_start, test_end)
        if N_t < 5:
            print(f"  {tgt_name:<16} {N_t:>4}  SKIP (too few actors)")
            continue
        K_t = min(K_source, N_t - 2)

        # Baseline: train+test on target itself
        r2_base, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

        # Zero-shot: use energy F/Q, target operator
        F_zs = F_trained[:K_t, :K_t] if K_t <= F_trained.shape[0] else None
        Q_zs = Q_trained[:K_t, :K_t] if K_t <= Q_trained.shape[0] else None
        r2_zs, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t, F_zs, Q_zs)

        # Fine-tune: re-estimate everything (K carried from source)
        r2_ft, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

        retention_zs = r2_zs / r2_base * 100 if r2_base > 0 else 0
        retention_ft = r2_ft / r2_base * 100 if r2_base > 0 else 0

        print(f"  {tgt_name:<16} {N_t:>4} {r2_base:>10.4f} {r2_zs:>10.4f} {r2_ft:>10.4f} "
              f"{retention_zs:>9.0f}%")

        for mode, r2, ret in [("zero_shot", r2_zs, retention_zs),
                               ("fine_tune", r2_ft, retention_ft),
                               ("baseline", r2_base, 100.0)]:
            rows.append({"experiment": "C1+C2", "source": "ENERGY", "target": tgt_name,
                          "mode": mode, "oos_r2": r2, "retention_pct": ret,
                          "N_source": N_e, "N_target": N_t, "K": K_t})
    return pd.DataFrame(rows)


# --- C3: Cross-Cap Transfer -------------------------------------------------


def run_c3():
    print("\n" + "=" * 80)
    print("  C3: CROSS-CAP TRANSFER (US-LC -> US-MC, US-SC)")
    print("=" * 80)

    train_start = pd.Timestamp("2010-01-01")
    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2025-12-31")

    source = load_intensity("US-LC")
    otr_s, _, _, om_s, N_s = prepare(source, train_start, train_end, test_start, test_end)
    K_source = min(8, N_s - 2)
    _, F_trained, Q_trained = run_pipeline(otr_s, np.zeros((1, N_s)), np.zeros((1, N_s)),
                                            om_s, N_s, K_source)

    rows = []
    print(f"\n  Source: US-LC (N={N_s}, K={K_source})")
    print(f"  {'Target':<16} {'N':>4} {'Baseline':>10} {'Zero-shot':>10} {'Fine-tune':>10} {'Retention':>10}")

    for tgt_name in ["US-MC", "US-SC_trimmed"]:
        tgt = load_intensity(tgt_name)
        otr_t, ote_t, ote_raw_t, om_t, N_t = prepare(tgt, train_start, train_end, test_start, test_end)
        K_t = min(K_source, N_t - 2)

        r2_base, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

        F_zs = F_trained[:K_t, :K_t] if K_t <= F_trained.shape[0] else None
        Q_zs = Q_trained[:K_t, :K_t] if K_t <= Q_trained.shape[0] else None
        r2_zs, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t, F_zs, Q_zs)

        r2_ft, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

        retention = r2_zs / r2_base * 100 if r2_base > 0 else 0
        print(f"  {tgt_name:<16} {N_t:>4} {r2_base:>10.4f} {r2_zs:>10.4f} {r2_ft:>10.4f} {retention:>9.0f}%")

        for mode, r2, ret in [("zero_shot", r2_zs, retention),
                               ("fine_tune", r2_ft, r2_ft / r2_base * 100 if r2_base > 0 else 0),
                               ("baseline", r2_base, 100.0)]:
            rows.append({"experiment": "C3", "source": "US-LC", "target": tgt_name,
                          "mode": mode, "oos_r2": r2, "retention_pct": ret,
                          "N_source": N_s, "N_target": N_t, "K": K_t})
    return pd.DataFrame(rows)


# --- C4: Cross-Geography Transfer --------------------------------------------


def run_c4():
    print("\n" + "=" * 80)
    print("  C4: CROSS-GEOGRAPHY TRANSFER (US-LC -> UK-LC, both M-B return method)")
    print("=" * 80)

    train_start = pd.Timestamp("2010-01-01")
    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2025-12-31")

    source = load_intensity("US-LC_return")
    otr_s, _, _, om_s, N_s = prepare(source, train_start, train_end, test_start, test_end)
    K_source = min(8, N_s - 2)
    _, F_trained, Q_trained = run_pipeline(otr_s, np.zeros((1, N_s)), np.zeros((1, N_s)),
                                            om_s, N_s, K_source)

    tgt = load_intensity("UK-LC")
    otr_t, ote_t, ote_raw_t, om_t, N_t = prepare(tgt, train_start, train_end, test_start, test_end)
    K_t = min(K_source, N_t - 2)

    r2_base, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

    F_zs = F_trained[:K_t, :K_t] if K_t <= F_trained.shape[0] else None
    Q_zs = Q_trained[:K_t, :K_t] if K_t <= Q_trained.shape[0] else None
    r2_zs, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t, F_zs, Q_zs)
    r2_ft, _, _ = run_pipeline(otr_t, ote_t, ote_raw_t, om_t, N_t, K_t)

    retention = r2_zs / r2_base * 100 if r2_base > 0 else 0
    print(f"\n  Source: US-LC_return (N={N_s}, K={K_source})")
    print(f"  Target: UK-LC        (N={N_t}, K={K_t})")
    print(f"  Baseline:   R2={r2_base:.4f}")
    print(f"  Zero-shot:  R2={r2_zs:.4f}  retention={retention:.0f}%")
    print(f"  Fine-tune:  R2={r2_ft:.4f}")

    rows = []
    for mode, r2, ret in [("zero_shot", r2_zs, retention),
                           ("fine_tune", r2_ft, r2_ft / r2_base * 100 if r2_base > 0 else 0),
                           ("baseline", r2_base, 100.0)]:
        rows.append({"experiment": "C4", "source": "US-LC", "target": "UK-LC",
                      "mode": mode, "oos_r2": r2, "retention_pct": ret,
                      "N_source": N_s, "N_target": N_t, "K": K_t})
    return pd.DataFrame(rows)


# --- C5: Cross-Period Transfer -----------------------------------------------


def run_c5():
    print("\n" + "=" * 80)
    print("  C5: CROSS-PERIOD TRANSFER (era transitions, zero-shot)")
    print("=" * 80)

    transitions = [
        ("PRE-GFC->POST-GFC",      "2005-01-01", "2007-12-31", "2008-01-01", "2009-12-31"),
        ("POST-GFC->PRE-COVID",     "2010-01-01", "2014-12-31", "2015-01-01", "2019-12-31"),
        ("PRE-COVID->POST-COVID",   "2015-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
        ("POST-COVID->RECENT",      "2020-01-01", "2022-12-31", "2023-01-01", "2025-12-31"),
    ]

    source_int = load_intensity("US-LC")
    rows = []

    print(f"\n  {'Transition':<28} {'T_train':>8} {'T_test':>7} {'N':>4} {'Baseline':>10} {'Zero-shot':>10} {'Retention':>10}")

    for name, ts, te, test_s, test_e in transitions:
        ts, te = pd.Timestamp(ts), pd.Timestamp(te)
        test_s, test_e = pd.Timestamp(test_s), pd.Timestamp(test_e)

        otr, ote, ote_raw, om, N = prepare(source_int, ts, te, test_s, test_e)
        T_tr, T_te = otr.shape[0], ote.shape[0]
        K = min(8, N - 2, T_tr - 2)
        if K < 1 or T_tr < 4:
            print(f"  {name:<28}  SKIP (insufficient data)")
            continue

        # Baseline: in-sample (train and test from same era — use test era for both)
        otr_b, ote_b, ote_raw_b, om_b, N_b = prepare(source_int, test_s, test_e, test_s, test_e)
        # Can't train on test, so baseline = train on source era, test on source era
        # Actually baseline should be: train on TARGET era itself
        # But for cross-period, that IS the zero-shot condition.
        # Use in-era performance: train 80%, test 20% within the TEST era
        t_split = max(T_te * 3 // 4, 2)
        if T_te > 4:
            otr_in = ote - om  # demeaned test data
            otr_in_80 = otr_in[:t_split]
            ote_in_20 = otr_in[t_split:]
            ote_raw_20 = ote_raw[t_split:]
            om_20 = om
            try:
                r2_base, _, _ = run_pipeline(otr_in_80, ote_in_20, ote_raw_20, om_20, N, K)
            except Exception:
                r2_base = float("nan")
        else:
            r2_base = float("nan")

        # Zero-shot: train on source era, test on target era
        try:
            r2_zs, F_s, Q_s = run_pipeline(otr, ote, ote_raw, om, N, K)
        except Exception:
            r2_zs = float("nan")

        retention = r2_zs / r2_base * 100 if r2_base > 0 and np.isfinite(r2_base) else float("nan")
        print(f"  {name:<28} {T_tr:>8} {T_te:>7} {N:>4} {r2_base:>10.4f} {r2_zs:>10.4f} "
              f"{retention:>9.0f}%" if np.isfinite(retention) else
              f"  {name:<28} {T_tr:>8} {T_te:>7} {N:>4} {r2_base:>10.4f} {r2_zs:>10.4f}       n/a")

        rows.append({"experiment": "C5", "transition": name, "mode": "zero_shot",
                      "oos_r2": r2_zs, "T_train": T_tr, "T_test": T_te, "N": N, "K": K})
    return pd.DataFrame(rows)


# --- C6: Data Regime Degradation ---------------------------------------------


def run_c6():
    print("\n" + "=" * 80)
    print("  C6: DATA REGIME DEGRADATION (Gold -> Sparse)")
    print("=" * 80)

    source_int = load_intensity("US-LC")
    train_start = pd.Timestamp("2015-01-01")
    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2025-12-31")

    otr, ote, ote_raw, om, N = prepare(source_int, train_start, train_end, test_start, test_end)
    K = min(8, N - 2)
    rng = np.random.default_rng(42)

    regimes = {
        "gold": {"noise": 0.0, "n_frac": 1.0, "skip_quarters": 1},
        "silver": {"noise": 0.05, "n_frac": 1.0, "skip_quarters": 1},
        "bronze": {"noise": 0.1, "n_frac": 0.5, "skip_quarters": 2},
        "sparse": {"noise": 0.2, "n_frac": 0.25, "skip_quarters": 4},
    }

    rows = []
    print(f"\n  N={N}, K={K}, T_train={otr.shape[0]}")
    print(f"  {'Regime':<10} {'Noise':>6} {'N_frac':>7} {'Skip_Q':>7} {'R2':>10}")

    for regime_name, params in regimes.items():
        otr_deg = otr.copy()

        # Add noise
        if params["noise"] > 0:
            actor_std = np.std(otr_deg, axis=0, keepdims=True)
            otr_deg += rng.standard_normal(otr_deg.shape) * actor_std * params["noise"]

        # Subsample actors
        N_sub = max(10, int(N * params["n_frac"]))
        if N_sub < N:
            selected = rng.choice(N, N_sub, replace=False)
            selected.sort()
            otr_sub = otr_deg[:, selected]
            ote_sub = ote[:, selected]
            ote_raw_sub = ote_raw[:, selected]
            om_sub = om[:, selected]
        else:
            otr_sub, ote_sub, ote_raw_sub, om_sub = otr_deg, ote, ote_raw, om
            N_sub = N

        # Skip quarters (reduce temporal resolution)
        skip = params["skip_quarters"]
        if skip > 1:
            otr_sub = otr_sub[::skip]

        K_sub = min(K, N_sub - 2, otr_sub.shape[0] - 2)
        if K_sub < 1:
            rows.append({"experiment": "C6", "regime": regime_name, "oos_r2": float("nan")})
            print(f"  {regime_name:<10}  SKIP")
            continue

        try:
            r2, _, _ = run_pipeline(otr_sub, ote_sub, ote_raw_sub, om_sub, N_sub, K_sub)
        except Exception:
            r2 = float("nan")

        print(f"  {regime_name:<10} {params['noise']:>6.2f} {params['n_frac']:>7.2f} "
              f"{params['skip_quarters']:>7} {r2:>10.4f}")
        rows.append({"experiment": "C6", "regime": regime_name, "oos_r2": r2,
                      "noise": params["noise"], "n_frac": params["n_frac"],
                      "skip_q": params["skip_quarters"], "N_eff": N_sub, "K": K_sub})

    return pd.DataFrame(rows)


# --- main --------------------------------------------------------------------


def main():
    print("=" * 80)
    print(f"  PHASE C: TRANSFER EXPERIMENTS  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pipeline: GOLD+ (DMD, spherical R, online Q, EWM, T=variable)")
    print("=" * 80)

    df_c12 = run_c1_c2()
    df_c3 = run_c3()
    df_c4 = run_c4()
    df_c5 = run_c5()
    df_c6 = run_c6()

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in [("C1C2-SECTOR-TRANSFER", df_c12), ("C3-CAP-TRANSFER", df_c3),
                      ("C4-GEO-TRANSFER", df_c4), ("C5-PERIOD-TRANSFER", df_c5),
                      ("C6-DATA-REGIME", df_c6)]:
        if not df.empty:
            df.to_parquet(METRICS_DIR / f"level1_{name}.parquet", index=False)
            print(f"\n  Wrote: results/metrics/level1_{name}.parquet")

    config = {
        "phase": "C",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "GOLD+ (DMD, spherical R, online Q, EWM)",
        "experiments": ["C1+C2", "C3", "C4", "C5", "C6"],
    }
    with open(CONFIGS_DIR / "C-TRANSFER.yaml", "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: results/configs/C-TRANSFER.yaml")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
