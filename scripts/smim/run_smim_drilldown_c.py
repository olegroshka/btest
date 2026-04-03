#!/usr/bin/env python
"""Phase C drill-down: GOLD++ validation + true zero-shot transfer."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K, K_MAX, T_YR = 8, 15, 5
rng = np.random.default_rng(42)


def load_panel():
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_ids = [a["actor_id"] for a in reg["actors"] if a["actor_id"] in actors]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(pd.date_range("2005-01-01", "2025-12-31", freq="QS"))
    return wide[[c for c in actor_ids if c in wide.columns]]


def prep(wide, ty):
    ts = pd.Timestamp(f"{ty - T_YR}-01-01")
    te = pd.Timestamp(f"{ty - 1}-12-31")
    test_s = pd.Timestamp(f"{ty}-01-01")
    test_e = pd.Timestamp(f"{ty}-12-31")
    tr = wide[(wide.index >= ts) & (wide.index <= te)]
    te_df = wide[(wide.index >= test_s) & (wide.index <= test_e)]
    valid = tr.columns[tr.notna().any()].intersection(te_df.columns[te_df.notna().any()])
    tr, te_df = tr[valid], te_df[valid]
    otr = tr.fillna(tr.mean()).values.astype(np.float64)
    ote = te_df.fillna(tr.mean()).values.astype(np.float64)
    N = otr.shape[1]
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / 8)
    om = (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()
    return otr - om, ote - om, ote, om, N, otr


def get_basis(otr_dm, N):
    k = min(K, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        return mf.basis[:, :min(k, mf.K)].real, mf
    except Exception:
        corr = np.corrcoef(otr_dm.T)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 0)
        corr[np.abs(corr) < 0.1] = 0
        mf = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
        return mf.basis[:, :min(k, mf.K)].real, mf


def train_model(otr_dm, N, noise_sigma=0.0):
    otr_aug = otr_dm.copy()
    if noise_sigma > 0:
        actor_std = np.std(otr_dm, axis=0, keepdims=True)
        otr_aug = otr_aug + rng.standard_normal(otr_aug.shape) * actor_std * noise_sigma
    U, mf = get_basis(otr_aug, N)
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=mf.eigenvalues[:k_act], method=mf.method)
    kf = KalmanFilter()
    st = kf.em_estimate(otr_aug, mf_k, max_iter=50)
    R_sph = np.eye(N) * np.trace(st.params["R"]) / N
    return U, st.params["F"], st.params["Q"].copy(), R_sph


def online_kalman(ote_dm, ote, om, N, U, F, Q, R_sph):
    k_act = U.shape[1]
    alpha_c = np.zeros(k_act)
    P_c = np.eye(k_act)
    Q_run = Q.copy()
    pred = np.zeros_like(ote)
    for t in range(ote.shape[0]):
        a_p = F @ alpha_c
        P_p = F @ P_c @ F.T + Q_run
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
        Q_run = 0.7 * Q_run + 0.3 * np.outer(innov, innov)
        Q_run = (Q_run + Q_run.T) / 2
        Q_run += np.eye(k_act) * 1e-6
        alpha_c = a_f
        P_c = P_f
    return pred


def frozen_kalman(ote_dm, ote, om, N, U_target, F_source, Q_source, R_sph):
    """True zero-shot: frozen F/Q from source, no online adaptation."""
    k_act = min(F_source.shape[0], U_target.shape[1])
    F = F_source[:k_act, :k_act]
    Q = Q_source[:k_act, :k_act]
    U = U_target[:, :k_act]
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
        alpha_c = a_f
        P_c = P_f
    return pred


def main():
    print("=" * 90)
    print(f"  PHASE C DRILL-DOWN  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    wide = load_panel()

    # ══════════════════════════════════════════════════════════════════════════
    print("\n  ITEM 1: GOLD++ (noise augmentation sigma=0.10) vs GOLD+ vs AR(1)")
    print("  " + "-" * 80)

    gold, goldpp, ar1s = [], [], []
    for ty in range(2015, 2025):
        otr_dm, ote_dm, ote, om, N, otr = prep(wide, ty)
        actual = ote.T.ravel()

        U, F, Q, R = train_model(otr_dm, N, noise_sigma=0.0)
        pred_g = online_kalman(ote_dm, ote, om, N, U, F, Q, R)
        gold.append(float(oos_r_squared(pred_g.T.ravel(), actual)))

        U2, F2, Q2, R2 = train_model(otr_dm, N, noise_sigma=0.10)
        pred_gpp = online_kalman(ote_dm, ote, om, N, U2, F2, Q2, R2)
        goldpp.append(float(oos_r_squared(pred_gpp.T.ravel(), actual)))

        pred_ar = np.zeros_like(ote)
        for i in range(N):
            y = otr[:, i]
            X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
            try:
                b = np.linalg.lstsq(X, y[1:], rcond=None)[0]
            except Exception:
                b = [y.mean(), 0]
            v = otr[-1, i]
            for t in range(4):
                pred_ar[t, i] = b[0] + b[1] * v
                v = pred_ar[t, i]
        ar1s.append(float(oos_r_squared(pred_ar.T.ravel(), actual)))

    print(f"\n  {'Window':>8} {'GOLD+':>10} {'GOLD++':>10} {'AR(1)':>10} {'++ delta':>10}")
    for i, ty in enumerate(range(2015, 2025)):
        d = goldpp[i] - gold[i]
        print(f"  W{ty}   {gold[i]:>10.4f} {goldpp[i]:>10.4f} {ar1s[i]:>10.4f} {d:>+10.4f}")
    print(f"  {'MEAN':>8} {np.mean(gold):>10.4f} {np.mean(goldpp):>10.4f} "
          f"{np.mean(ar1s):>10.4f} {np.mean(goldpp) - np.mean(gold):>+10.4f}")

    t_stat, p_val = ttest_rel(goldpp, ar1s)
    wins = sum(1 for i in range(10) if goldpp[i] > ar1s[i])
    print(f"\n  GOLD++ vs AR(1): delta={np.mean(goldpp) - np.mean(ar1s):+.4f}  "
          f"wins={wins}/10  paired-t p={p_val:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n  ITEM 2: TRUE ZERO-SHOT (freeze F/Q from adjacent window)")
    print("  " + "-" * 80)

    zs_rows = []
    for ty in range(2016, 2025):
        # Source: previous year window
        otr_src, _, _, om_src, N_src, _ = prep(wide, ty - 1)
        U_src, F_src, Q_src, R_src = train_model(otr_src, N_src, noise_sigma=0.10)

        # Target: current year
        otr_tgt, ote_tgt, ote_raw_tgt, om_tgt, N_tgt, _ = prep(wide, ty)
        actual = ote_raw_tgt.T.ravel()

        # Full retrain (GOLD++ baseline)
        U_t, F_t, Q_t, R_t = train_model(otr_tgt, N_tgt, noise_sigma=0.10)
        pred_full = online_kalman(ote_tgt, ote_raw_tgt, om_tgt, N_tgt, U_t, F_t, Q_t, R_t)
        r2_full = float(oos_r_squared(pred_full.T.ravel(), actual))

        # True zero-shot: target U, frozen source F/Q, no online adapt
        U_tgt_basis, _ = get_basis(otr_tgt, N_tgt)
        R_tgt = np.eye(N_tgt) * np.trace(R_src) / N_src
        pred_frozen = frozen_kalman(ote_tgt, ote_raw_tgt, om_tgt, N_tgt,
                                     U_tgt_basis, F_src, Q_src, R_tgt)
        r2_frozen = float(oos_r_squared(pred_frozen.T.ravel(), actual))

        # Zero-shot + online Q adapt
        pred_zs_online = online_kalman(ote_tgt, ote_raw_tgt, om_tgt, N_tgt,
                                        U_tgt_basis, F_src, Q_src, R_tgt)
        r2_zs_online = float(oos_r_squared(pred_zs_online.T.ravel(), actual))

        ret_f = r2_frozen / r2_full * 100 if r2_full > 0 else 0
        ret_o = r2_zs_online / r2_full * 100 if r2_full > 0 else 0
        print(f"  W{ty - 1}->W{ty}: full={r2_full:.4f}  frozen={r2_frozen:.4f}({ret_f:.0f}%)  "
              f"frozen+online={r2_zs_online:.4f}({ret_o:.0f}%)")
        zs_rows.append({
            "source": f"W{ty - 1}", "target": f"W{ty}",
            "r2_full": r2_full, "r2_frozen": r2_frozen, "r2_frozen_online": r2_zs_online,
            "retention_frozen": ret_f, "retention_online": ret_o,
        })

    df_zs = pd.DataFrame(zs_rows)
    print(f"\n  Mean retention (frozen F/Q, no adapt):    {df_zs['retention_frozen'].mean():.0f}%")
    print(f"  Mean retention (frozen F/Q + online Q):   {df_zs['retention_online'].mean():.0f}%")
    print(f"  Mean R2 full retrain:   {df_zs['r2_full'].mean():.4f}")
    print(f"  Mean R2 frozen:         {df_zs['r2_frozen'].mean():.4f}")
    print(f"  Mean R2 frozen+online:  {df_zs['r2_frozen_online'].mean():.4f}")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "window": [f"W{ty}" for ty in range(2015, 2025)],
        "gold_plus": gold, "gold_plus_plus": goldpp, "ar1": ar1s,
    }).to_parquet(METRICS_DIR / "drilldown_GOLD_PP.parquet", index=False)
    df_zs.to_parquet(METRICS_DIR / "drilldown_TRUE_ZERO_SHOT.parquet", index=False)
    print(f"\n  Saved results to results/metrics/")
    print("=" * 90)


if __name__ == "__main__":
    main()
