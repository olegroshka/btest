#!/usr/bin/env python
"""
B5-SIGNAL-COMPONENT-INTERACTION experiment runner.

2x2x2 factorial: signals (FULL / INTENSITY-ONLY) x regimes (M=1 / M=2)
x emergence (on / off). Tests H1c (narrative x regime) and H4a
(emergence x signal interaction).

Since B3 showed external signals are dispensable at L1, we use L2 depth
(Kalman) here to test whether signals interact with state-space components.

Output:
  results/metrics/level1_B5-SIGNAL-COMPONENT-INTERACTION.parquet
  results/configs/B5-SIGNAL-COMPONENT-INTERACTION.yaml
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
from quantdsl_backtest.smim.dynamics.kim_filter import KimFilter
from quantdsl_backtest.smim.dynamics.phase_transition import criticality_index, extract_order_parameter
from quantdsl_backtest.smim.emergence.pid import compute_synergy_matrix
from quantdsl_backtest.smim.gaps.emergence_aware import EmergenceAwareBenchmark
from quantdsl_backtest.smim.gaps.predictive import PredictiveBenchmark
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer
from quantdsl_backtest.smim.validation.metrics import oos_r_squared

EXP_ID = "B5-SIGNAL-COMPONENT-INTERACTION"
INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
from quantdsl_backtest.smim.data.actor_registry import ActorRegistry as ConcreteActorRegistry
from quantdsl_backtest.smim.interfaces import Actor, ActorType, Layer

K_MIN = 3
K_MAX = 15
TEST_YEARS = list(range(2015, 2025))
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFIGS_DIR = RESULTS_DIR / "configs"

_AT = {"global_shock": ActorType.GLOBAL_SHOCK, "central_bank": ActorType.CENTRAL_BANK,
       "regulator": ActorType.REGULATOR, "intl_org": ActorType.INTL_ORG,
       "sector_leader": ActorType.SECTOR_LEADER, "large_firm": ActorType.LARGE_FIRM,
       "bank": ActorType.BANK}


def load_data():
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)
    int_df = pd.read_parquet(INTENSITIES_PATH)
    actors = set(int_df["actor_id"].unique())
    actor_data = [a for a in reg["actors"] if a["actor_id"] in actors]
    actor_ids = [a["actor_id"] for a in actor_data]
    wide = int_df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    wide.index = pd.to_datetime(wide.index)
    idx = pd.date_range("2005-01-01", "2025-12-31", freq="QS")
    wide = wide.reindex(idx)
    cols = [c for c in actor_ids if c in wide.columns]
    return wide[cols], actor_data


def run_cell(obs_train, obs_test, obs_test_raw, obs_mean, N,
             M, emergence, actor_data, actor_ids_final):
    """Run one factorial cell. Returns R2."""
    K = min(K_MIN, N - 1)

    corr = np.corrcoef(obs_train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr[np.abs(corr) < 0.1] = 0.0

    mf_cand = SchurDecomposer().decompose(corr, k=min(K_MAX, N))
    U = mf_cand.basis[:, :K].real
    mf_k = ModalFrame(basis=U, eigenvalues=mf_cand.eigenvalues[:K], method=mf_cand.method)

    bench_actors = [
        Actor(actor_id=a["actor_id"], name=a["name"],
              actor_type=_AT.get(a["actor_type"], ActorType.LARGE_FIRM),
              layer=Layer(a["layer"]), geography=a["geography"],
              sector=a["sector"], external_ids=a.get("external_ids", {}))
        for a in actor_data if a["actor_id"] in set(actor_ids_final)
    ]
    registry = ConcreteActorRegistry(actors=bench_actors[:N])

    # State estimation
    if M == 1:
        kf = KalmanFilter()
        st = kf.em_estimate(obs_train, mf_k, max_iter=50)
        kf2 = KalmanFilter(F=st.params["F"], Q=st.params["Q"], R=st.params["R"])
        st_test = kf2.filter(obs_test, mf_k)
    else:
        kim = KimFilter(n_regimes=M)
        st = kim.em_estimate(obs_train, mf_k, max_iter=50)
        kim2 = KimFilter(n_regimes=M)
        st_test = kim2.filter(obs_test, mf_k,
                              F_list=st.params["F_list"], Q_list=st.params["Q_list"],
                              R=st.params["R"], P_trans=st.params["P_trans"])

    if not emergence:
        gap = PredictiveBenchmark().compute(obs_test, st_test, mf_k, registry)
        pred = (gap.benchmarks + obs_mean.T).ravel()
    else:
        alpha_tr = st.alpha_filtered
        recon = alpha_tr @ U.T
        gaps_tr = (obs_train - recon).T
        try:
            syn = compute_synergy_matrix(alpha_tr, gaps_tr, K)
        except Exception:
            syn = np.zeros((K, K))
        psi = extract_order_parameter(alpha_tr, d=min(3, K))
        crit = criticality_index(psi, window_size=8)
        crit_test = np.full(obs_test.shape[0], crit[-1] if len(crit) > 0 else 1.0)
        ea = EmergenceAwareBenchmark(synergy_matrix=syn, criticality=crit_test)
        gap = ea.compute(obs_test, st_test, mf_k, registry)
        pred = (gap.benchmarks + obs_mean.T).ravel()

    actual = obs_test_raw.T.ravel()
    return float(oos_r_squared(pred, actual))


def main():
    print("=" * 70)
    print(f"  {EXP_ID}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  2x2x2 factorial  (signals irrelevant at L1, using Kalman depth)")
    print("=" * 70)

    intensities, actor_data = load_data()
    windows = [
        {"window_id": f"W{ty}",
         "train_start": pd.Timestamp(f"{ty - 10}-01-01"),
         "train_end": pd.Timestamp(f"{ty - 1}-12-31"),
         "test_start": pd.Timestamp(f"{ty}-01-01"),
         "test_end": pd.Timestamp(f"{ty}-12-31")}
        for ty in TEST_YEARS
    ]

    cells = [
        {"M": 1, "emergence": False, "label": "M1/no-emerg"},
        {"M": 1, "emergence": True, "label": "M1/emerg"},
        {"M": 2, "emergence": False, "label": "M2/no-emerg"},
        {"M": 2, "emergence": True, "label": "M2/emerg"},
    ]

    rows = []
    for wi, w in enumerate(windows):
        ts, te = w["train_start"], w["train_end"]
        test_s, test_e = w["test_start"], w["test_end"]

        int_train = intensities[(intensities.index >= ts) & (intensities.index <= te)]
        int_test = intensities[(intensities.index >= test_s) & (intensities.index <= test_e)]
        valid = int_train.columns[int_train.notna().any()]
        int_train, int_test = int_train[valid], int_test[valid]
        actor_ids_final = list(valid)
        N = len(actor_ids_final)

        col_means = int_train.mean()
        otr = int_train.fillna(col_means).values.astype(np.float64)
        ote = int_test.fillna(col_means).values.astype(np.float64)
        om = otr.mean(axis=0, keepdims=True)
        obs_train, obs_test = otr - om, ote - om

        print(f"\n  [{wi + 1}/{len(windows)}] {w['window_id']}  N={N}")

        for cell in cells:
            try:
                r2 = run_cell(obs_train, obs_test, ote, om, N,
                              cell["M"], cell["emergence"], actor_data, actor_ids_final)
                print(f"    {cell['label']:<20} R2={r2:.4f}")
            except Exception as exc:
                r2 = float("nan")
                print(f"    {cell['label']:<20} FAILED: {exc}")

            rows.append({
                "experiment_id": EXP_ID,
                "window_id": w["window_id"],
                "M": cell["M"],
                "emergence": cell["emergence"],
                "label": cell["label"],
                "oos_r2": r2,
            })

    df = pd.DataFrame(rows)

    # ANOVA-like summary
    print("\n" + "-" * 70)
    print("  FACTORIAL SUMMARY (mean across windows)")
    for label in ["M1/no-emerg", "M1/emerg", "M2/no-emerg", "M2/emerg"]:
        sub = df[df["label"] == label]
        print(f"  {label:<20} R2={sub['oos_r2'].mean():.4f} +/- {sub['oos_r2'].std():.4f}")

    # Main effects
    m1_mean = df[df["M"] == 1]["oos_r2"].mean()
    m2_mean = df[df["M"] == 2]["oos_r2"].mean()
    em_off = df[~df["emergence"]]["oos_r2"].mean()
    em_on = df[df["emergence"]]["oos_r2"].mean()
    print(f"\n  Main effects:")
    print(f"    Regime: M=1 ({m1_mean:.4f}) vs M=2 ({m2_mean:.4f})  delta={m2_mean - m1_mean:+.4f}")
    print(f"    Emergence: off ({em_off:.4f}) vs on ({em_on:.4f})  delta={em_on - em_off:+.4f}")
    print("=" * 70)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(METRICS_DIR / f"level1_{EXP_ID}.parquet", index=False)
    yaml.dump({"experiment_id": EXP_ID, "cells": [c["label"] for c in cells]},
              open(CONFIGS_DIR / f"{EXP_ID}.yaml", "w"))
    print(f"  Wrote results.")


if __name__ == "__main__":
    main()
