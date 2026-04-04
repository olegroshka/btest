#!/usr/bin/env python
"""
SMIM Drilldown #2 — Venue 2: Recursive / Streaming DMD.

V2-1: Rolling DMD vs frozen DMD (does updating the basis help prediction?)
V2-2: K_effective tracking over full timeline (does mode count change?)
V2-3: Basis rotation speed (does fast rotation predict R^2 drops?)
V2-4: Mode birth/death events (do structural breaks align with market shocks?)

All run on the EXISTING experiment_a1 quarterly intensity panel.

Usage::
    uv run python scripts/smim/run_smim_dd2_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantdsl_backtest.smim.dynamics.kalman import KalmanFilter
from quantdsl_backtest.smim.interfaces import ModalFrame
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer
from quantdsl_backtest.smim.validation.metrics import diebold_mariano_test, oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "smim" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "smim" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

K_GOLD = 8
K_MAX = 15
T_YR = 5
TEST_YEARS = list(range(2015, 2025))


# =========================================================================
# Shared infrastructure
# =========================================================================


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


def ewm_demean(otr, halflife=8):
    w = np.exp(-np.arange(otr.shape[0])[::-1] * np.log(2) / halflife)
    return (otr * w[:, None]).sum(axis=0, keepdims=True) / w.sum()


def dmd_basis(otr_dm, N, k=K_GOLD):
    """Extract DMD basis. Returns (U, eigenvalues) or (None, None) on failure."""
    k_use = min(k, N - 2)
    try:
        mf = ExactDMDDecomposer().decompose_snapshots(otr_dm.T, k=min(K_MAX, N))
        U = mf.basis[:, :min(k_use, mf.K)].real
        eigs = mf.eigenvalues[:min(k_use, mf.K)]
        return U, eigs
    except Exception:
        return None, None


def subspace_angle(U1, U2, k=None):
    """Mean principal angle between column spaces (radians)."""
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svals, -1, 1))
    return float(np.mean(angles))


def train_sph_r(otr_dm, N, U):
    """EM estimation just to get spherical R scale."""
    k_act = U.shape[1]
    mf_k = ModalFrame(basis=U, eigenvalues=np.ones(k_act, dtype=complex), method="dmd")
    kf = KalmanFilter()
    st = kf.em_estimate(otr_dm, mf_k, max_iter=50)
    return np.eye(N) * np.trace(st.params["R"]) / N


def platinum_predict_quarter(obs_dm_quarter, om, N, U, F, Q, R_sph,
                              alpha_init, P_init, Q_run):
    """PLATINUM prediction for a single quarter.

    Returns (pred, pred_modal, a_f, P_f, Q_new) where:
      pred       = U @ a_p + om  (PREDICTIVE: genuine OOS forecast)
      pred_modal = U @ a_f + om  (MODAL: uses current obs — diagnostic only)
    """
    k_act = U.shape[1]
    a_p = F @ alpha_init
    P_p = F @ P_init @ F.T + Q_run
    v = obs_dm_quarter - U @ a_p
    S = U @ P_p @ U.T + R_sph
    try:
        Kg = P_p @ U.T @ np.linalg.solve(S, np.eye(N))
    except Exception:
        Kg = np.zeros((k_act, N))
    a_f = a_p + Kg @ v
    P_f = (np.eye(k_act) - Kg @ U) @ P_p
    pred = (a_p @ U.T + om).ravel()          # PREDICTIVE (genuine forecast)
    pred_modal = (a_f @ U.T + om).ravel()    # MODAL (diagnostic)
    # Online Q adaptation
    innov = a_f - F @ alpha_init
    Q_new = 0.7 * Q_run + 0.3 * np.outer(innov, innov)
    Q_new = (Q_new + Q_new.T) / 2 + np.eye(k_act) * 1e-6
    return pred, pred_modal, a_f, P_f, Q_new


# =========================================================================
# V2-1: Rolling DMD vs Frozen DMD
# =========================================================================


def run_v2_1(wide):
    print("\n" + "=" * 90)
    print("  V2-1: ROLLING DMD vs FROZEN DMD")
    print("  Does updating the basis each quarter improve prediction?")
    print("=" * 90)

    rows = []

    for ty in TEST_YEARS:
        t0 = time.time()
        train_start = pd.Timestamp(f"{ty - T_YR}-01-01")
        test_quarters = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")

        # Get all data for this window
        all_data = wide[(wide.index >= train_start) & (wide.index <= test_quarters[-1])]
        valid = all_data.columns[all_data.notna().any()]
        all_data = all_data[valid].fillna(all_data[valid].mean())
        N = len(valid)

        # Initial training window
        train_end = pd.Timestamp(f"{ty - 1}-12-31")
        otr = all_data[(all_data.index >= train_start) & (all_data.index <= train_end)].values.astype(np.float64)
        om = ewm_demean(otr)
        otr_dm = otr - om

        # Frozen basis (computed once on training window)
        U_frozen, eigs_frozen = dmd_basis(otr_dm, N)
        if U_frozen is None:
            print(f"  W{ty}: DMD failed, skipping")
            continue
        k_act = U_frozen.shape[1]
        R_sph = train_sph_r(otr_dm, N, U_frozen)

        # PLATINUM params
        F_plat = np.eye(k_act) * 0.99
        Q_init = np.eye(k_act) * 0.5

        # === Frozen path: use U_frozen for all 4 test quarters ===
        alpha_c = np.zeros(k_act)
        P_c = np.eye(k_act)
        Q_run = Q_init.copy()
        preds_frozen = []
        actuals = []

        for q_date in test_quarters:
            q_data = all_data.loc[[q_date]].values.astype(np.float64)
            if q_data.shape[0] == 0:
                continue
            q_dm = q_data[0] - om.ravel()
            pred, _modal, alpha_c, P_c, Q_run = platinum_predict_quarter(
                q_dm, om, N, U_frozen, F_plat, Q_init, R_sph,
                alpha_c, P_c, Q_run,
            )
            preds_frozen.append(pred)
            actuals.append(q_data[0])

        if not preds_frozen:
            continue
        preds_frozen = np.array(preds_frozen)
        actuals = np.array(actuals)
        r2_frozen = float(oos_r_squared(preds_frozen.ravel(), actuals.ravel()))

        # === Rolling path: recompute DMD after each test quarter ===
        alpha_c = np.zeros(k_act)
        P_c = np.eye(k_act)
        Q_run = Q_init.copy()
        preds_rolling = []
        U_current = U_frozen.copy()
        rolling_window_start = train_start
        rotation_angles = []

        for i, q_date in enumerate(test_quarters):
            q_data = all_data.loc[[q_date]].values.astype(np.float64)
            if q_data.shape[0] == 0:
                continue
            q_dm = q_data[0] - om.ravel()

            # Predict with CURRENT basis
            pred, _modal, alpha_c_new, P_c_new, Q_run_new = platinum_predict_quarter(
                q_dm, om, N, U_current, F_plat, Q_init, R_sph,
                alpha_c, P_c, Q_run,
            )
            preds_rolling.append(pred)

            # Now UPDATE: expand training window to include this quarter
            expanded_end = q_date
            otr_expanded = all_data[
                (all_data.index >= rolling_window_start) & (all_data.index <= expanded_end)
            ].values.astype(np.float64)
            om_new = ewm_demean(otr_expanded)
            otr_dm_new = otr_expanded - om_new

            U_new, eigs_new = dmd_basis(otr_dm_new, N)
            if U_new is not None:
                # Measure rotation
                k_common = min(U_current.shape[1], U_new.shape[1])
                angle = subspace_angle(U_current, U_new, k=k_common)
                rotation_angles.append(np.degrees(angle))

                # Update basis and reproject alpha state
                U_current = U_new
                k_act_new = U_new.shape[1]
                # Reproject alpha onto new basis
                alpha_c = U_new.T @ (U_current[:, :alpha_c_new.shape[0]] @ alpha_c_new
                                      if alpha_c_new.shape[0] != k_act_new
                                      else q_dm)
                # Simple: project last observation onto new basis
                alpha_c = U_new.T @ q_dm
                P_c = np.eye(k_act_new)
                Q_run = Q_init[:k_act_new, :k_act_new].copy() if k_act_new <= Q_init.shape[0] else np.eye(k_act_new) * 0.5
                F_plat_new = np.eye(k_act_new) * 0.99
                R_sph = train_sph_r(otr_dm_new, N, U_new)
                om = om_new
            else:
                # DMD failed on expanded window — keep current
                alpha_c = alpha_c_new
                P_c = P_c_new
                Q_run = Q_run_new
                rotation_angles.append(0.0)

        preds_rolling = np.array(preds_rolling)
        r2_rolling = float(oos_r_squared(preds_rolling.ravel(), actuals.ravel()))

        mean_rotation = np.mean(rotation_angles) if rotation_angles else 0.0

        # DM test
        frozen_err = preds_frozen.ravel() - actuals.ravel()
        rolling_err = preds_rolling.ravel() - actuals.ravel()
        try:
            dm_stat, dm_p = diebold_mariano_test(rolling_err, frozen_err)
        except Exception:
            dm_stat, dm_p = 0.0, 1.0

        elapsed = time.time() - t0
        print(f"  W{ty}: frozen={r2_frozen:.4f}  rolling={r2_rolling:.4f}  "
              f"delta={r2_rolling - r2_frozen:+.4f}  "
              f"mean_rot={mean_rotation:.1f}deg  ({elapsed:.1f}s)")

        rows.append({
            "window": f"W{ty}", "r2_frozen": r2_frozen, "r2_rolling": r2_rolling,
            "delta": r2_rolling - r2_frozen,
            "mean_rotation_deg": mean_rotation,
            "dm_stat": dm_stat, "dm_p": dm_p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    print(f"\n  === V2-1 SUMMARY ===")
    print(f"  Mean frozen R^2:  {df['r2_frozen'].mean():.4f}")
    print(f"  Mean rolling R^2: {df['r2_rolling'].mean():.4f}")
    print(f"  Mean delta:       {df['delta'].mean():+.4f}")
    print(f"  Rolling wins:     {(df['delta'] > 0).sum()}/{len(df)} windows")
    print(f"  Mean rotation:    {df['mean_rotation_deg'].mean():.1f} deg/quarter")

    verdict = "ROLLING HELPS" if df["delta"].mean() > 0.001 else \
              "ROLLING HURTS" if df["delta"].mean() < -0.001 else "NO DIFFERENCE"
    print(f"  VERDICT: {verdict}")
    return df


# =========================================================================
# V2-2/V2-3/V2-4: Basis Evolution Tracking
# =========================================================================


def run_v2_evolution(wide):
    print("\n" + "=" * 90)
    print("  V2-2/V2-3/V2-4: BASIS EVOLUTION TRACKING")
    print("  Rolling 5yr DMD window, 1Q steps, full timeline")
    print("=" * 90)

    # Generate quarterly dates for rolling windows
    all_quarters = pd.date_range("2010-01-01", "2024-12-31", freq="QS")
    window_size_q = T_YR * 4  # 20 quarters

    rows = []
    prev_U = None
    prev_eigs = None

    for i, q_end in enumerate(all_quarters):
        q_start = q_end - pd.DateOffset(years=T_YR)
        if q_start < pd.Timestamp("2005-01-01"):
            continue

        tr = wide[(wide.index >= q_start) & (wide.index <= q_end)]
        valid = tr.columns[tr.notna().any()]
        if len(valid) < 20:
            continue
        tr = tr[valid]
        otr = tr.fillna(tr.mean()).values.astype(np.float64)
        N = otr.shape[1]
        om = ewm_demean(otr)
        otr_dm = otr - om
        T = otr_dm.shape[0]

        U, eigs = dmd_basis(otr_dm, N)
        if U is None:
            continue

        k_act = U.shape[1]

        # V2-2: K_effective (eigenvalues above threshold)
        if eigs is not None:
            eig_mag = np.abs(eigs)
            threshold = 0.1 * eig_mag.max() if eig_mag.max() > 0 else 0.01
            k_eff = int(np.sum(eig_mag > threshold))
            # Spectral energy concentration
            total_energy = np.sum(eig_mag ** 2)
            top3_energy = np.sum(np.sort(eig_mag ** 2)[-3:]) / (total_energy + 1e-12)
        else:
            k_eff = k_act
            top3_energy = 1.0

        # V2-3: Basis rotation from previous quarter
        rotation_deg = 0.0
        if prev_U is not None and prev_U.shape[0] == U.shape[0]:
            k_common = min(prev_U.shape[1], U.shape[1])
            rotation_deg = np.degrees(subspace_angle(prev_U, U, k=k_common))

        # V2-4: Mode birth/death
        births = 0
        deaths = 0
        if prev_eigs is not None and eigs is not None:
            prev_mag = np.sort(np.abs(prev_eigs))[::-1]
            curr_mag = np.sort(np.abs(eigs))[::-1]
            # Count eigenvalues that crossed threshold
            prev_k = int(np.sum(prev_mag > threshold))
            curr_k = int(np.sum(curr_mag > threshold))
            if curr_k > prev_k:
                births = curr_k - prev_k
            elif curr_k < prev_k:
                deaths = prev_k - curr_k

        rows.append({
            "quarter": q_end, "N": N, "T": T, "K_actual": k_act,
            "K_effective": k_eff, "top3_energy_frac": top3_energy,
            "rotation_deg": rotation_deg,
            "births": births, "deaths": deaths,
            "max_eig_mag": float(np.abs(eigs).max()) if eigs is not None else 0,
            "min_eig_mag": float(np.abs(eigs).min()) if eigs is not None else 0,
        })

        prev_U = U
        prev_eigs = eigs

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No results.")
        return df

    # V2-2: K_effective evolution
    print(f"\n  === V2-2: K_EFFECTIVE OVER TIME ===")
    print(f"  {'Quarter':>12} {'K_eff':>6} {'Top3 energy':>12} {'Rotation':>10} {'B/D':>6}")
    for _, row in df.iterrows():
        bd = ""
        if row["births"] > 0:
            bd = f"+{row['births']}"
        elif row["deaths"] > 0:
            bd = f"-{row['deaths']}"
        print(f"  {str(row['quarter'])[:7]:>12} "
              f"{row['K_effective']:>6} {row['top3_energy_frac']:>12.3f} "
              f"{row['rotation_deg']:>10.1f} {bd:>6}")

    # V2-2 summary stats
    print(f"\n  K_effective: mean={df['K_effective'].mean():.1f}  "
          f"std={df['K_effective'].std():.1f}  "
          f"min={df['K_effective'].min()}  max={df['K_effective'].max()}")
    k_varies = df["K_effective"].std() > 0.5
    print(f"  Mode count {'VARIES' if k_varies else 'is STABLE'} over time")

    # V2-3: Rotation statistics
    rot = df[df["rotation_deg"] > 0]["rotation_deg"]
    print(f"\n  === V2-3: BASIS ROTATION ===")
    print(f"  Rotation per quarter: mean={rot.mean():.1f} deg  "
          f"std={rot.std():.1f}  min={rot.min():.1f}  max={rot.max():.1f}")

    # Correlate rotation with market events / K_effective changes
    # High rotation quarters
    high_rot_threshold = rot.mean() + rot.std()
    high_rot = df[df["rotation_deg"] > high_rot_threshold]
    if not high_rot.empty:
        print(f"  High-rotation quarters (> {high_rot_threshold:.0f} deg):")
        for _, row in high_rot.iterrows():
            q_str = str(row["quarter"])[:7]
            print(f"    {q_str}: {row['rotation_deg']:.1f} deg, K_eff={row['K_effective']}")

    # V2-4: Birth/death events
    print(f"\n  === V2-4: MODE BIRTH/DEATH EVENTS ===")
    events = df[(df["births"] > 0) | (df["deaths"] > 0)]
    if events.empty:
        print("  No mode birth/death events detected.")
    else:
        print(f"  Total births: {df['births'].sum()}, deaths: {df['deaths'].sum()}")
        for _, row in events.iterrows():
            q_str = str(row["quarter"])[:7]
            event_type = f"BIRTH(+{row['births']})" if row["births"] > 0 else f"DEATH(-{row['deaths']})"
            print(f"    {q_str}: {event_type}  K_eff={row['K_effective']}  rot={row['rotation_deg']:.1f}deg")

    # V2-3 extended: does rotation predict anything?
    # Merge with test-year R^2 from PLATINUM
    print(f"\n  === V2-3 EXTENDED: ROTATION vs PREDICTION ===")
    # Compute yearly average rotation for each test year
    df["year"] = df["quarter"].dt.year
    yearly_rot = df.groupby("year")["rotation_deg"].mean()
    print(f"  Yearly mean rotation:")
    for yr, rot_val in yearly_rot.items():
        print(f"    {yr}: {rot_val:.1f} deg")

    return df


# =========================================================================
# Main
# =========================================================================


def main():
    print("=" * 90)
    print(f"  SMIM DRILLDOWN #2 — VENUE 2: RECURSIVE DMD  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Baseline: PLATINUM R^2=0.543")
    print("=" * 90)

    wide = load_panel()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # V2-1: Rolling vs Frozen
    df_v2_1 = run_v2_1(wide)
    if not df_v2_1.empty:
        df_v2_1.to_parquet(METRICS_DIR / "dd2_V2-1_rolling.parquet", index=False)

    # V2-2/V2-3/V2-4: Evolution tracking
    df_v2_evo = run_v2_evolution(wide)
    if not df_v2_evo.empty:
        df_v2_evo.to_parquet(METRICS_DIR / "dd2_V2_evolution.parquet", index=False)

    # Final summary
    print("\n" + "=" * 90)
    print("  VENUE 2 FINDINGS")
    print("=" * 90)
    if not df_v2_1.empty:
        delta = df_v2_1["delta"].mean()
        wins = (df_v2_1["delta"] > 0).sum()
        print(f"  V2-1 Rolling vs Frozen: delta={delta:+.4f}, wins={wins}/{len(df_v2_1)}")
        print(f"        {'ADAPTIVE BASIS HELPS' if delta > 0.001 else 'Frozen basis is sufficient'}")
    if not df_v2_evo.empty:
        k_std = df_v2_evo["K_effective"].std()
        rot_mean = df_v2_evo[df_v2_evo["rotation_deg"] > 0]["rotation_deg"].mean()
        n_events = (df_v2_evo["births"] + df_v2_evo["deaths"]).sum()
        print(f"  V2-2 K_effective std: {k_std:.1f} "
              f"({'VARIES' if k_std > 0.5 else 'STABLE'})")
        print(f"  V2-3 Mean rotation: {rot_mean:.1f} deg/quarter")
        print(f"  V2-4 Birth/death events: {n_events}")
        if k_std > 0.5 or n_events > 0:
            print(f"  ==> STRUCTURAL EVOLUTION DETECTED")
        else:
            print(f"  ==> Basis is STATIC (no emergence)")


if __name__ == "__main__":
    main()
