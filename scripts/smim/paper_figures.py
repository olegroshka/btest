#!/usr/bin/env python
"""
Generate all paper figures from experiment results.

Saves to docs/smim/paper/img/ as both PDF and PNG.
Style: publication-quality, muted academic palette, clean typography.

Usage::
    uv run python scripts/smim/paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMG_DIR = PROJECT_ROOT / "docs" / "smim" / "paper" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
METRICS = PROJECT_ROOT / "results" / "metrics"

# ── Style ────────────────────────────────────────────────────────────────────

ACCENT = "#133D7A"
ACCENT_LIGHT = "#4A7FBF"
GOLD_COLOR = "#C4960C"
GREEN = "#2A8636"
RED = "#B8352A"
GREY = "#888888"
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(IMG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf/png")


# ── Figure 2: Performance Ladder (Waterfall) ─────────────────────────────────

def fig_ladder():
    steps = [
        ("Original\nA1", 0.339, None),
        ("+ EWM\ndemean", 0.381, 0.339),
        ("+ T=5yr\nK=5", 0.392, 0.381),
        ("+ Spherical\nR Kalman", 0.434, 0.392),
        ("+ DMD\nbasis", 0.467, 0.434),
        ("+ Online Q\nK=8", 0.524, 0.467),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(steps))
    colors = []
    bottoms = []
    heights = []
    for i, (label, val, prev) in enumerate(steps):
        if prev is None:
            bottoms.append(0)
            heights.append(val)
            colors.append(GREY)
        else:
            bottoms.append(prev)
            heights.append(val - prev)
            colors.append(ACCENT if val > prev else RED)

    bars = ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # AR(1) baseline
    ax.axhline(0.425, color=RED, linestyle="--", linewidth=1.5, zorder=2, label="AR(1) baseline")
    ax.text(5.35, 0.427, "AR(1) = 0.425", fontsize=8, color=RED, va="bottom")

    # Value labels
    for i, (label, val, prev) in enumerate(steps):
        ax.text(i, val + 0.008, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color=ACCENT)
        if prev is not None:
            delta = val - prev
            ax.text(i, (val + prev) / 2, f"+{delta:.3f}", ha="center", fontsize=7,
                    color="white", fontweight="bold", va="center")

    # Connector lines
    for i in range(len(steps) - 1):
        val = steps[i][1]
        ax.plot([i + 0.3, i + 0.7], [val, val], color="#999999", linewidth=0.8,
                linestyle=":", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Out-of-Sample R$^2$")
    ax.set_title("Performance Ladder: Cumulative R$^2$ Improvement", fontweight="bold", color=ACCENT)
    ax.set_ylim(0, 0.60)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig2_ladder")


# ── Figure 3: Per-Window R² Comparison ───────────────────────────────────────

def fig_per_window():
    # Data from DD-10 final results — correct AR(1) at T=10yr per window
    windows = [f"W{ty}" for ty in range(2015, 2025)]
    gold_plus = [0.5163, 0.4110, 0.4360, 0.6364, 0.5320, 0.6491, 0.4142, 0.5860, 0.4812, 0.5876]
    ar1_t10 = [0.3345, 0.2952, 0.3349, 0.4894, 0.4717, 0.5405, 0.3087, 0.4723, 0.4823, 0.5163]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(windows))
    w = 0.35

    ax.bar(x - w/2, gold_plus, w, color=ACCENT, label="GOLD+ (DMD + Sph. Kalman)", zorder=3, edgecolor="white")
    ax.bar(x + w/2, ar1_t10, w, color=GOLD_COLOR, label="AR(1) T=10yr", zorder=3, edgecolor="white", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_ylabel("Out-of-Sample R$^2$")
    ax.set_title("Per-Window Performance: GOLD+ vs AR(1)", fontweight="bold", color=ACCENT)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 0.72)
    fig.tight_layout()
    save(fig, "fig3_per_window")


# ── Figure 4: Component Ablation Heatmap (B1) ───────────────────────────────

def fig_ablation():
    try:
        df = pd.read_parquet(METRICS / "level1_B1-COMPONENT-ABLATION.parquet")
    except FileNotFoundError:
        print("  SKIP fig4 (B1 data not found)")
        return

    depths = ["L1_graph", "L2_kalman", "L3_regime", "L4_emergence", "L5_full"]
    windows = sorted(df["window_id"].unique())
    matrix = np.zeros((len(depths), len(windows)))
    for i, d in enumerate(depths):
        for j, w in enumerate(windows):
            row = df[(df["depth"] == d) & (df["window_id"] == w)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]["oos_r2"]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=0.5)

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(["L1: Graph\nfactors", "L2: + Kalman\nM=1", "L3: + Regime\nM=2",
                         "L4: + Emergence", "L5: Full\npipeline"], fontsize=8)
    # Value annotations
    for i in range(len(depths)):
        for j in range(len(windows)):
            val = matrix[i, j]
            color = "white" if val < 0.15 or val > 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("OOS R$^2$", fontsize=9)
    ax.set_title("Component Ablation: R$^2$ by Depth and Window", fontweight="bold", color=ACCENT)
    fig.tight_layout()
    save(fig, "fig4_ablation")


# ── Figure 5: Spectral Method Comparison (B2) ───────────────────────────────

def fig_spectral():
    try:
        df = pd.read_parquet(METRICS / "level1_B2-SPECTRAL-METHODS.parquet")
    except FileNotFoundError:
        print("  SKIP fig5 (B2 data not found)")
        return

    # Remove extended_dmd (failed, no data)
    df = df[df["method"] != "extended_dmd"]
    summary = df.groupby("method")["oos_r2"].agg(["mean", "std"]).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = range(len(summary))
    colors = [GOLD_COLOR if m == "dmd" else ACCENT_LIGHT if m == "pca" else ACCENT for m in summary.index]
    bars = ax.barh(y, summary["mean"], xerr=summary["std"], color=colors,
                    edgecolor="white", linewidth=1, capsize=3, zorder=3, height=0.6)

    ax.set_yticks(y)
    labels_map = {"dmd": "DMD", "pca": "PCA\n(symmetric)", "schur": "Schur",
                  "polar": "Polar", "hermitian_dilation": "Hermitian\ndilation",
                  "directed_variation": "Directed\nvariation"}
    ax.set_yticklabels([labels_map.get(m, m) for m in summary.index], fontsize=8)
    ax.set_xlabel("Mean OOS R$^2$ (10 windows)")
    ax.set_title("Spectral Method Comparison (B2)", fontweight="bold", color=ACCENT)

    for i, (method, row) in enumerate(summary.iterrows()):
        ax.text(row["mean"] + row["std"] + 0.005, i, f"{row['mean']:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    save(fig, "fig5_spectral")


# ── Figure 6: Regularisation Path (DD-9) ────────────────────────────────────

def fig_regularisation():
    try:
        df = pd.read_parquet(METRICS / "drilldown_DD-9.parquet")
    except FileNotFoundError:
        print("  SKIP fig6 (DD-9 data not found)")
        return

    by_s = df.groupby("shrinkage").agg(
        pred=("r2_kalman_pred", "mean"),
        modal=("r2_kalman_modal", "mean"),
        l1=("r2_l1", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(by_s.index, by_s["modal"], "o-", color=ACCENT, linewidth=2.5, markersize=7,
            label="Kalman modal", zorder=4)
    ax.plot(by_s.index, by_s["pred"], "s--", color=ACCENT_LIGHT, linewidth=1.5, markersize=5,
            label="Kalman predictive", zorder=3)
    ax.axhline(by_s["l1"].mean(), color=GREEN, linestyle=":", linewidth=1.5, label="L1 OLS (no Kalman)")
    # AR(1) removed: modal R² not comparable to forecast baselines

    ax.set_xlabel("Shrinkage parameter $s$ (0 = sample R, 1 = spherical R)")
    ax.set_ylabel("Mean OOS R$^2$")
    ax.set_title("Regularisation Path: Observation Covariance Shrinkage",
                  fontweight="bold", color=ACCENT)
    ax.legend(loc="lower right", framealpha=0.9)

    # Annotate the hockey stick
    ax.annotate("N$^2$ overfit\nzone", xy=(0.0, by_s.loc[0.0, "modal"]),
                xytext=(0.15, -4), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax.set_ylim(min(-1, by_s["modal"].min() - 0.5), max(0.5, by_s["modal"].max() + 0.05))
    fig.tight_layout()
    save(fig, "fig6_regularisation")


# ── Figure 7: Robustness 2x2 Panel ─────────────────────────────────────────

def fig_robustness():
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    # B6: N-sweep
    ax = axes[0, 0]
    ns = [20, 50, 99, 125]
    l1_schur = [0.8127, 0.7934, 0.7392, 0.7020]
    rw = [0.8033, 0.6469, 0.5493, 0.5475]
    ax.plot(ns, l1_schur, "o-", color=ACCENT, linewidth=2, label="L1 OLS")
    ax.plot(ns, rw, "s--", color=GOLD_COLOR, linewidth=1.5, label="Random walk")
    ax.set_xlabel("Number of actors (N)")
    ax.set_ylabel("OOS R$^2$")
    ax.set_title("(a) N-Sweep (B6)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 0.9)

    # B7: T-sweep
    ax = axes[0, 1]
    ts = [5, 8, 10, 15, 20]
    l1_vals = [0.2156, 0.4021, 0.3601, 0.3401, 0.3634]
    ar1_vals = [0.4336, 0.5193, 0.5163, 0.5468, 0.5468]
    ax.plot(ts, l1_vals, "o-", color=ACCENT, linewidth=2, label="L1 OLS")
    ax.plot(ts, ar1_vals, "s--", color=GOLD_COLOR, linewidth=1.5, label="AR(1)")
    ax.set_xlabel("Training window (years)")
    ax.set_ylabel("OOS R$^2$")
    ax.set_title("(b) T-Sweep (B7)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    # B8: Noise
    ax = axes[1, 0]
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    r2_noise = [0.3393, 0.3335, 0.3383, 0.3329, 0.3440, 0.3219]
    retention = [r / r2_noise[0] * 100 for r in r2_noise]
    ax.bar(range(len(sigmas)), retention, color=ACCENT, edgecolor="white", zorder=3, width=0.6)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:.1f}" for s in sigmas])
    ax.set_xlabel("Noise level ($\\sigma$ / $\\sigma_{signal}$)")
    ax.set_ylabel("Performance retention (%)")
    ax.set_title("(c) Noise Robustness (B8)", fontweight="bold", fontsize=10)
    ax.axhline(70, color=RED, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(80, 110)

    # B9: Edge corruption
    ax = axes[1, 1]
    fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    r2_edge = [0.3393] * 6  # all identical
    ax.bar(range(len(fracs)), [100] * len(fracs), color=GREEN, edgecolor="white", zorder=3, width=0.6)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{f:.1f}" for f in fracs])
    ax.set_xlabel("Edge corruption fraction")
    ax.set_ylabel("Performance retention (%)")
    ax.set_title("(d) Edge Robustness (B9)", fontweight="bold", fontsize=10)
    ax.axhline(70, color=RED, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 120)
    ax.text(2.5, 105, "100% at all levels\n(symmetric operator)", ha="center", fontsize=8,
            color=GREEN, fontstyle="italic")

    fig.suptitle("Robustness Analysis", fontweight="bold", fontsize=12, color=ACCENT, y=1.01)
    fig.tight_layout()
    save(fig, "fig7_robustness")


# ── Figure 8: True Zero-Shot Retention ──────────────────────────────────────

def fig_zero_shot():
    try:
        df = pd.read_parquet(METRICS / "drilldown_TRUE_ZERO_SHOT.parquet")
    except FileNotFoundError:
        print("  SKIP fig8 (zero-shot data not found)")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w/2, df["retention_frozen"], w, color=ACCENT_LIGHT, label="Frozen F/Q", zorder=3,
           edgecolor="white")
    ax.bar(x + w/2, df["retention_online"], w, color=ACCENT, label="Frozen + Online Q", zorder=3,
           edgecolor="white")

    ax.axhline(100, color=GREY, linestyle="--", linewidth=1, zorder=2)
    ax.set_xticks(x)
    labels = [f"{row['source']}\n->{row['target']}" for _, row in df.iterrows()]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Retention vs Full Retrain (%)")
    ax.set_title("True Zero-Shot Transfer: Dynamics Portability",
                  fontweight="bold", color=ACCENT)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_ylim(70, 180)

    # Mean annotations
    mean_f = df["retention_frozen"].mean()
    mean_o = df["retention_online"].mean()
    ax.text(len(df) - 0.5, mean_f, f"mean: {mean_f:.0f}%", fontsize=8, color=ACCENT_LIGHT, va="bottom")
    ax.text(len(df) - 0.5, mean_o + 3, f"mean: {mean_o:.0f}%", fontsize=8, color=ACCENT, va="bottom")

    fig.tight_layout()
    save(fig, "fig8_zero_shot")


# ── Figure 9: D2 Gap Prediction ─────────────────────────────────────────────

def fig_d2():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    quintiles = [1, 2, 3, 4, 5]
    outcomes = [0.1892, 0.0645, -0.0025, -0.0522, -0.2007]

    colors = [GREEN if v > 0 else RED for v in outcomes]
    ax.bar(quintiles, outcomes, color=colors, edgecolor="white", width=0.6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(quintiles)
    ax.set_xticklabels(["Q1\n(most under-\ninvesting)", "Q2", "Q3", "Q4",
                         "Q5\n(most over-\ninvesting)"], fontsize=8)
    ax.set_ylabel("Mean Next-4Q Intensity Change")
    ax.set_title("Gap Quintile vs Subsequent CapEx Revision (D2)\n"
                  "$\\beta$ = -0.27, $t$ = -6.95 after level control",
                  fontweight="bold", color=ACCENT, fontsize=10)

    # Spread annotation
    ax.annotate("", xy=(5, outcomes[-1]), xytext=(1, outcomes[0]),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
    ax.text(3, 0.15, "Q5-Q1 = -0.39", ha="center", fontsize=9, color=ACCENT, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig9_d2_quintile")


# ── Figure 10: D6 Gap Dispersion vs VIX ─────────────────────────────────────

def fig_d6():
    try:
        df = pd.read_parquet(METRICS / "level4_D6-EMERGENCE-TIMING.parquet")
    except FileNotFoundError:
        print("  SKIP fig10 (D6 data not found)")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    lags = df["lag_Q"].values
    corrs = df["corr_disp_vix"].values

    colors = [ACCENT if c < 0 else GOLD_COLOR for c in corrs]
    ax.bar(lags, corrs, color=colors, edgecolor="white", width=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color=GREY, linewidth=0.5, linestyle=":")

    ax.set_xlabel("Lag (quarters, negative = gap dispersion leads VIX)")
    ax.set_ylabel("Correlation")
    ax.set_title("Gap Dispersion vs VIX: Lead-Lag Analysis (D6)",
                  fontweight="bold", color=ACCENT)
    # Find lag=-1 index safely
    lag_list = lags.tolist()
    if -1 in lag_list:
        idx_neg1 = lag_list.index(-1)
        ax.annotate("Gaps emerge\nduring calm\nperiods", xy=(-1, corrs[idx_neg1]),
                    xytext=(-3.5, 0.05), fontsize=8, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    fig.tight_layout()
    save(fig, "fig10_d6_vix")


# ── Figure 11: Variance Decomposition ───────────────────────────────────────

def fig_variance():
    fig, ax = plt.subplots(figsize=(5, 5))

    # Three components of R²=0.524
    total = 0.524
    actor_mean = 0.281  # from per-actor mean alone
    spectral = total - actor_mean  # 0.243

    sizes = [actor_mean / total, spectral / total]
    labels = [f"Per-actor mean\n{actor_mean:.3f} ({actor_mean/total:.0%})",
              f"Spectral dynamics\n{spectral:.3f} ({spectral/total:.0%})"]
    colors = [GOLD_COLOR, ACCENT]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                        autopct="", startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"Variance Decomposition of R$^2$ = {total:.3f}",
                  fontweight="bold", color=ACCENT, fontsize=11, pad=15)

    # Center circle for R² value
    circle = plt.Circle((0, 0), 0.35, color="white", zorder=5)
    ax.add_patch(circle)
    ax.text(0, 0, f"R$^2$\n{total:.3f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=ACCENT, zorder=6)

    fig.tight_layout()
    save(fig, "fig11_variance")


# ── Figure 1: Framework Schematic ───────────────────────────────────────────

def fig_schematic():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 3.5)
    ax.axis("off")

    boxes = [
        (0.7, 2.0, "Intensity\nPanel\n$y_{i,t} \\in [0,1]$\n$N \\times T$", "#E8EEF5"),
        (2.8, 2.0, "EWM\nDemeaning\n$\\tilde{y} = y - \\hat{\\mu}$\nhalflife=8Q", "#E8F5E9"),
        (5.0, 2.0, "DMD\nDecomposition\n$Y' \\approx AU$\n$K=8$ modes", "#FFF3E0"),
        (7.2, 2.0, "Kalman Filter\n$R = \\frac{\\mathrm{tr}(\\hat{R})}{N} I$\n$\\alpha_t | \\tilde{y}_{1:t}$", "#E8EEF5"),
        (9.4, 2.0, "Online $Q$\nAdaptation\n$Q_{t+1} = 0.7Q_t$\n$+ 0.3 \\eta\\eta'$", "#FFF3E0"),
    ]

    gap_box = (9.4, 0.3, "Investment Gap\n$\\Delta_{i,t} = y_{i,t} - (\\hat{\\mu}_i + U\\alpha_t)$", "#FFEBEE")

    # Draw boxes
    bw, bh = 1.6, 1.6
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - bw/2, y - bh/2), bw, bh, linewidth=1.8,
                               edgecolor=ACCENT, facecolor=color, zorder=3,
                               clip_on=False, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=ACCENT,
                zorder=4, linespacing=1.3)

    # Gap box (wider, at bottom right)
    gw, gh = 2.0, 1.0
    x, y, text, color = gap_box
    rect = plt.Rectangle((x - gw/2, y - gh/2), gw, gh, linewidth=2,
                           edgecolor=RED, facecolor=color, zorder=3, clip_on=False)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=RED,
            zorder=4, fontweight="bold", linespacing=1.3)

    # Horizontal arrows between pipeline boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + bw/2
        x2 = boxes[i + 1][0] - bw/2
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

    # Arrow from Kalman down to Gap
    ax.annotate("", xy=(9.4, 0.3 + gh/2), xytext=(9.4, 2.0 - bh/2),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))

    # Delta-R² annotations below pipeline boxes
    deltas = [None, "+4.2pp", "+1.1pp", "+4.2pp", "+5.7pp"]
    for i, d in enumerate(deltas):
        if d:
            bx = boxes[i][0]
            ax.text(bx, 2.0 + bh/2 + 0.15, d, ha="center", fontsize=9,
                    color=GREEN, fontweight="bold")

    # R² progression above
    r2vals = ["R$^2$=0.339", "0.381", "0.392", "0.434", "0.524"]
    for i, r2 in enumerate(r2vals):
        bx = boxes[i][0]
        ax.text(bx, 2.0 + bh/2 + 0.45, r2, ha="center", fontsize=7.5,
                color=GREY, fontstyle="italic")

    ax.set_title("SMIM Pipeline Architecture with Component Contributions",
                  fontweight="bold", color=ACCENT, fontsize=13, pad=15)
    fig.tight_layout()
    save(fig, "fig1_schematic")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...")
    fig_schematic()
    fig_ladder()
    fig_per_window()
    fig_ablation()
    fig_spectral()
    fig_regularisation()
    fig_robustness()
    fig_zero_shot()
    fig_d2()
    fig_d6()
    fig_variance()
    print(f"\nAll figures saved to {IMG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
