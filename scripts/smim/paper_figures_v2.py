#!/usr/bin/env python
"""
Generate paper figures v2 — updated for rolling DMD (R^2=0.691).

Updates:
  - fig2_ladder: 8 steps ending at 0.691
  - fig3_per_window: 3 bars (rolling DMD, frozen PLATINUM, AR(1))
  - fig11_variance: updated for R^2=0.691
  - fig1_schematic: pipeline with rolling basis + F regularisation
  - NEW fig12_rotation: basis rotation over time with event annotations
  - NEW fig13_rolling_vs_frozen: paired comparison

Unchanged: fig4 ablation, fig5 spectral, fig6 regularisation, fig7 robustness,
  fig8 zero-shot, fig9 D2, fig10 D6 (these report pre-rolling results accurately)

Usage::
    uv run python scripts/smim/paper_figures_v2.py
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
TEAL = "#1A7A6D"
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


# ── Figure 2: Performance Ladder (8 steps to 0.691) ─────────────────────────

def fig_ladder():
    # Matches Table 5 (tab:ladder) in the paper.
    # Steps 1-2 (EWM + shorter T without spherical R) cause Kalman divergence
    # and are documented in the table but omitted from this figure.
    # The successful steps are: 0 (baseline), 3 (spherical R), 4 (DMD),
    # 5 (online Q + K=8), 6 (F reg), 7 (rolling DMD).
    steps = [
        ("Baseline\n(T=10yr, K=3\nSchur, full demean)", 0.266, None),
        ("+ Spherical R\n(Kalman rescued)", 0.440, 0.266),
        ("+ DMD\nbasis", 0.472, 0.440),
        ("+ Online Q\nK=8", 0.530, 0.472),
        ("+ F reg\nQ=0.5", 0.549, 0.530),
        ("+ Rolling\nDMD", 0.696, 0.549),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))

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
            colors.append(TEAL if i == len(steps) - 1 else ACCENT)

    bars = ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Phase annotations
    ax.axvspan(-0.5, 3.5, alpha=0.04, color=ACCENT, zorder=1)
    ax.axvspan(3.5, 5.5, alpha=0.04, color=TEAL, zorder=1)
    ax.text(1.5, 0.02, "Regularisation", fontsize=8, color=ACCENT, ha="center", alpha=0.6)
    ax.text(4.5, 0.02, "Dual reg + rolling", fontsize=8, color=TEAL, ha="center", alpha=0.6)

    # Value labels
    for i, (label, val, prev) in enumerate(steps):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9,
                fontweight="bold", color=ACCENT if i < len(steps) - 1 else TEAL)
        if prev is not None:
            delta = val - prev
            if delta > 0.015:
                ax.text(i, (val + prev) / 2, f"+{delta:.3f}", ha="center", fontsize=7,
                        color="white", fontweight="bold", va="center")

    # Connector lines
    for i in range(len(steps) - 1):
        val = steps[i][1]
        ax.plot([i + 0.3, i + 0.7], [val, val], color="#999999", linewidth=0.8,
                linestyle=":", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=7.5)
    ax.set_ylabel("Modal R$^2$ (93-actor CapEx/Assets panel)")
    ax.set_title("Performance Ladder: Successful Steps from Baseline to Rolling DMD",
                 fontweight="bold", color=ACCENT)
    ax.set_ylim(0, 0.78)
    fig.tight_layout()
    save(fig, "fig2_ladder")


# ── Figure 3: Per-Window R^2 (Rolling vs Frozen vs AR(1)) ───────────────────

def fig_per_window():
    windows = [f"W{ty}" for ty in range(2015, 2025)]

    # Verified values (independent re-run 2026-04-04)
    rolling = [0.6672, 0.6106, 0.6635, 0.7332, 0.6822, 0.7639, 0.6447, 0.7524, 0.6915, 0.7496]
    frozen =  [0.5610, 0.4320, 0.4775, 0.6552, 0.5580, 0.6588, 0.4392, 0.6011, 0.4931, 0.6107]
    # AR(1) removed: modal R² is reconstruction quality, not comparable to forecasts

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(windows))
    w = 0.3

    ax.bar(x - w/2, rolling, w, color=TEAL, label=f"Rolling DMD (mean={np.mean(rolling):.3f})",
           zorder=3, edgecolor="white")
    ax.bar(x + w/2, frozen, w, color=ACCENT, label=f"Static basis (mean={np.mean(frozen):.3f})",
           zorder=3, edgecolor="white")

    # Delta annotations above rolling bars
    for i in range(len(windows)):
        delta = rolling[i] - frozen[i]
        ax.text(i - w/2, rolling[i] + 0.008, f"+{delta:.2f}",
                ha="center", fontsize=6.5, color=TEAL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_ylabel("Modal R$^2$ (spectral reconstruction)")
    ax.set_title("Per-Window Reconstruction Quality: Rolling vs Static Basis",
                 fontweight="bold", color=ACCENT)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    save(fig, "fig3_per_window")


# ── Figure 12: Basis Rotation Over Time ──────────────────────────────────────

def fig_basis_rotation():
    try:
        df = pd.read_parquet(METRICS / "dd2_V2_evolution.parquet")
    except FileNotFoundError:
        print("  SKIP fig12 (V2 evolution data not found)")
        return

    df = df[df["rotation_deg"] > 0].copy()
    dates = pd.to_datetime(df["quarter"])
    rotation = df["rotation_deg"].values

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # Rotation bars
    colors = [RED if r > 32 else ACCENT for r in rotation]
    ax1.bar(dates, rotation, width=80, color=colors, alpha=0.8, zorder=3, edgecolor="none")

    # Mean line
    mean_rot = rotation.mean()
    ax1.axhline(mean_rot, color=GREY, linestyle="--", linewidth=1.2, zorder=2)
    ax1.text(dates.iloc[-1] + pd.Timedelta(days=60), mean_rot,
             f" mean={mean_rot:.0f} deg", fontsize=8, color=GREY, va="center",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))

    # Event annotations
    events = {
        "2012-07-01": "Euro\ncrisis",
        "2018-04-01": "Tariff\nwar",
        "2020-04-01": "COVID",
        "2022-07-01": "Fed\ntightening",
    }
    for date_str, label in events.items():
        event_date = pd.Timestamp(date_str)
        # Find closest quarter
        closest = dates.iloc[(dates - event_date).abs().argmin()]
        idx = dates.tolist().index(closest)
        rot_val = rotation[idx]
        ax1.annotate(label, xy=(closest, rot_val), xytext=(closest, rot_val + 6),
                     fontsize=7, ha="center", color=RED,
                     arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
                     fontweight="bold")

    ax1.set_ylabel("Basis Rotation (degrees per quarter)")
    ax1.set_xlabel("Quarter")
    ax1.set_title("Spectral Basis Rotation: The Cross-Sectional Structure Evolves Continuously",
                  fontweight="bold", color=ACCENT)
    ax1.set_ylim(0, 55)

    # Secondary info
    ax1.text(0.02, 0.95,
             "K$_{eff}$ = 8 across all 52 transitions (zero mode birth/death)",
             transform=ax1.transAxes, fontsize=8, va="top", color=GREY, fontstyle="italic")

    fig.tight_layout()
    save(fig, "fig12_rotation")


# ── Figure 11: Variance Decomposition (updated to 0.691) ────────────────────

def fig_variance():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): Static basis = 0.543
    total_frozen = 0.549
    mean_frozen = 0.281
    spectral_frozen = total_frozen - mean_frozen
    ax = axes[0]
    sizes = [mean_frozen / total_frozen, spectral_frozen / total_frozen]
    labels = [f"Per-actor mean\n{mean_frozen:.3f} ({mean_frozen/total_frozen:.0%})",
              f"Spectral dynamics\n{spectral_frozen:.3f} ({spectral_frozen/total_frozen:.0%})"]
    colors = [GOLD_COLOR, ACCENT]
    wedges, _, _ = ax.pie(sizes, labels=labels, colors=colors, explode=(0.03, 0.03),
                          autopct="", startangle=90, textprops={"fontsize": 8})
    circle = plt.Circle((0, 0), 0.35, color="white", zorder=5)
    ax.add_patch(circle)
    ax.text(0, 0, f"R$^2$\n{total_frozen:.3f}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=ACCENT, zorder=6)
    ax.set_title("(a) Static Basis", fontweight="bold", fontsize=10, color=ACCENT)

    # Panel (b): Rolling DMD = 0.696
    total_rolling = 0.696
    mean_rolling = 0.281  # EWM mean unchanged
    spectral_rolling = total_rolling - mean_rolling
    ax = axes[1]
    sizes = [mean_rolling / total_rolling, spectral_rolling / total_rolling]
    labels = [f"Per-actor mean\n{mean_rolling:.3f} ({mean_rolling/total_rolling:.0%})",
              f"Spectral dynamics\n{spectral_rolling:.3f} ({spectral_rolling/total_rolling:.0%})"]
    colors = [GOLD_COLOR, TEAL]
    wedges, _, _ = ax.pie(sizes, labels=labels, colors=colors, explode=(0.03, 0.03),
                          autopct="", startangle=90, textprops={"fontsize": 8})
    circle = plt.Circle((0, 0), 0.35, color="white", zorder=5)
    ax.add_patch(circle)
    ax.text(0, 0, f"R$^2$\n{total_rolling:.3f}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=TEAL, zorder=6)
    ax.set_title("(b) Rolling DMD Basis", fontweight="bold", fontsize=10, color=TEAL)

    fig.suptitle("Variance Decomposition: Static vs Rolling Basis",
                 fontweight="bold", color=ACCENT, fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig11_variance")


# ── Figure 1: Pipeline Schematic (updated) ──────────────────────────────────

def fig_schematic():
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-2.5, 4.8)
    ax.axis("off")

    pipeline_y = 2.5
    boxes = [
        (0.8, pipeline_y, "Intensity\nPanel\n$y_{i,t} \\in [0,1]$\n$N \\times T$", "#E8EEF5"),
        (3.0, pipeline_y, "EWM\nDemeaning\n$\\tilde{y} = y - \\hat{\\mu}$\n$\\tau=8$Q", "#E8F5E9"),
        (5.2, pipeline_y, "DMD\n$Y' \\approx AX$\n$K=8$ modes\n$U \\in \\mathbb{R}^{N\\times K}$", "#FFF3E0"),
        (7.6, pipeline_y, "Kalman Filter\n$R = cI$ (spherical)\n$F = 0.99I$ (fixed)\n$Q_0 = 0.5I$", "#E8EEF5"),
        (10.0, pipeline_y, "Online $Q$\nAdaptation\n$Q_{t+1} = 0.7Q_t$\n$+\\, 0.3\\eta\\eta'$", "#FFF3E0"),
    ]

    gap_box = (10.0, -0.8, "Investment Gap\n$\\Delta_{i,t} = y_{i,t} - (\\hat{\\mu}_i + U\\alpha_t)$", "#FFEBEE")

    bw, bh = 1.8, 1.7
    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - bw/2, y - bh/2), bw, bh, linewidth=1.8,
                               edgecolor=ACCENT, facecolor=color, zorder=3,
                               clip_on=False, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=ACCENT,
                zorder=4, linespacing=1.3)

    # Gap box
    gw, gh = 2.2, 0.9
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
        ax.annotate("", xy=(x2, pipeline_y), xytext=(x1, pipeline_y),
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

    # Arrow from Online Q down to Gap box
    ax.annotate("", xy=(10.0, -0.8 + gh/2), xytext=(10.0, pipeline_y - bh/2),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))

    # Rolling update loop: curved arrow from right side back to DMD
    ax.annotate("", xy=(5.2, pipeline_y + bh/2 + 0.25),
                xytext=(10.0 + bw/2 + 0.1, pipeline_y + bh/2 + 0.25),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(7.6, 4.15, "Rolling basis update (each quarter)",
            ha="center", fontsize=8.5, color=TEAL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F2F1", edgecolor=TEAL, linewidth=1.2))

    # R^2 progression below boxes
    r2_labels = [
        (0.8, "0.339"),
        (3.0, "0.381\n+4.2pp"),
        (5.2, "0.392\n+1.1pp"),
        (7.6, "0.543\n+15.1pp"),
        (10.0, "0.691\n+14.8pp"),
    ]
    for bx, label in r2_labels:
        ax.text(bx, pipeline_y - bh/2 - 0.2, label, ha="center", fontsize=7,
                color=GREY, fontstyle="italic", va="top")

    ax.set_title("SMIM Pipeline: Dual Regularisation with Rolling Spectral Basis",
                  fontweight="bold", color=ACCENT, fontsize=12, pad=15)
    # Don't use tight_layout — it clips patches outside axes.
    # Use explicit subplots_adjust to guarantee margins.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
    # Save with extra padding to prevent bbox clipping the gap box border
    for ext in ["pdf", "png"]:
        fig.savefig(IMG_DIR / f"fig1_schematic.{ext}", pad_inches=0.3)
    plt.close(fig)
    print(f"  fig1_schematic.pdf/png")


# ── Regenerate unchanged figures from original script ─────────────────────────
# Import and call original functions for unchanged figures

def fig_ablation_extended():
    """Extended ablation heatmap: original depths + regularised + rolling."""
    windows = [f"W{ty}" for ty in range(2015, 2025)]

    # Original B1 ablation (pre-regularisation)
    l1 = [0.326, 0.348, 0.129, 0.410, 0.425, 0.435, 0.215, 0.315, 0.266, 0.408]
    l2 = [0.283, 0.292, 0.088, 0.253, 0.387, 0.353, 0.286, 0.276, 0.341, 0.355]
    l3 = [0.286, 0.141, 0.117, 0.293, 0.331, 0.406, 0.274, 0.273, 0.324, 0.363]
    l5 = [0.286, 0.279, 0.117, 0.293, 0.385, 0.406, 0.284, 0.282, 0.341, 0.376]

    # Verified values (independent re-run 2026-04-04)
    # AR(1) removed: modal R² is reconstruction quality, not comparable to AR(1) forecasts
    plat = [0.561, 0.432, 0.478, 0.655, 0.558, 0.659, 0.439, 0.601, 0.493, 0.611]
    rolling = [0.667, 0.611, 0.664, 0.733, 0.682, 0.764, 0.645, 0.752, 0.692, 0.750]

    row_labels = [
        "L1: Graph factors\n(no Kalman)",
        "L2: + Kalman\n(unregularised R)",
        "L3: + Regime\nswitching",
        "L5: Full original\npipeline",
        "SMIM\n(static basis)",
        "SMIM\n(rolling basis)",
    ]
    data = np.array([l1, l2, l3, l5, plat, rolling])
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Custom diverging colormap: red (bad) -> white (AR1 level) -> green (good)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ablation",
        [(0.0, "#CC3333"), (0.35, "#FFCC66"), (0.55, "#FFFFCC"), (0.75, "#99CC99"), (1.0, "#1A7A6D")])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0, vmax=0.80)

    # Horizontal divider between diagnosis (rows 0-3) and solution (rows 4-6)
    ax.axhline(3.5, color="black", linewidth=2.5, zorder=5)

    # Grid lines between all rows
    for i in range(n_rows - 1):
        if i != 3:
            ax.axhline(i + 0.5, color="white", linewidth=1.5, zorder=5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(windows, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            # Use white text on dark cells, black on light
            color = "white" if val < 0.20 or val > 0.60 else "black"
            fontweight = "bold" if i >= 4 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight=fontweight)

    # Mean column on right
    for i in range(n_rows):
        mean_val = data[i].mean()
        fontweight = "bold" if i >= 4 else "normal"
        ax.text(n_cols + 0.3, i, f"{mean_val:.3f}", ha="left", va="center",
                fontsize=8, fontweight=fontweight, color=ACCENT if i >= 4 else GREY)

    ax.text(n_cols + 0.3, -0.7, "Mean", ha="left", fontsize=8, fontweight="bold", color=ACCENT)

    # Phase labels
    ax.text(-0.5, 1.5, "Problem\ndiagnosis", ha="right", va="center", fontsize=8,
            color=RED, fontweight="bold", rotation=90, transform=ax.transData)
    ax.text(-0.5, 4.5, "Solution", ha="right", va="center", fontsize=8,
            color=TEAL, fontweight="bold", rotation=90, transform=ax.transData)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("OOS R$^2$", fontsize=9)

    ax.set_title("Component Ablation: From Unregularised Kalman to SMIM (Rolling Basis)",
                 fontweight="bold", color=ACCENT, fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_ablation")


def regenerate_unchanged():
    """Regenerate figures that don't change (5, 6, 7, 8, 9, 10)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import paper_figures as pf_orig
        pf_orig.fig_spectral()
        pf_orig.fig_regularisation()
        pf_orig.fig_robustness()
        pf_orig.fig_zero_shot()
        pf_orig.fig_d2()
        pf_orig.fig_d6()
    except Exception as e:
        print(f"  Warning: could not regenerate unchanged figures: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures v2 (updated for rolling DMD)...")
    print("\n  Updated figures:")
    fig_schematic()
    fig_ladder()
    fig_per_window()
    fig_ablation_extended()
    fig_variance()
    fig_basis_rotation()

    print("\n  Unchanged figures (regenerating):")
    regenerate_unchanged()

    print(f"\nAll figures saved to {IMG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
