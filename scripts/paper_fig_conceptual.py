#!/usr/bin/env python
"""Generate the conceptual architecture figure (Figure 1) — clean version."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "smim" / "paper" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Colors
ACCENT = "#133D7A"
GOLD = "#C4960C"
GREEN = "#2A8636"
RED = "#B8352A"
ORANGE = "#D4770B"
PURPLE = "#6B3FA0"
GREY = "#666666"
LIGHT_GREY = "#999999"

L0_BG = "#FFF3E0"
L1_BG = "#F3E5F5"
L2_BG = "#E3F2FD"
PIPE_GREENBG = "#E8F5E9"
PIPE_ORANGEBG = "#FFF8E1"
GAP_BG = "#FFEBEE"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.facecolor": "white",
})


def rounded_box(ax, x, y, w, h, text, bg, ec, fontsize=7.5, fw="bold", tc=None):
    tc = tc or ec
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.06", lw=1.3,
                          edgecolor=ec, facecolor=bg, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fw, color=tc, zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.0, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=2)


def layer_stripe(ax, y, h, color, label):
    ax.axhspan(y - h/2, y + h/2, color=color, alpha=0.25, zorder=0)
    ax.text(0.15, y, label, ha="left", va="center", fontsize=8,
            fontweight="bold", color=GREY, zorder=1)


def main():
    fig = plt.figure(figsize=(13.5, 6))

    # Two panels with controlled spacing
    ax_left = fig.add_axes([0.02, 0.05, 0.58, 0.88])   # wider left
    ax_right = fig.add_axes([0.63, 0.05, 0.35, 0.88])   # narrower right

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL (a): Multilayer Actor Structure
    # ═══════════════════════════════════════════════════════════════════════
    ax = ax_left
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-1.2, 5.2)
    ax.axis("off")
    ax.text(5.0, 5.05, "(a) Multilayer Actor Hierarchy",
            ha="center", fontsize=11, fontweight="bold", color=ACCENT)

    # Layer stripes
    layer_stripe(ax, 4.0, 1.0, ORANGE, "Layer 0\nExogenous")
    layer_stripe(ax, 2.4, 1.0, PURPLE, "Layer 1\nInstitutional")
    layer_stripe(ax, 0.7, 1.2, ACCENT, "Layer 2\nFirms")

    # Layer 0 actors
    l0_x = [2.0, 4.0, 6.0, 8.0]
    l0_labels = ["Oil price\n(Brent)", "Fed funds\nrate", "VIX", "USD\nindex"]
    for x, label in zip(l0_x, l0_labels):
        rounded_box(ax, x, 4.0, 1.5, 0.7, label, L0_BG, ORANGE, fontsize=7, tc=ORANGE)

    # Layer 1 actors
    l1_x = [2.5, 5.0, 7.5]
    l1_labels = ["Federal\nReserve", "SEC", "Bank of\nEngland"]
    for x, label in zip(l1_x, l1_labels):
        rounded_box(ax, x, 2.4, 1.4, 0.7, label, L1_BG, PURPLE, fontsize=7, tc=PURPLE)

    # Layer 2: sector clusters
    sectors = [
        (1.3, "Energy\n(12)", L0_BG),
        (3.1, "Tech\n(15)", L2_BG),
        (4.9, "Financials\n(12)", PIPE_GREENBG),
        (6.7, "Healthcare\n(10)", GAP_BG),
        (8.5, "Industrials\n(12)", "#F5F5F5"),
    ]
    for x, label, bg in sectors:
        rounded_box(ax, x, 0.7, 1.4, 0.85, label, bg, ACCENT, fontsize=7, tc=ACCENT)

    # Strong arrows: L0 -> L2 (direct transmission — the D3 finding)
    for sx in [2.0, 4.0, 6.0]:
        targets = [x for x, _, _ in sectors if abs(sx - x) < 2.5]
        for tx in targets[:2]:
            arrow(ax, sx, 3.6, tx, 1.2, RED, lw=1.8)

    # Weak arrows: L0 -> L1
    for sx in l0_x:
        for ix in l1_x:
            if abs(sx - ix) < 2.5:
                arrow(ax, sx, 3.6, ix, 2.8, LIGHT_GREY, lw=0.6)

    # Very weak arrows: L1 -> L2 (insignificant in D3)
    arrow(ax, 2.5, 2.0, 1.3, 1.2, LIGHT_GREY, lw=0.4)
    arrow(ax, 5.0, 2.0, 4.9, 1.2, LIGHT_GREY, lw=0.4)
    arrow(ax, 7.5, 2.0, 6.7, 1.2, LIGHT_GREY, lw=0.4)

    # Annotations
    ax.text(9.5, 3.2, "Direct L0$\\to$L2\n$r = 0.86$\n($p < 0.001$)",
            fontsize=7, color=RED, fontstyle="italic", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=RED, alpha=0.95, lw=0.8))

    ax.text(9.5, 1.8, "L1$\\to$L2 weak\n$r = 0.03$, n.s.",
            fontsize=6, color=LIGHT_GREY, fontstyle="italic", ha="center")

    # Bottom: summary
    ax.text(5.0, -0.4, "$N = 93$ actors  |  6 sectors  |  84 quarters (2005\u20132025)",
            ha="center", fontsize=8, color=GREY, fontstyle="italic")
    ax.text(5.0, -0.8, "Intensity: CapEx/Assets cross-sectional rank $\\in [0, 1]$",
            ha="center", fontsize=8, color=GREY, fontstyle="italic")

    # ═══════════════════════════════════════════════════════════════════════
    # PANEL (b): Processing Pipeline
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = ax_right
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-1.5, 5.7)
    ax2.axis("off")
    ax2.text(2.5, 5.5, "(b) Processing Pipeline",
             ha="center", fontsize=11, fontweight="bold", color=ACCENT)

    stages = [
        (2.0, 4.6, "Intensity Panel\n$y_{i,t} \\in [0,1]$, $N \\times T$", L2_BG, ACCENT),
        (2.0, 3.4, "EWM Demeaning\n$\\tilde{y} = y - \\hat{\\mu}$, $\\tau\\!=\\!8$Q", PIPE_GREENBG, GREEN),
        (2.0, 2.2, "DMD Decomposition\n$Y' \\approx AU$, $K\\!=\\!8$", PIPE_ORANGEBG, ORANGE),
        (2.0, 1.0, "Kalman Filter\n$R = cI$ (spherical)", L2_BG, ACCENT),
        (2.0, -0.2, "Online $Q$ Adaptation\n$\\lambda = 0.3$", PIPE_ORANGEBG, ORANGE),
    ]

    bw, bh = 3.4, 0.82
    for x, y, text, bg, ec in stages:
        rounded_box(ax2, x, y, bw, bh, text, bg, ec, fontsize=8, fw="normal", tc=ec)

    # Arrows between stages
    for i in range(len(stages) - 1):
        y1 = stages[i][1] - bh/2
        y2 = stages[i + 1][1] + bh/2
        arrow(ax2, 2.0, y1, 2.0, y2, ACCENT, lw=1.8)

    # R² and delta annotations
    annotations = [
        (4.6, "$R^2\\!=\\!0.339$", ""),
        (3.4, "$R^2\\!=\\!0.381$", "+4.2pp"),
        (2.2, "$R^2\\!=\\!0.392$", "+1.1pp"),
        (1.0, "$R^2\\!=\\!0.434$", "+4.2pp"),
        (-0.2, "$R^2\\!=\\!0.524$", "+5.7pp"),
    ]
    for y, r2_text, delta in annotations:
        ax2.text(4.1, y + 0.08, r2_text, fontsize=7.5, color=GREY,
                ha="left", fontstyle="italic")
        if delta:
            ax2.text(4.1, y - 0.2, delta, fontsize=8, color=GREEN,
                    ha="left", fontweight="bold")

    # Output: gap box
    arrow(ax2, 2.0, -0.2 - bh/2, 2.0, -1.05, RED, lw=2.0)
    rounded_box(ax2, 2.0, -1.3, 3.4, 0.45,
                "$\\Delta_{i,t} = y_{i,t} - (\\hat{\\mu}_i + U\\alpha_t)$",
                GAP_BG, RED, fontsize=8, fw="bold", tc=RED)

    fig.savefig(IMG_DIR / "fig1_conceptual.pdf")
    fig.savefig(IMG_DIR / "fig1_conceptual.png")
    plt.close(fig)
    print("  fig1_conceptual.pdf/png")


if __name__ == "__main__":
    main()
