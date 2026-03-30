#!/usr/bin/env python
"""Generate the conceptual architecture figure (new Figure 1)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "smim" / "paper" / "img"

ACCENT = "#133D7A"
ACCENT_LIGHT = "#4A7FBF"
GOLD = "#C4960C"
GREEN = "#2A8636"
RED = "#B8352A"
ORANGE = "#D4770B"
PURPLE = "#6B3FA0"
GREY = "#888888"
LIGHT_BLUE = "#D6E4F5"
LIGHT_GREEN = "#D6F0DC"
LIGHT_ORANGE = "#FAEBD7"
LIGHT_RED = "#FDE8E8"
LIGHT_PURPLE = "#EDE4F7"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def draw_actor_box(ax, x, y, w, h, label, sublabel, color, text_color=ACCENT):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05", linewidth=1.2,
                          edgecolor=text_color, facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.12, label, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.15, sublabel, ha="center", va="center", fontsize=6,
                color=GREY, zorder=4, fontstyle="italic")


def draw_layer_band(ax, y, label, color, alpha=0.12):
    ax.axhspan(y - 0.55, y + 0.55, color=color, alpha=alpha, zorder=0)
    ax.text(-0.3, y, label, ha="right", va="center", fontsize=9,
            fontweight="bold", color=GREY, rotation=0)


def draw_arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                              gridspec_kw={"width_ratios": [3, 2], "wspace": 0.08})

    # ── LEFT PANEL: Multilayer Actor Structure ──────────────────────────────
    ax = axes[0]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.8, 4.5)
    ax.axis("off")
    ax.set_title("(a) Multilayer Actor Structure", fontweight="bold",
                  color=ACCENT, fontsize=11, pad=10)

    # Layer bands
    draw_layer_band(ax, 3.5, "Layer 0\nExogenous", ORANGE)
    draw_layer_band(ax, 1.8, "Layer 1\nInstitutional", PURPLE)
    draw_layer_band(ax, 0.0, "Layer 2\nFirms", ACCENT)

    # Layer 0: Shocks
    shocks = [
        (1.0, "Oil price", "Brent/WTI"),
        (3.0, "Fed funds", "FEDFUNDS"),
        (5.0, "VIX", "Volatility"),
        (7.0, "USD index", "DTWEXBGS"),
    ]
    for x, label, sub in shocks:
        draw_actor_box(ax, x, 3.5, 1.5, 0.7, label, sub, LIGHT_ORANGE, ORANGE)

    # Layer 1: Institutions
    institutions = [
        (1.5, "Federal\nReserve", "Central bank"),
        (4.0, "SEC", "Regulator"),
        (6.5, "Bank of\nEngland", "Central bank"),
    ]
    for x, label, sub in institutions:
        draw_actor_box(ax, x, 1.8, 1.5, 0.7, label, sub, LIGHT_PURPLE, PURPLE)

    # Layer 2: Firms (grouped by sector)
    sectors = [
        (0.8, "Energy\n(12)", "CVX, COP...", LIGHT_ORANGE),
        (2.5, "Tech\n(15)", "AAPL, MSFT...", LIGHT_BLUE),
        (4.2, "Financials\n(12)", "BAC, JPM...", LIGHT_GREEN),
        (5.9, "Healthcare\n(10)", "JNJ, PFE...", LIGHT_RED),
        (7.6, "Industrials\n(12)", "BA, GE...", "#F0F0F0"),
    ]
    for x, label, sub, color in sectors:
        draw_actor_box(ax, x, 0.0, 1.3, 0.8, label, sub, color, ACCENT)

    # Propagation arrows (Layer 0 -> Layer 1)
    for sx in [1.0, 3.0, 5.0, 7.0]:
        for ix in [1.5, 4.0, 6.5]:
            if abs(sx - ix) < 3:
                draw_arrow(ax, sx, 3.1, ix, 2.2, ORANGE, lw=0.6)

    # Propagation arrows (Layer 0 -> Layer 2, direct)
    draw_arrow(ax, 1.0, 3.1, 0.8, 0.45, RED, lw=1.5)
    draw_arrow(ax, 3.0, 3.1, 2.5, 0.45, RED, lw=1.5)
    draw_arrow(ax, 5.0, 3.1, 5.9, 0.45, RED, lw=1.0)
    ax.text(0.3, 1.6, "Direct\ntransmission\n(D3: r=0.86)", fontsize=6,
            color=RED, fontstyle="italic", ha="center")

    # Propagation arrows (Layer 1 -> Layer 2, weak)
    for ix in [1.5, 4.0, 6.5]:
        for fx in [0.8, 2.5, 4.2, 5.9, 7.6]:
            if abs(ix - fx) < 2.5:
                draw_arrow(ax, ix, 1.4, fx, 0.45, PURPLE, style="-|>", lw=0.4)

    # Gap annotation
    ax.annotate("", xy=(7.6, -0.8), xytext=(7.6, -0.5),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(7.6, -1.2, r"$\Delta_{i,t} = y_{i,t} - y^*_{i,t}$",
            ha="center", fontsize=10, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_RED, edgecolor=RED))
    ax.text(7.6, -1.65, "Investment Gap", ha="center", fontsize=8, color=RED)

    # N=93 annotation
    ax.text(4.0, -1.5, "N = 93 actors with intensity\n84 quarters (2005-2025)",
            ha="center", fontsize=8, color=GREY, fontstyle="italic")

    # ── RIGHT PANEL: Pipeline Flow ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-2.0, 6.5)
    ax2.axis("off")
    ax2.set_title("(b) Processing Pipeline", fontweight="bold",
                   color=ACCENT, fontsize=11, pad=10)

    stages = [
        (2.0, 5.5, "Intensity Panel\n$y_{i,t} \\in [0,1]$\n$N \\times T$", LIGHT_BLUE, ACCENT),
        (2.0, 4.0, "EWM Demeaning\n$\\tilde{y} = y - \\hat{\\mu}$\n$\\tau = 8Q$", LIGHT_GREEN, GREEN),
        (2.0, 2.5, "DMD Decomposition\n$Y' \\approx AU$\n$K = 8$ modes", LIGHT_ORANGE, ORANGE),
        (2.0, 1.0, "Kalman Filter\n$R = cI$ (spherical)\n$\\alpha_t | \\tilde{y}_{1:t}$", LIGHT_BLUE, ACCENT),
        (2.0, -0.5, "Online $Q$ Adapt\n$Q_{t+1} = 0.7Q_t + 0.3\\eta\\eta'$", LIGHT_ORANGE, ORANGE),
    ]

    for x, y, text, bg, ec in stages:
        box = FancyBboxPatch((x - 1.5, y - 0.55), 3.0, 1.1,
                              boxstyle="round,pad=0.08", linewidth=1.5,
                              edgecolor=ec, facecolor=bg, zorder=3)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha="center", va="center", fontsize=7.5,
                color=ec, zorder=4, linespacing=1.3)

    # Arrows between stages
    for i in range(len(stages) - 1):
        y1 = stages[i][1] - 0.55
        y2 = stages[i + 1][1] + 0.55
        ax2.annotate("", xy=(2.0, y2), xytext=(2.0, y1),
                     arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

    # R² annotations on the right
    r2_vals = [
        (5.5, "Start", "0.339"),
        (4.0, "+4.2pp", "0.381"),
        (2.5, "+1.1pp", "0.392"),
        (1.0, "+4.2pp", "0.434"),
        (-0.5, "+5.7pp", "0.524"),
    ]
    for y, delta, r2 in r2_vals:
        ax2.text(3.8, y + 0.15, f"$R^2$={r2}", fontsize=7, color=GREY,
                ha="left", fontstyle="italic")
        if delta != "Start":
            ax2.text(3.8, y - 0.15, delta, fontsize=7.5, color=GREEN,
                    ha="left", fontweight="bold")

    # Output arrow
    ax2.annotate("", xy=(2.0, -1.5), xytext=(2.0, -1.05),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    gap_box = FancyBboxPatch((0.3, -2.0), 3.4, 0.8,
                              boxstyle="round,pad=0.08", linewidth=2,
                              edgecolor=RED, facecolor=LIGHT_RED, zorder=3)
    ax2.add_patch(gap_box)
    ax2.text(2.0, -1.6, "Investment Gap\n$\\Delta_{i,t} = y_{i,t} - (\\hat{\\mu}_i + U\\alpha_t)$",
            ha="center", va="center", fontsize=8, color=RED, fontweight="bold", zorder=4)

    fig.savefig(IMG_DIR / "fig1_conceptual.pdf")
    fig.savefig(IMG_DIR / "fig1_conceptual.png")
    plt.close(fig)
    print("  fig1_conceptual.pdf/png")


if __name__ == "__main__":
    main()
