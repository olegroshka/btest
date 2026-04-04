#!/usr/bin/env python
"""Generate the conceptual architecture figure (Figure 1) — minimal clean version."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "smim" / "paper" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Palette
ACCENT = "#133D7A"
GOLD = "#C4960C"
GREEN = "#2A8636"
RED = "#B8352A"
ORANGE = "#D4770B"
PURPLE = "#6B3FA0"
GREY = "#555555"

L0_BG = "#FFF3E0"
L1_BG = "#F3E5F5"
L2_BG = "#E3F2FD"
PIPE_GREEN = "#E8F5E9"
PIPE_ORANGE = "#FFF8E1"
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


def box(ax, x, y, w, h, text, bg, ec, fs=8, bold=True):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.06", lw=1.3,
                        edgecolor=ec, facecolor=bg, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=ec,
            zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=2)


def main():
    fig = plt.figure(figsize=(14, 6.5))
    ax_l = fig.add_axes([0.01, 0.04, 0.57, 0.89])
    ax_r = fig.add_axes([0.62, 0.04, 0.36, 0.89])

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (a) — Multilayer Actor Hierarchy
    # ══════════════════════════════════════════════════════════════════════
    ax = ax_l
    ax.set_xlim(0, 11)
    ax.set_ylim(-1.6, 5.8)
    ax.axis("off")
    ax.text(5.5, 5.25, "(a)  Multilayer Actor Hierarchy",
            ha="center", fontsize=12, fontweight="bold", color=ACCENT)

    # ── Layer bands with labels ABOVE the band ─────────────────────────
    for y, h, color, label in [
        (4.0, 0.95, ORANGE, "Layer 0 \u2014 Exogenous Shocks  (7 actors)"),
        (2.5, 0.95, PURPLE, "Layer 1 \u2014 Institutional Actors  (5 actors)"),
        (0.8, 1.20, ACCENT, "Layer 2 \u2014 Firms  (81 actors, 5 sectors)"),
    ]:
        ax.axhspan(y - h / 2, y + h / 2, color=color, alpha=0.10, zorder=0)
        ax.text(5.5, y + h / 2 + 0.12, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color, zorder=6)

    # ── Layer 0 actors ───────────────────────────────────────────────────
    l0 = [(2.0, "Oil price"), (4.2, "Fed funds"), (6.4, "VIX"), (8.6, "USD index")]
    for x, label in l0:
        box(ax, x, 4.0, 1.7, 0.6, label, L0_BG, ORANGE, fs=8)

    # ── Layer 1 actors ───────────────────────────────────────────────────
    l1 = [(2.0, "Federal Reserve"), (4.7, "SEC"), (7.4, "Bank of England")]
    for x, label in l1:
        box(ax, x, 2.5, 2.0, 0.6, label, L1_BG, PURPLE, fs=8)

    # ── Layer 2 sectors ──────────────────────────────────────────────────
    secs = [
        (1.2, "Energy\n(12)"),
        (3.2, "Tech\n(15)"),
        (5.2, "Financials\n(12)"),
        (7.2, "Healthcare\n(10)"),
        (9.2, "Industrials\n(12)"),
    ]
    for x, label in secs:
        box(ax, x, 0.8, 1.6, 0.85, label, L2_BG, ACCENT, fs=8)

    # ── Single clean transmission arrow (the D3 finding) ─────────────────
    # One bold arrow L0 -> L2 with annotation, nothing else
    ax.annotate(
        "", xy=(5.2, 1.25), xytext=(4.2, 3.65),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5,
                         connectionstyle="arc3,rad=-0.15"),
        zorder=5,
    )
    ax.text(2.7, 2.15,
            "Direct transmission\n$r = 0.86$, $p < 0.001$\n(D3 finding)",
            fontsize=8, color=RED, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=RED, alpha=0.95, lw=0.8))

    # Dashed line for weak L1->L2
    ax.annotate(
        "", xy=(5.2, 1.25), xytext=(4.7, 2.15),
        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.0,
                         linestyle="dashed"),
        zorder=2,
    )
    ax.text(6.6, 1.8, "$r = 0.03$, n.s.",
            fontsize=7, color="#AAAAAA", fontstyle="italic")

    # ── Bottom annotation ────────────────────────────────────────────────
    ax.text(5.5, -0.3,
            "Structural panel: $N = 93$ actors  |  5 sectors  |  84 quarters (2005 \u2013 2025)",
            ha="center", fontsize=8.5, color=GREY, fontstyle="italic")
    ax.text(5.5, -0.7,
            "Headline panel: $N = 146$ US firms  |  CapEx / Revenue rank $\\in [0, 1]$",
            ha="center", fontsize=8.5, color=GREY, fontstyle="italic")

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (b) — Processing Pipeline (updated for dual reg + rolling DMD)
    # ══════════════════════════════════════════════════════════════════════
    ax2 = ax_r
    ax2.set_xlim(-0.5, 6.0)
    ax2.set_ylim(-3.5, 6.0)
    ax2.axis("off")
    ax2.text(2.2, 5.7, "(b)  Processing Pipeline",
             ha="center", fontsize=12, fontweight="bold", color=ACCENT)

    TEAL = "#1A7A6D"

    stages = [
        (2.0, 4.8, "Intensity Panel\n$y_{i,t} \\in [0,1]$,  $N \\times T$",
         L2_BG, ACCENT),
        (2.0, 3.5, "EWM Demeaning\n$\\tilde{y} = y - \\hat{\\mu}$,  $\\tau \\in \\{8, 12\\}$Q",
         PIPE_GREEN, GREEN),
        (2.0, 2.2, "DMD Decomposition\n$Y' \\approx AU$,  $K = 2$ modes",
         PIPE_ORANGE, ORANGE),
        (2.0, 0.9, "Kalman Filter (dual reg.)\n$R = cI$,  $F = 0.99I$,  $Q_0 = 0.5I$",
         L2_BG, ACCENT),
        (2.0, -0.4, "Online $Q$ Adaptation\n$\\lambda = 0.3$",
         PIPE_ORANGE, ORANGE),
    ]

    bw, bh = 3.8, 0.85
    for x, y, text, bg, ec in stages:
        box(ax2, x, y, bw, bh, text, bg, ec, fs=8, bold=False)

    for i in range(len(stages) - 1):
        y1 = stages[i][1] - bh / 2
        y2 = stages[i + 1][1] + bh / 2
        arrow(ax2, 2.0, y1, 2.0, y2, ACCENT, lw=1.8)

    # No R^2 annotations — the ablation numbers are in Table 3 (structural panel).
    # The conceptual figure shows architecture only.

    # Rolling basis update loop — just outside the left box edge
    loop_x = 2.0 - bw / 2 - 0.1  # just outside the left edge
    ax2.annotate(
        "", xy=(loop_x, 2.2),
        xytext=(loop_x, -0.4),
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2,
                         connectionstyle="arc3,rad=-0.35"),
        zorder=5,
    )
    ax2.text(loop_x - 0.55, 0.9, "Rolling\nbasis\nupdate",
             ha="center", fontsize=7, color=TEAL, fontweight="bold")

    # Output gap box — positioned with enough clearance
    gap_y = -1.6
    arrow(ax2, 2.0, -0.4 - bh / 2, 2.0, gap_y + 0.35, RED, lw=2.2)
    box(ax2, 2.0, gap_y, 3.8, 0.6,
        "$\\Delta_{i,t} = y_{i,t} - (\\hat{\\mu}_i + U\\alpha_t)$",
        GAP_BG, RED, fs=9, bold=True)

    # Save with explicit padding to prevent bottom clipping
    fig.savefig(IMG_DIR / "fig1_conceptual.pdf", pad_inches=0.25)
    fig.savefig(IMG_DIR / "fig1_conceptual.png", pad_inches=0.25)
    plt.close(fig)
    print("  fig1_conceptual.pdf/png")


if __name__ == "__main__":
    main()
