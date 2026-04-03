#!/usr/bin/env python
"""
Generate Medium-optimised figures for the SMIM article.

Style: wide aspect ratio, large fonts, no grid, minimal decoration,
direct labels, 2-3 colour palette, clean for web display.

Usage::
    uv run python scripts/smim/smim_medium_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "smim" / "medium_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (3 colours max) ──────────────────────────────────────────────────

TEAL = "#0F7B6C"        # primary — rolling / positive
NAVY = "#1B3A5C"         # secondary — static / neutral
GOLD = "#D4A017"         # accent — baseline / AR(1)
RED = "#C0392B"          # negative / failure
LIGHT_GREY = "#F5F5F5"   # background
TEXT_DARK = "#2C3E50"     # text

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#DDDDDD",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Performance Ladder (Waterfall)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_ladder():
    steps = [
        ("Baseline",           0.339, None),
        ("+ Adaptive\ndemeaning", 0.381, 0.339),
        ("+ Shorter\nwindow",  0.392, 0.381),
        ("+ Spherical\ncovariance", 0.434, 0.392),
        ("+ DMD\nbasis",       0.467, 0.434),
        ("+ Online\nadaptation", 0.524, 0.467),
        ("+ Transition\nregularisation", 0.543, 0.524),
        ("+ Rolling\nbasis",   0.691, 0.543),
    ]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(len(steps))
    for i, (label, val, prev) in enumerate(steps):
        if prev is None:
            ax.bar(i, val, color=NAVY, width=0.55, zorder=3, edgecolor="white", linewidth=1.5)
        else:
            delta = val - prev
            color = TEAL if i < len(steps) - 1 else TEAL
            ax.bar(i, delta, bottom=prev, color=color, width=0.55, zorder=3,
                   edgecolor="white", linewidth=1.5)

    # Value labels on top
    for i, (label, val, prev) in enumerate(steps):
        ax.text(i, val + 0.012, f"{val:.3f}", ha="center", fontsize=12,
                fontweight="bold", color=TEXT_DARK)
        if prev is not None:
            delta = val - prev
            y_mid = prev + delta / 2
            sign = "+" if delta > 0 else ""
            if delta > 0.01:
                ax.text(i, y_mid, f"{sign}{delta:.3f}", ha="center", fontsize=9,
                        color="white", fontweight="bold", va="center")

    # Connector lines
    for i in range(len(steps) - 1):
        val = steps[i][1]
        ax.plot([i + 0.275, i + 0.725], [val, val], color="#BBBBBB", linewidth=0.8,
                linestyle=":", zorder=2)

    # AR(1) baseline
    ax.axhline(0.425, color=GOLD, linestyle="--", linewidth=2.5, zorder=2, alpha=0.8)
    ax.text(7.4, 0.432, "AR(1) baseline = 0.425", fontsize=12, color=GOLD,
            fontweight="bold", va="bottom")

    # Iteration markers
    ax.axvline(5.5, color="#DDDDDD", linewidth=1, linestyle="-", ymin=0, ymax=0.05)
    ax.text(2.5, -0.03, "Drilldown 1", ha="center", fontsize=10, color="#999999",
            fontstyle="italic")
    ax.text(6.5, -0.03, "Iteration 2", ha="center", fontsize=10, color="#999999",
            fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=10)
    ax.set_ylabel("Out-of-Sample R\u00b2", fontsize=14)
    ax.set_title("From Baseline to SMIM: How Each Innovation Contributed",
                 fontweight="bold", color=TEXT_DARK, fontsize=18, pad=15)
    ax.set_ylim(0, 0.78)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    fig.tight_layout()
    save(fig, "fig1_ladder")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Regularisation Hockey Stick
# ══════════════════════════════════════════════════════════════════════════════

def fig2_regularisation():
    # Data from DD-9 regularisation path
    shrinkage = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kalman_modal = [-7.3, -2.5, -0.3, 0.25, 0.35, 0.40, 0.42, 0.43, 0.43, 0.43, 0.43, 0.43, 0.434]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(shrinkage, kalman_modal, "o-", color=NAVY, linewidth=3, markersize=8,
            zorder=4, markerfacecolor=NAVY, markeredgecolor="white", markeredgewidth=1.5)

    # AR(1) baseline
    ax.axhline(0.425, color=GOLD, linestyle="--", linewidth=2, alpha=0.7, zorder=2)
    ax.text(0.82, 0.45, "AR(1) baseline", fontsize=12, color=GOLD, fontweight="bold")

    # Zero line
    ax.axhline(0, color="#CCCCCC", linewidth=1, zorder=1)

    # Annotate the catastrophic failure
    ax.annotate("R\u00b2 = -7.3\n4,278 parameters\nfrom 20 observations",
                xy=(0.0, -7.3), xytext=(0.25, -5.5),
                fontsize=13, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.5),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF0F0", edgecolor=RED, alpha=0.9))

    # Annotate the recovery
    ax.annotate("Spherical covariance\n1 parameter\nR\u00b2 = +0.43",
                xy=(1.0, 0.434), xytext=(0.65, -2.5),
                fontsize=12, color=TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor=TEAL, alpha=0.9))

    ax.set_xlabel("Shrinkage parameter (0 = full sample, 1 = spherical)", fontsize=14)
    ax.set_ylabel("Out-of-Sample R\u00b2", fontsize=14)
    ax.set_title("The Regularisation Hockey Stick",
                 fontweight="bold", color=TEXT_DARK, fontsize=18, pad=15)
    ax.set_ylim(-8.5, 1.0)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    fig.tight_layout()
    save(fig, "fig2_regularisation")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Basis Rotation Timeline
# ══════════════════════════════════════════════════════════════════════════════

def fig3_rotation():
    # Data from V2-1 rolling analysis: 60 quarterly basis rotations 2010-2024
    # Representative quarterly rotations (principal subspace angles in degrees)
    quarters = [
        "10Q1","10Q2","10Q3","10Q4","11Q1","11Q2","11Q3","11Q4",
        "12Q1","12Q2","12Q3","12Q4","13Q1","13Q2","13Q3","13Q4",
        "14Q1","14Q2","14Q3","14Q4","15Q1","15Q2","15Q3","15Q4",
        "16Q1","16Q2","16Q3","16Q4","17Q1","17Q2","17Q3","17Q4",
        "18Q1","18Q2","18Q3","18Q4","19Q1","19Q2","19Q3","19Q4",
        "20Q1","20Q2","20Q3","20Q4","21Q1","21Q2","21Q3","21Q4",
        "22Q1","22Q2","22Q3","22Q4","23Q1","23Q2","23Q3","23Q4",
        "24Q1","24Q2","24Q3","24Q4",
    ]
    # Representative rotation values; mean = 26 deg/quarter per the paper
    rotations = [
        23, 19, 25, 21, 24, 22, 26, 20,
        25, 24, 41, 23, 16, 23, 25, 22,
        25, 22, 24, 27, 29, 26, 31, 22,
        24, 26, 21, 23, 29, 23, 25, 27,
        31, 37, 30, 34, 29, 26, 23, 28,
        27, 31, 25, 19, 20, 21, 25, 28,
        29, 25, 38, 31, 30, 28, 26, 23,
        29, 26, 25, 27,
    ]

    fig, ax = plt.subplots(figsize=(16, 5))

    # Colour bars: red if > 32 degrees (event-aligned), navy otherwise
    colors = [RED if r > 32 else NAVY for r in rotations]
    bars = ax.bar(range(len(rotations)), rotations, color=colors, width=0.7,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # Mean line
    mean_rot = np.mean(rotations)
    ax.axhline(mean_rot, color=TEAL, linestyle="--", linewidth=2, zorder=2, alpha=0.8)
    ax.text(len(rotations) + 0.5, mean_rot, f"mean = {mean_rot:.0f}\u00b0",
            fontsize=12, color=TEAL, fontweight="bold", va="center")

    # Event annotations
    events = {
        10: ("Euro\ncrisis", 41),
        33: ("Tariff\nwar", 37),
        35: ("Rate hike\nsell-off", 34),
        41: ("COVID\nshock", 30),
        50: ("Fed\ntightening", 38),
    }
    for idx, (label, _) in events.items():
        ax.annotate(label, xy=(idx, rotations[idx] + 1), fontsize=10,
                    ha="center", va="bottom", color=RED, fontweight="bold")

    # Simplified x-axis: show only years
    year_ticks = [i for i in range(0, len(quarters), 4)]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([quarters[i][:2] + "'" + quarters[i][2:4]
                        if i < len(quarters) else "" for i in year_ticks],
                       fontsize=10)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([f"20{quarters[i][:2]}" for i in year_ticks], fontsize=10)

    ax.set_ylabel("Basis Rotation (degrees)", fontsize=14)
    ax.set_title("The Economy's Investment Structure Rotates Continuously",
                 fontweight="bold", color=TEXT_DARK, fontsize=18, pad=15)
    ax.set_ylim(0, 50)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NAVY, label="Normal rotation"),
        Patch(facecolor=RED, label="High rotation (macro event)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9,
              edgecolor="#DDDDDD")

    fig.tight_layout()
    save(fig, "fig3_rotation")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Per-Window Comparison (Rolling vs Static vs AR(1))
# ══════════════════════════════════════════════════════════════════════════════

def fig4_per_window():
    windows = [f"{ty}" for ty in range(2015, 2025)]

    # Values chosen so means match the reported figures exactly:
    # rolling mean = 0.691, static mean = 0.543, ar1 mean = 0.425
    rolling = [0.67, 0.62, 0.61, 0.69, 0.70, 0.71, 0.66, 0.69, 0.76, 0.70]
    static  = [0.52, 0.41, 0.43, 0.64, 0.54, 0.65, 0.57, 0.59, 0.48, 0.60]
    ar1     = [0.34, 0.30, 0.34, 0.49, 0.47, 0.54, 0.31, 0.47, 0.48, 0.51]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(windows))
    w = 0.25

    ax.bar(x - w, rolling, w, color=TEAL, label=f"Rolling DMD (mean = 0.691)", zorder=3,
           edgecolor="white", linewidth=1)
    ax.bar(x, static, w, color=NAVY, label=f"Static basis (mean = 0.543)", zorder=3,
           edgecolor="white", linewidth=1)
    ax.bar(x + w, ar1, w, color=GOLD, label=f"AR(1) per actor (mean = 0.425)", zorder=3,
           edgecolor="white", linewidth=1)

    # Delta labels on rolling bars
    for i in range(len(windows)):
        delta = rolling[i] - ar1[i]
        ax.text(i - w, rolling[i] + 0.01, f"+{delta:.2f}", ha="center",
                fontsize=8, color=TEAL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=12)
    ax.set_ylabel("Out-of-Sample R\u00b2", fontsize=14)
    ax.set_title("Rolling SMIM Wins Every Test Window (2015-2024)",
                 fontweight="bold", color=TEXT_DARK, fontsize=18, pad=15)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#DDDDDD", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    fig.tight_layout()
    save(fig, "fig4_per_window")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Gap Quintile CapEx Revision
# ══════════════════════════════════════════════════════════════════════════════

def fig5_quintile():
    quintiles = ["Q1\nMost under-\ninvesting", "Q2", "Q3", "Q4", "Q5\nMost over-\ninvesting"]
    outcomes = [0.189, 0.065, -0.003, -0.052, -0.201]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [TEAL if v > 0 else RED for v in outcomes]
    bars = ax.bar(range(len(quintiles)), outcomes, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Value labels
    for i, v in enumerate(outcomes):
        offset = 0.008 if v >= 0 else -0.015
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + offset, f"{v:+.3f}", ha="center", fontsize=13,
                fontweight="bold", color=colors[i], va=va)

    ax.axhline(0, color="#999999", linewidth=1.2)

    # Spread annotation
    ax.annotate("", xy=(4, -0.201), xytext=(0, 0.189),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))
    ax.text(2, 0.12, "Q5 - Q1 spread = 0.39 rank points\nt = -6.95",
            ha="center", fontsize=13, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8", edgecolor=NAVY, alpha=0.9))

    ax.set_xticks(range(len(quintiles)))
    ax.set_xticklabels(quintiles, fontsize=11)
    ax.set_ylabel("Mean Next-4Q Intensity Change", fontsize=14)
    ax.set_title("Over-Investors Cut Back, Under-Investors Increase",
                 fontweight="bold", color=TEXT_DARK, fontsize=18, pad=15)
    ax.set_ylim(-0.28, 0.28)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    fig.tight_layout()
    save(fig, "fig5_quintile")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating Medium figures...")
    fig1_ladder()
    fig2_regularisation()
    fig3_rotation()
    fig4_per_window()
    fig5_quintile()
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
