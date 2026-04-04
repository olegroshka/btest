#!/usr/bin/env python
"""Generate the T-sweep figure showing delta-R² vs training window length."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "smim" / "paper" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

ACCENT = "#133D7A"
TEAL = "#1A7A6D"
GOLD = "#C4960C"
GREY = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# Data from run_smim_iter5_1_cv2.py T-sweep output
T_years = [2, 3, 4, 5, 7]
T_quarters = [8, 12, 16, 20, 28]
mean_delta = [0.0571, 0.0335, 0.0114, -0.0050, -0.0061]
ci_lo = [0.0415, 0.0210, 0.0002, -0.0128, -0.0152]
ci_hi = [0.0740, 0.0473, 0.0219, 0.0036, 0.0037]
wins = [10, 10, 8, 3, 4]
perm_p = [0.0013, 0.0013, 0.0425, 0.8619, 0.8664]


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.array(T_years)
    y = np.array(mean_delta) * 100  # convert to percentage points
    lo = np.array(ci_lo) * 100
    hi = np.array(ci_hi) * 100
    err_lo = y - lo
    err_hi = hi - y

    # Color bars by significance
    colors = [TEAL if p < 0.05 else GREY for p in perm_p]

    bars = ax.bar(x, y, width=0.6, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
                elinewidth=1.5, capsize=5, capthick=1.5, zorder=5)

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    # Annotate wins and p-values
    for i, (xi, yi, wi, pi) in enumerate(zip(x, y, wins, perm_p)):
        label = f"{wi}/10"
        if pi < 0.01:
            label += f"\n$p={pi:.3f}$"
        elif pi < 0.05:
            label += f"\n$p={pi:.2f}$"
        ypos = hi[i] + 0.3 if yi > 0 else lo[i] - 0.3
        va = "bottom" if yi > 0 else "top"
        ax.text(xi, ypos, label, ha="center", va=va, fontsize=8, color=colors[i],
                fontweight="bold")

    # Arrow annotation for the cross-sectional pooling story
    ax.annotate(
        "Cross-sectional\npooling advantage",
        xy=(2, mean_delta[0] * 100), xytext=(4.5, 6.5),
        fontsize=9, color=TEAL, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5),
    )

    # Grey bars + legend + win counts already communicate non-significance

    ax.set_xlabel("Training window $T$ (years)", fontsize=11)
    ax.set_ylabel("$\\Delta R^2$ (SMIM $-$ AR(1), percentage points)", fontsize=11)
    ax.set_xticks(T_years)
    ax.set_xticklabels([f"{t}yr\n({q}Q)" for t, q in zip(T_years, T_quarters)])
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(-2.5, 9.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAL, alpha=0.85, label="Significant ($p < 0.05$)"),
        Patch(facecolor=GREY, alpha=0.85, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    fig.savefig(IMG_DIR / "fig13_tsweep.pdf")
    fig.savefig(IMG_DIR / "fig13_tsweep.png")
    plt.close(fig)
    print("  fig13_tsweep.pdf/png")


if __name__ == "__main__":
    main()
