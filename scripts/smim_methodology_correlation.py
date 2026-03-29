"""Compute per-actor Spearman correlation between capex_assets_xsrank and
return_12m_xsrank intensities for all US equity universes.

Reads:
  data/smim/intensities/{universe_id}_intensities.parquet        (capex)
  data/smim/intensities/{universe_id}_return_intensities.parquet (return)

Writes:
  docs/smim/reports/intensity_methodology_correlation.md

Decision gate: if US-LC median Spearman rho(capex, return) >= 0.4 then
cross-geography experiments (C4) are analytically defensible.

Usage:
    uv run python scripts/smim_methodology_correlation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INTENSITY_DIR = _ROOT / "data" / "smim" / "intensities"
REPORT_DIR    = _ROOT / "docs" / "smim" / "reports"

# US universes — UK already uses return as primary method
US_UNIVERSES = [
    "US-LC",
    "US-LC-ENERGY",
    "US-LC-TECH",
    "US-LC-FINS",
    "US-LC-HEALTH",
    "US-LC-INDUS",
    "US-MC",
    "US-SC",
]
MIN_OVERLAPPING_QUARTERS = 8


def _compute_universe_correlation(uni_id: str) -> dict | None:
    capex_path  = INTENSITY_DIR / f"{uni_id}_intensities.parquet"
    return_path = INTENSITY_DIR / f"{uni_id}_return_intensities.parquet"

    if not capex_path.exists():
        log.warning("  Missing capex file: %s", capex_path.name)
        return None
    if not return_path.exists():
        log.warning("  Missing return file: %s", return_path.name)
        return None

    df_c = pd.read_parquet(capex_path)[["actor_id", "period", "intensity_value"]].rename(
        columns={"intensity_value": "capex"}
    )
    df_r = pd.read_parquet(return_path)[["actor_id", "period", "intensity_value"]].rename(
        columns={"intensity_value": "return_"}
    )

    merged = df_c.merge(df_r, on=["actor_id", "period"], how="inner")

    actor_rhos: list[float] = []
    n_positive = 0
    n_negative = 0

    for actor_id, grp in merged.groupby("actor_id"):
        grp = grp.dropna(subset=["capex", "return_"])
        if len(grp) < MIN_OVERLAPPING_QUARTERS:
            continue
        rho, _ = spearmanr(grp["capex"], grp["return_"])
        if not np.isnan(rho):
            actor_rhos.append(rho)
            if rho > 0.4:
                n_positive += 1
            if rho <= 0.0:
                n_negative += 1

    if not actor_rhos:
        return None

    arr = np.array(actor_rhos)
    return {
        "n_actors": len(arr),
        "median_rho": float(np.median(arr)),
        "p25_rho": float(np.percentile(arr, 25)),
        "p75_rho": float(np.percentile(arr, 75)),
        "n_rho_gt_04": n_positive,
        "n_rho_le_00": n_negative,
        "pct_rho_gt_04": n_positive / len(arr),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for uni_id in US_UNIVERSES:
        log.info("Processing %s …", uni_id)
        r = _compute_universe_correlation(uni_id)
        if r is not None:
            results[uni_id] = r
            log.info(
                "  N=%d actors | median ρ=%.3f [%.3f, %.3f] | "
                "ρ>0.4: %d/%d (%.0f%%)",
                r["n_actors"], r["median_rho"],
                r["p25_rho"], r["p75_rho"],
                r["n_rho_gt_04"], r["n_actors"],
                r["pct_rho_gt_04"] * 100,
            )

    # Decision gate: based on US-LC
    uslc = results.get("US-LC")
    if uslc is None:
        c4_recommendation = "UNKNOWN — US-LC data not available"
        c4_defensible = False
    elif uslc["median_rho"] >= 0.4:
        c4_defensible = True
        c4_recommendation = (
            f"**C4 DEFENSIBLE WITH DISCLOSURE NOTE** — "
            f"US-LC median ρ(capex, return) = {uslc['median_rho']:.3f} ≥ 0.4. "
            f"The two intensity proxies are moderately correlated. "
            f"C4a (return vs return) is the primary cross-geography experiment; "
            f"C4b (capex vs return) is the robustness check."
        )
    else:
        c4_defensible = False
        c4_recommendation = (
            f"**C4 REQUIRES HOMOGENEOUS METHODOLOGY** — "
            f"US-LC median ρ(capex, return) = {uslc['median_rho']:.3f} < 0.4. "
            f"CapEx and return intensities diverge too much for cross-geography "
            f"comparison. Run C4a only (return vs return) until a Companies House "
            f"CapEx adapter is built for UK equities."
        )

    # Render markdown report
    lines = [
        "# SMIM Intensity Methodology Correlation Report",
        "",
        "> Generated: 2026-03-28",
        "> Methodology A: `capex_assets_xsrank` (CapEx/Assets, EDGAR, US only)",
        "> Methodology B: `return_12m_xsrank` (12-month price return, OHLCV)",
        "> Per-actor Spearman ρ across shared quarterly periods (min 8 quarters)",
        "",
        "## Per-Universe Summary",
        "",
        "| Universe | N actors | Median ρ | P25–P75 | ρ > 0.4 | ρ ≤ 0 |",
        "|----------|----------|----------|---------|---------|-------|",
    ]
    for uni_id in US_UNIVERSES:
        r = results.get(uni_id)
        if r is None:
            lines.append(f"| {uni_id} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {uni_id} | {r['n_actors']} "
                f"| {r['median_rho']:.3f} "
                f"| [{r['p25_rho']:.3f}, {r['p75_rho']:.3f}] "
                f"| {r['n_rho_gt_04']}/{r['n_actors']} ({r['pct_rho_gt_04']:.0%}) "
                f"| {r['n_rho_le_00']} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **ρ > 0.4**: the two methodologies agree on relative actor ranking — "
        "cross-geography comparison is defensible.",
        "- **ρ ≤ 0**: methodologies are uncorrelated or inversely correlated — "
        "inclusion in cross-geography experiments would be misleading.",
        "- The decision gate is based on **US-LC median ρ** (largest, most diverse universe).",
        "",
        "## C4 Decision Gate",
        "",
        c4_recommendation,
        "",
        "## Experiment Variants",
        "",
        "| Variant | US intensity | UK intensity | Type |",
        "|---------|-------------|-------------|------|",
        "| C4a (primary) | `return_12m_xsrank` | `return_12m_xsrank` | Homogeneous |",
        "| C4b (robustness) | `capex_assets_xsrank` | `return_12m_xsrank` | Heterogeneous |",
        "",
        "| Variant | US universe | US intensity | SC universe | SC intensity | Type |",
        "|---------|------------|-------------|------------|-------------|------|",
        "| C3a (primary) | US-LC | `capex_assets_xsrank` | US-SC_trimmed | `capex_assets_xsrank` | Homogeneous |",
        "| C3b (robustness) | US-LC | `return_12m_xsrank` | US-SC_trimmed | `return_12m_xsrank` | Homogeneous |",
    ]

    report_md = "\n".join(lines) + "\n"
    out_path = REPORT_DIR / "intensity_methodology_correlation.md"
    out_path.write_text(report_md, encoding="utf-8")
    log.info("Report written → %s", out_path.name)
    uslc_rho_str = f"{uslc['median_rho']:.3f}" if uslc else "n/a"
    log.info("C4 defensible: %s | US-LC median rho: %s", c4_defensible, uslc_rho_str)
    log.info("Report: %s", out_path.relative_to(_ROOT))


if __name__ == "__main__":
    main()
