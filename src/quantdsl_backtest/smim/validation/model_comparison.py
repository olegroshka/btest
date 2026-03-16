"""Model comparison suite for SMIM Gate G5 evaluation.

M5.5-T1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantdsl_backtest.smim.validation.metrics import (
    diebold_mariano_test,
    oos_r_squared,
    rmse,
    spearman_rho,
)


@dataclass
class ModelComparisonResult:
    """Predictions and metrics for one model."""

    model_name: str
    r2: float
    rmse: float
    spearman_rho: float
    errors: np.ndarray  # residuals, for DM test


@dataclass
class ComparisonTableRow:
    """Pairwise DM test comparison between two models."""

    model_a: str
    model_b: str
    dm_stat: float
    dm_pvalue: float
    r2_diff: float  # model_a.r2 - model_b.r2
    winner: str  # "A", "B", or "tie"


class ModelComparisonSuite:
    """Compare SMIM against baseline models using OOS metrics and DM tests.

    Baseline names correspond to common comparison models.
    """

    BASELINE_NAMES = [
        "historical_mean",
        "random_walk",
        "dfm",
        "graph_free",
        "symmetric_laplacian",
    ]
    SMIM_NAME = "smim"

    def __init__(self) -> None:
        self._results: dict[str, ModelComparisonResult] = {}

    def add_result(
        self,
        name: str,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        """Register a model's predictions.

        Args:
            name: Model name (use SMIM_NAME or one of BASELINE_NAMES).
            predicted: Predicted values (T,).
            actual: Actual values (T,).
        """
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        errors = actual - predicted

        self._results[name] = ModelComparisonResult(
            model_name=name,
            r2=oos_r_squared(predicted, actual),
            rmse=rmse(predicted, actual),
            spearman_rho=spearman_rho(predicted, actual),
            errors=errors,
        )

    def compare_all(
        self,
    ) -> tuple[pd.DataFrame, list[ComparisonTableRow]]:
        """Compare SMIM against all registered baselines.

        Returns:
            - metrics_df: DataFrame with columns [model, r2, rmse, spearman_rho]
            - comparisons: List of pairwise DM test results (SMIM vs each baseline)
        """
        # Build metrics table
        rows = []
        for name, result in self._results.items():
            rows.append({
                "model": name,
                "r2": result.r2,
                "rmse": result.rmse,
                "spearman_rho": result.spearman_rho,
            })
        metrics_df = pd.DataFrame(rows)

        # Pairwise DM tests: SMIM vs each baseline
        comparisons = []
        smim = self._results.get(self.SMIM_NAME)
        if smim is None:
            return metrics_df, comparisons

        for baseline_name, baseline in self._results.items():
            if baseline_name == self.SMIM_NAME:
                continue

            dm_stat, dm_pval = diebold_mariano_test(
                smim.errors, baseline.errors
            )
            r2_diff = smim.r2 - baseline.r2

            # Determine winner
            if dm_pval < 0.05:
                winner = "A" if dm_stat < 0 else "B"
            else:
                winner = "tie"

            comparisons.append(ComparisonTableRow(
                model_a=self.SMIM_NAME,
                model_b=baseline_name,
                dm_stat=dm_stat,
                dm_pvalue=dm_pval,
                r2_diff=r2_diff,
                winner=winner,
            ))

        return metrics_df, comparisons

    def smim_beats_baselines(self, alpha: float = 0.05) -> bool:
        """True if SMIM significantly beats at least 3 baselines by DM test.

        "Beats" means DM test is significant (p < alpha) AND SMIM has lower
        error (negative dm_stat after correcting for direction: SMIM is model A,
        lower error means d_t = e_smim^2 - e_baseline^2 < 0 → negative stat).

        Args:
            alpha: Significance level for DM test.

        Returns:
            True if SMIM beats ≥3 baselines.
        """
        _, comparisons = self.compare_all()
        beat_count = sum(
            1 for c in comparisons
            if c.dm_pvalue < alpha and c.winner == "A"
        )
        return beat_count >= 3
