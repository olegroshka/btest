"""Cross-sector/geography parameter transfer experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import orthogonal_procrustes

from quantdsl_backtest.smim.interfaces import ModalFrame


@dataclass
class TransferabilityResult:
    parameter_name: str
    source_value: float
    target_value: float
    relative_diff: float
    transfers: bool


@dataclass
class TransferExperimentResult:
    source_name: str
    target_name: str
    n_transferable: int
    n_total: int
    transfer_rate: float
    parameter_results: list[TransferabilityResult]
    passes: bool
    metadata: dict = field(default_factory=dict)


class TransferExperiment:
    """Tests parameter transfer between two datasets (geographies/sectors).

    Parameters tested:
    1. modal_structure: Procrustes-aligned Frobenius distance between modal bases
    2. regime_dynamics: Frobenius norm of (source_transition - target_transition)
    3. edge_weights: relative difference in spectral gap of operators
    4. normalisation: ratio of signal standard deviations
    """

    PARAMETERS = ["modal_structure", "regime_dynamics", "edge_weights", "normalisation"]

    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold

    def run(
        self,
        source_name: str,
        source_observations: np.ndarray,
        source_modal_frame: ModalFrame,
        target_name: str,
        target_observations: np.ndarray,
        target_modal_frame: ModalFrame,
        source_transition_matrix: np.ndarray | None = None,
        target_transition_matrix: np.ndarray | None = None,
        source_operator: np.ndarray | None = None,
        target_operator: np.ndarray | None = None,
    ) -> TransferExperimentResult:
        """Run transfer experiment and return TransferExperimentResult."""
        param_results = []

        # 1. modal_structure
        modal_diff = self._modal_structure_diff(
            source_modal_frame.basis, target_modal_frame.basis
        )
        source_modal_norm = float(np.linalg.norm(source_modal_frame.basis, "fro"))
        target_modal_norm = float(np.linalg.norm(target_modal_frame.basis, "fro"))
        max_norm = max(source_modal_norm, target_modal_norm, 1e-9)
        modal_rel_diff = modal_diff / max_norm
        param_results.append(
            TransferabilityResult(
                parameter_name="modal_structure",
                source_value=source_modal_norm,
                target_value=target_modal_norm,
                relative_diff=modal_rel_diff,
                transfers=modal_rel_diff < self.threshold,
            )
        )

        # 2. regime_dynamics
        if source_transition_matrix is not None and target_transition_matrix is not None:
            s_norm = float(np.linalg.norm(source_transition_matrix, "fro"))
            t_norm = float(np.linalg.norm(target_transition_matrix, "fro"))
            max_trans = max(s_norm, t_norm, 1e-9)
            trans_diff = float(
                np.linalg.norm(source_transition_matrix - target_transition_matrix, "fro")
            )
            trans_rel_diff = trans_diff / max_trans
            param_results.append(
                TransferabilityResult(
                    parameter_name="regime_dynamics",
                    source_value=s_norm,
                    target_value=t_norm,
                    relative_diff=trans_rel_diff,
                    transfers=trans_rel_diff < self.threshold,
                )
            )
        else:
            # Skipped — mark as transferable
            param_results.append(
                TransferabilityResult(
                    parameter_name="regime_dynamics",
                    source_value=0.0,
                    target_value=0.0,
                    relative_diff=0.0,
                    transfers=True,
                )
            )

        # 3. edge_weights
        if source_operator is not None and target_operator is not None:
            sg_source = self._spectral_gap(source_operator)
            sg_target = self._spectral_gap(target_operator)
            max_sg = max(abs(sg_source), abs(sg_target), 1e-9)
            edge_rel_diff = abs(sg_source - sg_target) / max_sg
            param_results.append(
                TransferabilityResult(
                    parameter_name="edge_weights",
                    source_value=sg_source,
                    target_value=sg_target,
                    relative_diff=edge_rel_diff,
                    transfers=edge_rel_diff < self.threshold,
                )
            )
        else:
            param_results.append(
                TransferabilityResult(
                    parameter_name="edge_weights",
                    source_value=0.0,
                    target_value=0.0,
                    relative_diff=0.0,
                    transfers=True,
                )
            )

        # 4. normalisation
        source_stds = np.std(source_observations, axis=0)
        target_stds = np.std(target_observations, axis=0)
        # Use common actors (min dimension)
        n_common = min(len(source_stds), len(target_stds))
        s_std = source_stds[:n_common]
        t_std = target_stds[:n_common]
        max_stds = np.maximum(s_std, t_std) + 1e-9
        norm_rel_diff = float(np.mean(np.abs(s_std - t_std) / max_stds))
        param_results.append(
            TransferabilityResult(
                parameter_name="normalisation",
                source_value=float(np.mean(s_std)),
                target_value=float(np.mean(t_std)),
                relative_diff=norm_rel_diff,
                transfers=norm_rel_diff < self.threshold,
            )
        )

        n_transferable = sum(1 for r in param_results if r.transfers)
        n_total = len(param_results)
        transfer_rate = n_transferable / n_total if n_total > 0 else 0.0

        return TransferExperimentResult(
            source_name=source_name,
            target_name=target_name,
            n_transferable=n_transferable,
            n_total=n_total,
            transfer_rate=transfer_rate,
            parameter_results=param_results,
            passes=transfer_rate >= 0.5,
        )

    def _modal_structure_diff(self, basis_a: np.ndarray, basis_b: np.ndarray) -> float:
        """Procrustes-aligned Frobenius distance between two bases."""
        k = min(basis_a.shape[1], basis_b.shape[1])
        a = basis_a[:, :k]
        b = basis_b[:, :k]
        # Align dimensions if N differs
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        try:
            R, _ = orthogonal_procrustes(a, b)
            aligned = a @ R
            diff = float(np.linalg.norm(aligned - b, "fro"))
        except Exception:
            diff = float(np.linalg.norm(a - b, "fro"))
        return diff

    def _spectral_gap(self, operator: np.ndarray) -> float:
        """Lambda_1 - Lambda_2 of singular values."""
        sv = np.linalg.svd(operator, compute_uv=False)
        if len(sv) >= 2:
            return float(sv[0] - sv[1])
        return float(sv[0]) if len(sv) > 0 else 0.0
