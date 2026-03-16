"""
Persistence helpers for edge matrices and aggregate operators.

M2.1-T8
"""

from __future__ import annotations

from pathlib import Path

import scipy.sparse as sp
from scipy.sparse import csr_matrix, load_npz, save_npz

from quantdsl_backtest.smim.interfaces import RelationChannel


def save_edges(
    edges: dict[RelationChannel, csr_matrix],
    path: Path,
    period: str,
) -> None:
    """Save per-channel edge matrices as .npz files.

    Creates ``{path}/{period}/edges_{channel.value}.npz`` for each channel.

    Args:
        edges: Mapping from RelationChannel to its csr_matrix.
        path: Root directory.
        period: Sub-directory label (e.g., "2023Q1").
    """
    out_dir = Path(path) / period
    out_dir.mkdir(parents=True, exist_ok=True)
    for channel, matrix in edges.items():
        file_path = out_dir / f"edges_{channel.value}.npz"
        save_npz(str(file_path), matrix)


def load_edges(path: Path, period: str) -> dict[RelationChannel, csr_matrix]:
    """Load per-channel edge matrices from .npz files.

    Args:
        path: Root directory.
        period: Sub-directory label matching what was used in save_edges.

    Returns:
        Mapping from RelationChannel to csr_matrix.
    """
    out_dir = Path(path) / period
    result: dict[RelationChannel, csr_matrix] = {}
    for channel in RelationChannel:
        file_path = out_dir / f"edges_{channel.value}.npz"
        if file_path.exists():
            result[channel] = csr_matrix(load_npz(str(file_path)))
    return result


def save_operator(op: csr_matrix, path: Path, period: str) -> None:
    """Save aggregate operator matrix to {path}/{period}/operator.npz.

    Args:
        op: Aggregate operator csr_matrix.
        path: Root directory.
        period: Sub-directory label.
    """
    out_dir = Path(path) / period
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / "operator.npz"
    save_npz(str(file_path), op)


def load_operator(path: Path, period: str) -> csr_matrix:
    """Load aggregate operator matrix from {path}/{period}/operator.npz.

    Args:
        path: Root directory.
        period: Sub-directory label.

    Returns:
        Loaded csr_matrix.
    """
    file_path = Path(path) / period / "operator.npz"
    return csr_matrix(load_npz(str(file_path)))
