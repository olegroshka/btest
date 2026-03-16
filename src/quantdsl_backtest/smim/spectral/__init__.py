"""SMIM spectral decomposition: Schur, polar, Hermitian-dilation, DMD."""

from quantdsl_backtest.smim.spectral.base import AbstractSpectralDecomposer, SpectralDecomposerFactory
from quantdsl_backtest.smim.spectral.comparison import MethodComparisonRow, SpectralComparisonHarness
from quantdsl_backtest.smim.spectral.dmd import ExactDMDDecomposer, ExtendedDMDDecomposer
from quantdsl_backtest.smim.spectral.dv_basis import DirectedVariationDecomposer
from quantdsl_backtest.smim.spectral.hermitian import HermitianDilationDecomposer
from quantdsl_backtest.smim.spectral.mode_selection import (
    CombinedModeSelector,
    CompressibilityFilter,
    MDLModeSelector,
    RGRelevanceFilter,
    lz_complexity,
)
from quantdsl_backtest.smim.spectral.modal_report import build_modal_report, gate_g3_passes
from quantdsl_backtest.smim.spectral.oos_evaluation import (
    OOSReconstructionResult,
    rolling_oos_reconstruction,
)
from quantdsl_backtest.smim.spectral.polar import PolarDecomposer
from quantdsl_backtest.smim.spectral.schur import SchurDecomposer

__all__ = [
    "AbstractSpectralDecomposer",
    "SpectralDecomposerFactory",
    "SchurDecomposer",
    "PolarDecomposer",
    "HermitianDilationDecomposer",
    "DirectedVariationDecomposer",
    "ExactDMDDecomposer",
    "ExtendedDMDDecomposer",
    "MethodComparisonRow",
    "SpectralComparisonHarness",
    "MDLModeSelector",
    "CompressibilityFilter",
    "RGRelevanceFilter",
    "CombinedModeSelector",
    "lz_complexity",
    "OOSReconstructionResult",
    "rolling_oos_reconstruction",
    "build_modal_report",
    "gate_g3_passes",
]
