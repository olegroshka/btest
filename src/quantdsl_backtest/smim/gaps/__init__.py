"""SMIM gap computation: benchmark construction and Δ_{i,t} calculation.

Public API
----------
``BenchmarkFactory``  — maps ``BenchmarkClass`` → ``BenchmarkComputer`` instance.
``get_benchmark(cls)`` — convenience wrapper around the default factory.

During WP0 all factory entries are placeholder implementations that raise
``NotImplementedError``.  Concrete implementations are registered as each
WP milestone completes (see ``docs/smim/benchmark_specs.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantdsl_backtest.smim.interfaces import (
    Actor,
    ActorRegistry,
    BenchmarkClass,
    BenchmarkComputer,
    FilteredState,
    GapResult,
    ModalFrame,
)


# ═══════════════════════════════════════════════════════════
# Placeholder implementation
# ═══════════════════════════════════════════════════════════

class _PlaceholderBenchmark:
    """Placeholder that raises NotImplementedError on compute().

    Satisfies the BenchmarkComputer protocol structurally so type-checkers
    are happy, but signals clearly that the implementation is pending.
    """

    def __init__(self, cls: BenchmarkClass) -> None:
        self._cls = cls

    @property
    def benchmark_class(self) -> BenchmarkClass:
        return self._cls

    def compute(
        self,
        observations: NDArray[np.floating],
        filtered_state: FilteredState,
        modal_frame: ModalFrame,
        actors: ActorRegistry,
    ) -> GapResult:
        raise NotImplementedError(
            f"Benchmark {self._cls.value!r} is not yet implemented. "
            f"See docs/smim/benchmark_specs.md for the target milestone."
        )


# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════

class BenchmarkFactory:
    """Registry mapping BenchmarkClass → BenchmarkComputer.

    Usage::

        factory = BenchmarkFactory()
        computer = factory.get(BenchmarkClass.PREDICTIVE)
        gap = computer.compute(observations, filtered_state, modal_frame, actors)

    The default factory populates all five benchmark classes with
    ``_PlaceholderBenchmark`` instances.  Concrete implementations are
    registered via ``register()`` as each WP milestone completes.
    """

    def __init__(self) -> None:
        self._registry: dict[BenchmarkClass, BenchmarkComputer] = {
            cls: _PlaceholderBenchmark(cls)  # type: ignore[misc]
            for cls in BenchmarkClass
        }

    def get(self, cls: BenchmarkClass) -> BenchmarkComputer:
        """Return the BenchmarkComputer for the given BenchmarkClass.

        Args:
            cls: One of the BenchmarkClass enum values.

        Returns:
            The registered BenchmarkComputer instance.

        Raises:
            KeyError: If cls is not registered (should never happen with the
                default factory, which pre-registers all five classes).
        """
        return self._registry[cls]

    def register(self, cls: BenchmarkClass, computer: BenchmarkComputer) -> None:
        """Replace the registered BenchmarkComputer for cls.

        Args:
            cls: BenchmarkClass enum value.
            computer: Object satisfying the BenchmarkComputer protocol.
        """
        self._registry[cls] = computer

    def registered_classes(self) -> list[BenchmarkClass]:
        """Return all registered BenchmarkClass values."""
        return list(self._registry)


# ═══════════════════════════════════════════════════════════
# Module-level default factory
# ═══════════════════════════════════════════════════════════

_default_factory = BenchmarkFactory()


def get_benchmark(cls: BenchmarkClass) -> BenchmarkComputer:
    """Return the default-factory BenchmarkComputer for cls.

    Args:
        cls: BenchmarkClass enum value.

    Returns:
        Registered BenchmarkComputer (placeholder until WP4).
    """
    return _default_factory.get(cls)
