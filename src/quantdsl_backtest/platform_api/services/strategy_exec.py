from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

from quantdsl_backtest.dsl.strategy import Strategy


@dataclass(frozen=True)
class StrategyBuildResult:
    strategy: Strategy


def build_strategy_from_source(*, source: str, strategy_id: str) -> StrategyBuildResult:
    """Execute a strategy source snapshot and extract a Strategy instance.

    Contract (MVP):
      - Strategy source must define either:
          - `strategy` global variable, or
          - `build_strategy()` function returning a Strategy

    This will evolve into a stricter notebook/DSL compilation step later.
    """

    m = types.ModuleType(f"strategy_{strategy_id}")
    g: dict[str, Any] = m.__dict__
    exec(compile(source, filename=f"{strategy_id}.py", mode="exec"), g, g)

    if isinstance(g.get("strategy"), Strategy):
        strat = g["strategy"]
    elif callable(g.get("build_strategy")):
        strat = g["build_strategy"]()
        if not isinstance(strat, Strategy):
            raise TypeError("build_strategy() must return a Strategy")
    else:
        raise ValueError("Strategy source must define `strategy` or `build_strategy()`")

    # Ensure stable name
    try:
        if not getattr(strat, "name", None):
            strat.name = strategy_id  # type: ignore[attr-defined]
    except Exception:
        pass

    return StrategyBuildResult(strategy=strat)

