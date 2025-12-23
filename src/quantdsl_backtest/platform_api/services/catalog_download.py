from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List

from ...dsl.universe import Universe
from ...data.orchestrator import load_bundle, default_registry
from ...data.requests import DataRequest


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    request: DataRequest
    universe_name: Optional[str]
    dry_run: bool


def download_bundle(
    *,
    request: DataRequest,
    universe: Optional[Universe] = None,
    dry_run: bool = False,
    entities: Optional[List[str]] = None,
) -> dict:
    """Download/fill missing data for a request.

    Contract:
      - Uses the existing data provider layer (registry + cache + tail-fetch logic).
      - With dry_run=True, it performs a no-op and only returns the plan.
      - If `entities` is provided, it is applied as `Universe.static_instruments` for providers
        that respect universe selection (e.g., yf://).
    """

    if entities:
        # Universe drives provider selection logic for multi-entity sources.
        universe = Universe(
            name=getattr(universe, "name", "adhoc") if universe is not None else "adhoc",
            filters=getattr(universe, "filters", []) if universe is not None else [],
            id_field=getattr(universe, "id_field", "ticker") if universe is not None else "ticker",
            static_instruments=list(entities),
        )

    if dry_run:
        plan = None
        try:
            from quantdsl_backtest.platform_api.services.catalog_download_plan import (
                plan_download_for_request,
            )

            plan = plan_download_for_request(request=request, entities=list(entities or []))
        except Exception:
            plan = None

        return {
            "dry_run": True,
            "request": asdict(request),
            "universe": getattr(universe, "name", None) if universe is not None else None,
            "entities": list(entities) if entities else None,
            "plan": plan,
        }

    reg = default_registry()

    bundle = load_bundle(request, universe, registry=reg)

    out = {
        "dry_run": False,
        "kind": getattr(bundle, "kind", None),
        "source": getattr(bundle, "source", None),
        "start": getattr(bundle, "start", None),
        "end": getattr(bundle, "end", None),
        "frequency": getattr(bundle, "frequency", None),
    }

    # Bundle-specific summary
    if hasattr(bundle, "instruments"):
        out["entities"] = list(getattr(bundle, "instruments") or [])
    elif hasattr(bundle, "entities"):
        out["entities"] = list(getattr(bundle, "entities") or [])

    # Provider cache stats (best-effort)
    try:
        provider = reg.resolve(request)
        stats_fn = getattr(provider, "tail_cache_stats", None)
        if callable(stats_fn):
            out["cache_stats"] = stats_fn()

        per_entity_fn = getattr(provider, "last_entity_cache_stats", None)
        if callable(per_entity_fn):
            out["stats_by_entity"] = per_entity_fn()

            from quantdsl_backtest.platform_api.services.catalog_stats_classify import (
                classify_actions_by_entity,
            )

            out["actions_by_entity"] = classify_actions_by_entity(out["stats_by_entity"])
    except Exception:
        pass

    return out
