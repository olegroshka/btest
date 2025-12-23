from __future__ import annotations

from typing import Any, Dict, Optional


def classify_entity_action(stats: Dict[str, Any]) -> Optional[str]:
    """Classify a per-entity cache action based on TailCacheStats delta.

    Heuristics (mutually exclusive preference):
      - if tail_fetches > 0 -> 'tail_fetch'
      - elif hits > 0 -> 'cache_hit'
      - elif misses > 0 -> 'full_fetch'
      - else -> None

    Notes:
      - For some providers, a miss implies a full fetch. For tail fetch paths, the loader
        will record tail_fetches.
      - Values may be JSON-decoded strings; we coerce via int().
    """

    if not stats:
        return None

    def _to_int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    tail = _to_int(stats.get("tail_fetches", 0))
    hits = _to_int(stats.get("hits", 0))
    misses = _to_int(stats.get("misses", 0))

    if tail > 0:
        return "tail_fetch"
    if hits > 0:
        return "cache_hit"
    if misses > 0:
        return "full_fetch"
    return None


def classify_actions_by_entity(stats_by_entity: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ent, stats in (stats_by_entity or {}).items():
        action = classify_entity_action(stats)
        if action is not None:
            out[str(ent)] = action
    return out
