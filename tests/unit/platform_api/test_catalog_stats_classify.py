from __future__ import annotations

from quantdsl_backtest.platform_api.services.catalog_stats_classify import (
    classify_actions_by_entity,
    classify_entity_action,
)


def test_classify_entity_action_priority_tail_over_hit_over_miss():
    assert classify_entity_action({"tail_fetches": 1, "hits": 1, "misses": 1}) == "tail_fetch"
    assert classify_entity_action({"hits": 1, "misses": 1}) == "cache_hit"
    assert classify_entity_action({"misses": 1}) == "full_fetch"
    assert classify_entity_action({}) is None


def test_classify_entity_action_ignores_reads_writes_only():
    # reads/writes are useful diagnostics but do not define the high-level action
    assert classify_entity_action({"reads": 1, "writes": 1}) is None


def test_classify_entity_action_handles_string_numbers():
    # defensive: allow stats values that come from JSON-ish decoding
    stats1: dict[str, object] = {"misses": "1"}
    stats2: dict[str, object] = {"hits": "2"}
    assert classify_entity_action(stats1) == "full_fetch"
    assert classify_entity_action(stats2) == "cache_hit"


def test_classify_actions_by_entity_filters_unknowns():
    out = classify_actions_by_entity({"A": {"misses": 1}, "B": {}})
    assert out == {"A": "full_fetch"}
