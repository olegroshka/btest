from __future__ import annotations

from quantdsl_backtest.data.sources.cache import TailCacheStats


def test_tail_cache_stats_snapshot_and_delta():
    s = TailCacheStats(reads=1, writes=2, hits=3, misses=4, tail_fetches=5)
    before = s.snapshot()

    # mutate
    s.reads += 10
    s.hits += 1

    after = s.snapshot()
    d = after.delta(before)

    assert d.reads == 10
    assert d.hits == 1
    assert d.writes == 0

