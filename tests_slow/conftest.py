from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_arctic_cache_for_slow_tests(tmp_path_factory: pytest.TempPathFactory):
    """Isolate ArcticDB LMDB cache per test session.

    Why:
      - Slow tests are often run with xdist (parallel).
      - The default cache location is repo-local `local_cache/`, which is not safe for
        concurrent writers on Windows LMDB.

    We set QUANTDSL_ARCTIC_URI to a temp folder under pytest's base temp dir.
    This keeps slow tests hermetic and avoids corrupting the developer cache.

    Note: unit tests already avoid heavy cache usage; we keep this constrained to tests_slow.
    """

    # Respect explicit override (e.g., debugging a real cache)
    if os.environ.get("QUANTDSL_ARCTIC_URI"):
        return

    root = tmp_path_factory.mktemp("arctic_slow")

    # If xdist is enabled, isolate per worker too
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        root = Path(root) / worker
        root.mkdir(parents=True, exist_ok=True)

    os.environ["QUANTDSL_ARCTIC_URI"] = f"lmdb://{root.as_posix()}"


def pytest_collection_modifyitems(config, items):
    """Ensure 'manual' tests are skipped unless explicitly requested via -m manual."""
    marker_expr = config.getoption("-m")
    if marker_expr and "manual" in marker_expr:
        return

    skip_manual = pytest.mark.skip(reason="manually triggered test; use -m manual to run")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)

